import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable

import httpx2
import numpy as np
from mcp.client.auth import OAuthClientProvider
from mcp.types import Tool
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelRequest,
    ModelSettings,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    TextPartDelta,
    UsageLimits,
    UserPromptPart,
)
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider
from pydantic_ai.toolsets import AbstractToolset

from sirin.config import LLMConfig, McpServerConfig
from sirin.isolation import IsolatedToolset
from sirin.oauth import OAuthManager, OAuthRequired
from sirin.prompt import compile_prompt
from sirin.status import ItemStatus

_log = logging.getLogger(__name__)

# Tool-call rounds before the agent gives up on an answer.
MAX_TOOL_ROUNDS = 8
LLM_TIMEOUT_S = 60
# User turns of a dialog that the model sees again; older ones are forgotten.
MAX_HISTORY_TURNS = 20
# Pause before an MCP server that failed or was lost is tried again; it doubles up to the maximum.
RECONNECT_MIN_S = 2.0
RECONNECT_MAX_S = 60.0
# What the voice tag of a user turn says when the voice is none of the remembered ones.
UNKNOWN_VOICE = "not recognized"
# Added to the system prompt while voices are recognized. The name is in the turn and not here because the
# people of one dialog can differ from turn to turn, and a model reads a fact next to the question more
# reliably than one at the end of a long prompt.
VOICE_INSTRUCTIONS = (
    "Each user turn starts with a tag such as [Voice: Anna]: the name of the person whose voice said it, as "
    f"recognized from the audio, or [Voice: {UNKNOWN_VOICE}] for a voice that is not remembered. The tag is "
    "reliable: trust it over your own guesses about who speaks. Never say the tag aloud or write one yourself."
)


class TurnDeps(BaseModel):
    """Who the agent is answering; tools that act on the device or the speaker read it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: str
    # The name of the remembered voice that speaks, when it was recognized.
    speaker: str | None = None
    # The speech of this turn as 16 kHz floats; None for a turn nobody spoke, such as a reminder.
    speech: np.ndarray | None = None

    @property
    def voice(self) -> str:
        """Whose voice speaks, as the model is told."""
        return self.speaker or UNKNOWN_VOICE

    def tagged(self, text: str) -> str:
        """text as the model gets it: with whose voice said it, when voices are recognized."""
        if self.speech is None or not len(self.speech):
            return text
        return f"[Voice: {self.voice}] {text}"


class Agent:
    """LLM with MCP tools; streams the spoken part of the answer as text deltas."""

    def __init__(
        self,
        cfg: LLMConfig,
        mcp_servers: list[McpServerConfig],
        oauth: OAuthManager,
        on_status: Callable[[], None],
        builtin_tools: list[AbstractToolset[TurnDeps]],
    ) -> None:
        self._cfg = cfg
        self.error_reply = cfg.error_reply
        self._on_status = on_status
        self._prompt = compile_prompt(cfg.system_prompt)
        self._oauth = oauth
        self._mcp_servers = mcp_servers
        self._builtin_tools = builtin_tools
        self._history: dict[str, tuple[float, list[ModelMessage]]] = {}
        # The toolsets of the servers connected now, by server name.
        self._live: dict[str, AbstractToolset[TurnDeps]] = {}
        self._holders: list[asyncio.Task[None]] = []
        self.tool_status: dict[str, ItemStatus] = {}
        provider = OpenAICompatibleProvider("local", base_url=cfg.base_url, api_key=cfg.api_key or "none")
        self._agent = PydanticAgent(
            OpenAIChatModel(cfg.model, provider=provider),
            deps_type=TurnDeps,
            instructions=self._render_prompt,
            model_settings=ModelSettings(temperature=cfg.temperature, timeout=LLM_TIMEOUT_S),
        )

    async def start(self) -> None:
        """Returns once every server was tried, so the first dialog has the tools of the servers that are up."""
        tried = [asyncio.Event() for _ in self._mcp_servers]
        self._holders = [
            asyncio.create_task(self._hold(server, event))
            for server, event in zip(self._mcp_servers, tried, strict=True)
        ]
        await asyncio.gather(*(event.wait() for event in tried))

    async def _hold(self, server: McpServerConfig, tried: asyncio.Event) -> None:
        """Keeps one server connected until close(): a failed or lost connection is opened again after a pause."""
        self._set_status(server.name, ItemStatus(state="pending", detail="connecting"))
        delay = RECONNECT_MIN_S
        while True:
            lost: asyncio.Queue[Exception] = asyncio.Queue()
            try:
                async with build_toolset(server, self._oauth, on_lost=lost.put_nowait) as toolset:
                    count = len(await toolset.list_tools())
                    self._live[server.name] = toolset.prefixed(server.name)
                    _log.info("MCP server %s: %d tools", server.name, count)
                    self._set_status(server.name, ItemStatus(state="ok", detail=f"{count} tools"))
                    tried.set()
                    delay = RECONNECT_MIN_S
                    reason = f"connection lost: {await lost.get()}"
            except Exception as exc:
                reason = await explain_connect_error(server, exc)
            finally:
                self._live.pop(server.name, None)
            if self._set_status(server.name, ItemStatus(state="error", detail=reason)):
                _log.error(
                    "MCP server %s is unavailable, its tools are skipped until it answers: %s", server.name, reason
                )
            tried.set()
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_S)

    def _set_status(self, name: str, status: ItemStatus) -> bool:
        """True if the status changed."""
        if self.tool_status.get(name) == status:
            return False
        self.tool_status[name] = status
        self._on_status()
        return True

    def _render_prompt(self, ctx: RunContext[TurnDeps]) -> str:
        prompt = self._prompt.render()
        return prompt if ctx.deps.speech is None else f"{prompt}\n\n{VOICE_INSTRUCTIONS}"

    async def close(self) -> None:
        for holder in self._holders:
            holder.cancel()
        await asyncio.gather(*self._holders, return_exceptions=True)

    async def reply(self, conversation_id: str, text: str, turn: TurnDeps) -> str:
        """The whole answer at once, for speech nobody is waiting on."""
        return "".join([delta async for delta in self.stream_reply(conversation_id, text, turn)]).strip()

    async def stream_reply(self, conversation_id: str, text: str, turn: TurnDeps) -> AsyncIterator[str]:
        async with self._agent.run_stream_events(
            turn.tagged(text),
            message_history=self._messages(conversation_id),
            deps=turn,
            usage_limits=UsageLimits(request_limit=MAX_TOOL_ROUNDS + 1),
            # The built-in tools go last: their descriptions carry the current time, and an LLM server reuses its
            # prompt cache only up to the first changed token.
            toolsets=[*self._live.values(), *self._builtin_tools],
        ) as events:
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    if event.part.content:
                        yield event.part.content
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    if event.delta.content_delta:
                        yield event.delta.content_delta
                elif isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
                    # Ends the sentence so text said before a tool call is spoken while the tool runs.
                    yield "\n"
                elif isinstance(event, FunctionToolCallEvent):
                    args = json.dumps(event.part.args_as_dict(), ensure_ascii=False)
                    _log.info("[%s] tool %s(%s)", conversation_id, event.part.tool_name, args)
                elif isinstance(event, AgentRunResultEvent):
                    self._history[conversation_id] = (time.monotonic(), _last_turns(event.result.all_messages()))
                    _log.info("[%s] %r -> %r", conversation_id, text, event.result.output)

    def _messages(self, conversation_id: str) -> list[ModelMessage]:
        now = time.monotonic()
        self._history = {k: v for k, v in self._history.items() if now - v[0] < self._cfg.conversation_ttl_s}
        return self._history.get(conversation_id, (now, []))[1]


def _last_turns(messages: list[ModelMessage]) -> list[ModelMessage]:
    """The last MAX_HISTORY_TURNS turns. A turn starts at a user prompt, so a tool call stays with its result."""
    starts = [
        i
        for i, message in enumerate(messages)
        if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)
    ]
    return messages[starts[-MAX_HISTORY_TURNS] :] if len(starts) > MAX_HISTORY_TURNS else messages


async def explain_connect_error(server: McpServerConfig, exc: Exception) -> str:
    """Turns an opaque MCP connection error into one the settings page can act on.

    A 401 from the server surfaces from the MCP client only as "Server returned an error
    response", so the endpoint is asked once more without the MCP layer.
    """
    reason = f"{type(exc).__name__}: {exc}"
    if server.oauth or isinstance(exc, OAuthRequired):
        return reason
    try:
        async with httpx2.AsyncClient(timeout=5) as client:
            response = await client.post(server.url, headers=server.headers, json={})
    except httpx2.HTTPError:
        return reason
    if response.status_code == 401:
        return "Unauthorized (401): turn on oauth, or add an Authorization header"
    if response.status_code == 403:
        return "Forbidden (403): the credentials have no access to this server"
    return reason


def build_toolset(
    server: McpServerConfig,
    oauth: OAuthManager,
    auth: OAuthClientProvider | None = None,
    on_lost: Callable[[Exception], None] | None = None,
) -> IsolatedToolset:
    """auth overrides the stored-token OAuth provider, for the login flow itself."""
    if server.oauth and auth is None:
        auth = oauth.provider(server)
    mcp = MCPToolset(server.url, id=server.name, headers=server.headers or None, auth=auth)
    return IsolatedToolset(mcp, on_lost)


async def list_server_tools(
    server: McpServerConfig, oauth: OAuthManager, auth: OAuthClientProvider | None = None
) -> list[Tool]:
    """Connects once and lists the server's tools."""
    async with build_toolset(server, oauth, auth) as toolset:
        return await toolset.list_tools()
