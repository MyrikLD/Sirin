import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any

import httpx2
import numpy as np
from mcp.client.auth import OAuthClientProvider
from mcp.types import Tool
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    UsageLimits,
    UserPromptPart,
    capture_run_messages,
)
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ERROR_OUTCOMES
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
# Times the model is asked again after a response with neither text nor a tool call. Some OpenAI-compatible
# servers give such an empty response, for example after a tool result, and more than once in a row.
OUTPUT_RETRIES = 3
# User turns of a dialog that the model sees again; older ones are forgotten.
MAX_HISTORY_TURNS = 20
# Pause before an MCP server that failed or was lost is tried again; it doubles up to the maximum.
RECONNECT_MIN_S = 2.0
RECONNECT_MAX_S = 60.0
# What the voice tag of a user turn says when the voice is none of the remembered ones.
UNKNOWN_VOICE = "not recognized"
# Added to the system prompt of every turn.
TIME_INSTRUCTIONS = (
    "Each user turn carries a tag such as [Time: 2026-01-31 18:05, Saturday]: the local time it was said. The last "
    "one is the current time. Never say the tag aloud or write one yourself."
)
# Added to the system prompt while voices are recognized. The name is in the turn and not here because the
# people of one dialog can differ from turn to turn, and a model reads a fact next to the question more
# reliably than one at the end of a long prompt.
VOICE_INSTRUCTIONS = (
    "Each user turn starts with a tag such as [Voice: Anna]: the name of the person whose voice said it, as "
    f"recognized from the audio, or [Voice: {UNKNOWN_VOICE}] for a voice no person has claimed. The tag is "
    "reliable: trust it over your own guesses about who speaks. Never say the tag aloud or write one yourself."
)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: str | None = Field(default=None, description="What the tool returned to the model; None until it returns")
    failed: bool = False


class TurnDeps(BaseModel):
    """Who the agent is answering; tools that act on the device or the speaker read it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: str
    # The history row of this turn; None for a turn not kept in the history, such as a reminder.
    turn_id: int | None = None
    # The voice that speaks, when the gate recognized one, and the person it is linked to: their personal
    # MCP servers join the tools of the turn.
    voice_id: int | None = None
    person_id: int | None = None
    speaker: str | None = None
    # The speech of this turn as 16 kHz floats; None for a turn nobody spoke, such as a reminder.
    speech: np.ndarray | None = None
    # The tool calls of the answer, filled in by the agent as they happen.
    calls: list[ToolCall] = Field(default_factory=list)

    @property
    def voice(self) -> str:
        """Whose voice speaks, as the model is told."""
        return self.speaker or UNKNOWN_VOICE

    def tagged(self, text: str) -> str:
        """text as the model gets it: with the local time and, when voices are recognized, whose voice said it.

        The time is here because the message keeps its text in the history, so the prompt cache of the LLM server
        stays valid from one minute to the next.
        """
        text = f"[Time: {datetime.now().astimezone():%Y-%m-%d %H:%M, %A}] {text}"
        if self.speech is None or not len(self.speech):
            return text
        return f"[Voice: {self.voice}] {text}"


class PersonalServer(BaseModel):
    """An MCP server of one person: its tools are offered only in the turns of that person's voice."""

    id: int
    owner_id: int
    cfg: McpServerConfig


class Agent:
    """LLM with MCP tools; streams the spoken part of the answer as text deltas."""

    def __init__(
        self,
        cfg: LLMConfig,
        mcp_servers: list[McpServerConfig],
        oauth: OAuthManager,
        on_status: Callable[[], None],
        builtin_tools: list[AbstractToolset[TurnDeps]],
        personal: list[PersonalServer] | None = None,
    ) -> None:
        self._cfg = cfg
        self.error_reply = cfg.error_reply
        self._on_status = on_status
        self._prompt = compile_prompt(cfg.system_prompt)
        self._oauth = oauth
        # (owner, personal server id, config); the shared servers have neither and come first.
        self._servers: list[tuple[int | None, int | None, McpServerConfig]] = [
            *((None, None, server) for server in mcp_servers),
            *((p.owner_id, p.id, p.cfg) for p in personal or []),
        ]
        self._builtin_tools = builtin_tools
        self._history: dict[str, tuple[float, list[ModelMessage]]] = {}
        # The toolsets of the servers connected now, by (owner, server name).
        self._live: dict[tuple[int | None, str], AbstractToolset[TurnDeps]] = {}
        self._holders: list[asyncio.Task[None]] = []
        # Shared servers by name, for the settings page; personal ones by id.
        self.tool_status: dict[str, ItemStatus] = {}
        self.personal_status: dict[int, ItemStatus] = {}
        model = cfg.current
        provider = OpenAICompatibleProvider("local", base_url=model.base_url, api_key=model.api_key or "none")
        self._agent = PydanticAgent(
            OpenAIChatModel(model.model, provider=provider),
            deps_type=TurnDeps,
            instructions=self._render_prompt,
            model_settings=ModelSettings(temperature=model.temperature, timeout=LLM_TIMEOUT_S),
            retries={"output": OUTPUT_RETRIES},
        )

    async def start(self) -> None:
        """Returns once every server was tried, so the first dialog has the tools of the servers that are up."""
        tried = [asyncio.Event() for _ in self._servers]
        self._holders = [
            asyncio.create_task(self._hold(owner, server_id, server, event))
            for (owner, server_id, server), event in zip(self._servers, tried, strict=True)
        ]
        await asyncio.gather(*(event.wait() for event in tried))

    async def _hold(
        self, owner: int | None, server_id: int | None, server: McpServerConfig, tried: asyncio.Event
    ) -> None:
        """Keeps one server connected until close(): a failed or lost connection is opened again after a pause."""
        key = (owner, server.name)
        label = server.name if owner is None else f"{server.name} of person {owner}"

        def set_status(status: ItemStatus) -> bool:
            return self._set_status(server.name if server_id is None else server_id, status)

        set_status(ItemStatus(state="pending", detail="connecting"))
        delay = RECONNECT_MIN_S
        while True:
            lost: asyncio.Queue[Exception] = asyncio.Queue()
            try:
                async with await build_toolset(server, self._oauth, on_lost=lost.put_nowait, owner_id=owner) as toolset:
                    count = len(await toolset.list_tools())
                    self._live[key] = toolset.prefixed(server.name)
                    _log.info("MCP server %s: %d tools", label, count)
                    set_status(ItemStatus(state="ok", detail=f"{count} tools"))
                    tried.set()
                    delay = RECONNECT_MIN_S
                    reason = f"connection lost: {await lost.get()}"
            except Exception as exc:
                reason = await explain_connect_error(server, exc)
            finally:
                self._live.pop(key, None)
            if set_status(ItemStatus(state="error", detail=reason)):
                _log.error("MCP server %s is unavailable, its tools are skipped until it answers: %s", label, reason)
            tried.set()
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_S)

    def _set_status(self, key: str | int, status: ItemStatus) -> bool:
        """key: a shared server's name or a personal server's id. True if the status changed."""
        statuses: dict = self.tool_status if isinstance(key, str) else self.personal_status
        if statuses.get(key) == status:
            return False
        statuses[key] = status
        self._on_status()
        return True

    def _render_prompt(self, ctx: RunContext[TurnDeps]) -> str:
        prompt = f"{self._prompt.render()}\n\n{TIME_INSTRUCTIONS}"
        return prompt if ctx.deps.speech is None else f"{prompt}\n\n{VOICE_INSTRUCTIONS}"

    async def close(self) -> None:
        for holder in self._holders:
            holder.cancel()
        await asyncio.gather(*self._holders, return_exceptions=True)

    async def reply(self, conversation_id: str, text: str, turn: TurnDeps) -> str:
        """The whole answer at once, for speech nobody is waiting on."""
        return "".join([delta async for delta in self.stream_reply(conversation_id, text, turn)]).strip()

    async def stream_reply(self, conversation_id: str, text: str, turn: TurnDeps) -> AsyncIterator[str]:
        calls: dict[str, ToolCall] = {}
        history = self._messages(conversation_id)
        with capture_run_messages() as messages:
            try:
                async for delta in self._stream(conversation_id, text, turn, history, calls):
                    yield delta
            finally:
                _log_empty_responses(conversation_id, messages, history)

    async def _stream(
        self,
        conversation_id: str,
        text: str,
        turn: TurnDeps,
        history: list[ModelMessage],
        calls: dict[str, ToolCall],
    ) -> AsyncIterator[str]:
        async with self._agent.run_stream_events(
            turn.tagged(text),
            message_history=history,
            deps=turn,
            usage_limits=UsageLimits(request_limit=MAX_TOOL_ROUNDS + 1),
            toolsets=[*self._toolsets_for(turn), *self._builtin_tools],
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
                    call = ToolCall(name=event.part.tool_name, args=event.part.args_as_dict())
                    calls[event.part.tool_call_id] = call
                    turn.calls.append(call)
                    _log.info("[%s] tool %s(%s)", conversation_id, call.name, json.dumps(call.args, ensure_ascii=False))
                elif isinstance(event, FunctionToolResultEvent) and (call := calls.get(event.tool_call_id)):
                    call.result = event.part.model_response_str(wrap_if_error=False)
                    call.failed = event.part.outcome in ERROR_OUTCOMES
                elif isinstance(event, AgentRunResultEvent):
                    self._history[conversation_id] = (time.monotonic(), _last_turns(event.result.all_messages()))
                    _log.info("[%s] %r -> %r", conversation_id, text, event.result.output)

    def _toolsets_for(self, turn: TurnDeps) -> list[AbstractToolset[TurnDeps]]:
        """The shared servers, then the personal ones of the person whose voice speaks."""
        mine = [(key, toolset) for key, toolset in self._live.items() if key[0] is None or key[0] == turn.person_id]
        return [toolset for _, toolset in sorted(mine, key=lambda item: (item[0][0] is not None, item[0][1]))]

    def _messages(self, conversation_id: str) -> list[ModelMessage]:
        now = time.monotonic()
        self._history = {k: v for k, v in self._history.items() if now - v[0] < self._cfg.conversation_ttl_s}
        return self._history.get(conversation_id, (now, []))[1]


def _log_empty_responses(conversation_id: str, messages: list[ModelMessage], history: list[ModelMessage]) -> None:
    """Warns about each response of this run with neither text nor a tool call: the model was asked again."""
    old = {id(message) for message in history}
    for message in messages:
        if id(message) in old or not isinstance(message, ModelResponse):
            continue
        if any(isinstance(part, ToolCallPart) or isinstance(part, TextPart) and part.content for part in message.parts):
            continue
        _log.warning(
            "[%s] empty model response: finish_reason=%s, parts=%s, usage=%s, details=%s",
            conversation_id,
            message.finish_reason,
            [part.part_kind for part in message.parts],
            message.usage,
            message.provider_details,
        )


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


async def build_toolset(
    server: McpServerConfig,
    oauth: OAuthManager,
    auth: OAuthClientProvider | None = None,
    on_lost: Callable[[Exception], None] | None = None,
    owner_id: int | None = None,
) -> IsolatedToolset:
    """auth overrides the stored-token OAuth provider, for the login flow itself; owner_id picks a personal server."""
    if server.oauth and auth is None:
        auth = await oauth.provider(server, owner_id)
    mcp = MCPToolset(server.url, id=server.name, headers=server.headers or None, auth=auth)
    return IsolatedToolset(mcp, on_lost)


async def list_server_tools(
    server: McpServerConfig,
    oauth: OAuthManager,
    auth: OAuthClientProvider | None = None,
    owner_id: int | None = None,
) -> list[Tool]:
    """Connects once and lists the server's tools."""
    async with await build_toolset(server, oauth, auth, owner_id=owner_id) as toolset:
        return await toolset.list_tools()
