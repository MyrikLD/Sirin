import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import httpx2
from mcp.client.auth import OAuthClientProvider
from mcp.types import Tool
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelSettings,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    UsageLimits,
)
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider
from pydantic_ai.toolsets import AbstractToolset

from voice_orchestrator.config import LLMConfig, McpServerConfig
from voice_orchestrator.isolation import run_isolated
from voice_orchestrator.oauth import OAuthManager, OAuthRequired
from voice_orchestrator.prompt import compile_prompt

_log = logging.getLogger(__name__)

# Tool-call rounds before the agent gives up on an answer.
MAX_TOOL_ROUNDS = 8
LLM_TIMEOUT_S = 60


class Agent:
    """LLM with MCP tools; streams the spoken part of the answer as text deltas."""

    def __init__(self, cfg: LLMConfig, mcp_servers: list[McpServerConfig], oauth: OAuthManager) -> None:
        self._cfg = cfg
        self._prompt = compile_prompt(cfg.system_prompt)
        self._oauth = oauth
        self._mcp_servers = mcp_servers
        self._history: dict[str, tuple[float, list[ModelMessage]]] = {}
        self._closing = asyncio.Event()
        self._holders: list[asyncio.Task[None]] = []
        self._agent: PydanticAgent[None, str] | None = None
        self.tool_servers: list[str] = []
        self.tool_failures: dict[str, str] = {}

    async def start(self) -> None:
        # All servers connect at once, so startup waits for the slowest one rather than their sum.
        results = await asyncio.gather(*(self._connect(server) for server in self._mcp_servers))
        toolsets: list[AbstractToolset[None]] = []
        for server, result in zip(self._mcp_servers, results, strict=True):
            if isinstance(result, str):
                _log.error("MCP server %s is unavailable, its tools are skipped: %s", server.name, result)
                self.tool_failures[server.name] = result
            else:
                toolsets.append(result.prefixed(server.name))
                self.tool_servers.append(server.name)

        provider = OpenAICompatibleProvider("local", base_url=self._cfg.base_url, api_key=self._cfg.api_key or "none")
        self._agent = PydanticAgent(
            OpenAIChatModel(self._cfg.model, provider=provider),
            instructions=self._render_prompt,
            toolsets=toolsets,
            model_settings=ModelSettings(temperature=self._cfg.temperature, timeout=LLM_TIMEOUT_S),
        )

    async def _connect(self, server: McpServerConfig) -> MCPToolset[None] | str:
        """The connected toolset, or why it could not connect."""
        ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        try:
            toolset = build_toolset(server, self._oauth)
            self._holders.append(asyncio.create_task(self._hold(server.name, toolset, ready)))
            _log.info("MCP server %s: %d tools", server.name, await ready)
            return toolset
        except Exception as exc:  # noqa: BLE001
            return await explain_connect_error(server, exc)

    def _render_prompt(self) -> str:
        return self._prompt.render()

    async def close(self) -> None:
        self._closing.set()
        await asyncio.gather(*self._holders, return_exceptions=True)

    async def stream_reply(self, conversation_id: str, text: str) -> AsyncIterator[str]:
        if self._agent is None:
            raise RuntimeError("Agent.start() was not called")

        async with self._agent.run_stream_events(
            text,
            message_history=self._messages(conversation_id),
            usage_limits=UsageLimits(request_limit=MAX_TOOL_ROUNDS + 1),
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
                    self._history[conversation_id] = (time.monotonic(), event.result.all_messages())
                    _log.info("[%s] %r -> %r", conversation_id, text, event.result.output)

    def _messages(self, conversation_id: str) -> list[ModelMessage]:
        now = time.monotonic()
        self._history = {k: v for k, v in self._history.items() if now - v[0] < self._cfg.conversation_ttl_s}
        return self._history.get(conversation_id, (now, []))[1]

    async def _hold(self, name: str, toolset: MCPToolset[None], ready: "asyncio.Future[int]") -> None:
        # Each MCP connection lives in its own task: a failed anyio transport otherwise leaks
        # its cancel scope into the caller and silently cancels it.
        try:
            async with toolset:
                ready.set_result(len(await toolset.list_tools()))
                await self._closing.wait()
        except Exception as exc:  # noqa: BLE001
            if not ready.done():
                ready.set_exception(exc)
            else:
                _log.exception("MCP server %s connection lost", name)


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
    server: McpServerConfig, oauth: OAuthManager, auth: OAuthClientProvider | None = None
) -> MCPToolset[None]:
    """auth overrides the stored-token OAuth provider, for the login flow itself."""
    if server.oauth and auth is None:
        auth = oauth.provider(server)
    return MCPToolset(server.url, id=server.name, headers=server.headers or None, auth=auth)


async def list_server_tools(
    server: McpServerConfig, oauth: OAuthManager, auth: OAuthClientProvider | None = None
) -> list[Tool]:
    """Connects once and lists the server's tools, outside the caller's task (see run_isolated)."""

    async def connect() -> list[Tool]:
        async with build_toolset(server, oauth, auth) as toolset:
            return await toolset.list_tools()

    return await run_isolated(connect())
