import asyncio

from pydantic_ai import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.toolsets import FunctionToolset

from sirin import agent as agent_module
from sirin.agent import MAX_HISTORY_TURNS, Agent, _last_turns
from sirin.config import LLMConfig, McpServerConfig
from sirin.oauth import OAuthManager

SERVER = McpServerConfig(name="home", url="http://127.0.0.1:1/mcp")


class _FakeToolset(FunctionToolset):
    async def list_tools(self) -> list:
        return []


async def _ignore(_: str) -> None:
    pass


async def _until(changed: asyncio.Event, condition) -> None:
    async with asyncio.timeout(2):
        while not condition():
            changed.clear()
            await changed.wait()


async def test_mcp_server_is_retried_after_a_failure_and_a_lost_connection(tmp_path, monkeypatch):
    lost_callbacks = []

    def build(server, oauth, auth=None, on_lost=None):
        lost_callbacks.append(on_lost)
        if len(lost_callbacks) == 1:
            raise ConnectionError("down")
        return _FakeToolset()

    monkeypatch.setattr(agent_module, "build_toolset", build)
    monkeypatch.setattr(agent_module, "RECONNECT_MIN_S", 0.01)
    changed = asyncio.Event()
    agent = Agent(LLMConfig(), [SERVER], OAuthManager(tmp_path, _ignore), changed.set, [])
    try:
        await agent.start()
        assert agent.tool_status["home"].state == "error"
        assert not agent._live

        await _until(changed, lambda: agent.tool_status["home"].state == "ok")
        assert list(agent._live) == ["home"]

        lost_callbacks[-1](ConnectionError("gone"))
        await _until(changed, lambda: agent.tool_status["home"].state == "error")
        await _until(changed, lambda: agent.tool_status["home"].state == "ok")
        assert len(lost_callbacks) == 3
    finally:
        await agent.close()
    assert not agent._live


def test_history_keeps_the_last_turns_whole():
    turn = [ModelRequest(parts=[UserPromptPart(content="hi")]), ModelResponse(parts=[TextPart(content="hello")])]
    kept = _last_turns(turn * (MAX_HISTORY_TURNS + 3))
    assert len(kept) == MAX_HISTORY_TURNS * 2
    assert isinstance(kept[0], ModelRequest)
    assert _last_turns(turn * 2) == turn * 2
