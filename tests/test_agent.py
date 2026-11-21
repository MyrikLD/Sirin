import asyncio

import pytest
from pydantic_ai import ModelRequest, ModelResponse, TextPart, UnexpectedModelBehavior, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaThinkingPart, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from sirin import agent as agent_module
from sirin.agent import MAX_HISTORY_TURNS, OUTPUT_RETRIES, Agent, PersonalServer, TurnDeps, _last_turns
from sirin.config import LLMConfig, McpServerConfig
from sirin.oauth import OAuthManager

SERVER = McpServerConfig(name="home", url="http://127.0.0.1:1/mcp")


class _FakeToolset(FunctionToolset):
    async def list_tools(self) -> list:
        return []


async def _ignore(*_) -> None:
    pass


async def _until(changed: asyncio.Event, condition) -> None:
    async with asyncio.timeout(2):
        while not condition():
            changed.clear()
            await changed.wait()


async def test_mcp_server_is_retried_after_a_failure_and_a_lost_connection(db, monkeypatch):
    lost_callbacks = []

    async def build(server, oauth, auth=None, on_lost=None, owner_id=None):
        lost_callbacks.append(on_lost)
        if len(lost_callbacks) == 1:
            raise ConnectionError("down")
        return _FakeToolset()

    monkeypatch.setattr(agent_module, "build_toolset", build)
    monkeypatch.setattr(agent_module, "RECONNECT_MIN_S", 0.01)
    changed = asyncio.Event()
    agent = Agent(LLMConfig(), [SERVER], OAuthManager(db, _ignore), changed.set, [])
    try:
        await agent.start()
        assert agent.tool_status["home"].state == "error"
        assert not agent._live

        await _until(changed, lambda: agent.tool_status["home"].state == "ok")
        assert list(agent._live) == [(None, "home")]

        lost_callbacks[-1](ConnectionError("gone"))
        await _until(changed, lambda: agent.tool_status["home"].state == "error")
        await _until(changed, lambda: agent.tool_status["home"].state == "ok")
        assert len(lost_callbacks) == 3
    finally:
        await agent.close()
    assert not agent._live


async def test_personal_tools_go_only_to_their_person(db, monkeypatch):
    toolsets = {}

    async def build(server, oauth, auth=None, on_lost=None, owner_id=None):
        toolsets[(owner_id, server.name)] = toolset = _FakeToolset()
        return toolset

    monkeypatch.setattr(agent_module, "build_toolset", build)
    personal = [
        PersonalServer(id=1, owner_id=7, cfg=McpServerConfig(name="mail", url="http://127.0.0.1:1/mail")),
        PersonalServer(id=2, owner_id=8, cfg=McpServerConfig(name="mail", url="http://127.0.0.1:1/other")),
    ]
    agent = Agent(LLMConfig(), [SERVER], OAuthManager(db, _ignore), lambda: None, [], personal)
    try:
        await agent.start()
        assert agent.personal_status[1].state == "ok" and agent.tool_status["home"].state == "ok"

        def owners(turn: TurnDeps) -> list:
            return [
                key[0] for toolset in agent._toolsets_for(turn) for key, t in toolsets.items() if t is toolset.wrapped
            ]

        assert owners(TurnDeps(device="k")) == [None]
        assert owners(TurnDeps(device="k", person_id=7)) == [None, 7]
    finally:
        await agent.close()


def test_history_keeps_the_last_turns_whole():
    turn = [ModelRequest(parts=[UserPromptPart(content="hi")]), ModelResponse(parts=[TextPart(content="hello")])]
    kept = _last_turns(turn * (MAX_HISTORY_TURNS + 3))
    assert len(kept) == MAX_HISTORY_TURNS * 2
    assert isinstance(kept[0], ModelRequest)
    assert _last_turns(turn * 2) == turn * 2


async def test_an_empty_model_response_is_asked_again_and_logged(db, caplog):
    answers = iter([{0: DeltaThinkingPart(content="hmm")}, "Done."])

    async def stream(messages: list, info: AgentInfo):
        yield next(answers)

    agent = Agent(LLMConfig(), [], OAuthManager(db, _ignore), lambda: None, [])
    with agent._agent.override(model=FunctionModel(stream_function=stream)):
        assert await agent.reply("dialog", "Hello", TurnDeps(device="kitchen")) == "Done."
    [record] = [r for r in caplog.records if "empty model response" in r.message]
    assert "[dialog]" in record.message and "'thinking'" in record.message


async def test_a_model_that_stays_empty_fails_and_every_response_is_logged(db, caplog):
    async def stream(messages: list, info: AgentInfo):
        yield {0: DeltaThinkingPart(content="hmm")}

    agent = Agent(LLMConfig(), [], OAuthManager(db, _ignore), lambda: None, [])
    with agent._agent.override(model=FunctionModel(stream_function=stream)), pytest.raises(UnexpectedModelBehavior):
        await agent.reply("dialog", "Hello", TurnDeps(device="kitchen"))
    assert len([r for r in caplog.records if "empty model response" in r.message]) == OUTPUT_RETRIES + 1
