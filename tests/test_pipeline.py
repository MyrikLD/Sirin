from types import SimpleNamespace

from aioesphomeapi import VoiceAssistantEventType as Ev

from sirin.accounts import Account, Role
from sirin.agent import ToolCall
from sirin.clips import ClipStore
from sirin.history import History, list_turns
from sirin.pipeline import SILENCE, PipelineRun

ADMIN = Account(id=0, login="admin", name="Admin", role=Role.ADMIN, created_at="2026-01-01T00:00:00Z")


class _Client:
    def __init__(self) -> None:
        self.events: list[tuple[Ev, dict | None]] = []

    def send_voice_assistant_event(self, event: Ev, data: dict | None) -> None:
        self.events.append((event, data))


class _Agent:
    def __init__(self, answer: str, fails: bool = False, error_reply: str = "") -> None:
        self.error_reply = error_reply
        self._answer = answer
        self._fails = fails
        self.heard: list[str] = []
        self.turns: list = []

    async def stream_reply(self, conversation_id: str, text: str, turn):
        self.heard.append(text)
        self.turns.append(turn)
        turn.calls.append(ToolCall(name="light_on", args={"room": "kitchen"}, result="done"))
        yield self._answer
        if self._fails:
            raise ConnectionError("LLM is down")


class _Tts:
    def __init__(self) -> None:
        self.spoken = ""

    async def speak(self, text, clip) -> None:
        async for delta in text:
            self.spoken += delta
        clip.set_format(16000, 2, 1)
        await clip.write(b"\x00\x00")


async def _run(
    monkeypatch, agent: _Agent, heard: str = "turn on the light", answers_question: bool = False, history=None
) -> tuple[_Client, _Tts]:
    async def listen(self) -> str:
        return heard

    monkeypatch.setattr(PipelineRun, "_listen", listen)
    client, tts = _Client(), _Tts()
    services = SimpleNamespace(agent=agent, tts=tts, clips=ClipStore("http://sirin", 8765, 300), history=history)
    await PipelineRun(client, services, "c", "kitchen", "http://sirin", answers_question).run()
    return client, tts


async def test_failed_agent_is_answered_with_the_error_reply(monkeypatch):
    client, tts = await _run(monkeypatch, _Agent("Turning it", fails=True, error_reply="Sorry, try again."))
    assert tts.spoken == "Turning it\nSorry, try again."
    events = [event for event, _ in client.events]
    assert Ev.VOICE_ASSISTANT_ERROR not in events
    assert events[-2:] == [Ev.VOICE_ASSISTANT_TTS_END, Ev.VOICE_ASSISTANT_RUN_END]
    assert (Ev.VOICE_ASSISTANT_TTS_START, {"text": "Turning it Sorry, try again."}) in client.events


async def test_failed_agent_is_an_error_event_without_an_error_reply(monkeypatch):
    client, _ = await _run(monkeypatch, _Agent("Turning it", fails=True))
    events = [event for event, _ in client.events]
    assert Ev.VOICE_ASSISTANT_ERROR in events
    assert Ev.VOICE_ASSISTANT_TTS_END not in events


async def test_device_listens_again_only_after_a_question(monkeypatch):
    async def follow_up(answer: str) -> str:
        client, _ = await _run(monkeypatch, _Agent(answer))
        return next(data for event, data in client.events if event is Ev.VOICE_ASSISTANT_INTENT_END)[
            "continue_conversation"
        ]

    assert await follow_up("Which lamp?\n") == "1"
    assert await follow_up("Done.") == "0"


async def test_silence_after_a_question_goes_to_the_agent_and_is_not_asked_again(monkeypatch):
    agent = _Agent("Are you there?")
    client, _ = await _run(monkeypatch, agent, heard="", answers_question=True)
    assert agent.heard == [SILENCE]
    events = dict(client.events)
    assert Ev.VOICE_ASSISTANT_ERROR not in events
    assert events[Ev.VOICE_ASSISTANT_INTENT_END]["continue_conversation"] == "0"


async def test_silence_after_the_wake_word_is_an_error(monkeypatch):
    agent = _Agent("Hello.")
    client, _ = await _run(monkeypatch, agent, heard="")
    assert agent.heard == []
    assert Ev.VOICE_ASSISTANT_ERROR in dict(client.events)


async def test_the_turn_is_kept_in_the_history(monkeypatch, db):
    history = History(db, None)
    watched = history.watch()
    agent = _Agent("Done.", fails=True, error_reply="Sorry.")
    await _run(monkeypatch, agent, history=history)
    await history.stop()

    async with db.session() as session:
        (turn,) = await list_turns(session, ADMIN)
    assert agent.turns[0].turn_id == turn.id
    assert (turn.device, turn.text, turn.answer) == ("kitchen", "turn on the light", "Done. Sorry.")
    assert [call.name for call in turn.tools] == ["light_on"]
    assert turn.error == "agent: ConnectionError: LLM is down"
    assert {"heard", "first_text", "first_audio", "done"} <= turn.timings.keys()
    # Once when heard, once when answered.
    assert [watched.get_nowait(), watched.get_nowait()] == [turn.id, turn.id]


async def test_a_run_that_heard_nothing_is_not_kept(monkeypatch, db):
    history = History(db, None)
    await _run(monkeypatch, _Agent("Hello."), heard="", history=history)
    await history.stop()
    async with db.session() as session:
        assert await list_turns(session, ADMIN) == []
