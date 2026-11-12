import json

import numpy as np
import pytest
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from sirin.agent import TurnDeps
from sirin.vad import FRAME_SAMPLES, SAMPLE_RATE
from sirin.voices import (
    SAMPLE_SAMPLES,
    IsolatedStream,
    Isolator,
    SpeakerGate,
    SpeakerModel,
    TooShort,
    VoiceModels,
    Voices,
    fbank,
    to_floats,
    to_pcm,
    voice_tools,
)

# The fakes know two voices: a positive level and a negative one.
ANNA, STRANGER = 0.5, -0.5


class _Speaker(SpeakerModel):
    def __init__(self) -> None:
        pass

    def embed(self, speech: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0] if speech.mean() > 0 else [0.0, 1.0], np.float32)


class _Stream(IsolatedStream):
    """Lets everything through, the way the network does while its voice is silent."""

    def __init__(self) -> None:
        pass

    def push(self, samples: np.ndarray) -> np.ndarray:
        return samples

    def flush(self) -> np.ndarray:
        return np.zeros(0, np.float32)


class _Isolator(Isolator):
    def __init__(self) -> None:
        pass

    def voiceprint(self, sample: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def stream(self, voiceprint: dict[str, np.ndarray]) -> IsolatedStream:
        return _Stream()


class _Vad:
    """Speech is anything loud."""

    def stream(self):
        return lambda frame: float(np.abs(to_floats(frame)).mean() > 0.1)


def _audio(*parts: tuple[float, float]) -> np.ndarray:
    """parts: (level, seconds) stretches, rounded to whole frames."""
    return np.concatenate(
        [
            np.full(round(seconds * SAMPLE_RATE / FRAME_SAMPLES) * FRAME_SAMPLES, level, np.float32)
            for level, seconds in parts
        ]
    )


@pytest.fixture
def voices(tmp_path) -> Voices:
    return Voices(tmp_path, VoiceModels(speaker=_Speaker(), isolator=_Isolator()))


def _through_gate(voices: Voices, audio: np.ndarray) -> tuple[SpeakerGate, np.ndarray]:
    gate = SpeakerGate(voices, _Vad())
    pcm = to_pcm(audio)
    # Chunks that are not whole frames, as a device may send them.
    out = b"".join(gate.feed(pcm[i : i + 700]) for i in range(0, len(pcm), 700)) + gate.finish()
    return gate, to_floats(out)


def test_known_voice_locks_the_gate_and_other_voices_are_muted(voices):
    voices.remember("Anna", _audio((ANNA, 3)))
    audio = _audio((0, 0.5), (ANNA, 2), (0, 0.5), (STRANGER, 3), (0, 0.5))
    gate, out = _through_gate(voices, audio)

    assert gate.speaker == "Anna"
    assert len(out) == len(audio)
    anna, stranger = slice(SAMPLE_RATE // 2, SAMPLE_RATE * 5 // 2), slice(SAMPLE_RATE * 3, SAMPLE_RATE * 6)
    assert np.array_equal(out[anna], audio[anna])
    assert not out[stranger].any()
    assert np.all(gate.speech > 0)


def test_unknown_voice_passes_unchanged(voices):
    voices.remember("Anna", _audio((ANNA, 3)))
    audio = _audio((0, 0.5), (STRANGER, 2.2), (0, 0.5))
    gate, out = _through_gate(voices, audio)

    assert gate.speaker is None
    assert np.array_equal(out, audio)
    assert len(gate.speech) == round(2.2 * SAMPLE_RATE / FRAME_SAMPLES) * FRAME_SAMPLES


def test_gate_reports_the_closest_voice_it_compared(voices):
    voices.remember("Anna", _audio((ANNA, 3)))
    gate, _ = _through_gate(voices, _audio((0, 0.5), (ANNA, 0.4), (0, 1)))
    assert gate.closest is None

    gate, _ = _through_gate(voices, _audio((0, 0.5), (STRANGER, 2), (0, 1)))
    assert gate.speaker is None
    assert gate.closest is not None and gate.closest[:2] == ("Anna", 0.0)


def test_known_voice_after_a_stranger_still_locks(voices):
    voices.remember("Anna", _audio((ANNA, 3)))
    audio = _audio((0, 0.5), (STRANGER, 2), (0, 0.5), (ANNA, 2), (0, 0.5))
    gate, out = _through_gate(voices, audio)

    assert gate.speaker == "Anna"
    assert np.array_equal(out, audio)


def test_voices_survive_a_restart_and_can_be_forgotten(tmp_path, voices):
    with pytest.raises(TooShort):
        voices.remember("Anna", _audio((ANNA, 2)))
    voices.remember("Anna", _audio((ANNA, 4)))
    (saved,) = json.loads((tmp_path / "voices" / "voices.json").read_text())
    assert saved["name"] == "Anna"
    sample = tmp_path / "voices" / saved["sample"]
    assert sample.stat().st_size == 44 + SAMPLE_SAMPLES * 2

    reloaded = Voices(tmp_path, VoiceModels(speaker=_Speaker(), isolator=_Isolator()))
    assert reloaded.identify(_audio((ANNA, 1))) == "Anna"
    assert reloaded.identify(_audio((STRANGER, 1))) is None
    assert reloaded.forget("anna")
    assert not reloaded.forget("Anna")
    assert reloaded.names == []
    assert not sample.exists()


def _calling(tool: str, args: dict) -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1].parts[-1]
        if isinstance(last, ToolReturnPart):
            return ModelResponse(parts=[TextPart(str(last.content))])
        return ModelResponse(parts=[ToolCallPart(tool, args)])

    return FunctionModel(model)


async def test_voice_is_remembered_from_the_speech_of_the_turn(voices):
    agent = Agent(deps_type=TurnDeps, toolsets=[voice_tools(voices)])
    remember = _calling("remember_voice", {"name": "Anna"})

    short = TurnDeps(device="kitchen", speech=_audio((ANNA, 1)))
    result = await agent.run("remember me", model=remember, deps=short)
    assert "longer phrase" in result.output
    assert voices.names == []

    await agent.run("remember me", model=remember, deps=TurnDeps(device="kitchen", speech=_audio((ANNA, 4))))
    assert voices.names == ["Anna"]


def test_turn_carries_the_voice_only_when_voices_are_recognized():
    speech = _audio((ANNA, 1))
    assert TurnDeps(device="kitchen", speaker="Anna", speech=speech).tagged("hi") == "[Voice: Anna] hi"
    assert TurnDeps(device="kitchen", speech=speech).tagged("hi") == "[Voice: not recognized] hi"
    assert TurnDeps(device="kitchen").tagged("hi") == "hi"
    assert TurnDeps(device="kitchen", speech=np.zeros(0, np.float32)).tagged("(silence)") == "(silence)"


def test_fbank_has_a_row_per_10_ms_and_no_mean():
    features = fbank(np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1)
    assert features.shape == (98, 80)
    assert np.abs(features.mean(axis=0)).max() < 1e-3


def test_sample_is_the_loudest_three_seconds(tmp_path, voices):
    voices.remember("Anna", _audio((0.1, 2), (ANNA, 3), (0.1, 2)))
    reloaded = Voices(tmp_path, VoiceModels(speaker=_Speaker(), isolator=_Isolator()))
    assert reloaded.isolate("Anna").push(np.zeros(4, np.float32)).shape == (4,)
    assert SAMPLE_SAMPLES == 3 * SAMPLE_RATE
