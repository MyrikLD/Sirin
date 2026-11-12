import asyncio
from datetime import UTC, datetime

import numpy as np
import pytest
import sqlalchemy as sa

from sirin.accounts import Role, create_account
from sirin.agent import TurnDeps
from sirin.history import History
from sirin.tables import Turn, Voice
from sirin.vad import FRAME_SAMPLES, SAMPLE_RATE
from sirin.voice_links import NotAllowed, claim_turn, merge_voices, release_turn
from sirin.voice_models import IsolatedStream, Isolator, SpeakerModel, VoiceModels, fbank, to_floats, to_pcm
from sirin.voices import CONFIDENT_MS, SpeakerGate, Voices, voice_profile

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
        return {"k": np.full(4, sample.mean(), np.float32)}

    def stream(self, voiceprint: dict[str, np.ndarray]) -> IsolatedStream:
        return _Stream()


class _Vad:
    """Speech is anything loud."""

    def stream(self):
        return lambda frame: float(np.abs(to_floats(frame)).mean() > 0.1)


def _models(version: str = "v1") -> VoiceModels:
    return VoiceModels(speaker=_Speaker(), isolator=_Isolator(), version=version)


def _audio(*parts: tuple[float, float]) -> np.ndarray:
    """parts: (level, seconds) stretches, rounded to whole frames."""
    return np.concatenate(
        [
            np.full(round(seconds * SAMPLE_RATE / FRAME_SAMPLES) * FRAME_SAMPLES, level, np.float32)
            for level, seconds in parts
        ]
    )


@pytest.fixture
async def voices(db) -> Voices:
    voices = Voices(db, _models())
    await voices.start()
    return voices


async def _turn(db, voices: Voices, speech: np.ndarray, voice_id: int | None = None) -> int:
    """A history turn of speech, learned the way the pipeline does it."""
    turn_id = await History(db, voices).open_turn("kitchen", "c", datetime.now(UTC), voice_id, "hi", None, None)
    await voices.learn(turn_id, voice_id, speech)
    return turn_id


async def _voice_of(db, turn_id: int) -> int | None:
    async with db.session() as session:
        return await session.scalar(sa.select(Turn.voice_id).where(Turn.id == turn_id))


async def _person(db, login: str, role: Role = Role.MEMBER):
    async with db.session() as session, session.begin():
        return await create_account(session, login, login.title(), "x", role)


async def _through_gate(voices: Voices, audio: np.ndarray) -> tuple[SpeakerGate, np.ndarray]:
    """The gate runs in a worker thread, as in the pipeline: it reads voiceprints through the event loop."""

    def run() -> tuple[SpeakerGate, bytes]:
        gate = SpeakerGate(voices, _Vad())
        pcm = to_pcm(audio)
        # Chunks that are not whole frames, as a device may send them.
        return gate, b"".join(gate.feed(pcm[i : i + 700]) for i in range(0, len(pcm), 700)) + gate.finish()

    gate, out = await asyncio.to_thread(run)
    return gate, to_floats(out)


async def test_an_unknown_speaker_becomes_a_voice_that_is_recognized_later(db, voices):
    assert await _voice_of(db, await _turn(db, voices, _audio((ANNA, 1)))) is None
    first = await _turn(db, voices, _audio((ANNA, 2)))
    voice_id = await _voice_of(db, first)
    assert voice_id is not None and voices.count == 1

    assert voices.closest(_audio((ANNA, 1))) == (voice_id, 1.0)
    assert voices.closest(_audio((STRANGER, 1))) == (voice_id, 0.0)
    gate, _ = await _through_gate(voices, _audio((0, 0.5), (ANNA, 2), (0, 0.5)))
    assert gate.speaker == voice_id


async def test_only_a_linked_voice_mutes_other_speech(db, voices):
    voice_id = await _voice_of(db, await _turn(db, voices, _audio((ANNA, 4))))
    audio = _audio((0, 0.5), (ANNA, 2), (0, 0.5), (STRANGER, 3), (0, 0.5))

    gate, out = await _through_gate(voices, audio)
    assert gate.speaker == voice_id
    assert np.array_equal(out, audio)

    anna = await _person(db, "anna")
    async with db.session() as session, session.begin():
        await session.execute(sa.update(Voice).where(Voice.id == voice_id).values(user_id=anna.id))
    await voices.reload()
    gate, out = await _through_gate(voices, audio)
    assert gate.speaker == voice_id
    said, stranger = slice(SAMPLE_RATE // 2, SAMPLE_RATE * 5 // 2), slice(SAMPLE_RATE * 3, SAMPLE_RATE * 6)
    assert np.array_equal(out[said], audio[said])
    assert not out[stranger].any()
    assert np.all(gate.speech > 0)
    assert await voices.person_name(voice_id) == "Anna"


async def test_a_short_run_is_not_compared(db, voices):
    await _turn(db, voices, _audio((ANNA, 2)))
    gate, _ = await _through_gate(voices, _audio((0, 0.5), (ANNA, 0.4), (0, 1)))
    assert gate.closest is None and gate.speaker is None


def test_profile_follows_confident_turns_and_never_leaves_the_anchors():
    a, b = np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)
    near_a = np.array([0.8, 0.6], np.float32)
    assert np.array_equal(voice_profile(a, [(b, CONFIDENT_MS, False)]), a)
    assert np.array_equal(voice_profile(a, [(near_a, CONFIDENT_MS - 1, False)]), a)
    moved = voice_profile(a, [(near_a, CONFIDENT_MS, False)])
    assert np.allclose(moved, near_a)
    # An anchor counts whatever it is like; a turn far from the anchor does not.
    anchored = voice_profile(near_a, [(b, 0, True), (near_a, CONFIDENT_MS, False)])
    assert np.allclose(anchored, (b + near_a) / np.linalg.norm(b + near_a))
    assert np.allclose(voice_profile(a, [(b, 0, True), (a, CONFIDENT_MS, False)]), b)


async def test_other_models_drop_every_voice_and_embedding(db, voices):
    turn_id = await _turn(db, voices, _audio((ANNA, 4)))
    again = Voices(db, _models("v1"))
    await again.start()
    assert again.count == 1

    other = Voices(db, _models("v2"))
    await other.start()
    assert other.count == 0
    async with db.session() as session:
        assert await session.scalar(sa.select(Turn.embedding).where(Turn.id == turn_id)) is None


async def test_an_unclaimed_voice_goes_with_its_last_turn(db, voices):
    turn_id = await _turn(db, voices, _audio((ANNA, 2)))
    assert await voices.clean_up() == 0
    async with db.session() as session, session.begin():
        await session.execute(sa.delete(Turn).where(Turn.id == turn_id))
    assert await voices.clean_up() == 1
    assert voices.count == 0


async def test_claiming_a_turn_links_its_voice_and_only_its_owner_changes_it(db, voices):
    anna, boris = await _person(db, "anna"), await _person(db, "boris")
    first = await _turn(db, voices, _audio((ANNA, 2)))
    voice_id = await _voice_of(db, first)
    second = await _turn(db, voices, _audio((ANNA, 2)), voice_id)

    async with db.session() as session, session.begin():
        with pytest.raises(NotAllowed):
            await release_turn(session, anna, second)
        assert await claim_turn(session, anna, first) == voice_id
    async with db.session() as session, session.begin():
        with pytest.raises(NotAllowed):
            await claim_turn(session, boris, second)
        await release_turn(session, anna, second)
    assert await _voice_of(db, second) is None
    assert await voices.person_name(voice_id) == "Anna"


async def test_merged_voices_keep_one_profile_and_all_turns(db, voices):
    admin = await _person(db, "boss", Role.ADMIN)
    first = await _turn(db, voices, _audio((ANNA, 2)))
    other = await _turn(db, voices, _audio((STRANGER, 4)))
    keep, gone = await _voice_of(db, first), await _voice_of(db, other)
    async with db.session() as session, session.begin():
        await merge_voices(session, admin, gone, keep)
    await voices.reload()
    assert voices.count == 1
    assert await _voice_of(db, other) == keep
    async with db.session() as session:
        assert await session.scalar(sa.select(Voice.voiceprint.is_not(None)).where(Voice.id == keep))


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
