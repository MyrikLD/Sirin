import asyncio
import logging
import threading
from collections import deque
from datetime import UTC, datetime
from typing import Protocol

import numpy as np
import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from sirin.database import Database
from sirin.tables import Setting, Turn, User, Voice
from sirin.vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, SAMPLE_WIDTH, START_THRESHOLD, SileroModel
from sirin.voice_models import (
    SAMPLE_SAMPLES,
    IsolatedStream,
    VoiceModels,
    Voiceprint,
    loudest_sample,
    pack_voiceprint,
    to_floats,
    to_pcm,
    unpack_voiceprint,
)

_log = logging.getLogger(__name__)

# Cosine similarity with a known voice from which a run of speech counts as that voice. Measured on clean
# recordings: 1.5 s of the same voice scores 0.4 to 0.6, of another voice below 0.3.
MATCH_THRESHOLD = 0.35
# Similarity below which an isolated run is somebody else and is muted. Lower than MATCH_THRESHOLD: the
# runs are shorter and what is left of an overlapping voice lowers the score of the right one.
KEEP_THRESHOLD = 0.15
# Speech the gate collects before it asks whose voice it is, and the least that can answer that.
IDENTIFY_MS = 1500
MIN_IDENTIFY_MS = 600
# The same for checking that an isolated run is still the locked voice.
VERIFY_MS = 1200
MIN_VERIFY_MS = 800
# A pause this long ends a run early, so a short command is not held until the device gives up.
HOLD_SILENCE_MS = 400
# Audio from before the speech that the isolation stream hears first, so it does not start in mid-word.
WARM_UP_MS = 500

# Less speech than this gives no useful embedding.
MIN_EMBED_MS = 500
# The least speech of an unknown speaker that creates a new voice.
NEW_VOICE_MS = 1500
# A turn of a voice joins its profile with at least this much speech and this similarity to the profile.
CONFIDENT_MS = 2000
CONFIDENT_SIMILARITY = 0.5

# The `settings` row naming the networks the stored embeddings and voiceprints come from.
MODELS_SETTING = "voice_models"


def voice_profile(current: np.ndarray, turns: list[tuple[np.ndarray, int, bool]]) -> np.ndarray:
    """The profile of a voice from its turns, (embedding, ms of speech, confirmed by the person).

    Confirmed turns are anchors and always count. Another turn counts when it is long, close to the current
    profile and close to every anchor, so a turn matched by mistake does not pull the profile away. With no
    turn that counts, the profile stays as it is.
    """
    anchors = [embedding for embedding, _, confirmed in turns if confirmed]
    chosen = anchors + [
        embedding
        for embedding, ms, confirmed in turns
        if not confirmed
        and ms >= CONFIDENT_MS
        and float(embedding @ current) >= CONFIDENT_SIMILARITY
        and all(float(embedding @ anchor) >= MATCH_THRESHOLD for anchor in anchors)
    ]
    if not chosen:
        return current
    mean = np.mean(chosen, axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def _vector(data: bytes) -> np.ndarray:
    return np.frombuffer(data, np.float32)


class KnownVoices(Protocol):
    """What the speaker gate asks about the known voices; it runs in a worker thread."""

    def any(self) -> bool: ...

    def closest(self, speech: np.ndarray) -> tuple[int, float] | None: ...

    def can_isolate(self, voice_id: int) -> bool: ...

    def isolate(self, voice_id: int) -> IsolatedStream: ...

    def is_voice(self, voice_id: int, speech: np.ndarray) -> bool: ...


class Voices:
    """The voices in the `voices` table, with their profiles held in memory for the speaker gate.

    A turn of an unknown speaker creates a voice; later turns of that voice refine its profile. A voice
    isolates (mutes the other speech) only when a person has linked it and its voiceprint exists.
    """

    def __init__(self, db: Database, models: VoiceModels) -> None:
        self._db = db
        self._models = models
        # Gate threads read these while the event loop replaces them.
        self._lock = threading.Lock()
        self._profiles: dict[int, np.ndarray] = {}
        self._isolatable: set[int] = set()
        self._voiceprints: dict[int, Voiceprint] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        async with self._db.session() as session, session.begin():
            stored = await session.scalar(sa.select(Setting.value).where(Setting.name == MODELS_SETTING))
            if stored != {"version": self._models.version}:
                await _drop_voice_data(session, had_version=stored is not None)
                stmt = insert(Setting).values(name=MODELS_SETTING, value={"version": self._models.version})
                await session.execute(
                    stmt.on_conflict_do_update(index_elements=[Setting.name], set_={"value": stmt.excluded.value})
                )
        await self.reload()

    async def reload(self) -> None:
        """Reads the profiles again after voices were added, linked, merged or deleted."""
        async with self._db.session() as session:
            rows = (
                await session.execute(
                    sa.select(Voice.id, Voice.embedding, Voice.user_id, Voice.voiceprint.is_not(None).label("has_vp"))
                )
            ).all()
        profiles = {r.id: _vector(r.embedding) for r in rows}
        isolatable = {r.id for r in rows if r.user_id is not None and r.has_vp}
        with self._lock:
            self._profiles, self._isolatable = profiles, isolatable
            self._voiceprints = {k: v for k, v in self._voiceprints.items() if k in isolatable}

    @property
    def count(self) -> int:
        return len(self._profiles)

    def any(self) -> bool:
        return bool(self._profiles)

    def embed(self, speech: np.ndarray) -> np.ndarray:
        return self._models.speaker.embed(speech)

    def closest(self, speech: np.ndarray) -> tuple[int, float] | None:
        """The known voice most like speech and its similarity; None when no voice is known."""
        with self._lock:
            profiles = self._profiles
        if not profiles:
            return None
        embedding = self._models.speaker.embed(speech)
        scores = {voice_id: float(profile @ embedding) for voice_id, profile in profiles.items()}
        best = max(scores, key=scores.__getitem__)
        return best, scores[best]

    def can_isolate(self, voice_id: int) -> bool:
        with self._lock:
            return voice_id in self._isolatable

    def is_voice(self, voice_id: int, speech: np.ndarray) -> bool:
        """Whether speech is in that voice; True for a voice deleted meanwhile."""
        with self._lock:
            profile = self._profiles.get(voice_id)
        return profile is None or float(profile @ self._models.speaker.embed(speech)) >= KEEP_THRESHOLD

    def isolate(self, voice_id: int) -> IsolatedStream:
        """A stream that keeps only that voice. Called from a worker thread: the voiceprint is read on first use."""
        with self._lock:
            voiceprint = self._voiceprints.get(voice_id)
        if voiceprint is None:
            if self._loop is None:
                raise RuntimeError("Voices.start() was not called")
            voiceprint = asyncio.run_coroutine_threadsafe(self._read_voiceprint(voice_id), self._loop).result()
            with self._lock:
                self._voiceprints[voice_id] = voiceprint
        return self._models.isolator.stream(voiceprint)

    async def _read_voiceprint(self, voice_id: int) -> Voiceprint:
        async with self._db.session() as session:
            data = await session.scalar(sa.select(Voice.voiceprint).where(Voice.id == voice_id))
        if data is None:
            raise LookupError(f"voice {voice_id} has no voiceprint")
        return unpack_voiceprint(data)

    async def person(self, voice_id: int | None) -> tuple[int, str] | None:
        """The id and name of the person the voice is linked to."""
        if voice_id is None:
            return None
        async with self._db.session() as session:
            row = (
                await session.execute(
                    sa.select(User.id, User.name).join(Voice, Voice.user_id == User.id).where(Voice.id == voice_id)
                )
            ).one_or_none()
        return (row.id, row.name) if row is not None else None

    async def learn(self, turn_id: int, voice_id: int | None, speech: np.ndarray) -> None:
        """Stores the embedding of a turn and assigns it a voice: the one the gate found, or a new one.

        The first turn of a voice with SAMPLE_SAMPLES of speech gives it its voiceprint; the turn's audio
        is not kept.
        """
        speech_ms = len(speech) * 1000 // SAMPLE_RATE
        if speech_ms < MIN_EMBED_MS:
            return
        embedding = (await asyncio.to_thread(self._models.speaker.embed, speech)).astype(np.float32)
        async with self._db.session() as session, session.begin():
            if voice_id is not None:
                # The gate's voice may have been deleted while the turn was answered.
                voice_id = await session.scalar(sa.select(Voice.id).where(Voice.id == voice_id))
            elif speech_ms >= NEW_VOICE_MS:
                result = await session.execute(
                    sa.insert(Voice)
                    .values(embedding=embedding.tobytes(), created_at=datetime.now(UTC))
                    .returning(Voice.id)
                )
                voice_id = result.scalar_one()
                _log.info("turn %d: a new voice %d", turn_id, voice_id)
            await session.execute(
                sa.update(Turn).where(Turn.id == turn_id).values(embedding=embedding.tobytes(), voice_id=voice_id)
            )
            needs_voiceprint = voice_id is not None and await session.scalar(
                sa.select(Voice.voiceprint.is_(None)).where(Voice.id == voice_id)
            )
        if voice_id is None:
            return
        if needs_voiceprint and len(speech) >= SAMPLE_SAMPLES:
            voiceprint = await asyncio.to_thread(self._models.isolator.voiceprint, loudest_sample(speech))
            async with self._db.session() as session, session.begin():
                await session.execute(
                    sa.update(Voice).where(Voice.id == voice_id).values(voiceprint=pack_voiceprint(voiceprint))
                )
        await self.recompute(voice_id)

    async def recompute(self, voice_id: int) -> None:
        async with self._db.session() as session, session.begin():
            await recompute_profile(session, voice_id)
        await self.reload()

    async def clean_up(self) -> int:
        """Deletes the voices nobody linked that no turn points to any more."""
        async with self._db.session() as session, session.begin():
            used = sa.select(Turn.voice_id).where(Turn.voice_id.is_not(None))
            deleted = await session.scalars(
                sa.delete(Voice).where(Voice.user_id.is_(None), Voice.id.not_in(used)).returning(Voice.id)
            )
            count = len(deleted.all())
        if count:
            _log.info("deleted %d unlinked voice(s) without turns", count)
            await self.reload()
        return count


async def recompute_profile(session: AsyncSession, voice_id: int) -> None:
    current = await session.scalar(sa.select(Voice.embedding).where(Voice.id == voice_id))
    if current is None:
        return
    rows = await session.execute(
        sa.select(Turn.embedding, Turn.speech_ms, Turn.voice_confirmed).where(
            Turn.voice_id == voice_id, Turn.embedding.is_not(None)
        )
    )
    turns = [(_vector(r.embedding), r.speech_ms or 0, r.voice_confirmed) for r in rows]
    profile = voice_profile(_vector(current), turns)
    await session.execute(sa.update(Voice).where(Voice.id == voice_id).values(embedding=profile.tobytes()))


async def _drop_voice_data(session: AsyncSession, had_version: bool) -> None:
    """Embeddings and voiceprints of other networks cannot be compared with these: all of them go."""
    voices = await session.scalar(sa.select(sa.func.count()).select_from(Voice))
    await session.execute(sa.update(Turn).values(embedding=None, voice_id=None, voice_confirmed=False))
    await session.execute(sa.delete(Voice))
    if had_version or voices:
        _log.warning("the voice models changed: %d voice(s) and the embeddings of all turns are dropped", voices)


class SpeakerGate:
    """One utterance on its way to the recognizer: finds out who speaks and, for a linked voice, lets only it through.

    Audio passes unchanged until speech starts. A run of speech is held until there is enough of it to
    judge, then compared with the known voices. A match names the speaker; when that voice can isolate,
    the gate locks, and everything from the start of that run on goes through the isolation network.
    Other matches and unknown voices are released unchanged and the next run is tried again. Once locked,
    each isolated run is compared with the locked voice and muted when it is somebody else: with its voice
    absent the network lets another one through.
    """

    def __init__(self, voices: KnownVoices, vad: SileroModel) -> None:
        self._voices = voices
        self._vad = vad.stream()
        self._pending = np.zeros(0, np.float32)
        self._recent: deque[np.ndarray] = deque(maxlen=WARM_UP_MS // FRAME_MS)
        self._held: list[np.ndarray] = []
        self._held_speech: list[np.ndarray] = []
        self._silence_ms = 0
        self._stream: IsolatedStream | None = None
        # Whether the run before this one was kept, while the speech goes on without a pause; None after a pause.
        self._kept_so_far: bool | None = None
        self._speech: list[np.ndarray] = []
        # The voice that speaks: the locked one, else the best match.
        self.speaker: int | None = None
        # The best match among the runs compared with the known voices: (voice id, similarity, ms of speech).
        self.closest: tuple[int, float, int] | None = None

    @property
    def speech(self) -> np.ndarray:
        """The speech let through so far, without the pauses."""
        return np.concatenate(self._speech) if self._speech else np.zeros(0, np.float32)

    def feed(self, chunk: bytes) -> bytes:
        """Microphone audio in; the audio that is ready for the recognizer out, possibly none."""
        samples = to_floats(chunk)
        return self._frames(samples if self._stream is None else self._stream.push(samples))

    def finish(self) -> bytes:
        """The audio still held when the microphone stream ends."""
        was_locked = self._stream is not None
        ready = self._judge() if self._held else b""
        if self._stream is not None:
            if not was_locked:
                self._pending = self._stream.push(self._pending)
            ready += self._frames(self._stream.flush())
            if self._held:
                ready += self._judge()
        return ready + to_pcm(self._pending)

    def _frames(self, samples: np.ndarray) -> bytes:
        data = np.concatenate([self._pending, samples])
        ready = []
        while len(data) >= FRAME_SAMPLES:
            was_locked = self._stream is not None
            ready.append(self._frame(data[:FRAME_SAMPLES]))
            data = data[FRAME_SAMPLES:]
            if self._stream is not None and not was_locked:
                data = self._stream.push(data)
        self._pending = data
        return b"".join(ready)

    def _frame(self, frame: np.ndarray) -> bytes:
        is_speech = self._vad(to_pcm(frame)) > START_THRESHOLD
        locked = self._stream is not None
        if not self._held and not (is_speech and (locked or self._voices.any())):
            if is_speech:
                self._speech.append(frame)
            if not locked:
                self._recent.append(frame)
            return to_pcm(frame)
        self._held.append(frame)
        if is_speech:
            self._held_speech.append(frame)
            self._silence_ms = 0
        else:
            self._silence_ms += FRAME_MS
        enough_ms = VERIFY_MS if locked else IDENTIFY_MS
        if len(self._held_speech) * FRAME_MS >= enough_ms:
            return self._judge(goes_on=True)
        if self._silence_ms >= HOLD_SILENCE_MS:
            return self._judge()
        return b""

    def _judge(self, goes_on: bool = False) -> bytes:
        """goes_on: the run was cut at its full length, so the next one continues the same speech."""
        audio, speech = np.concatenate(self._held), self._held_speech
        self._held, self._held_speech, self._silence_ms = [], [], 0
        speech_ms = len(speech) * FRAME_MS
        if self._stream is not None:
            assert self.speaker is not None
            if speech_ms >= MIN_VERIFY_MS:
                keep = self._voices.is_voice(self.speaker, np.concatenate(speech))
            else:
                # Too short to tell: the end of a longer speech shares its verdict.
                keep = self._kept_so_far is not False
            self._kept_so_far = keep if goes_on else None
            if not keep:
                return bytes(len(audio) * SAMPLE_WIDTH)
            self._speech += speech
            return to_pcm(audio)
        self._speech += speech
        closest = self._voices.closest(np.concatenate(speech)) if speech_ms >= MIN_IDENTIFY_MS else None
        if closest is not None and (self.closest is None or closest[1] > self.closest[1]):
            self.closest = (*closest, speech_ms)
            if closest[1] >= MATCH_THRESHOLD:
                self.speaker = closest[0]
        if closest is None or closest[1] < MATCH_THRESHOLD or not self._voices.can_isolate(closest[0]):
            self._recent.extend(audio.reshape(-1, FRAME_SAMPLES))
            return to_pcm(audio)
        self.speaker = closest[0]
        self._stream = stream = self._voices.isolate(closest[0])
        warm_up = np.concatenate(self._recent) if self._recent else np.zeros(0, np.float32)
        return to_pcm(stream.push(np.concatenate([warm_up, audio]))[len(warm_up) :])
