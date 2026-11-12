import asyncio
import io
import logging
import threading
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
from pydantic import BaseModel, TypeAdapter
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from sirin.agent import TurnDeps
from sirin.files import write_private
from sirin.vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, SAMPLE_WIDTH, START_THRESHOLD, SileroModel

_log = logging.getLogger(__name__)

# Files in the SIRIN_VOICE_MODELS directory.
SPEAKER_FILE = "speaker.onnx"
ENROLL_FILE = "enroll.onnx"
CHUNK_FILE = "chunk.onnx"

# The isolation network takes a voice sample of exactly this length.
SAMPLE_SAMPLES = 3 * SAMPLE_RATE
# Cosine similarity with a remembered voice from which a run of speech counts as that person. Measured on
# clean recordings: 1.5 s of the same voice scores 0.4 to 0.6, of another voice below 0.3.
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

# Kaldi filter bank features as WeSpeaker computes them: 25 ms Hamming frames every 10 ms, 80 mel bins from 20 Hz.
_FRAME, _SHIFT, _FFT, _MELS = 400, 160, 512, 80
_PREEMPHASIS = 0.97
_LOG_FLOOR = 1.1920929e-07
_WINDOW = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(_FRAME) / (_FRAME - 1))).astype(np.float32)


def _mel_filters() -> np.ndarray:
    def mel(hz: np.ndarray | float) -> np.ndarray:
        return 1127.0 * np.log1p(np.asarray(hz) / 700.0)

    edges = np.linspace(mel(20.0), mel(SAMPLE_RATE / 2), _MELS + 2)
    bins = mel(np.arange(_FFT // 2 + 1) * SAMPLE_RATE / _FFT)
    left, center, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising, falling = (bins - left) / (center - left), (right - bins) / (right - center)
    return np.maximum(0.0, np.minimum(rising, falling)).astype(np.float32)


_FILTERS = _mel_filters()


def fbank(samples: np.ndarray) -> np.ndarray:
    """Mean-normalized log mel features, one row per 10 ms; samples are floats in [-1, 1]."""
    scaled = samples.astype(np.float32) * 32768.0
    count = 1 + (len(scaled) - _FRAME) // _SHIFT
    frames = scaled[np.arange(_FRAME)[None, :] + _SHIFT * np.arange(count)[:, None]]
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames = np.concatenate([frames[:, :1] * (1 - _PREEMPHASIS), frames[:, 1:] - _PREEMPHASIS * frames[:, :-1]], axis=1)
    power = np.abs(np.fft.rfft(frames * _WINDOW, _FFT)) ** 2
    features = np.log(np.maximum(power @ _FILTERS.T, _LOG_FLOOR))
    return (features - features.mean(axis=0)).astype(np.float32)


def _session(path: Path, threads: int) -> onnxruntime.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(f"voice model not found at {path}; SIRIN_VOICE_MODELS must hold {path.name}")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class SpeakerModel:
    """A WeSpeaker network: speech in, a vector out that is close for the same voice and far for another."""

    def __init__(self, path: Path) -> None:
        self._session = _session(path, threads=1)
        self._input = self._session.get_inputs()[0].name

    def embed(self, speech: np.ndarray) -> np.ndarray:
        """A unit vector; speech is floats in [-1, 1]."""
        (embedding,) = self._session.run(None, {self._input: fbank(speech)[None]})
        vector = embedding[0]
        return vector / np.linalg.norm(vector)


# What the isolation network keeps of a voice sample: its inputs besides the audio.
Voiceprint = dict[str, np.ndarray]

# Isolation works on 32 ms frames every 8 ms; the output is their overlap-add.
_HOP, _WIN, _PAD = 128, 512, 256
_WINDOW_SQUARED = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_WIN) / _WIN)).astype(np.float32) ** 2
# The network returns the voice inverted and about twice as loud.
_GAIN = -0.5
# Frames per network call: enough to make the call overhead small.
_FRAMES_PER_RUN = 100


class Isolator:
    """The VoxLock target-speaker network: from a sample of one voice and a mix, that voice alone."""

    def __init__(self, enroll: Path, chunk: Path) -> None:
        self._enroll = _session(enroll, threads=1)
        self._chunk = _session(chunk, threads=1)
        shapes = {i.name: i.shape for i in self._chunk.get_inputs()}
        self._state_shape = tuple(shapes["h"])

    def voiceprint(self, sample: np.ndarray) -> Voiceprint:
        """sample: SAMPLE_SAMPLES floats in [-1, 1] of the voice."""
        outputs = self._enroll.run(None, {"enroll": sample.astype(np.float32)[None]})
        return dict(zip(("enroll_n", "k", "v"), outputs, strict=True))

    def stream(self, voiceprint: Voiceprint) -> IsolatedStream:
        return IsolatedStream(self._chunk, voiceprint, self._state_shape)


class IsolatedStream:
    """One audio stream with every voice but one removed; the output runs 24 ms behind the input.

    The streaming loop of the voxlock package (Apache-2.0): only the LSTM state crosses network calls.
    """

    def __init__(
        self, session: onnxruntime.InferenceSession, voiceprint: Voiceprint, state_shape: tuple[int, ...]
    ) -> None:
        self._session = session
        self._voiceprint = voiceprint
        self._h = np.zeros(state_shape, np.float32)
        self._c = np.zeros(state_shape, np.float32)
        # The first samples, kept until there are enough to build the reflected padding in front of them.
        self._head = np.zeros(0, np.float32)
        self._input: np.ndarray | None = None
        self._overlap = np.zeros(_WIN, np.float32)
        self._envelope = np.zeros(_WIN, np.float32)
        self._padding_left = 0
        self._samples_in = 0
        self._samples_out = 0

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Floats in [-1, 1] in; the isolated audio that is ready out."""
        self._samples_in += len(samples)
        if self._input is None:
            self._head = np.concatenate([self._head, samples])
            if len(self._head) <= _PAD:
                return np.zeros(0, np.float32)
            self._input = np.concatenate([self._head[1 : _PAD + 1][::-1], self._head])
            self._padding_left = _PAD
        else:
            self._input = np.concatenate([self._input, samples])
        ready = self._run()
        self._samples_out += len(ready)
        return ready

    def flush(self) -> np.ndarray:
        """The samples still held back at the end of the audio."""
        if self._input is None:
            return np.zeros(0, np.float32)
        padded = np.concatenate([self._input, self._input[-_PAD - 1 : -1][::-1]])
        frames = 1 + (len(padded) - _WIN) // _HOP
        self._input = np.pad(padded, (0, max(0, (frames - 1) * _HOP + _WIN - len(padded))))
        tail = self._overlap_add_tail(self._run())
        tail = tail[: max(0, self._samples_in - self._samples_out)]
        self._samples_out += len(tail)
        self._input = None
        return tail

    def _overlap_add_tail(self, ready: np.ndarray) -> np.ndarray:
        rest = self._overlap / np.where(self._envelope > 1e-11, self._envelope, 1.0) * _GAIN
        return np.concatenate([ready, rest])

    def _run(self) -> np.ndarray:
        ready = []
        while self._input is not None and len(self._input) >= _WIN:
            count = min(_FRAMES_PER_RUN, 1 + (len(self._input) - _WIN) // _HOP)
            segment = self._input[: (count - 1) * _HOP + _WIN]
            frames, self._h, self._c = self._session.run(
                None, {"wav": segment[None], **self._voiceprint, "h": self._h, "c": self._c}
            )
            overlap = np.concatenate([self._overlap, np.zeros(count * _HOP, np.float32)])
            envelope = np.concatenate([self._envelope, np.zeros(count * _HOP, np.float32)])
            for i in range(count):
                overlap[i * _HOP : i * _HOP + _WIN] += frames[0, i]
                envelope[i * _HOP : i * _HOP + _WIN] += _WINDOW_SQUARED
            # Samples that no later frame touches.
            done = count * _HOP
            samples = overlap[:done] / np.where(envelope[:done] > 1e-11, envelope[:done], 1.0)
            self._overlap, self._envelope = overlap[done:], envelope[done:]
            self._input = self._input[done:]
            cut = min(self._padding_left, len(samples))
            self._padding_left -= cut
            ready.append(samples[cut:] * _GAIN)
        return np.concatenate(ready) if ready else np.zeros(0, np.float32)


@dataclass(frozen=True)
class VoiceModels:
    speaker: SpeakerModel
    isolator: Isolator

    @classmethod
    def load(cls, directory: Path) -> VoiceModels:
        return cls(
            speaker=SpeakerModel(directory / SPEAKER_FILE),
            isolator=Isolator(directory / ENROLL_FILE, directory / CHUNK_FILE),
        )


class Voice(BaseModel):
    """One person's voice as remembered: the embedding tells who speaks, the sample isolates their voice."""

    name: str
    embedding: list[float]
    # The WAV file next to voices.json with SAMPLE_SAMPLES of speech.
    sample: str


_voice_list = TypeAdapter(list[Voice])


class TooShort(ValueError):
    """Less speech than a voice sample needs."""


def to_floats(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def to_pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32768.0, -32768, 32767).astype(np.int16).tobytes()


def loudest_sample(speech: np.ndarray) -> np.ndarray:
    """The SAMPLE_SAMPLES of speech with the most energy."""
    step = SAMPLE_RATE // 100
    steps = len(speech) // step
    width = SAMPLE_SAMPLES // step
    if steps <= width:
        return speech[:SAMPLE_SAMPLES]
    energy = (speech[: steps * step].reshape(steps, step) ** 2).mean(axis=1)
    start = int(np.convolve(energy, np.ones(width), "valid").argmax()) * step
    return speech[start : start + SAMPLE_SAMPLES]


class Voices:
    """The voices the household has introduced, kept in data/voices: voices.json and a WAV sample of each."""

    def __init__(self, data_dir: Path, models: VoiceModels) -> None:
        self._dir = data_dir / "voices"
        self._path = self._dir / "voices.json"
        self._models = models
        # Pipeline threads identify while a tool call in another thread remembers or forgets.
        self._lock = threading.Lock()
        saved = _voice_list.validate_json(self._path.read_text()) if self._path.exists() else []
        self._voices = {_key(voice.name): voice for voice in saved}
        self._embeddings = {key: np.asarray(voice.embedding, np.float32) for key, voice in self._voices.items()}
        self._voiceprints: dict[str, Voiceprint] = {}

    @property
    def names(self) -> list[str]:
        return [voice.name for voice in self._voices.values()]

    def closest(self, speech: np.ndarray) -> tuple[str, float] | None:
        """The remembered voice most like speech and its similarity; None when no voice is remembered."""
        if not self._embeddings:
            return None
        embedding = self._models.speaker.embed(speech)
        with self._lock:
            scores = {key: float(known @ embedding) for key, known in self._embeddings.items()}
            key = max(scores, key=scores.__getitem__)
            return self._voices[key].name, scores[key]

    def identify(self, speech: np.ndarray) -> str | None:
        """The name of the remembered voice that speech is in, if any."""
        closest = self.closest(speech)
        return closest[0] if closest is not None and closest[1] >= MATCH_THRESHOLD else None

    def is_voice(self, name: str | None, speech: np.ndarray) -> bool:
        """Whether speech is in the voice remembered under name; True for a voice forgotten meanwhile."""
        with self._lock:
            known = self._embeddings.get(_key(name or ""))
        return known is None or float(known @ self._models.speaker.embed(speech)) >= KEEP_THRESHOLD

    def isolate(self, name: str) -> IsolatedStream:
        """A stream that keeps only the voice remembered under name."""
        key = _key(name)
        with self._lock:
            voiceprint = self._voiceprints.get(key)
            if voiceprint is None:
                voiceprint = self._models.isolator.voiceprint(_read_wav(self._dir / self._voices[key].sample))
                self._voiceprints[key] = voiceprint
        return self._models.isolator.stream(voiceprint)

    def remember(self, name: str, speech: np.ndarray) -> None:
        """Stores speech as the voice of name, replacing what was known under that name."""
        if len(speech) < SAMPLE_SAMPLES:
            raise TooShort(f"{len(speech) / SAMPLE_RATE:.1f} s of speech, a voice sample needs 3 s")
        voice = Voice(
            name=name, embedding=self._models.speaker.embed(speech).tolist(), sample=f"{uuid.uuid4().hex[:8]}.wav"
        )
        key = _key(name)
        with self._lock:
            write_private(self._dir / voice.sample, _wav(loudest_sample(speech)))
            self._drop_sample(key)
            self._voices[key] = voice
            self._embeddings[key] = np.asarray(voice.embedding, np.float32)
            self._voiceprints.pop(key, None)
            self._save()

    def forget(self, name: str) -> bool:
        key = _key(name)
        with self._lock:
            if key not in self._voices:
                return False
            self._drop_sample(key)
            del self._voices[key], self._embeddings[key]
            self._voiceprints.pop(key, None)
            self._save()
        return True

    def _drop_sample(self, key: str) -> None:
        if (voice := self._voices.get(key)) is not None:
            (self._dir / voice.sample).unlink(missing_ok=True)

    def _save(self) -> None:
        write_private(self._path, _voice_list.dump_json(list(self._voices.values())).decode())


def _wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(SAMPLE_WIDTH)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(to_pcm(samples))
    return buffer.getvalue()


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as file:
        return to_floats(file.readframes(file.getnframes()))


def _key(name: str) -> str:
    return name.strip().casefold()


class SpeakerGate:
    """One utterance on its way to the recognizer: finds out who speaks, then lets only that voice through.

    Audio passes unchanged until speech starts. A run of speech is held until there is enough of it to
    judge. Before a voice is known the run is compared with the remembered voices: a match locks the gate,
    and everything from the start of that run on goes through the isolation network; an unknown voice is
    released unchanged and the next run is tried again. Once locked, each isolated run is compared with
    the locked voice and muted when it is somebody else: with its voice absent the network lets another
    one through.
    """

    def __init__(self, voices: Voices, vad: SileroModel) -> None:
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
        self.speaker: str | None = None
        # The best match among the runs compared with the remembered voices: (name, similarity, ms of speech).
        self.closest: tuple[str, float, int] | None = None

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
        if not self._held and not (is_speech and (locked or self._voices.names)):
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
        if closest is None or closest[1] < MATCH_THRESHOLD:
            self._recent.extend(audio.reshape(-1, FRAME_SAMPLES))
            return to_pcm(audio)
        self.speaker = name = closest[0]
        self._stream = self._voices.isolate(name)
        warm_up = np.concatenate(self._recent) if self._recent else np.zeros(0, np.float32)
        return to_pcm(self._stream.push(np.concatenate([warm_up, audio]))[len(warm_up) :])


def voice_tools(voices: Voices) -> FunctionToolset[TurnDeps]:
    tools = FunctionToolset[TurnDeps]()

    @tools.tool
    async def remember_voice(ctx: RunContext[TurnDeps], name: str) -> str:
        """Remembers the voice of the person speaking now under their name.

        From then on you are told who speaks, and other voices and background talk are removed from what that
        person says. Call it when the user asks you to remember or recognize their voice. It learns from the
        user's last phrase and needs at least 3 seconds of speech in it: when the phrase was too short, ask
        the user to say something longer, for example to count to ten, and call the tool again after they do.

        Args:
            name: The person's name as you will be told it later.
        """
        if ctx.deps.speech is None:
            raise ModelRetry("Nobody is speaking in this turn.")
        try:
            # The networks run for a moment; the event loop keeps serving the other devices meanwhile.
            await asyncio.to_thread(voices.remember, name, ctx.deps.speech)
        except TooShort as e:
            return f"Not remembered: {e}. Ask the user to say a longer phrase, then call this tool again."
        return f"The voice of {name} is remembered."

    @tools.tool
    async def forget_voice(ctx: RunContext[TurnDeps], name: str) -> str:
        """Forgets a remembered voice; the names come from known_voices."""
        if not voices.forget(name):
            raise ModelRetry(f"No voice is remembered as {name}; call known_voices.")
        return f"The voice of {name} is forgotten."

    @tools.tool
    async def known_voices(ctx: RunContext[TurnDeps]) -> dict[str, str | list[str]]:
        """Lists the names of the people whose voices are remembered, and whose voice speaks now."""
        return {"remembered": voices.names, "speaking_now": ctx.deps.voice}

    return tools
