import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime

from sirin.vad import SAMPLE_RATE

# Files in the SIRIN_VOICE_MODELS directory.
SPEAKER_FILE = "speaker.onnx"
ENROLL_FILE = "enroll.onnx"
CHUNK_FILE = "chunk.onnx"

# The isolation network takes a voice sample of exactly this length.
SAMPLE_SAMPLES = 3 * SAMPLE_RATE

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
    # Identifies the networks: embeddings and voiceprints of other networks mean nothing to these.
    version: str

    @classmethod
    def load(cls, directory: Path) -> VoiceModels:
        digest = hashlib.sha256()
        for name in (SPEAKER_FILE, ENROLL_FILE, CHUNK_FILE):
            with (directory / name).open("rb") as file:
                digest.update(hashlib.file_digest(file, "sha256").digest())
        return cls(
            speaker=SpeakerModel(directory / SPEAKER_FILE),
            isolator=Isolator(directory / ENROLL_FILE, directory / CHUNK_FILE),
            version=digest.hexdigest(),
        )


def pack_voiceprint(voiceprint: Voiceprint) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **voiceprint)
    return buffer.getvalue()


def unpack_voiceprint(data: bytes) -> Voiceprint:
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


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
