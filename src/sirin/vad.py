import enum
from pathlib import Path

import numpy as np
import onnxruntime

from sirin.config import VadConfig

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
# Silero scores 512-sample frames (32 ms at 16 kHz), the size of the satellite's audio chunks.
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE
# Samples of the previous frame the model expects in front of each frame.
CONTEXT_SAMPLES = 64

# Home Assistant's segmenter timings.
SPEECH_MS = 300  # speech needed before an utterance starts
COMMAND_MS = 1000  # shortest utterance: silence before this does not end it
RESET_MS = 1000  # steady speech or silence that resets the counters
# Silero's own defaults: speech starts above 0.5 and goes on while it stays above 0.35.
START_THRESHOLD = 0.5
SPEECH_THRESHOLD = 0.35


class VadEvent(enum.Enum):
    NONE = enum.auto()
    SPEECH_START = enum.auto()
    SPEECH_END = enum.auto()
    TIMEOUT = enum.auto()


class SileroModel:
    """The Silero VAD network, loaded once and shared by every utterance."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found at {path}; set SIRIN_VAD_MODEL to its path")
        options = onnxruntime.SessionOptions()
        # One frame is tiny: extra threads only add scheduling overhead.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def stream(self) -> _Silero:
        return _Silero(self._session)


class _Silero:
    """Speech probability per frame of one audio stream."""

    def __init__(self, session: onnxruntime.InferenceSession) -> None:
        self._session = session
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._rate = np.array(SAMPLE_RATE, dtype=np.int64)

    def __call__(self, frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32).reshape(1, -1) / 32768.0
        model_input = np.concatenate([self._context, samples], axis=1)
        probability, self._state = self._session.run(
            None, {"input": model_input, "state": self._state, "sr": self._rate}
        )
        self._context = model_input[:, -CONTEXT_SAMPLES:]
        return float(probability[0, 0])


class Endpointer:
    """Finds where a spoken command starts and ends; the satellite streams until the server decides.

    Home Assistant's VoiceCommandSegmenter on top of the Silero VAD model. Counters only reset after
    RESET_MS of steady speech or silence, so a short pause or a short noise does not end or start a command.
    """

    def __init__(self, cfg: VadConfig, model: SileroModel) -> None:
        self._cfg = cfg
        self._vad = model.stream()
        self._pending = b""
        self._elapsed_ms = 0
        self._in_command = False
        self._speech_left = SPEECH_MS
        self._command_left = COMMAND_MS - SPEECH_MS
        self._silence_left = cfg.silence_ms
        self._reset_left = RESET_MS
        self.heard_speech = False

    def feed(self, chunk: bytes) -> VadEvent:
        data = self._pending + chunk
        usable = len(data) - len(data) % FRAME_BYTES
        self._pending = data[usable:]
        result = VadEvent.NONE
        for start in range(0, usable, FRAME_BYTES):
            event = self._frame(self._vad(data[start : start + FRAME_BYTES]))
            if event is VadEvent.SPEECH_START:
                result = event
            elif event is not VadEvent.NONE:
                return event
        return result

    def _frame(self, probability: float) -> VadEvent:
        self._elapsed_ms += FRAME_MS
        limit = self._cfg.max_utterance_ms if self.heard_speech else self._cfg.no_speech_timeout_ms
        if self._elapsed_ms >= limit:
            return VadEvent.SPEECH_END if self.heard_speech else VadEvent.TIMEOUT
        return self._in_utterance(probability) if self._in_command else self._before_utterance(probability)

    def _before_utterance(self, probability: float) -> VadEvent:
        if probability > START_THRESHOLD:
            self._reset_left = RESET_MS
            self._speech_left -= FRAME_MS
            if self._speech_left <= 0:
                self._in_command = True
                self._command_left = COMMAND_MS - SPEECH_MS
                self._silence_left = self._cfg.silence_ms
                if not self.heard_speech:
                    self.heard_speech = True
                    return VadEvent.SPEECH_START
        else:
            self._reset_left -= FRAME_MS
            if self._reset_left <= 0:
                self._speech_left = SPEECH_MS
                self._reset_left = RESET_MS
        return VadEvent.NONE

    def _in_utterance(self, probability: float) -> VadEvent:
        self._command_left -= FRAME_MS
        if probability > SPEECH_THRESHOLD:
            self._reset_left -= FRAME_MS
            if self._reset_left <= 0:
                self._silence_left = self._cfg.silence_ms
                self._reset_left = RESET_MS
            return VadEvent.NONE
        self._reset_left = RESET_MS
        self._silence_left -= FRAME_MS
        if self._silence_left <= 0 and self._command_left <= 0:
            return VadEvent.SPEECH_END
        return VadEvent.NONE
