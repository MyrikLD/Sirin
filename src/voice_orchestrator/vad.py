import enum

from pymicro_vad import MicroVad

from voice_orchestrator.config import VadConfig

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
# microVAD scores 10 ms frames of 16 kHz mono int16.
FRAME_BYTES = SAMPLE_RATE // 100 * SAMPLE_WIDTH
FRAME_MS = 10

# Home Assistant's segmenter values, tuned for the microVAD model.
SPEECH_MS = 300  # speech needed before an utterance starts
COMMAND_MS = 1000  # shortest utterance: silence before this does not end it
RESET_MS = 1000  # steady speech or silence that resets the counters
START_THRESHOLD = 0.2  # speech probability that counts before the utterance
SPEECH_THRESHOLD = 0.5  # speech probability that counts inside the utterance


class VadEvent(enum.Enum):
    NONE = enum.auto()
    SPEECH_START = enum.auto()
    SPEECH_END = enum.auto()
    TIMEOUT = enum.auto()


class Endpointer:
    """Finds where a spoken command starts and ends; the satellite streams until the server decides.

    A port of Home Assistant's VoiceCommandSegmenter on top of the same microVAD model. Counters
    only reset after RESET_MS of steady speech or silence, so a short pause or a short noise does
    not end or start a command.
    """

    def __init__(self, cfg: VadConfig) -> None:
        self._cfg = cfg
        self._vad = MicroVad()
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
            event = self._frame(self._vad.process_10ms(data[start : start + FRAME_BYTES]))
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
        if probability < 0:
            # microVAD returns a negative score until it has heard enough audio.
            return VadEvent.NONE
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
