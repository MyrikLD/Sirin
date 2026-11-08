from voice_orchestrator.config import VadConfig
from voice_orchestrator.vad import Endpointer, VadEvent

CHUNK = bytes(1024)  # 32 ms of 16 kHz mono silence, as the satellite sends it


def test_silence_times_out_without_speech():
    endpointer = Endpointer(VadConfig(no_speech_timeout_ms=500))
    events = [endpointer.feed(CHUNK) for _ in range(20)]
    first_timeout = events.index(VadEvent.TIMEOUT)
    assert 14 <= first_timeout <= 16  # about 500 ms in
    assert VadEvent.SPEECH_START not in events
    assert not endpointer.heard_speech
