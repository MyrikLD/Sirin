import pytest

from voice_orchestrator.config import VadConfig
from voice_orchestrator.vad import Endpointer, SileroModel, VadEvent

CHUNK = bytes(1024)  # 32 ms of 16 kHz mono silence, as the satellite sends it


def test_silence_times_out_without_speech(vad_model):
    endpointer = Endpointer(VadConfig(no_speech_timeout_ms=500), vad_model)
    events = [endpointer.feed(CHUNK) for _ in range(20)]
    first_timeout = events.index(VadEvent.TIMEOUT)
    assert 14 <= first_timeout <= 16  # about 500 ms in
    assert VadEvent.SPEECH_START not in events
    assert not endpointer.heard_speech


def test_missing_model_file_names_the_setting(tmp_path):
    with pytest.raises(FileNotFoundError, match="VO_VAD_MODEL"):
        SileroModel(tmp_path / "silero_vad.onnx")
