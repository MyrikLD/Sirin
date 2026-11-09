import hashlib
import urllib.request
from pathlib import Path

import pytest

from sirin.config import AppSettings
from sirin.vad import SileroModel

# Same release and hash as the Dockerfile.
MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


@pytest.fixture(scope="session")
def vad_model_path(pytestconfig) -> Path:
    # Downloaded on the first run only: the pytest cache keeps the file between runs.
    path = pytestconfig.cache.mkdir("silero") / "silero_vad.onnx"
    if not path.exists():
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            content = response.read()
        assert hashlib.sha256(content).hexdigest() == MODEL_SHA256, "unexpected Silero VAD model"
        path.write_bytes(content)
    return path


@pytest.fixture(scope="session")
def vad_model(vad_model_path) -> SileroModel:
    return SileroModel(vad_model_path)


@pytest.fixture
def app_settings(tmp_path, vad_model_path) -> AppSettings:
    # Explicit values win over SIRIN_* variables and .env of the machine running the tests.
    return AppSettings(host="127.0.0.1", port=8765, public_url=None, data_dir=tmp_path, vad_model=vad_model_path)
