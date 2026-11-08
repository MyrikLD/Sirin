import pytest

from voice_orchestrator.config import AppSettings


@pytest.fixture
def app_settings(tmp_path) -> AppSettings:
    # Explicit values win over VO_* variables and .env of the machine running the tests.
    return AppSettings(host="127.0.0.1", port=8765, public_url=None, data_dir=tmp_path)
