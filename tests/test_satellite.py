from types import SimpleNamespace

from aioesphomeapi import VoiceAssistantConfigurationResponse, VoiceAssistantWakeWord

from sirin.config import DeviceConfig
from sirin.satellite import Satellite


class _Device:
    def __init__(self, active: str) -> None:
        self.active = active
        self.selected: list[list[str]] = []

    async def get_voice_assistant_configuration(self, **_: float) -> VoiceAssistantConfigurationResponse:
        words = [VoiceAssistantWakeWord(id=w, wake_word=w, trained_languages=["ru"]) for w in ("alisa", "glados")]
        return VoiceAssistantConfigurationResponse(available_wake_words=words, active_wake_words=[self.active])

    async def set_voice_assistant_configuration(self, active_wake_words: list[str]) -> None:
        self.selected.append(active_wake_words)


async def _select(word: str, active: str = "alisa") -> tuple[_Device, str | None]:
    satellite = Satellite(DeviceConfig(name="kitchen", host="127.0.0.1"), SimpleNamespace(), lambda: None)
    satellite._client = device = _Device(active)
    return device, await satellite._select_wake_word(word)


async def test_wake_word_is_selected_unless_already_active():
    device, problem = await _select("glados")
    assert (device.selected, problem) == ([["glados"]], None)

    device, problem = await _select("glados", active="glados")
    assert (device.selected, problem) == ([], None)


async def test_wake_word_the_device_lacks_is_reported():
    device, problem = await _select("jarvis")
    assert device.selected == []
    assert "alisa, glados" in problem
