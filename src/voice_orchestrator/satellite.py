import asyncio
import contextlib
import logging
from collections.abc import Callable

from aioesphomeapi import (
    APIClient,
    APIVersion,
    DeviceInfo,
    ReconnectLogic,
    VoiceAssistantAudioSettings,
    VoiceAssistantFeature,
)

from voice_orchestrator.config import DeviceConfig
from voice_orchestrator.pipeline import PipelineRun, Services
from voice_orchestrator.status import ItemStatus

_log = logging.getLogger(__name__)


def device_client(cfg: DeviceConfig, client_info: str) -> APIClient:
    return APIClient(cfg.host, cfg.port, None, noise_psk=cfg.noise_psk, client_info=client_info)


def voice_support_problem(info: DeviceInfo, api_version: APIVersion | None) -> str | None:
    """Why this device cannot serve as a satellite, or None if it can."""
    features = info.voice_assistant_feature_flags_compat(api_version) if api_version else 0
    if not features & VoiceAssistantFeature.VOICE_ASSISTANT:
        return "the device has no voice_assistant component"
    if not features & VoiceAssistantFeature.API_AUDIO:
        return "the firmware is too old: no audio over the API"
    return None


class Satellite:
    """Holds the API connection to one ESPHome satellite and runs its pipelines."""

    def __init__(self, cfg: DeviceConfig, services: Services) -> None:
        self.cfg = cfg
        self.status = ItemStatus(state="pending", detail="connecting")
        self.audio_base_url: str | None = None
        self.services = services
        self._client = device_client(cfg, "voice-orchestrator")
        self._reconnect = ReconnectLogic(
            client=self._client,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
            on_connect_error=self._on_connect_error,
            name=cfg.host,
        )
        self._unsubscribe: Callable[[], None] | None = None
        self._run: PipelineRun | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._reconnect.start()

    async def stop(self) -> None:
        await self._cancel_run()
        await self._reconnect.stop()
        await self._client.disconnect()

    async def _on_connect(self) -> None:
        info = await self._client.device_info()
        if problem := voice_support_problem(info, self._client.api_version):
            self.status = ItemStatus(state="error", detail=problem)
            _log.error("[%s] %s: %s", self.cfg.name, info.name, problem)
            return

        self._unsubscribe = self._client.subscribe_voice_assistant(
            handle_start=self._handle_start,
            handle_stop=self._handle_stop,
            handle_audio=self._handle_audio,
        )
        self.audio_base_url = await self.services.clips.base_url_for(self.cfg.host)
        self.status = ItemStatus(state="ok", detail=f"connected to {info.friendly_name or info.name}")
        _log.info("[%s] answers are served from %s", self.cfg.name, self.audio_base_url)
        _log.info("[%s] connected to %s (ESPHome %s)", self.cfg.name, info.name, info.esphome_version)

    async def _on_connect_error(self, exc: Exception) -> None:
        self.status = ItemStatus(state="error", detail=f"{type(exc).__name__}: {exc}")

    async def _on_disconnect(self, expected_disconnect: bool) -> None:
        _log.info("[%s] disconnected (expected=%s)", self.cfg.name, expected_disconnect)
        self._unsubscribe = None
        self.status = ItemStatus(state="pending", detail="reconnecting")
        await self._cancel_run()

    async def _handle_start(
        self,
        conversation_id: str,
        flags: int,
        audio_settings: VoiceAssistantAudioSettings,
        wake_word_phrase: str | None,
    ) -> int | None:
        await self._cancel_run()
        _log.info("[%s] pipeline start, wake word %r", self.cfg.name, wake_word_phrase)
        if self.audio_base_url is None:
            return None
        # Firmware that does not echo the conversation id back (pulsar) would start every dialog from
        # scratch; keying by device keeps the history until conversation_ttl_s of silence, as ESPHome does.
        conversation = conversation_id or f"device:{self.cfg.name}"
        run = PipelineRun(self._client, self.services, conversation, self.cfg.name, self.audio_base_url)
        self._run = run
        self._task = asyncio.create_task(run.run())
        # Port 0 tells the device to stream audio over the API connection instead of UDP.
        return 0

    async def _handle_audio(self, data: bytes, data2: bytes | None) -> None:
        if self._run is not None:
            self._run.feed(data)

    async def _handle_stop(self, abort: bool) -> None:
        if abort:
            if self._run is not None:
                _log.info("[%s] the device stopped the pipeline", self.cfg.name)
            await self._cancel_run()
        elif self._run is not None:
            self._run.end_audio()

    async def _cancel_run(self) -> None:
        # Awaited so the old run's RUN_END reaches the device before a new RUN_START.
        task, self._task, self._run = self._task, None, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
