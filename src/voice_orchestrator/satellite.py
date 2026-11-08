import asyncio
import contextlib
import logging
from collections.abc import Callable

from aioesphomeapi import (
    APIClient,
    ReconnectLogic,
    VoiceAssistantAudioSettings,
    VoiceAssistantFeature,
)

from voice_orchestrator.config import DeviceConfig
from voice_orchestrator.pipeline import PipelineRun, Services

_log = logging.getLogger(__name__)


class Satellite:
    """Holds the API connection to one ESPHome satellite and runs its pipelines."""

    def __init__(self, cfg: DeviceConfig, services: Services) -> None:
        self.cfg = cfg
        self.state = "connecting"
        self.device_name: str | None = None
        self.audio_base_url: str | None = None
        self._services = services
        self._client = APIClient(
            cfg.host, cfg.port, None, noise_psk=cfg.noise_psk, client_info="voice-orchestrator"
        )
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
        api_version = self._client.api_version
        if api_version is None:
            raise ConnectionError("API version unknown after connect")
        self.device_name = info.friendly_name or info.name
        features = info.voice_assistant_feature_flags_compat(api_version)
        if not features & VoiceAssistantFeature.VOICE_ASSISTANT:
            self.state = "no voice_assistant component"
            _log.error("[%s] %s has no voice_assistant component", self.cfg.name, info.name)
            return
        if not features & VoiceAssistantFeature.API_AUDIO:
            self.state = "firmware too old: no audio over the API"
            _log.error("[%s] firmware is too old: no audio over the API", self.cfg.name)
            return

        self._unsubscribe = self._client.subscribe_voice_assistant(
            handle_start=self._handle_start,
            handle_stop=self._handle_stop,
            handle_audio=self._handle_audio,
        )
        self.audio_base_url = await self._services.clips.base_url_for(self.cfg.host)
        self.state = "connected"
        _log.info("[%s] answers are served from %s", self.cfg.name, self.audio_base_url)
        _log.info("[%s] connected to %s (ESPHome %s)", self.cfg.name, info.name, info.esphome_version)

    async def _on_connect_error(self, exc: Exception) -> None:
        self.state = f"{type(exc).__name__}: {exc}"

    async def _on_disconnect(self, expected_disconnect: bool) -> None:
        _log.info("[%s] disconnected (expected=%s)", self.cfg.name, expected_disconnect)
        self._unsubscribe = None
        self.state = "connecting"
        await self._cancel_run()

    async def _handle_start(
        self,
        conversation_id: str,
        flags: int,  # noqa: ARG002
        audio_settings: VoiceAssistantAudioSettings,  # noqa: ARG002
        wake_word_phrase: str | None,
    ) -> int | None:
        await self._cancel_run()
        _log.info("[%s] pipeline start, wake word %r", self.cfg.name, wake_word_phrase)
        if self.audio_base_url is None:
            return None
        # Firmware that does not echo the conversation id back (pulsar) would start every dialog from
        # scratch; keying by device keeps the history until conversation_ttl_s of silence, as ESPHome does.
        conversation = conversation_id or f"device:{self.cfg.name}"
        run = PipelineRun(self._client, self._services, conversation, self.cfg.name, self.audio_base_url)
        self._run = run
        self._task = asyncio.create_task(run.run())
        # Port 0 tells the device to stream audio over the API connection instead of UDP.
        return 0

    async def _handle_audio(self, data: bytes, data2: bytes | None) -> None:  # noqa: ARG002
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
