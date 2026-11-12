import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable

from aioesphomeapi import (
    APIClient,
    APIConnectionError,
    APIVersion,
    DeviceInfo,
    ReconnectLogic,
    VoiceAssistantAudioSettings,
    VoiceAssistantFeature,
)

from sirin.agent import TurnDeps
from sirin.config import DeviceConfig
from sirin.pipeline import PipelineRun, Services
from sirin.status import ItemStatus

_log = logging.getLogger(__name__)

# How long the device may take to play an announcement.
ANNOUNCE_TIMEOUT_S = 120


def device_client(cfg: DeviceConfig, client_info: str) -> APIClient:
    return APIClient(cfg.host, cfg.port, None, noise_psk=cfg.noise_psk, client_info=client_info)


def voice_features(info: DeviceInfo, api_version: APIVersion | None) -> VoiceAssistantFeature:
    return VoiceAssistantFeature(info.voice_assistant_feature_flags_compat(api_version) if api_version else 0)


def voice_support_problem(features: VoiceAssistantFeature) -> str | None:
    """Why a device with these features cannot serve as a satellite, or None if it can."""
    if not features & VoiceAssistantFeature.VOICE_ASSISTANT:
        return "the device has no voice_assistant component"
    if not features & VoiceAssistantFeature.API_AUDIO:
        return "the firmware is too old: no audio over the API"
    return None


class Satellite:
    """Holds the API connection to one ESPHome satellite and runs its pipelines."""

    def __init__(self, cfg: DeviceConfig, services: Services, on_status: Callable[[], None]) -> None:
        self.cfg = cfg
        self._on_status = on_status
        self._status = ItemStatus(state="pending", detail="connecting")
        self.audio_base_url: str | None = None
        self.services = services
        self._client = device_client(cfg, "sirin")
        self._reconnect = ReconnectLogic(
            client=self._client,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
            on_connect_error=self._on_connect_error,
            name=cfg.host,
        )
        self._subscribed = False
        self._can_announce = False
        self._run: PipelineRun | None = None
        self._task: asyncio.Task[None] | None = None
        self._announcing = asyncio.Lock()

    @property
    def status(self) -> ItemStatus:
        return self._status

    @status.setter
    def status(self, value: ItemStatus) -> None:
        if value != self._status:
            self._status = value
            self._on_status()

    @property
    def _device_conversation(self) -> str:
        """The dialog of a device that sends no conversation id, which announcements also join."""
        return f"device:{self.cfg.name}"

    async def start(self) -> None:
        await self._reconnect.start()

    async def stop(self) -> None:
        await self._cancel_run()
        await self._reconnect.stop()
        await self._client.disconnect()

    async def _on_connect(self) -> None:
        info = await self._client.device_info()
        features = voice_features(info, self._client.api_version)
        if problem := voice_support_problem(features):
            self.status = ItemStatus(state="error", detail=problem)
            _log.error("[%s] %s: %s", self.cfg.name, info.name, problem)
            return

        self._can_announce = bool(features & VoiceAssistantFeature.ANNOUNCE)
        self._client.subscribe_voice_assistant(
            handle_start=self._handle_start,
            handle_stop=self._handle_stop,
            handle_audio=self._handle_audio,
        )
        self._subscribed = True
        self.audio_base_url = await self.services.clips.base_url_for(self.cfg.host)
        self.status = ItemStatus(state="ok", detail=f"connected to {info.friendly_name or info.name}")
        _log.info("[%s] answers are served from %s", self.cfg.name, self.audio_base_url)
        _log.info("[%s] connected to %s (ESPHome %s)", self.cfg.name, info.name, info.esphome_version)

    async def _on_connect_error(self, exc: Exception) -> None:
        self.status = ItemStatus(state="error", detail=f"{type(exc).__name__}: {exc}")

    async def _on_disconnect(self, expected_disconnect: bool) -> None:
        _log.info("[%s] disconnected (expected=%s)", self.cfg.name, expected_disconnect)
        self._subscribed = False
        self.status = ItemStatus(state="pending", detail="reconnecting")
        await self._cancel_run()

    async def _handle_start(
        self,
        conversation_id: str,
        flags: int,
        audio_settings: VoiceAssistantAudioSettings,
        wake_word_phrase: str | None,
    ) -> int | None:
        previous = self._run
        await self._cancel_run()
        _log.info("[%s] pipeline start, wake word %r", self.cfg.name, wake_word_phrase)
        if self.audio_base_url is None:
            return None
        # Firmware that does not echo the conversation id back (pulsar) would start every dialog from
        # scratch; keying by device keeps the history until conversation_ttl_s of silence, as ESPHome does.
        conversation = conversation_id or self._device_conversation
        run = PipelineRun(
            self._client,
            self.services,
            conversation,
            self.cfg.name,
            self.audio_base_url,
            answers_question=previous is not None and previous.awaits_reply,
        )
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

    async def announce(self, instruction: str) -> bool:
        """Runs instruction through the agent and has the device announce the answer.

        False if the device is offline now; raises if it will never be able to announce.
        """
        async with self._announcing:
            if not self._subscribed or self.audio_base_url is None:
                return False
            if not self._can_announce:
                raise RuntimeError(f"{self.cfg.name}: the firmware has no voice_assistant announce support")
            # Waits for the dialog in progress instead of talking over it.
            while (task := self._task) is not None and not task.done():
                await asyncio.wait({task})

            services = self.services
            text = await services.agent.reply(self._device_conversation, instruction, TurnDeps(device=self.cfg.name))
            if not text:
                _log.warning("[%s] the agent has nothing to announce", self.cfg.name)
                return True

            clip, url = services.clips.create(self.audio_base_url)

            async def chunks() -> AsyncIterator[str]:
                yield text

            synthesis = asyncio.create_task(services.tts.speak(chunks(), clip))
            try:
                if not await clip.wait_audio():
                    raise RuntimeError("TTS produced no audio")
                _log.info("[%s] announces %r from %s", self.cfg.name, text, url)
                await self._client.send_voice_assistant_announcement_await_response(url, ANNOUNCE_TIMEOUT_S, text=text)
            except APIConnectionError:
                return False
            except TimeoutError:
                _log.warning("[%s] no end of the announcement within %d s", self.cfg.name, ANNOUNCE_TIMEOUT_S)
            finally:
                await synthesis
            return True

    async def _cancel_run(self) -> None:
        # Awaited so the old run's RUN_END reaches the device before a new RUN_START.
        task, self._task, self._run = self._task, None, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
