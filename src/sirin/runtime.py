import asyncio
import logging

from sirin.agent import Agent
from sirin.clips import ClipStore
from sirin.config import AppSettings, RuntimeSettings, SettingsStore
from sirin.oauth import OAuthManager
from sirin.pipeline import Services
from sirin.satellite import Satellite
from sirin.speech import SpeechToText, TextToSpeech
from sirin.status import Status, StatusBoard
from sirin.vad import SileroModel

_log = logging.getLogger(__name__)


class Runtime:
    """Owns the components built from RuntimeSettings and rebuilds only those whose settings changed.

    A device whose own settings are unchanged keeps its connection and any dialog in progress, so
    tuning VAD or the voice does not drop the satellites or reconnect every MCP server.
    """

    def __init__(self, app_settings: AppSettings) -> None:
        public_url = str(app_settings.public_url) if app_settings.public_url else None
        self.clips = ClipStore(public_url, app_settings.port, app_settings.audio_ttl_s)
        self.oauth = OAuthManager(app_settings.data_dir, self._on_authorized)
        self.board = StatusBoard()
        self._store = SettingsStore(app_settings.data_dir)
        self._vad_model = SileroModel(app_settings.vad_model)
        self._lock = asyncio.Lock()
        self._services: Services | None = None
        self._satellites: list[Satellite] = []
        self.settings = RuntimeSettings()

    async def start(self) -> None:
        async with self._lock:
            await self._apply(self._store.load(), reconnect_tools=False)

    async def apply(self, settings: RuntimeSettings) -> None:
        """Saves the settings and rebuilds what they changed."""
        self._store.save(settings)
        async with self._lock:
            await self._apply(settings, reconnect_tools=False)

    async def reconnect_tools(self) -> None:
        """Rebuilds the agent with the current settings, e.g. after an MCP server was authorized."""
        async with self._lock:
            await self._apply(self.settings, reconnect_tools=True)

    async def _on_authorized(self, server_name: str) -> None:
        if any(m.name == server_name for m in self.settings.mcp_servers):
            await self.reconnect_tools()

    async def _apply(self, settings: RuntimeSettings, reconnect_tools: bool) -> None:
        old, previous = self._services, self.settings
        new_agent = (
            old is None
            or reconnect_tools
            or settings.llm != previous.llm
            or settings.mcp_servers != previous.mcp_servers
        )
        if new_agent:
            agent = Agent(settings.llm, settings.mcp_servers, self.oauth, self.board.touch)
            await agent.start()
        else:
            agent = old.agent
        services = Services(
            stt=old.stt if old and settings.stt == previous.stt else SpeechToText(settings.stt),
            tts=old.tts if old and settings.tts == previous.tts else TextToSpeech(settings.tts),
            agent=agent,
            clips=self.clips,
            vad=settings.vad,
            vad_model=self._vad_model,
        )

        running = {s.cfg.name: s for s in self._satellites}
        satellites = []
        for device in settings.devices:
            satellite = running.pop(device.name, None)
            if satellite is not None and satellite.cfg == device:
                satellite.services = services
            else:
                if satellite is not None:
                    await satellite.stop()
                satellite = Satellite(device, services, self.board.touch)
                await satellite.start()
            satellites.append(satellite)
        for removed in running.values():
            await removed.stop()

        if new_agent and old is not None:
            await old.agent.close()
        self._services, self._satellites, self.settings = services, satellites, settings
        self.board.touch()
        _log.info("applied settings: %d device(s), %d MCP server(s)", len(satellites), len(agent.tool_status))

    async def stop(self) -> None:
        async with self._lock:
            await self._stop()

    def status(self) -> Status:
        return Status(
            items={
                "devices": {s.cfg.name: s.status for s in self._satellites},
                "mcp_servers": dict(self._services.agent.tool_status) if self._services else {},
            }
        )

    async def _stop(self) -> None:
        for satellite in self._satellites:
            await satellite.stop()
        if self._services is not None:
            await self._services.agent.close()
        self._services, self._satellites = None, []
        self.board.touch()
