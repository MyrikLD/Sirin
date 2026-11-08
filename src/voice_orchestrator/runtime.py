import asyncio
import logging

from voice_orchestrator.agent import Agent
from voice_orchestrator.clips import ClipStore
from voice_orchestrator.config import RuntimeSettings
from voice_orchestrator.oauth import OAuthManager
from voice_orchestrator.pipeline import Services
from voice_orchestrator.satellite import Satellite
from voice_orchestrator.speech import SpeechToText, TextToSpeech
from voice_orchestrator.status import Status, StatusBoard

_log = logging.getLogger(__name__)


class Runtime:
    """Owns the components built from RuntimeSettings and rebuilds only those whose settings changed.

    A device whose own settings are unchanged keeps its connection and any dialog in progress, so
    tuning VAD or the voice does not drop the satellites or reconnect every MCP server.
    """

    def __init__(self, clips: ClipStore, oauth: OAuthManager, board: StatusBoard) -> None:
        self._clips = clips
        self._oauth = oauth
        self._board = board
        self._lock = asyncio.Lock()
        self._services: Services | None = None
        self._satellites: list[Satellite] = []
        self.settings = RuntimeSettings()

    async def apply(self, settings: RuntimeSettings) -> None:
        async with self._lock:
            await self._apply(settings, reconnect_tools=False)

    async def reconnect_tools(self) -> None:
        """Rebuilds the agent with the current settings, e.g. after an MCP server was authorized."""
        async with self._lock:
            await self._apply(self.settings, reconnect_tools=True)

    async def _apply(self, settings: RuntimeSettings, reconnect_tools: bool) -> None:
        old, previous = self._services, self.settings
        new_agent = (
            old is None
            or reconnect_tools
            or settings.llm != previous.llm
            or settings.mcp_servers != previous.mcp_servers
        )
        if new_agent:
            agent = Agent(settings.llm, settings.mcp_servers, self._oauth, self._board.touch)
            await agent.start()
        else:
            agent = old.agent
        services = Services(
            stt=old.stt if old and settings.stt == previous.stt else SpeechToText(settings.stt),
            tts=old.tts if old and settings.tts == previous.tts else TextToSpeech(settings.tts),
            agent=agent,
            clips=self._clips,
            vad=settings.vad,
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
                satellite = Satellite(device, services, self._board.touch)
                await satellite.start()
            satellites.append(satellite)
        for removed in running.values():
            await removed.stop()

        if new_agent and old is not None:
            await old.agent.close()
        self._services, self._satellites, self.settings = services, satellites, settings
        self._board.touch()
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
        self._board.touch()
