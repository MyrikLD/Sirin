import asyncio
import logging

from pydantic import BaseModel, Field

from voice_orchestrator.agent import Agent
from voice_orchestrator.clips import ClipStore
from voice_orchestrator.config import RuntimeSettings
from voice_orchestrator.oauth import OAuthManager
from voice_orchestrator.pipeline import Services
from voice_orchestrator.satellite import Satellite
from voice_orchestrator.speech import SpeechToText, TextToSpeech

_log = logging.getLogger(__name__)


class DeviceStatus(BaseModel):
    name: str = Field(description="Device label from the settings")
    host: str = Field(description="Device address")
    state: str = Field(description="connecting, connected or an error")
    device: str | None = Field(default=None, description="Name the device reports about itself")


class Status(BaseModel):
    devices: list[DeviceStatus]
    mcp_servers: list[str] = Field(description="MCP servers whose tools are available")
    mcp_failed: dict[str, str] = Field(description="MCP servers that failed to connect, with the error")


class Runtime:
    """Owns the components built from RuntimeSettings and rebuilds them when the settings change."""

    def __init__(self, clips: ClipStore, oauth: OAuthManager) -> None:
        self._clips = clips
        self._oauth = oauth
        self._lock = asyncio.Lock()
        self._agent: Agent | None = None
        self._satellites: list[Satellite] = []
        self.settings = RuntimeSettings()

    async def apply(self, settings: RuntimeSettings) -> None:
        async with self._lock:
            await self._stop()
            agent = Agent(settings.llm, settings.mcp_servers, self._oauth)
            await agent.start()
            services = Services(
                stt=SpeechToText(settings.stt),
                tts=TextToSpeech(settings.tts),
                agent=agent,
                clips=self._clips,
                vad=settings.vad,
            )
            satellites = [Satellite(device, services) for device in settings.devices]
            for satellite in satellites:
                await satellite.start()
            self._agent, self._satellites, self.settings = agent, satellites, settings
            _log.info("applied settings: %d device(s), %d MCP server(s)", len(satellites), len(agent.tool_servers))

    async def stop(self) -> None:
        async with self._lock:
            await self._stop()

    def status(self) -> Status:
        return Status(
            devices=[
                DeviceStatus(name=s.cfg.name, host=s.cfg.host, state=s.state, device=s.device_name)
                for s in self._satellites
            ],
            mcp_servers=self._agent.tool_servers if self._agent else [],
            mcp_failed=self._agent.tool_failures if self._agent else {},
        )

    async def _stop(self) -> None:
        for satellite in self._satellites:
            await satellite.stop()
        if self._agent is not None:
            await self._agent.close()
        self._agent, self._satellites = None, []
