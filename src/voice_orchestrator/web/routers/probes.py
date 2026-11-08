from fastapi import APIRouter

from voice_orchestrator.config import DeviceConfig, LLMConfig, McpServerConfig, SttConfig, TtsConfig
from voice_orchestrator.probes import ProbeResult, probe_device, probe_llm, probe_mcp, probe_stt, probe_tts
from voice_orchestrator.web.deps import RuntimeDep

# Probes take the unsaved form values, so a block can be checked before it is applied.
router = APIRouter(prefix="/api/probe")


@router.post("/stt")
async def post_probe_stt(cfg: SttConfig) -> ProbeResult:
    return await probe_stt(cfg)


@router.post("/tts")
async def post_probe_tts(cfg: TtsConfig) -> ProbeResult:
    return await probe_tts(cfg)


@router.post("/llm")
async def post_probe_llm(cfg: LLMConfig) -> ProbeResult:
    return await probe_llm(cfg)


@router.post("/mcp")
async def post_probe_mcp(cfg: McpServerConfig, runtime: RuntimeDep) -> ProbeResult:
    return await probe_mcp(cfg, runtime.oauth)


@router.post("/device")
async def post_probe_device(cfg: DeviceConfig) -> ProbeResult:
    return await probe_device(cfg)
