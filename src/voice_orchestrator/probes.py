import asyncio
from collections.abc import Awaitable, Callable

from aioesphomeapi import APIClient, VoiceAssistantFeature
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from voice_orchestrator.agent import build_toolset, explain_connect_error
from voice_orchestrator.config import DeviceConfig, LLMConfig, McpServerConfig, SttConfig, TtsConfig
from voice_orchestrator.isolation import run_isolated
from voice_orchestrator.oauth import OAuthManager
from voice_orchestrator.speech import describe

PROBE_TIMEOUT_S = 10.0


class ProbeResult(BaseModel):
    ok: bool = Field(description="Whether the service answered")
    message: str = Field(description="One-line summary or the error")
    details: list[str] = Field(default_factory=list, description="What the service reported, one item per line")
    options: dict[str, list[str]] = Field(
        default_factory=dict, description="Suggested values per settings field, e.g. voice names for tts.voice"
    )


async def probe_stt(cfg: SttConfig) -> ProbeResult:
    return await _guard(lambda: _probe_stt(cfg))


async def probe_tts(cfg: TtsConfig) -> ProbeResult:
    return await _guard(lambda: _probe_tts(cfg))


async def probe_llm(cfg: LLMConfig) -> ProbeResult:
    return await _guard(lambda: _probe_llm(cfg))


async def probe_mcp(cfg: McpServerConfig, oauth: OAuthManager) -> ProbeResult:
    return await _guard(lambda: run_isolated(_probe_mcp(cfg, oauth)))


async def probe_device(cfg: DeviceConfig) -> ProbeResult:
    return await _guard(lambda: _probe_device(cfg))


async def _guard(probe: Callable[[], Awaitable[ProbeResult]]) -> ProbeResult:
    try:
        return await asyncio.wait_for(probe(), PROBE_TIMEOUT_S)
    except TimeoutError:
        return ProbeResult(ok=False, message=f"No answer within {PROBE_TIMEOUT_S:.0f} s")
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(ok=False, message=f"{type(exc).__name__}: {exc}")


async def _probe_stt(cfg: SttConfig) -> ProbeResult:
    info = await describe(cfg.host, cfg.port)
    if not info.asr:
        return ProbeResult(ok=False, message="The server has no speech recognition")
    details, languages = [], set()
    for program in info.asr:
        streaming = ", streaming" if program.supports_transcript_streaming else ""
        details.append(f"{program.name}{streaming}")
        for model in program.models:
            if model.installed:
                languages.update(model.languages)
                details.append(f"  model {model.name}: {', '.join(model.languages) or 'any language'}")
    return ProbeResult(
        ok=True,
        message=f"{len(info.asr)} ASR program(s), {len(languages)} language(s)",
        details=details,
        options={"language": sorted(languages)},
    )


async def _probe_tts(cfg: TtsConfig) -> ProbeResult:
    info = await describe(cfg.host, cfg.port)
    if not info.tts:
        return ProbeResult(ok=False, message="The server has no speech synthesis")
    details, voices = [], []
    for program in info.tts:
        streaming = "streaming input" if program.supports_synthesize_streaming else "no streaming input"
        details.append(f"{program.name}: {streaming}")
        for voice in program.voices:
            if voice.installed:
                voices.append(voice.name)
                speakers = f", speakers: {', '.join(s.name for s in voice.speakers)}" if voice.speakers else ""
                details.append(f"  {voice.name} ({', '.join(voice.languages)}){speakers}")
    return ProbeResult(
        ok=True, message=f"{len(voices)} installed voice(s)", details=details, options={"voice": sorted(voices)}
    )


async def _probe_llm(cfg: LLMConfig) -> ProbeResult:
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "none", timeout=PROBE_TIMEOUT_S, max_retries=0)
    async with client:
        ids = sorted([model.id async for model in client.models.list()])
    if cfg.model and cfg.model not in ids:
        message = f"Server is up, but model {cfg.model!r} is not in its list"
        return ProbeResult(ok=False, message=message, details=ids, options={"model": ids})
    return ProbeResult(ok=True, message=f"{len(ids)} model(s)", details=ids, options={"model": ids})


async def _probe_mcp(cfg: McpServerConfig, oauth: OAuthManager) -> ProbeResult:
    try:
        toolset = build_toolset(cfg, oauth)
        async with toolset:
            tools = await toolset.list_tools()
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(ok=False, message=await explain_connect_error(cfg, exc))
    details = [f"{cfg.name}_{t.name}: {(t.description or '').splitlines()[0] if t.description else ''}" for t in tools]
    return ProbeResult(ok=True, message=f"{len(tools)} tool(s)", details=details)


async def _probe_device(cfg: DeviceConfig) -> ProbeResult:
    client = APIClient(cfg.host, cfg.port, None, noise_psk=cfg.noise_psk, client_info="voice-orchestrator-probe")
    try:
        await client.connect(login=True)
        info = await client.device_info()
        api_version = client.api_version
        features = info.voice_assistant_feature_flags_compat(api_version) if api_version else 0
        details = [
            f"name: {info.friendly_name or info.name}",
            f"model: {info.manufacturer} {info.model}".strip(),
            f"ESPHome: {info.esphome_version}",
            "features: " + (", ".join(f.name.lower() for f in VoiceAssistantFeature if features & f) or "none"),
        ]
        if not features & VoiceAssistantFeature.VOICE_ASSISTANT:
            return ProbeResult(ok=False, message="Connected, but the device has no voice_assistant", details=details)
        if not features & VoiceAssistantFeature.API_AUDIO:
            message = "Connected, but the firmware cannot send audio over the API"
            return ProbeResult(ok=False, message=message, details=details)
        wake = await client.get_voice_assistant_configuration(timeout=5)
        if wake.available_wake_words:
            names = ", ".join(w.wake_word for w in wake.available_wake_words)
            details.append(f"wake words: {names} (active: {', '.join(wake.active_wake_words) or 'none'})")
        return ProbeResult(ok=True, message=f"Connected to {info.friendly_name or info.name}", details=details)
    finally:
        await client.disconnect()
