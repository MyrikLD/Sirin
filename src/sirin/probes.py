import asyncio
import functools
from collections.abc import Awaitable, Callable

from aioesphomeapi import VoiceAssistantFeature
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from sirin.agent import explain_connect_error, list_server_tools
from sirin.config import DeviceConfig, LLMConfig, McpServerConfig, SttConfig, TtsConfig
from sirin.oauth import OAuthManager
from sirin.satellite import device_client, voice_features, voice_support_problem
from sirin.speech import describe

PROBE_TIMEOUT_S = 10.0


class ProbeResult(BaseModel):
    ok: bool = Field(description="Whether the service answered")
    message: str = Field(description="One-line summary or the error")
    details: list[str] = Field(default_factory=list, description="What the service reported, one item per line")
    options: dict[str, list[str]] = Field(
        default_factory=dict, description="Suggested values per settings field, e.g. voice names for tts.voice"
    )


def _guard[**P](probe: Callable[P, Awaitable[ProbeResult]]) -> Callable[P, Awaitable[ProbeResult]]:
    """Turns a timeout or any error into a failed ProbeResult, so the page always gets an answer."""

    @functools.wraps(probe)
    async def guarded(*args: P.args, **kwargs: P.kwargs) -> ProbeResult:
        try:
            return await asyncio.wait_for(probe(*args, **kwargs), PROBE_TIMEOUT_S)
        except TimeoutError:
            return ProbeResult(ok=False, message=f"No answer within {PROBE_TIMEOUT_S:.0f} s")
        except Exception as exc:
            return ProbeResult(ok=False, message=f"{type(exc).__name__}: {exc}")

    return guarded


@_guard
async def probe_stt(cfg: SttConfig) -> ProbeResult:
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


@_guard
async def probe_tts(cfg: TtsConfig) -> ProbeResult:
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


@_guard
async def probe_llm(cfg: LLMConfig) -> ProbeResult:
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "none", timeout=PROBE_TIMEOUT_S, max_retries=0)
    async with client:
        ids = sorted([model.id async for model in client.models.list()])
    ok = not cfg.model or cfg.model in ids
    message = f"{len(ids)} model(s)" if ok else f"Server is up, but model {cfg.model!r} is not in its list"
    return ProbeResult(ok=ok, message=message, details=ids, options={"model": ids})


@_guard
async def probe_mcp(cfg: McpServerConfig, oauth: OAuthManager) -> ProbeResult:
    try:
        tools = await list_server_tools(cfg, oauth)
    except Exception as exc:
        return ProbeResult(ok=False, message=await explain_connect_error(cfg, exc))
    details = [f"{cfg.name}_{t.name}: {(t.description or '').partition('\n')[0]}" for t in tools]
    return ProbeResult(ok=True, message=f"{len(tools)} tool(s)", details=details)


@_guard
async def probe_device(cfg: DeviceConfig) -> ProbeResult:
    client = device_client(cfg, "sirin-probe")
    try:
        await client.connect(login=True)
        info = await client.device_info()
        features = voice_features(info, client.api_version)
        details = [
            f"name: {info.friendly_name or info.name}",
            f"model: {info.manufacturer} {info.model}".strip(),
            f"ESPHome: {info.esphome_version}",
            "features: " + (", ".join(f.name.lower() for f in VoiceAssistantFeature if features & f) or "none"),
        ]
        if problem := voice_support_problem(features):
            return ProbeResult(ok=False, message=f"Connected, but {problem}", details=details)
        wake = await client.get_voice_assistant_configuration(timeout=5)
        if wake.available_wake_words:
            names = ", ".join(w.wake_word for w in wake.available_wake_words)
            details.append(f"wake words: {names} (active: {', '.join(wake.active_wake_words) or 'none'})")
        return ProbeResult(ok=True, message=f"Connected to {info.friendly_name or info.name}", details=details)
    finally:
        await client.disconnect()
