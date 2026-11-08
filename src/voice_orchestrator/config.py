import os
from pathlib import Path

from jinja2 import TemplateSyntaxError
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from voice_orchestrator.prompt import compile_prompt

# Marks a field the settings page shows as a dropdown filled from the block's probe.
CHOICES = {"x-choices": True}


class AppSettings(BaseSettings):
    """Process-level settings from the environment (prefix VO_) or a .env file."""

    model_config = SettingsConfigDict(env_prefix="VO_", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0", description="Bind address of the HTTP server")
    port: int = Field(default=8765, description="Bind port of the HTTP server")
    public_url: str | None = Field(
        default=None,
        description="Base URL satellites download answers from; by default this host's address on the "
        "route to each satellite. Set it only behind a proxy or NAT.",
    )
    audio_ttl_s: int = Field(default=300, description="How long a finished answer stays downloadable")
    data_dir: Path = Field(default=Path("data"), description="Where the settings page stores its settings")


class DeviceConfig(BaseModel):
    name: str = Field(description="Label used in logs")
    host: str = Field(description="Satellite IP or hostname")
    port: int = Field(default=6053, description="ESPHome native API port")
    noise_psk: str | None = Field(default=None, description="api.encryption.key from the ESPHome YAML")


class SttConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 10300
    language: str = Field(default="ru", json_schema_extra=CHOICES)


class TtsConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 10200
    voice: str | None = Field(
        default=None, description="Voice name; server default when empty", json_schema_extra=CHOICES
    )


class VadConfig(BaseModel):
    """End-of-utterance detection; defaults match Home Assistant's Assist pipeline."""

    silence_ms: int = Field(
        default=700, description="Silence that ends the utterance (HA: 250 aggressive, 700 default, 1250 relaxed)"
    )
    no_speech_timeout_ms: int = Field(default=5000, description="Give up when nobody speaks for this long")
    max_utterance_ms: int = Field(default=15000, description="Hard cap on one utterance")


class LLMConfig(BaseModel):
    base_url: str = Field(default="http://127.0.0.1:8080/v1", description="OpenAI-compatible base URL")
    model: str = Field(default="", json_schema_extra=CHOICES)
    api_key: str | None = Field(default=None, description="Bearer token, if the server needs one")
    system_prompt: str = Field(
        default=(
            "You are a smart home voice assistant. Reply in the language the user speaks, briefly, "
            "in one or two sentences, without markdown or lists: the reply is spoken aloud. "
            "Use the tools for actions and data."
        ),
        description="System prompt; a Jinja template rendered before each request, e.g. "
        '{{ now().strftime("%H:%M, %d.%m.%Y") }}',
    )
    temperature: float = 0.3
    conversation_ttl_s: int = Field(default=300, description="Idle time after which dialog history is dropped")

    @field_validator("system_prompt")
    @classmethod
    def _valid_template(cls, value: str) -> str:
        try:
            compile_prompt(value)
        except TemplateSyntaxError as exc:
            raise ValueError(f"template error on line {exc.lineno}: {exc.message}") from exc
        return value


class McpServerConfig(BaseModel):
    name: str = Field(description="Prefix for the server's tool names")
    url: str = Field(description="Streamable HTTP endpoint")
    headers: dict[str, str] = Field(default_factory=dict, description="Extra HTTP headers, e.g. Authorization")
    oauth: bool = Field(default=False, description="Log in with OAuth; use the Authorize button")


class RuntimeSettings(BaseModel):
    """Everything edited on the settings page; applied without a restart."""

    devices: list[DeviceConfig] = Field(default_factory=list, description="ESPHome satellites to serve")
    stt: SttConfig = Field(default_factory=SttConfig, description="Turns the user's speech into text")
    tts: TtsConfig = Field(default_factory=TtsConfig, description="Speaks the answer")
    vad: VadConfig = Field(default_factory=VadConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list, description="Tool servers for the LLM")

    @model_validator(mode="after")
    def _unique_names(self) -> "RuntimeSettings":
        groups = {"device": [d.name for d in self.devices], "MCP server": [m.name for m in self.mcp_servers]}
        for kind, names in groups.items():
            if len(names) != len(set(names)):
                raise ValueError(f"{kind} names must be unique")
        return self


class SettingsStore:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "settings.json"

    def load(self) -> RuntimeSettings:
        if not self._path.exists():
            return RuntimeSettings()
        return RuntimeSettings.model_validate_json(self._path.read_text())

    def save(self, settings: RuntimeSettings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(settings.model_dump_json(indent=2))
        os.replace(tmp, self._path)
