from pathlib import Path
from typing import Self

from jinja2 import TemplateSyntaxError
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sirin.prompt import compile_prompt

# Schema extensions the settings page renders from.
# A field shown as a dropdown, filled from the block's probe suggestions.
CHOICES = {"x-choices": True}
# A field shown masked.
SECRET = {"format": "password"}


def choices_from(list_field: str) -> dict[str, str]:
    """A field shown as a dropdown of the names of the items in another list field of the same block."""
    return {"x-choices-from": list_field}


def probed(path: str) -> ConfigDict:
    """A block with a Test button that POSTs the block's unsaved values to path."""
    return ConfigDict(json_schema_extra={"x-probe": path})


class AppSettings(BaseSettings):
    """Process-level settings from the environment (prefix SIRIN_) or a .env file."""

    model_config = SettingsConfigDict(env_prefix="SIRIN_", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0", description="Bind address of the HTTP server")
    port: int = Field(default=8765, description="Bind port of the HTTP server")
    public_url: AnyHttpUrl | None = Field(
        default=None,
        description="Base URL satellites download answers from; by default this host's address on the "
        "route to each satellite. Set it only behind a proxy or NAT.",
    )
    audio_ttl_s: int = Field(default=300, description="How long a finished answer stays downloadable")
    data_dir: Path = Field(default=Path("data"), description="Where the settings page stores its settings")
    vad_model: Path = Field(default=Path("models/silero_vad.onnx"), description="Silero VAD ONNX model file")
    voice_models: Path = Field(
        default=Path("models/voices"),
        description="Directory with speaker.onnx, enroll.onnx and chunk.onnx; voices are recognized and isolated "
        "when it exists",
    )


class DeviceConfig(BaseModel):
    model_config = probed("/api/probe/device")

    name: str = Field(description="Label used in logs")
    host: str = Field(description="Satellite IP or hostname")
    port: int = Field(default=6053, description="ESPHome native API port")
    noise_psk: str | None = Field(
        default=None, description="api.encryption.key from the ESPHome YAML", json_schema_extra=SECRET
    )


class SttConfig(BaseModel):
    model_config = probed("/api/probe/stt")

    host: str = "127.0.0.1"
    port: int = 10300
    language: str = Field(default="ru", json_schema_extra=CHOICES)


class TtsConfig(BaseModel):
    model_config = probed("/api/probe/tts")

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


class LLMModelConfig(BaseModel):
    """One language model the assistant can use; the block's active field picks the one in use."""

    model_config = probed("/api/probe/llm")

    name: str = Field(description="A name to pick it by")
    base_url: str = Field(default="http://127.0.0.1:8080/v1", description="OpenAI-compatible base URL")
    model: str = Field(default="", json_schema_extra=CHOICES)
    api_key: str | None = Field(
        default=None, description="Bearer token, if the server needs one", json_schema_extra=SECRET
    )
    temperature: float = 0.3


class LLMConfig(BaseModel):
    models: list[LLMModelConfig] = Field(
        default_factory=lambda: [LLMModelConfig(name="local")], min_length=1, description="The models to choose from"
    )
    active: str | None = Field(
        default=None,
        description="The model in use; the first one when empty",
        json_schema_extra=choices_from("models"),
    )
    system_prompt: str = Field(
        default=(
            "You are a smart home voice assistant. Reply in the language the user speaks, briefly, "
            "in one or two sentences, without markdown or lists: the reply is spoken aloud. "
            "Use the tools for actions and data."
        ),
        description="System prompt; a Jinja template rendered before each request",
    )
    error_reply: str = Field(
        default="Sorry, something went wrong. Please try again.",
        description="Spoken when the model fails; when empty, the device signals an error instead",
    )
    conversation_ttl_s: int = Field(default=300, description="Idle time after which dialog history is dropped")

    @property
    def current(self) -> LLMModelConfig:
        """The model in use."""
        return next((m for m in self.models if m.name == self.active), self.models[0])

    @model_validator(mode="after")
    def _known_active(self) -> Self:
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        if self.active is not None and self.active not in names:
            raise ValueError(f"the active model {self.active!r} is not in the list")
        return self

    @field_validator("system_prompt")
    @classmethod
    def _valid_template(cls, value: str) -> str:
        try:
            compile_prompt(value)
        except TemplateSyntaxError as exc:
            raise ValueError(f"template error on line {exc.lineno}: {exc.message}") from exc
        return value


class HistoryConfig(BaseModel):
    retention_days: int = Field(
        default=30,
        ge=1,
        description="Turns older than this are deleted, except a turn that set a reminder still pending",
    )


class McpServerConfig(BaseModel):
    model_config = probed("/api/probe/mcp")

    name: str = Field(description="Prefix for the server's tool names")
    url: str = Field(description="Streamable HTTP endpoint")
    headers: dict[str, str] = Field(default_factory=dict, description="Extra HTTP headers, e.g. Authorization")
    oauth: bool = Field(
        default=False,
        description="Log in with OAuth; use the Authorize button",
        # While this field is on, the block gets an Authorize button that POSTs its values here.
        json_schema_extra={"x-authorize": "/api/oauth/authorize"},
    )


class RuntimeSettings(BaseModel):
    """Everything edited on the settings page; applied without a restart."""

    devices: list[DeviceConfig] = Field(
        default_factory=list, title="Devices", description="ESPHome satellites to serve"
    )
    stt: SttConfig = Field(default_factory=SttConfig, title="Speech-to-text")
    tts: TtsConfig = Field(default_factory=TtsConfig, title="Text-to-speech")
    vad: VadConfig = Field(default_factory=VadConfig, title="End of utterance (VAD)")
    llm: LLMConfig = Field(default_factory=LLMConfig, title="Language model")
    history: HistoryConfig = Field(default_factory=HistoryConfig, title="History")
    mcp_servers: list[McpServerConfig] = Field(
        default_factory=list, title="MCP servers", description="Tool servers for the LLM"
    )

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        groups = {"device": [d.name for d in self.devices], "MCP server": [m.name for m in self.mcp_servers]}
        for kind, names in groups.items():
            if len(names) != len(set(names)):
                raise ValueError(f"{kind} names must be unique")
        return self
