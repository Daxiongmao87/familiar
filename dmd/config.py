"""Role-based model configuration. No model names hardcoded in code.

Every AI capability routes through a RoleConfig. `${VAR}` in string values is
expanded from the environment at load time (secrets never live in the file).
"""

from __future__ import annotations

import os
import re
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised when config.yaml is missing required roles or fails validation."""


def _expand(value: Any) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            var = m.group(1)
            out = os.environ.get(var)
            if out is None:
                raise ConfigError(
                    f"environment variable {var!r} referenced in config is not set"
                )
            return out

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class EndpointConfig(BaseModel):
    """Base shape for an OpenAI-compatible endpoint (URL, optional key/model/extra body)."""

    base_url: str
    api_key: str | None = None
    model_id: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class SynthesisRole(EndpointConfig):
    """Generation role endpoint; must name a model_id."""

    model_id: str  # required for generation roles


class FastRole(SynthesisRole):
    """Fast/cheap synthesis endpoint for low-latency roles."""

    pass


class VisionRole(BaseModel):
    """Optional vision-model endpoint (disabled by default)."""

    enabled: bool = False
    base_url: str | None = None
    api_key: str | None = None
    model_id: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class SttRole(EndpointConfig):
    """Speech-to-text endpoint (openai or whisperx dialect)."""

    base_url: str
    dialect: str = (
        "openai"  # "openai" = /v1/audio/transcriptions; "whisperx" = POST /transcribe
    )


class EmbeddingsRole(BaseModel):
    """Embedding provider: local fastembed model or remote endpoint."""

    provider: str = "local"  # local | endpoint
    model_id: str = "BAAI/bge-small-en-v1.5"
    base_url: str | None = None
    api_key: str | None = None


class ModelsConfig(BaseModel):
    """All model-role endpoints used by the live path."""

    synthesis: SynthesisRole
    fast: FastRole | None = None
    vision: VisionRole | None = None
    stt: SttRole
    embeddings: EmbeddingsRole = EmbeddingsRole()


class OrchestrationConfig(BaseModel):
    """Bounded async job-orchestration tuning."""

    max_concurrent: int = 3
    job_timeout_s: float = 20.0
    stale_after_s: float = 120.0


class AgentConfig(BaseModel):
    """Worker-agent and monitor tuning (v2 agentic layer)."""

    max_tool_calls: int = 6  # tool-call budget per agent loop (card tier)
    ephemeral_max_tool_calls: int = 3  # lighter budget for the ephemeral tier
    agent_timeout_s: float = 45.0  # whole agent-loop wall-clock budget
    web_timeout_s: float = 8.0  # per web_search/web_fetch call
    monitor_cadence_s: float = 30.0  # transcript monitor poll interval
    monitor_lookback: int = 20  # recent utterances the monitor reasons over
    card_kinds: list[str] = [
        "loot",
        "rules",
    ]
    # Fast-lane trigger kinds that map to the durable card tier
    # (detect_trigger yields loot/lore/rules/other); the rest go ephemeral.


class SttPipelineConfig(BaseModel):
    """STT chunking and VAD timing parameters."""

    sample_rate: int = 16000
    silence_ms: int = 700
    min_utterance_ms: int = 400
    max_chunk_s: int = 25


class DiscordConfig(BaseModel):
    """Discord bot credentials and voice-presence options."""

    token: str | None = None
    guild_id: str | None = None
    channel_id: str | None = None  # optional pin; otherwise follow DM / auto-join
    dm_user_id: str | None = None  # authority to follow — Familiar joins their VC
    self_mute: bool = True
    self_deaf: bool = False


class ProjectConfig(BaseModel):
    """The campaign project path and display name."""

    path: str | None = None
    name: str = "Untitled Campaign"


class ServerConfig(BaseModel):
    """Web server bind host/port, optional TLS cert/key, and shutdown bound."""

    host: str = "0.0.0.0"
    port: int = 8760
    https_enabled: bool = False
    cert_file: str | None = None
    key_file: str | None = None
    # Max seconds voice_presence.stop() may block before shutdown abandons it.
    # Prevents a stuck Discord login from wedging SIGTERM handling.
    shutdown_timeout_s: float = 10.0


class AppConfig(BaseModel):
    """Top-level application configuration model."""

    project: ProjectConfig = ProjectConfig()
    models: ModelsConfig
    orchestration: OrchestrationConfig = OrchestrationConfig()
    agent: AgentConfig = AgentConfig()
    stt_pipeline: SttPipelineConfig = SttPipelineConfig()
    discord: DiscordConfig = DiscordConfig()
    server: ServerConfig = ServerConfig()


def load_config(path: str) -> AppConfig:
    """Load and validate config.yaml. Raises ConfigError on any problem."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("config file must contain a YAML mapping")
    expanded = _expand(raw)
    try:
        return AppConfig.model_validate(expanded)
    except ValidationError as e:
        raise ConfigError(str(e)) from e


def load_config_dict(raw: dict) -> AppConfig:
    """Validate an already-parsed config mapping (tests)."""
    try:
        return AppConfig.model_validate(_expand(raw))
    except ValidationError as e:
        raise ConfigError(str(e)) from e
