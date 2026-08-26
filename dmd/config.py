"""Role-based model configuration. No model names hardcoded in code.

Every AI capability routes through a RoleConfig. `${VAR}` in string values is
expanded from the environment at load time (secrets never live in the file).
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

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
                raise ConfigError(f"environment variable {var!r} referenced in config is not set")
            return out

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class EndpointConfig(BaseModel):
    base_url: str
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class SynthesisRole(EndpointConfig):
    model_id: str  # required for generation roles


class FastRole(SynthesisRole):
    pass


class VisionRole(BaseModel):
    enabled: bool = False
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class SttRole(EndpointConfig):
    base_url: str


class EmbeddingsRole(BaseModel):
    provider: str = "local"  # local | endpoint
    model_id: str = "BAAI/bge-small-en-v1.5"
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class ModelsConfig(BaseModel):
    synthesis: SynthesisRole
    fast: Optional[FastRole] = None
    vision: Optional[VisionRole] = None
    stt: SttRole
    embeddings: EmbeddingsRole = EmbeddingsRole()


class OrchestrationConfig(BaseModel):
    max_concurrent: int = 3
    job_timeout_s: float = 20.0
    stale_after_s: float = 120.0


class SttPipelineConfig(BaseModel):
    sample_rate: int = 16000
    silence_ms: int = 700
    min_utterance_ms: int = 400
    max_chunk_s: int = 25


class DiscordConfig(BaseModel):
    token: Optional[str] = None
    guild_id: Optional[str] = None
    channel_id: Optional[str] = None  # optional pin; otherwise follow DM / auto-join
    dm_user_id: Optional[str] = None  # authority to follow — Familiar joins their VC
    self_mute: bool = True
    self_deaf: bool = False


class ProjectConfig(BaseModel):
    path: Optional[str] = None
    name: str = "Untitled Campaign"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8760
    https_enabled: bool = False
    cert_file: Optional[str] = None
    key_file: Optional[str] = None


class AppConfig(BaseModel):
    project: ProjectConfig = ProjectConfig()
    models: ModelsConfig
    orchestration: OrchestrationConfig = OrchestrationConfig()
    stt_pipeline: SttPipelineConfig = SttPipelineConfig()
    discord: DiscordConfig = DiscordConfig()
    server: ServerConfig = ServerConfig()


def load_config(path: str) -> AppConfig:
    """Load and validate config.yaml. Raises ConfigError on any problem."""
    with open(path, "r", encoding="utf-8") as f:
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
