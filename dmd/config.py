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
    # Upper bound on generated tokens for every chat call on this endpoint.
    # None falls back to the per-role default in dmd.gateway. A chat request
    # without max_tokens lets llama.cpp run with n_predict=-1: a repeating
    # model never releases its inference slot (2026-09-05 slot-leak incident).
    max_tokens: int | None = None
    # Read timeout in seconds for chat requests on this endpoint; None keeps
    # the client default. Short caps make an abandoned call cancel cleanly
    # instead of holding the connection for the full client timeout.
    request_timeout_s: float | None = None


class SynthesisRole(EndpointConfig):
    """Generation role endpoint; must name a model_id."""

    model_id: str  # required for generation roles
    # Inference provider: "remote" (base_url as configured) or "local"
    # (the Electron inference bridge; base_url is kept and reused when
    # the user switches back, cached models are never deleted).
    provider: str = "remote"


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
    max_tokens: int | None = None
    request_timeout_s: float | None = None


class SttRole(EndpointConfig):
    """Speech-to-text endpoint (openai, whisperx, or streaming dialect)."""

    base_url: str
    dialect: str = (
        "openai"  # "openai" = /v1/audio/transcriptions; "whisperx" = POST /transcribe;
        # "streaming" = SimulStreaming TCP server (raw s16le PCM in,
        # newline JSON partials/finals out) — the live-voice path
    )
    # WhisperX request options, exposed from config (SPEC §2 zero-hardcoding:
    # the gateway used to hardcode diarize=false&align=false, suppressing the
    # whisperx-server's own defaults).
    # diarization is DEAD (owner decision 2026-09-05): pyannote never beat 60%
    # on the mixed capture and costs 3.6 s/utterance. Identity comes from JIT
    # mic-state (§7a attribution on the final) + post-context correction.
    # The flag survives only for the batch queue's non-streaming dialects.
    diarize: bool = False
    align: bool = False
    # Streaming STT (SimulStreaming whisper server): host/port for the live
    # voice path. sample_rate must match the source (16 kHz mono int16).
    stream_host: str = "127.0.0.1"
    stream_port: int = 43007


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
    # The pool must not kill a job before the agent loop's own budget can
    # produce a result: job_timeout_s has to exceed AgentConfig.agent_timeout_s
    # (20 < 45 silently dropped every card on a slow local endpoint — live
    # defect, 2026-09-05).
    job_timeout_s: float = 60.0
    stale_after_s: float = 120.0


class SearchConfig(BaseModel):
    """Web-search provider for worker agents.

    Default: the bundled SearXNG instance (project-owned module, run via
    ``tools/searxng.sh``) on localhost. DDG HTML scraping stays as an
    automatic fallback when SearXNG is unreachable, so a search outage
    degrades rather than breaks the agent.
    """

    endpoint: str = "http://127.0.0.1:8888"  # bundled SearXNG JSON API
    key: str | None = None  # SearXNG static token (empty = open localhost)
    language: str = "en"
    timeout_s: float = 10.0


class AgentConfig(BaseModel):
    """Worker-agent and monitor tuning (v2 agentic layer)."""

    max_tool_calls: int = 6  # tool-call budget per agent loop (card tier)
    ephemeral_max_tool_calls: int = 3  # lighter budget for the ephemeral tier
    agent_timeout_s: float = 45.0  # whole agent-loop wall-clock budget
    web_timeout_s: float = 8.0  # per web_search/web_fetch call
    monitor_cadence_s: float = 30.0  # transcript monitor poll interval
    monitor_lookback: int = 20  # recent utterances the monitor reasons over
    # Seconds a card_done verdict must survive before the card is actually
    # marked done. A single monitor tick can misfire on a still-ongoing beat;
    # requiring the verdict to repeat (or persist) across a grace window keeps
    # auto-resolve from closing cards early.
    resolve_grace_s: float = 20.0
    search: SearchConfig = SearchConfig()


class StagingConfig(BaseModel):
    """Predictive-retrieval staging ("Predictive RAG", Priority-1 design).

    The transcript monitor predicts likely-next entities and their campaign
    excerpts are pre-fetched into a small RAM LRU so an actual turn can pull
    already-embedded context instead of paying retrieval latency inline. Staged
    data is advisory only — it never mutates canonical state — and a miss is a
    no-op fallback to the normal path, so these knobs only bound the cache and
    the per-tick fleet budget, never correctness.
    """

    enabled: bool = True
    ttl_s: float = 120.0  # staged excerpts expire; re-fetching is cheap
    max_entries: int = 32  # RAM LRU bound (the memory guard for this cache)
    prefetch_k: int = 4  # excerpts stored per predicted entity
    max_predicted: int = 6  # cap predictions prefetched per monitor tick
    max_inject_chars: int = 6000  # ceiling on the injected staged block


class OpenjevConfig(BaseModel):
    """Openjev decision gate (binary deploy/wait trigger, no taxonomy)."""

    enabled: bool = False  # off = legacy regex + fast-LLM trigger path
    # Inference provider: "remote" (base_url as configured) or "local"
    # (the bundled JEV sidecar at desktop.jev_local_url, identical /score).
    provider: str = "remote"
    base_url: str = "http://127.0.0.1:8199"  # openjev-serve endpoint
    threshold: float = 0.5  # P(deploy) at or above this deploys a worker
    timeout_s: float = 3.0  # per scoring call; failures fail closed to wait
    recent_n: int = 8  # transcript lines in the gate state window
    debounce_s: float = 30.0  # redeploy suppression window in seconds (0 off)
    directed_worker: bool = False  # JEV-routed deterministic worker (no agent loop)


class SttPipelineConfig(BaseModel):
    """STT chunking and VAD timing parameters."""

    sample_rate: int = 16000
    # Trailing-silence hangover before an utterance is endpointed. The
    # voice->transcript budget starts at speech-STOP, so every ms of hangover
    # is spent before STT even begins (a long tail silently eats the whole
    # 5s). Tuned low per the Priority-1 latency directive (400-600ms window);
    # 500ms is the midpoint — short enough to protect the budget, long enough
    # to survive intra-sentence pauses.
    silence_ms: int = 500
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


class DesktopConfig(BaseModel):
    """Local-provider endpoints owned by the Electron desktop shell.

    Python never dials WebGPU or model files directly; local inference
    is always reached through these localhost URLs, so the desktop
    runtime can be supervised, restarted, or degraded without Python
    knowing any implementation detail.
    """

    enabled: bool = False  # true when running under the Electron shell
    # OpenAI-compatible bridge (Electron WebLLM worker): chat/completions.
    bridge_url: str = "http://127.0.0.1:8791"
    # Bundled JEV sidecar (openjev-serve wire protocol): /score + /health.
    # 8299, not 8199: a user-run remote scorer may already hold 8199.
    jev_local_url: str = "http://127.0.0.1:8299"


class AppConfig(BaseModel):
    """Top-level application configuration model."""

    project: ProjectConfig = ProjectConfig()
    models: ModelsConfig
    orchestration: OrchestrationConfig = OrchestrationConfig()
    agent: AgentConfig = AgentConfig()
    stt_pipeline: SttPipelineConfig = SttPipelineConfig()
    staging: StagingConfig = StagingConfig()
    openjev: OpenjevConfig = OpenjevConfig()
    discord: DiscordConfig = DiscordConfig()
    server: ServerConfig = ServerConfig()
    desktop: DesktopConfig = DesktopConfig()


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
