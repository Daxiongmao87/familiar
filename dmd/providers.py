"""Local-vs-remote provider routing beneath the existing chat()/score() calls.

Familiar's Python code keeps perceiving synthesis as ``Gateway.chat(...)``
and JEV as ``OpenjevGate.score/decide(...)``. This module only decides
*which localhost or remote URL* each call goes to:

- synthesis provider ``remote`` → the configured OpenAI-compatible
  ``base_url`` (today's behavior, byte-identical);
- synthesis provider ``local`` → the Electron inference bridge, which
  speaks the same ``POST /chat/completions`` dialect;
- JEV provider ``remote`` → the configured ``openjev.base_url`` ``/score``;
- JEV provider ``local`` → the bundled JEV sidecar, which speaks the
  identical ``/score`` + ``/health`` wire protocol (same algorithm and
  weights as the pinned remote scorer).

Switching providers never deletes cached models: it only flips routing.
No Chromium/WebGPU detail leaks into Python — only localhost URLs.
"""

from __future__ import annotations

from typing import Any

from .config import AppConfig, EndpointConfig

#: Model id the Electron local bridge advertises. The bridge ignores the
#: id for routing (one resident model) but echoes it for diagnostics.
LOCAL_SYNTHESIS_MODEL = "minicpm5-2b-mlc"


def synthesis_provider_of(cfg: AppConfig) -> str:
    """Return ``local`` or ``remote`` for the synthesis role (default remote)."""
    mode = str(getattr(cfg.models.synthesis, "provider", "remote") or "remote")
    return "local" if mode == "local" else "remote"


def jev_provider_of(cfg: AppConfig) -> str:
    """Return ``local`` or ``remote`` for the JEV gate (default remote)."""
    oj = getattr(cfg, "openjev", None)
    mode = str(getattr(oj, "provider", "remote") or "remote")
    return "local" if mode == "local" else "remote"


def synthesis_endpoint_for(cfg: AppConfig, role: str = "synthesis") -> EndpointConfig:
    """Resolve the effective endpoint for a generation role.

    Remote mode returns the configured endpoint unchanged. Local mode
    returns a copy whose ``base_url`` points at the Electron inference
    bridge and whose ``model_id`` names the resident local model; the
    caller's ``extra_body``/``max_tokens``/``request_timeout_s`` are
    preserved so Gateway behavior (slot-leak guard, thinking switch,
    json_schema) is identical on both paths.
    """
    src: EndpointConfig
    if role == "fast":
        fast = cfg.models.fast
        if fast is None:
            raise ValueError("fast role not configured")
        src = fast
    else:
        src = cfg.models.synthesis
    if str(getattr(src, "provider", "remote") or "remote") != "local":
        return src
    desktop = getattr(cfg, "desktop", None)
    bridge = str(getattr(desktop, "bridge_url", "") or "").rstrip("/")
    if not bridge:
        raise ValueError("local synthesis selected but desktop.bridge_url is empty")
    return EndpointConfig(
        base_url=bridge,
        api_key=None,
        model_id=LOCAL_SYNTHESIS_MODEL,
        extra_body=dict(src.extra_body),
        max_tokens=src.max_tokens,
        request_timeout_s=src.request_timeout_s,
    )


def jev_base_url_for(cfg: AppConfig) -> str:
    """Resolve the effective JEV ``/score`` base URL for the provider mode."""
    oj = getattr(cfg, "openjev", None)
    if oj is None:
        raise ValueError("openjev config missing")
    if str(getattr(oj, "provider", "remote") or "remote") == "local":
        desktop = getattr(cfg, "desktop", None)
        local = str(getattr(desktop, "jev_local_url", "") or "").rstrip("/")
        if not local:
            raise ValueError("local JEV selected but desktop.jev_local_url is empty")
        return local
    return str(oj.base_url).rstrip("/")


class EndpointRouter:
    """Live provider router held by ``Gateway``.

    The router keeps a reference to the app config and resolves the
    effective endpoint on every call, so flipping
    ``cfg.models.synthesis.provider`` (and persisting it) reroutes the
    next ``chat()`` with no rebuild and no restart. Roles without a
    provider flag (vision, embeddings) always resolve to config. STT is
    not an HTTP role at all — it streams to a TCP server (see SttRole).
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

    @property
    def cfg(self) -> AppConfig:
        """The live config this router resolves against."""
        return self._cfg

    def resolve(self, role: str) -> EndpointConfig:
        """Return the effective endpoint for ``role``."""
        if role in ("synthesis", "fast"):
            return synthesis_endpoint_for(self._cfg, role)
        m = self._cfg.models
        if role == "vision":
            v = m.vision
            if v is None:  # pragma: no cover - Gateway raises the typed error
                raise ValueError("vision role not configured")
            return v  # type: ignore[return-value]
        raise ValueError(f"unknown role: {role!r}")

    def jev_base_url(self) -> str:
        """Return the effective JEV base URL for the current provider mode."""
        return jev_base_url_for(self._cfg)

    def describe(self) -> dict[str, Any]:
        """Provider state for status UIs. Never includes secrets."""

        def _ep(ep: EndpointConfig) -> dict[str, Any]:
            return {
                "provider": str(getattr(ep, "provider", "remote") or "remote"),
                "base_url": ep.base_url,
                "model_id": ep.model_id,
                "has_api_key": bool(ep.api_key),
            }

        oj = getattr(self._cfg, "openjev", None)
        desktop = getattr(self._cfg, "desktop", None)
        return {
            "synthesis": _ep(self._cfg.models.synthesis),
            "synthesis_effective_base_url": self.resolve("synthesis").base_url,
            "jev": {
                "provider": str(getattr(oj, "provider", "remote") or "remote"),
                "base_url": getattr(oj, "base_url", ""),
                "enabled": bool(getattr(oj, "enabled", False)),
            },
            "jev_effective_base_url": self.jev_base_url(),
            "desktop": {
                "enabled": bool(getattr(desktop, "enabled", False)),
                "bridge_url": str(getattr(desktop, "bridge_url", "")),
                "jev_local_url": str(getattr(desktop, "jev_local_url", "")),
            },
        }
