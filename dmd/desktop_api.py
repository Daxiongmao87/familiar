"""Desktop provider endpoints: status and local/remote switching.

Mounted onto the FastAPI app by ``main()``; ``server.py`` itself stays
untouched. All responses are secret-free (only ``has_api_key`` booleans).
Provider switches apply live (no restart) and persist to the config file
without touching cached models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import Request

PROVIDERS = ("local", "remote")
_SYNTH_KEYS = ("provider", "base_url", "model_id", "api_key", "extra_body")
_JEV_KEYS = ("provider", "base_url")


def _is_url(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.strip() != ""
        and value.startswith(("http://", "https://"))
    )


def _validate_patch(payload: Any) -> tuple[dict[str, Any], str | None]:
    """Validate a provider patch; return (clean_patch, error)."""
    if not isinstance(payload, dict):
        return {}, "body must be a JSON object"
    clean: dict[str, Any] = {}
    synth = payload.get("synthesis")
    if synth is not None:
        if not isinstance(synth, dict):
            return {}, "synthesis must be an object"
        for key in synth:
            if key not in _SYNTH_KEYS:
                return {}, f"unknown synthesis key: {key}"
        one: dict[str, Any] = {}
        if "provider" in synth:
            if synth["provider"] not in PROVIDERS:
                return {}, "synthesis.provider must be local or remote"
            one["provider"] = synth["provider"]
        if "base_url" in synth:
            if not _is_url(synth["base_url"]):
                return {}, "synthesis.base_url must be an http(s) URL"
            one["base_url"] = synth["base_url"].rstrip("/")
        if "model_id" in synth:
            if not isinstance(synth["model_id"], str) or not synth["model_id"].strip():
                return {}, "synthesis.model_id must be a non-empty string"
            one["model_id"] = synth["model_id"]
        if "api_key" in synth:
            if synth["api_key"] is not None and not isinstance(synth["api_key"], str):
                return {}, "synthesis.api_key must be a string or null"
            one["api_key"] = synth["api_key"]
        if "extra_body" in synth:
            if not isinstance(synth["extra_body"], dict):
                return {}, "synthesis.extra_body must be an object"
            one["extra_body"] = synth["extra_body"]
        clean["synthesis"] = one
    jev = payload.get("jev")
    if jev is not None:
        if not isinstance(jev, dict):
            return {}, "jev must be an object"
        for key in jev:
            if key not in _JEV_KEYS:
                return {}, f"unknown jev key: {key}"
        one = {}
        if "provider" in jev:
            if jev["provider"] not in PROVIDERS:
                return {}, "jev.provider must be local or remote"
            one["provider"] = jev["provider"]
        if "base_url" in jev:
            if not _is_url(jev["base_url"]):
                return {}, "jev.base_url must be an http(s) URL"
            one["base_url"] = jev["base_url"].rstrip("/")
        clean["jev"] = one
    if not clean:
        return {}, "nothing to update (synthesis and/or jev required)"
    return clean, None


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _persist(config_path: str | None, patch: dict[str, Any]) -> bool:
    """Merge the patch into the raw YAML config file. False when unwritable."""
    if not config_path:
        return False
    try:
        path = Path(config_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(raw, dict):
            return False
        raw_patch: dict[str, Any] = {}
        if "synthesis" in patch:
            raw_patch.setdefault("models", {})["synthesis"] = patch["synthesis"]
        if "jev" in patch:
            raw_patch["openjev"] = patch["jev"]
        merged = _deep_merge(raw, raw_patch)
        path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
        return True
    except OSError:
        return False


async def _probe_url(base_url: str) -> tuple[bool, str]:
    """Light reachability probe: GET {base}/models with a short timeout."""
    url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url)
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}"
    if resp.status_code < 500:
        return True, f"http {resp.status_code}"
    return False, f"http {resp.status_code}"


async def _probe_jev(base_url: str, timeout_s: float) -> tuple[bool, str]:
    """Probe a JEV endpoint via its /health document."""
    from .openjev import OpenjevGate

    gate = OpenjevGate(base_url, timeout_s=min(3.0, float(timeout_s or 3.0)))
    try:
        await gate.health()
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}"
    finally:
        await gate.aclose()


def mount_desktop_api(
    app: Any,
    *,
    cfg: Any = None,
    engine: Any = None,
    gateway: Any = None,
    config_path: str | None = None,
) -> None:
    """Register /api/desktop/* routes on an existing FastAPI app."""

    @app.get("/api/desktop/status")
    async def desktop_status() -> dict[str, Any]:
        from .providers import EndpointRouter

        if cfg is None:
            return {"ok": False, "detail": "no config"}
        router = EndpointRouter(cfg)
        state = router.describe()
        synth_url = str(state.get("synthesis_effective_base_url") or "")
        jev_url = str(state.get("jev_effective_base_url") or "")
        synth_ok, synth_detail = await _probe_url(synth_url) if synth_url else (False, "no url")
        oj = getattr(cfg, "openjev", None)
        jev_enabled = bool(getattr(oj, "enabled", False)) if oj is not None else False
        if jev_enabled and jev_url:
            jev_ok, jev_detail = await _probe_jev(
                jev_url, float(getattr(oj, "timeout_s", 3.0) or 3.0)
            )
        else:
            jev_ok, jev_detail = False, "disabled" if not jev_enabled else "no url"
        state["synthesis_reachable"] = synth_ok
        state["synthesis_detail"] = synth_detail
        state["jev_reachable"] = jev_ok
        state["jev_detail"] = jev_detail
        return {"ok": True, **state}

    @app.post("/api/desktop/providers")
    async def desktop_providers(req: Request) -> dict[str, Any]:
        if cfg is None:
            return {"ok": False, "detail": "no config"}
        try:
            payload = await req.json()
        except Exception:
            return {"ok": False, "detail": "invalid json"}
        patch, error = _validate_patch(payload)
        if error is not None:
            return {"ok": False, "detail": error}
        try:
            if "synthesis" in patch:
                for key, value in patch["synthesis"].items():
                    setattr(cfg.models.synthesis, key, value)
            if "jev" in patch:
                for key, value in patch["jev"].items():
                    setattr(cfg.openjev, key, value)
        except AttributeError as exc:
            return {"ok": False, "detail": f"config shape: {exc}"}
        effective: dict[str, str] = {}
        if engine is not None:
            apply_fn = getattr(engine, "apply_providers", None)
            if callable(apply_fn):
                try:
                    effective = dict(apply_fn())
                except Exception:
                    effective = {}
        persisted = _persist(config_path, patch)
        return {"ok": True, "effective": effective, "persisted": persisted}
