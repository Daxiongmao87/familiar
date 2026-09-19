"""OpenAI-compatible HTTP gateway, role-keyed.

One httpx.AsyncClient per Gateway, per-role EndpointConfig resolution,
zero vendor assumptions beyond the OpenAI REST dialect.

Slot-leak guard (2026-09-05 incident): every chat request MUST carry a
``max_tokens``. A llama.cpp request without one runs with ``n_predict=-1``;
when a small model repeats forever the client times out and abandons, but
the server slot keeps generating — one surrendered slot per timeout cadence.
``chat()`` therefore always bounds generation (caller arg > endpoint config >
per-role default), and honors a per-endpoint ``request_timeout_s`` so an
abandoned call cancels cleanly instead of holding the 180 s client default.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import httpx
import numpy as np

from .config import AppConfig, EndpointConfig

logger = logging.getLogger(__name__)

# Per-role generation caps used when neither the caller nor the endpoint
# config specifies max_tokens. Judge/trigger schemas are small; 1024 is
# generous for the fast lane. Card/agent prose gets a full card budget.
ROLE_DEFAULT_MAX_TOKENS: dict[str, int] = {
    "fast": 1024,
    "synthesis": 4096,
    "vision": 2048,
}
DEFAULT_MAX_TOKENS = 4096


class GatewayError(RuntimeError):
    """Raised for HTTP-level gateway problems (non-200, missing config)."""


class Gateway:
    """OpenAI-compatible async client keyed by AppConfig roles."""

    def __init__(self, cfg: AppConfig, router: Any = None) -> None:
        self._cfg = cfg
        # Optional provider router (dmd.providers.EndpointRouter). None
        # preserves the original behavior exactly: roles resolve straight
        # from config. A router reroutes synthesis/fast to the local
        # bridge when those roles select provider "local".
        self._router = router
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=180.0, pool=5.0)
        )

    @property
    def cfg(self) -> AppConfig:
        """Public access to the resolved config (roles, extra_body)."""
        return self._cfg

    async def aclose(self) -> None:
        await self._client.aclose()

    def _resolve(self, role: str) -> EndpointConfig:
        if self._router is not None and role in ("synthesis", "fast"):
            try:
                return self._router.resolve(role)
            except ValueError as exc:
                raise GatewayError(str(exc)) from exc
        m = self._cfg.models
        if role == "synthesis":
            return m.synthesis
        if role == "fast":
            if m.fast is None:
                raise GatewayError("fast role not configured")
            return m.fast
        if role == "vision":
            v = m.vision
            if v is None or not v.enabled or not v.base_url:
                raise GatewayError("vision role not enabled or not configured")
            return cast(EndpointConfig, v)
        raise GatewayError(f"unknown role: {role!r}")

    @staticmethod
    def _auth_headers(ep: EndpointConfig) -> dict[str, str]:
        if ep.api_key:
            return {"Authorization": f"Bearer {ep.api_key}"}
        return {}

    def _cap_max_tokens(self, role: str, ep: EndpointConfig, max_tokens: int | None) -> int:
        """Never let a chat request reach the wire without a generation cap."""
        if max_tokens is not None:
            return int(max_tokens)
        ep_cap = getattr(ep, "max_tokens", None)
        if ep_cap is not None:
            return int(ep_cap)
        return ROLE_DEFAULT_MAX_TOKENS.get(role, DEFAULT_MAX_TOKENS)

    async def chat(
        self,
        role: str,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> str | dict:
        ep = self._resolve(role)
        if not ep.base_url:
            raise GatewayError(f"role {role!r} has no base_url")
        if not ep.model_id:
            raise GatewayError(f"role {role!r} has no model_id")
        body: dict[str, Any] = {"model": ep.model_id, "messages": messages}
        body.update(ep.extra_body)
        if temperature is not None:
            body["temperature"] = temperature
        if thinking is not None:
            # Optional per-call thinking switch (llama.cpp honors it via
            # chat_template_kwargs; a top-level flag is ignored). Explicit
            # callers win over config extra_body.
            merged = dict(body.get("chat_template_kwargs") or {})
            merged["enable_thinking"] = thinking
            body["chat_template_kwargs"] = merged
        # Slot-leak guard: max_tokens is ALWAYS sent (see module docstring).
        body["max_tokens"] = self._cap_max_tokens(role, ep, max_tokens)
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        url = ep.base_url.rstrip("/") + "/chat/completions"
        req_timeout: Any = None
        ep_to = getattr(ep, "request_timeout_s", None)
        if ep_to is not None:
            req_timeout = httpx.Timeout(
                connect=5.0, read=float(ep_to), write=180.0, pool=5.0
            )
        try:
            r = await self._client.post(
                url, json=body, headers=self._auth_headers(ep), timeout=req_timeout
            )
        except httpx.TimeoutException as exc:
            # Abandoning mid-generation must not leave the caller hanging on a
            # half-read stream; close out as a clean, typed gateway failure.
            raise GatewayError(f"{role} chat timed out: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise GatewayError(f"{role} chat http {r.status_code}: {r.text[:200]}")
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, str):
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError, ValueError):
                return content
        return content

    async def stt_health(self) -> tuple[bool, str]:
        """Probe the streaming STT server reachability (TCP connect).

        Returns ``(reachable, detail)``. ``reachable is False`` means the UI
        must treat transcription as unavailable (degraded mode), which is the
        difference between "nobody is talking" and "transcription is dead".
        Uses a short timeout so a dead server can't stall callers.
        """
        from .streaming_stt import probe_server

        stt = self._cfg.models.stt
        host = getattr(stt, "stream_host", "")
        port = int(getattr(stt, "stream_port", 0) or 0)
        if not host or not port:
            return False, "no STT stream_host/stream_port configured"
        try:
            ok = await probe_server(host, port)
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            return False, f"{type(exc).__name__}: {exc}"
        if ok:
            return True, f"tcp {host}:{port} ok"
        return False, f"tcp {host}:{port} refused"

    async def embed(self, texts: list[str]) -> np.ndarray:
        emb = self._cfg.models.embeddings
        if emb.provider != "endpoint":
            raise NotImplementedError(
                "local embeddings handled by dmd.embedder.Embedder"
            )
        if not emb.base_url:
            raise GatewayError("embeddings endpoint has no base_url")
        url = emb.base_url.rstrip("/") + "/embeddings"
        body = {"model": emb.model_id, "input": texts}
        headers = {"Authorization": f"Bearer {emb.api_key}"} if emb.api_key else {}
        r = await self._client.post(url, json=body, headers=headers)
        if r.status_code != 200:
            raise GatewayError(f"embed http {r.status_code}: {r.text[:200]}")
        data = r.json()
        vectors = [d["embedding"] for d in data["data"]]
        return np.array(vectors, dtype=np.float32)

    async def probe_all(self) -> dict[str, str]:
        results: dict[str, str] = {}
        m = self._cfg.models
        roles: list[tuple[str, EndpointConfig]] = [("synthesis", m.synthesis)]
        if m.fast is not None:
            roles.append(("fast", m.fast))
        if m.vision is not None and m.vision.enabled and m.vision.base_url:
            roles.append(("vision", cast(EndpointConfig, m.vision)))

        for name, ep in roles:
            if not ep.base_url:
                continue
            url = ep.base_url.rstrip("/") + "/models"
            try:
                r = await self._client.get(url, headers=self._auth_headers(ep))
            except httpx.HTTPError as e:
                results[name] = f"error:{type(e).__name__}"
                continue
            if r.status_code != 200:
                results[name] = f"http_{r.status_code}"
                continue
            results[name] = "ok"
            try:
                payload = r.json()
                ids = [d.get("id") for d in payload.get("data", [])]
            except (json.JSONDecodeError, ValueError):
                ids = []
            if ep.model_id:
                results[f"{name}.model"] = (
                    "ok" if ep.model_id in ids else "missing_model"
                )
            if ep.extra_body and ep.model_id:
                body: dict[str, Any] = {
                    "model": ep.model_id,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                }
                body.update(ep.extra_body)
                chat_url = ep.base_url.rstrip("/") + "/chat/completions"
                try:
                    cr = await self._client.post(
                        chat_url, json=body, headers=self._auth_headers(ep)
                    )
                except httpx.HTTPError as e:
                    results[f"{name}.extra_body"] = (
                        f"extra_body_rejected:error:{type(e).__name__}"
                    )
                    continue
                if cr.status_code == 200:
                    results[f"{name}.extra_body"] = "extra_body_ok"
                else:
                    results[f"{name}.extra_body"] = (
                        f"extra_body_rejected:{cr.status_code}"
                    )
        return results
