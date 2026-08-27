"""OpenAI-compatible HTTP gateway, role-keyed.

One httpx.AsyncClient per Gateway, per-role EndpointConfig resolution,
zero vendor assumptions beyond the OpenAI REST dialect.
"""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import numpy as np

from .config import AppConfig, EndpointConfig


class GatewayError(RuntimeError):
    """Raised for HTTP-level gateway problems (non-200, missing config)."""


class Gateway:
    """OpenAI-compatible async client keyed by AppConfig roles."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
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
        if role == "stt":
            return m.stt
        raise GatewayError(f"unknown role: {role!r}")

    @staticmethod
    def _auth_headers(ep: EndpointConfig) -> dict[str, str]:
        if ep.api_key:
            return {"Authorization": f"Bearer {ep.api_key}"}
        return {}

    async def chat(
        self,
        role: str,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
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
        r = await self._client.post(url, json=body, headers=self._auth_headers(ep))
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

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "chunk.wav",
        prompt: str | None = None,
        extra: dict | None = None,
    ) -> str:
        ep = self._resolve("stt")
        if not ep.base_url:
            raise GatewayError("stt role has no base_url")
        if getattr(ep, "dialect", "openai") == "whisperx":
            wurl = ep.base_url.rstrip("/") + "/transcribe"
            r = await self._client.post(
                wurl,
                params={"diarize": "false", "align": "false"},
                content=audio_bytes,
                headers={"Content-Type": "audio/wav", **self._auth_headers(ep)},
            )
            if r.status_code != 200:
                raise GatewayError(f"stt http {r.status_code}: {r.text[:200]}")
            data = r.json()
            text = data.get("text") or ""
            if not text.strip():
                text = " ".join(
                    (seg.get("text") or "") for seg in data.get("segments", [])
                )
            return text.strip()
        url = ep.base_url.rstrip("/") + "/audio/transcriptions"
        files = {"file": (filename, audio_bytes, "audio/wav")}
        form: dict[str, str] = {}
        if prompt is not None:
            form["prompt"] = prompt
        if extra:
            for k, v in extra.items():
                form[k] = str(v)
        for k, v in ep.extra_body.items():
            form[k] = str(v)
        r = await self._client.post(
            url, files=files, data=form, headers=self._auth_headers(ep)
        )
        if r.status_code != 200:
            raise GatewayError(f"stt http {r.status_code}: {r.text[:200]}")
        return r.json().get("text", "")

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
        headers = (
            {"Authorization": f"Bearer {emb.api_key}"} if emb.api_key else {}
        )
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
        roles.append(("stt", m.stt))

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
            if name != "stt" and ep.extra_body and ep.model_id:
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