"""Deterministic OpenAI-compatible mock server for regression and E2E tests.

Scripted behavior, no randomness:
  GET  /v1/models                      -> lists mock-synthesis / mock-fast / mock-embed
  POST /v1/chat/completions            -> card JSON for synthesis role
  POST /score                          -> deterministic OpenJEV decisions
  POST /v1/embeddings                  -> stable hash-derived vectors, dim 8
  GET  /v1/_mock/requests              -> recorded request log for assertions

(STT is streaming-only over raw TCP — it has no HTTP mock here. Tests
that need transcription run a fake SimulStreaming TCP server instead.)
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def _stable_vec(text: str, dim: int = 8) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = [struct.unpack("<I", digest[i * 4 : i * 4 + 4])[0] for i in range(dim)]
    scale = max(vals) or 1
    return [round(v / scale, 6) for v in vals]


class MockState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []

    def record(self, path: str, body: Any) -> None:
        with self.lock:
            self.requests.append({"path": path, "body": body})


def _content_of(req: dict[str, Any]) -> str:
    msgs = req.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _card_json(user_text: str) -> dict[str, Any]:
    lowered = user_text.lower()
    if "rule" in lowered or "grapple" in lowered:
        body = "**Grappled** (SRD 5.1): speed 0, ends if grappler conditions break."
        title = "Rules — Grappled Condition"
    elif "history" in lowered or "lore" in lowered:
        body = "The Ashforge lies east of the temple square, where Vex'ahlia's seal was found."
        title = "Lore Brief"
    elif "search" in lowered or "loot" in lowered:
        body = (
            "## Loot Table — Fallen Scout\n\n"
            "| Skill | DC | On Success |\n|---|---|---|\n"
            "| Investigation | 12 | Hidden pouch (5 gp, coded note) |\n"
            "| Perception | 14 | Boot dagger, unworn |\n"
            "| Medicine | 15 | Cause of death: crossbow bolt |\n\n"
            "**Passive Insight 15+** — the note bears Vex'ahlia's seal."
        )
        title = "Loot Table — Fallen Scout"
    else:
        body = "The forge district lies east of the temple square."
        title = "Lore Brief"
    kind = "rules" if title.startswith("Rules") else "skill_table"
    return {"kind": kind, "title": title, "body_md": body}


def create_mock_app(state: MockState | None = None) -> FastAPI:
    state = state or MockState()
    app = FastAPI(title="mock-openai")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        state.record("/v1/models", None)
        return {
            "object": "list",
            "data": [
                {"id": "mock-synthesis", "object": "model"},
                {"id": "mock-fast", "object": "model"},
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        req = await request.json()
        state.record("/v1/chat/completions", {k: v for k, v in req.items() if k != "messages"})
        model = str(req.get("model", ""))
        rf = req.get("response_format")
        if isinstance(rf, dict):
            schema = rf.get("json_schema", {})
            props = sorted((schema.get("schema", {}).get("properties") or {}).keys())
        else:
            props = []
        if "fast" in model:
            # Agent ephemeral tier (fast role, free-form): grounded scene note.
            content = json.dumps(
                {
                    "text": (
                        "The Ashforge is a dwarven foundry district east of the temple "
                        "square, governed by the Forge-master; Vex'ahlia bears the scouts' "
                        "sealed signet."
                    )
                }
            )
        else:
            content = json.dumps(_card_json(_content_of(req)))
        return JSONResponse(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
            }
        )

    @app.post("/score")
    async def score(request: Request) -> dict[str, Any]:
        """Deterministic OpenJEV wire-compatible scorer for full-path tests."""
        row = await request.json()
        state.record("/score", row)
        ids = [str(option["id"]) for option in row.get("options", [])]
        state_text = str(row.get("state", ""))
        text = state_text.splitlines()[-1].lower() if state_text else ""
        if ids == ["deploy", "wait"]:
            deploy = any(
                word in text
                for word in ("search", "loot", "history", "lore", "grapple", "rules")
            )
            probabilities = [0.9, 0.1] if deploy else [0.1, 0.9]
        elif ids == ["card", "ephemeral"]:
            probabilities = [0.9, 0.1]
        elif ids == ["offline", "online", "both"]:
            probabilities = [0.1, 0.1, 0.8]
        else:
            probability = 1.0 / max(len(ids), 1)
            probabilities = [probability for _ in ids]
        return {
            "id": row.get("id", "mock"),
            "option_ids": ids,
            "probabilities": probabilities,
        }

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> dict[str, Any]:
        req = await request.json()
        inputs = req.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": _stable_vec(str(t))}
                for i, t in enumerate(inputs)
            ],
            "model": req.get("model", "mock-embed"),
        }

    @app.get("/v1/_mock/requests")
    async def recorded() -> dict[str, Any]:
        with state.lock:
            return {"requests": list(state.requests)}

    return app
