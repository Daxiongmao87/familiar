"""FastAPI app, WebSocket event bus, REST endpoints, and entrypoint.

Duck-typed dependencies so the file imports even before pipeline.py and
init_pass.py exist; create_app() and main() handle the missing-engine case
gracefully and serve the UI in degraded mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

QUEUE_MAX = 512
BACKGROUND_TASKS: set[asyncio.Task] = set()


class EventBus:
    """Per-client asyncio.Queue fan-out with backpressure protection."""

    def __init__(self, max_pending: int = QUEUE_MAX) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._max_pending = max_pending
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._preloop: list[dict[str, Any]] = []

    def attach_loop(self) -> None:
        """Capture the running loop and flush events buffered before startup."""
        self._loop = asyncio.get_running_loop()
        if self._preloop:
            buffered, self._preloop = self._preloop, []
            _flushed: list = []
            for ev in buffered:
                _flushed.extend(asyncio.ensure_future(self.publish(ev)))

    def publish_sync(self, event: dict[str, Any]) -> None:
        """Sync/thread-safe bridge for producers without an event loop handle."""
        if not isinstance(event, dict):
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            self._preloop.append(event)
            return
        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self.publish(event))
            )
        except RuntimeError:
            self._preloop.append(event)

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_pending)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    async def publish(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        async with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            if q.qsize() >= self._max_pending:
                await self._drop_slow(q)
                continue
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                await self._drop_slow(q)

    async def _drop_slow(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(q)
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break


def _default_status_provider() -> dict[str, Any]:
    return {
        "project": "",
        "indexed_docs": 0,
        "entities": 0,
        "tools": 0,
        "roles": {},
    }


def _card_to_dict(card: Any) -> dict[str, Any]:
    if isinstance(card, dict):
        return {
            "id": str(card.get("id", "")),
            "kind": str(card.get("kind", "info")),
            "title": str(card.get("title", "")),
            "body_md": str(card.get("body_md", "")),
            "t_context": float(card.get("t_context", 0.0) or 0.0),
            "meta": dict(card.get("meta") or {}),
            "status": str(card.get("status") or "active"),
            "player_ids": [str(p) for p in (card.get("player_ids") or [])],
        }
    if is_dataclass(card):
        data = asdict(card)
    else:
        data = {
            "id": getattr(card, "id", ""),
            "kind": getattr(card, "kind", "info"),
            "title": getattr(card, "title", ""),
            "body_md": getattr(card, "body_md", ""),
            "t_context": getattr(card, "t_context", 0.0),
            "meta": getattr(card, "meta", {}),
            "status": getattr(card, "status", "active"),
            "player_ids": getattr(card, "player_ids", []),
        }
    return {
        "id": str(data.get("id", "")),
        "kind": str(data.get("kind", "info")),
        "title": str(data.get("title", "")),
        "body_md": str(data.get("body_md", "")),
        "t_context": float(data.get("t_context", 0.0) or 0.0),
        "meta": dict(data.get("meta") or {}),
        "status": str(data.get("status") or "active"),
        "player_ids": [str(p) for p in (data.get("player_ids") or [])],
    }


def create_app(
    cfg: Any = None,
    engine: Any = None,
    init_runner: Any = None,
    status_provider: Callable[[], dict[str, Any]] | None = None,
    web_dir: str | Path | None = None,
    bus: EventBus | None = None,
    browser_source: Any = None,
    speaking_tracker: Any = None,
) -> FastAPI:
    """Build the FastAPI app with optional injected dependencies.

    Pass `bus` when publishing events from externally-wired components
    (engine.on_event, pool.on_card); otherwise /ws gets a private bus and
    external publishes go nowhere.
    """
    bus = bus or EventBus()
    provider = status_provider or _default_status_provider
    web_path = (
        Path(web_dir)
        if web_dir is not None
        else Path(__file__).resolve().parent.parent / "web"
    )
    if not web_path.exists():
        web_path.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="DM Copilot", version="0.1.0")
    if browser_source is not None:
        try:
            app.state.browser_source = browser_source
        except Exception:
            pass
    if speaking_tracker is not None:
        try:
            app.state.speaking_tracker = speaking_tracker
        except Exception:
            pass

    @app.on_event("startup")
    async def _attach_bus_loop() -> None:
        bus.attach_loop()

    if cfg is not None:
        try:
            discord_cfg = getattr(cfg, "discord", None)
            token = getattr(discord_cfg, "token", None) if discord_cfg else None
            guild_id = getattr(discord_cfg, "guild_id", None) if discord_cfg else None
            dm_user_id = (
                getattr(discord_cfg, "dm_user_id", None) if discord_cfg else None
            )
            if (
                isinstance(token, str)
                and token.startswith("${")
                and token.endswith("}")
            ):
                import os

                token = os.environ.get(token[2:-1])
            if token and guild_id and dm_user_id:
                from dmd.speaking_tracker import SpeakingTracker
                from dmd.voice_presence import VoicePresence

                tracker = getattr(app.state, "speaking_tracker", None)
                if tracker is None:
                    tracker = SpeakingTracker()
                    app.state.speaking_tracker = tracker
                presence = VoicePresence(
                    str(token), int(guild_id), int(dm_user_id), tracker
                )
                app.state.voice_presence = presence

                @app.on_event("startup")
                async def _start_voice_presence() -> None:
                    await presence.start()

                @app.on_event("shutdown")
                async def _stop_voice_presence() -> None:
                    await presence.stop(getattr(cfg, "shutdown_timeout_s", 10.0))
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("voice presence not started: %s", exc)

    if web_path.exists():
        try:
            app.mount("/static", StaticFiles(directory=str(web_path)), name="static")
        except Exception:
            pass

    index_file = web_path / "index.html"

    @app.get("/")
    async def root() -> Any:
        if index_file.exists():
            return FileResponse(str(index_file), media_type="text/html")
        return JSONResponse(
            {"ok": False, "detail": "index.html missing"},
            status_code=500,
        )

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        try:
            data = provider()
            if not isinstance(data, dict):
                return _default_status_provider()
            return data
        except Exception:
            return _default_status_provider()

    @app.post("/api/query")
    async def api_query(req: Request) -> dict[str, Any]:
        if engine is None:
            return {"ok": False, "detail": "no engine"}
        try:
            payload = await req.json()
        except Exception:
            return {"ok": False, "detail": "invalid json"}
        text = ""
        if isinstance(payload, dict):
            text = str(payload.get("text", "") or "")
        if not text.strip():
            return {"ok": False, "detail": "empty text"}
        manual_query = getattr(engine, "manual_query", None)
        if manual_query is None:
            return {"ok": False, "detail": "no engine"}
        try:
            await manual_query(text)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "detail": f"error:{type(exc).__name__}"}

    @app.post("/api/init")
    async def api_init(req: Request) -> dict[str, Any]:
        if init_runner is None:
            return {"ok": False, "detail": "no init runner"}
        try:
            payload = await req.json()
        except Exception:
            return {"ok": False, "detail": "invalid json"}
        path = ""
        if isinstance(payload, dict):
            path = str(payload.get("path", "") or "")
        if not path.strip():
            return {"ok": False, "detail": "empty path"}
        run_init = getattr(init_runner, "run_init", None)
        if run_init is None:
            return {"ok": False, "detail": "no init runner"}

        def _progress(stage: Any) -> None:
            # run_init's progress_cb is sync (see dmd.init_pass._cb): publishing
            # through the sync bridge keeps an async cb from being fire-and-
            # forgotten as an un-awaited coroutine.
            bus.publish_sync({"type": "init_progress", "stage": str(stage)})

        async def _run() -> None:
            try:
                result = await run_init(path, progress_cb=_progress)
                bus.publish_sync(
                    {
                        "type": "init_progress",
                        "stage": "done",
                        "n_docs": int(getattr(result, "n_docs", 0) or 0),
                        "n_chunks": int(getattr(result, "n_chunks", 0) or 0),
                        "n_entities": int(getattr(result, "n_entities", 0) or 0),
                    }
                )
            except Exception as exc:
                logging.getLogger(__name__).exception("init failed")
                await bus.publish(
                    {
                        "type": "init_progress",
                        "stage": f"error:{type(exc).__name__}",
                    }
                )

        _t = asyncio.create_task(_run())
        BACKGROUND_TASKS.add(_t)
        _t.add_done_callback(BACKGROUND_TASKS.discard)
        return {"ok": True}

    @app.get("/api/guild/members")
    async def api_guild_members() -> dict[str, Any]:
        if cfg is None:
            return {"ok": False, "detail": "no config", "members": []}
        discord_cfg = getattr(cfg, "discord", None)
        token = getattr(discord_cfg, "token", None) if discord_cfg else None
        guild_id = getattr(discord_cfg, "guild_id", None) if discord_cfg else None
        if not token or not guild_id:
            return {"ok": False, "detail": "discord not configured", "members": []}
        if isinstance(token, str) and token.startswith("${") and token.endswith("}"):
            import os

            token = os.environ.get(token[2:-1])
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"https://discord.com/api/v10/guilds/{guild_id}/members",
                    headers={"Authorization": f"Bot {token}"},
                    params={"limit": "1000"},
                )
                if resp.status_code != 200:
                    return {
                        "ok": False,
                        "detail": f"discord {resp.status_code}",
                        "members": [],
                    }
                raw = resp.json()
                members = []
                for m in raw if isinstance(raw, list) else []:
                    user = m.get("user") or {}
                    members.append(
                        {
                            "id": str(user.get("id", "")),
                            "username": str(user.get("username", "")),
                            "display_name": str(
                                m.get("nick")
                                or user.get("global_name")
                                or user.get("username")
                                or ""
                            ),
                            "avatar": str(user.get("avatar") or ""),
                        }
                    )
                return {"ok": True, "members": members}
        except Exception as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}", "members": []}

    @app.get("/api/config/dm")
    async def api_get_dm() -> dict[str, Any]:
        if cfg is None:
            return {"ok": False, "dm_user_id": None}
        discord_cfg = getattr(cfg, "discord", None)
        dm = getattr(discord_cfg, "dm_user_id", None) if discord_cfg else None
        return {"ok": True, "dm_user_id": dm}

    @app.post("/api/config/dm")
    async def api_set_dm(req: Request) -> dict[str, Any]:
        if cfg is None:
            return {"ok": False, "detail": "no config"}
        try:
            payload = await req.json()
        except Exception:
            return {"ok": False, "detail": "invalid json"}
        dm_id = payload.get("dm_user_id")
        if dm_id is not None and not isinstance(dm_id, str):
            dm_id = str(dm_id)
        if dm_id is not None:
            dm_id = dm_id.strip() or None
            if dm_id is not None and not dm_id.isdigit():
                # treat as username — try to resolve via guild members
                discord_cfg = getattr(cfg, "discord", None)
                token = getattr(discord_cfg, "token", None) if discord_cfg else None
                guild_id = (
                    getattr(discord_cfg, "guild_id", None) if discord_cfg else None
                )
                if token and guild_id:
                    try:
                        import httpx

                        async with httpx.AsyncClient(timeout=10.0) as client:
                            resp = await client.get(
                                f"https://discord.com/api/v10/guilds/{guild_id}/members",
                                headers={"Authorization": f"Bot {token}"},
                                params={"limit": "1000"},
                            )
                            if resp.status_code == 200:
                                for m in resp.json():
                                    user = m.get("user") or {}
                                    if (
                                        str(user.get("username", "")).lower()
                                        == dm_id.lower()
                                        or str(m.get("nick") or "").lower()
                                        == dm_id.lower()
                                    ):
                                        dm_id = str(user.get("id"))
                                        break
                    except Exception:
                        pass
        try:
            cfg.discord.dm_user_id = dm_id  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            import yaml

            raw_path = str(getattr(cfg, "_config_path", None) or "config.yaml")
            p = Path(raw_path)
            if not p.exists():
                p = Path("config.yaml")
            if p.exists():
                raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                if "discord" not in raw:
                    raw["discord"] = {}
                raw["discord"]["dm_user_id"] = dm_id
                p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        except Exception:
            pass
        return {"ok": True, "dm_user_id": dm_id}

    # --- Full config editor (GET/POST /api/config) ---
    SECRET_PATHS = (
        "discord.token",
        "models.synthesis.api_key",
        "models.fast.api_key",
        "models.stt.api_key",
        "models.embeddings.api_key",
        "models.vision.api_key",
    )

    def _mask_secrets(d: dict) -> dict:
        import copy

        out = copy.deepcopy(d)
        for path in SECRET_PATHS:
            parts = path.split(".")
            cur = out
            for part in parts[:-1]:
                cur = cur.get(part) if isinstance(cur, dict) else None
                if not isinstance(cur, dict):
                    cur = None
                    break
            if (
                isinstance(cur, dict)
                and isinstance(cur.get(parts[-1]), str)
                and cur.get(parts[-1])
            ):
                cur[parts[-1]] = "__MASKED__"
        return out

    def _effective(raw: dict) -> dict:
        """Config with all pydantic defaults applied — the values the app actually uses."""
        try:
            from dmd.config import AppConfig

            return AppConfig.model_validate(raw).model_dump()
        except Exception:
            return raw

    def _deep_merge(base: dict, patch: dict) -> dict:
        import copy

        out = copy.deepcopy(base)
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)
            elif v == "__MASKED__" and out.get(k) is not None:
                continue
            else:
                out[k] = v
        return out

    def _config_file() -> Path:
        p = Path(str(getattr(cfg, "_config_path", None) or "config.yaml"))
        return p if p.exists() else Path("config.yaml")

    @app.get("/api/config")
    async def api_get_config() -> dict[str, Any]:
        try:
            import yaml

            p = _config_file()
            if not p.exists():
                return {"ok": False, "detail": "config.yaml not found"}
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return {"ok": True, "config": _mask_secrets(_effective(raw))}
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    @app.post("/api/config")
    async def api_set_config(req: Request) -> dict[str, Any]:
        try:
            payload = await req.json()
        except Exception:
            return {"ok": False, "detail": "invalid json"}
        patch = payload.get("config")
        if not isinstance(patch, dict):
            return {"ok": False, "detail": "missing config object"}
        try:
            import yaml

            p = _config_file()
            if not p.exists():
                return {"ok": False, "detail": "config.yaml not found"}
            base_raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            base = _effective(base_raw)
            merged = _deep_merge(base, patch)
            if "server" in base_raw:
                merged["server"] = base_raw["server"]
            changed = [k for k in patch if merged.get(k) != base.get(k)]
            p.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
            return {"ok": True, "changed": changed, "restart_required": bool(changed)}
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    @app.post("/api/capture")
    async def api_capture(req: Request) -> dict[str, Any]:
        """Pause/resume audio capture into the pipeline (SPEC §15 control)."""
        if engine is None:
            return {"ok": False, "detail": "no engine"}
        try:
            payload = await req.json()
        except Exception:
            payload = {}
        paused = bool(payload.get("paused", False)) if isinstance(payload, dict) else False
        setter = getattr(engine, "set_capture_paused", None)
        if setter is None:
            return {"ok": False, "detail": "engine has no capture control"}
        try:
            state = setter(paused)
            return {"ok": True, "paused": bool(state.get("paused"))}
        except Exception as exc:
            return {"ok": False, "detail": f"error:{type(exc).__name__}"}

    @app.post("/api/ooc")
    async def api_ooc(req: Request) -> dict[str, Any]:
        """Toggle out-of-character mode: transcript keeps flowing, triggers
        and the proactive monitor stay silent (SPEC §15 control)."""
        if engine is None:
            return {"ok": False, "detail": "no engine"}
        try:
            payload = await req.json()
        except Exception:
            payload = {}
        on = bool(payload.get("on", False)) if isinstance(payload, dict) else False
        setter = getattr(engine, "set_ooc", None)
        if setter is None:
            return {"ok": False, "detail": "engine has no ooc control"}
        try:
            state = setter(on)
            return {"ok": True, "on": bool(state.get("on"))}
        except Exception as exc:
            return {"ok": False, "detail": f"error:{type(exc).__name__}"}

    @app.post("/api/card/done")
    async def api_card_done(req: Request) -> dict[str, Any]:
        """Mark a card done — set aside, never deleted (SPEC §9 lifecycle)."""
        if engine is None:
            return {"ok": False, "detail": "no engine"}
        try:
            payload = await req.json()
        except Exception:
            return {"ok": False, "detail": "invalid json"}
        card_id = str(payload.get("card_id", "") or "") if isinstance(payload, dict) else ""
        if not card_id.strip():
            return {"ok": False, "detail": "empty card_id"}
        marker = getattr(engine, "mark_card_done", None)
        if marker is None:
            return {"ok": False, "detail": "engine has no card lifecycle"}
        try:
            marked = await marker(card_id)
            return {"ok": bool(marked), "card_id": card_id}
        except Exception as exc:
            return {"ok": False, "detail": f"error:{type(exc).__name__}"}

    @app.get("/api/players")
    async def api_players() -> dict[str, Any]:
        """Seeded players (id + display name) for card player-badges (§9/§15)."""
        ps = getattr(engine, "player_state", None) if engine is not None else None
        if ps is None:
            return {"ok": True, "players": []}
        try:
            players = [
                {"id": str(p.get("player_id", "")), "name": str(p.get("name", "") or p.get("player_id", ""))}
                for p in ps.all_players()
            ]
            return {"ok": True, "players": players}
        except Exception as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}", "players": []}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        q = await bus.subscribe()
        try:
            while True:
                try:
                    event = await q.get()
                except asyncio.CancelledError:
                    break
                try:
                    await websocket.send_text(json.dumps(event, ensure_ascii=False))
                except Exception:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            await bus.unsubscribe(q)
            try:
                await websocket.close()
            except Exception:
                pass

    @app.websocket("/ws/audio")
    async def ws_audio(websocket: WebSocket) -> None:
        await websocket.accept()
        browser_source = getattr(app.state, "browser_source", None)
        if browser_source is None:
            try:
                from dmd.sources.browser import BrowserAudioSource

                browser_source = BrowserAudioSource()
                app.state.browser_source = browser_source
            except Exception:
                await websocket.close(code=1011)
                return
        try:
            while True:
                msg = await websocket.receive()
                data = msg.get("bytes")
                if data is not None:
                    browser_source.push_chunk(bytes(data))
                elif "text" in msg and msg["text"] is not None:
                    try:
                        import base64

                        raw = base64.b64decode(msg["text"])
                        browser_source.push_chunk(raw)
                    except Exception:
                        pass
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    return app


def _build_status_provider(
    cfg: Any,
    store: Any,
    gateway: Any,
    stt_monitor: Any = None,
    engine: Any = None,
) -> Callable[[], dict[str, Any]]:

    def _snapshot() -> dict[str, Any]:
        project = str(getattr(getattr(cfg, "project", None), "name", "") or "")
        indexed_docs = 0
        entities = 0
        tools = 0
        if store is not None:
            try:
                counts = store.counts()
                indexed_docs = int(counts.get("docs", 0))
                entities = int(counts.get("entities", 0))
            except Exception:
                indexed_docs = 0
                entities = 0
        roles: dict[str, bool] = {}
        if gateway is not None:
            for role in ("synthesis", "fast", "vision", "stt"):
                try:
                    ep = gateway._resolve(role)
                    roles[role] = bool(getattr(ep, "base_url", None))
                except Exception:
                    roles[role] = False
        else:
            roles = {"synthesis": False, "fast": False, "vision": False, "stt": False}
        out: dict[str, Any] = {
            "project": project,
            "indexed_docs": indexed_docs,
            "entities": entities,
            "stt_health": (
                stt_monitor.snapshot()
                if stt_monitor is not None
                else {"healthy": None, "detail": ""}
            ),
        }
        if engine is not None:
            try:
                out["controls"] = {
                    "capture_paused": bool(getattr(engine, "capture_paused", False)),
                    "ooc": bool(getattr(engine, "ooc", False)),
                }
            except Exception:
                pass
            tracker = getattr(engine, "speaking_tracker", None)
            if tracker is not None:
                try:
                    snap = tracker.snapshot()
                    names = snap.get("named", {}) if isinstance(snap, dict) else {}
                    out["speakers"] = [
                        {"id": uid, "name": names.get(uid)}
                        for uid in sorted(snap.get("active", {}))
                    ]
                except Exception:
                    pass
        return out

    return _snapshot


def _make_init_runner(
    cfg: Any,
    store: Any,
    gateway: Any,
    embedder: Any,
    engine: Any = None,
) -> Any:
    """Build the /api/init runner over dmd.init_pass.run_init.

    The init_pass signature is positional-named `project_path/cfg/store/gw/
    embedder/progress_cb`; a mismatched keyword here once made every live
    init die as an invisible TypeError inside a background task (found
    2026-09-05: the shipped index stayed empty because of it). On success,
    the live engine's lexicon is refreshed in place so a re-init takes effect
    without a server restart.
    """

    try:
        from dmd.init_pass import run_init as _run_init

        class _Runner:
            async def run_init(
                self,
                path: str,
                progress_cb: Callable[[str], None] | None = None,
            ) -> Any:
                result = await _run_init(
                    project_path=path,
                    cfg=cfg,
                    store=store,
                    gw=gateway,
                    embedder=embedder,
                    progress_cb=progress_cb,
                )
                if engine is not None:
                    refresh = getattr(engine, "refresh_lexicon", None)
                    if callable(refresh):
                        try:
                            refresh(list(getattr(result, "lexicon", []) or []))
                        except Exception:
                            pass
                return result

        return _Runner()
    except Exception:
        return None


def _seed_players(player_state: Any, project_path: str) -> None:
    """Seed the player store from the campaign's characters/ dir (if present)."""
    if not project_path:
        return
    chardir = Path(project_path) / "characters"
    if not chardir.is_dir():
        return
    for f in sorted(chardir.glob("*.md")):
        pid = f.stem
        name = pid.replace("_", " ").replace("-", " ").title()
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if text.startswith("---"):
                end = text.find("---", 3)
                if end != -1:
                    for line in text[3:end].splitlines():
                        if line.lower().startswith("title:"):
                            name = line.split(":", 1)[1].strip().strip('"')
        except Exception:
            pass
        try:
            player_state.seed(
                [{"id": pid, "name": name, "sheet": f"characters/{f.name}"}]
            )
        except Exception:
            pass


def _make_engine(
    cfg: Any,
    store: Any,
    gateway: Any,
    embedder: Any,
    lexicon_entries: Any,
    pool: Any,
    bus: EventBus,
    db_dir: Any = None,
    speaking_tracker: Any = None,
) -> Any:
    try:
        from dmd.pipeline import SessionEngine

        project_path = ""
        if cfg is not None:
            project_path = str(getattr(getattr(cfg, "project", None), "path", "") or "")

        def _on_event(event: dict[str, Any]) -> None:
            bus.publish_sync(event)

        player_state = None
        if db_dir is not None:
            try:
                from dmd.player_state import PlayerState

                player_state = PlayerState(str(Path(db_dir) / "players.db"))
                _seed_players(player_state, project_path)
            except Exception:
                player_state = None

        # The agent's run_tool drives the existing tools_reg (async-probed repo
        # scripts); wiring it needs a startup hook, so the agent runs on its
        # built-in tools (retrieve / repo_read / web) for now.
        tool_registry = None

        world_map = ""
        try:
            from dmd.world_map import build_world_map

            players = player_state.all_players() if player_state is not None else None
            tool_names = []
            world_map = build_world_map(
                project_path, store, players=players, tools=tool_names
            )
        except Exception:
            world_map = ""

        engine = SessionEngine(
            cfg=cfg,
            store=store,
            gw=gateway,
            entries=lexicon_entries,
            embedder=embedder,
            pool=pool,
            on_event=_on_event,
            project_path=project_path,
            tool_registry=tool_registry,
            player_state=player_state,
            world_map=world_map,
            speaking_tracker=speaking_tracker,
        )
        return engine
    except Exception:
        return None


def _make_pool(cfg: Any, bus: EventBus) -> Any:
    try:
        from dmd.orchestrator import JobPool

        orch_cfg = getattr(cfg, "orchestration", None)
        max_concurrent = int(getattr(orch_cfg, "max_concurrent", 3) or 3)
        job_timeout_s = float(getattr(orch_cfg, "job_timeout_s", 20.0) or 20.0)
        stale_after_s = float(getattr(orch_cfg, "stale_after_s", 120.0) or 120.0)

        async def _on_card(card: Any) -> None:
            await bus.publish({"type": "card", "card": _card_to_dict(card)})

        async def _on_drop(job: Any, reason: str) -> None:
            # A dropped synthesis job must be visible, not silent (a card that
            # never arrives is indistinguishable from a hung session to the DM).
            await bus.publish(
                {
                    "type": "job_dropped",
                    "kind": str(getattr(job, "kind", "")),
                    "reason": str(reason),
                    "t": time.time(),
                }
            )

        return JobPool(
            max_concurrent=max_concurrent,
            job_timeout_s=job_timeout_s,
            stale_after_s=stale_after_s,
            on_card=_on_card,
            on_drop=_on_drop,
        )
    except Exception:
        return None


def main(config_path: str) -> None:
    """Entry point: build app with real or degraded dependencies, then serve."""
    cfg: Any = None
    store: Any = None
    gateway: Any = None
    embedder: Any = None
    lexicon_entries: Any = []
    pool: Any = None
    engine: Any = None
    init_runner: Any = None
    db_dir: Any = None
    bus = EventBus()

    try:
        from dmd.config import load_config

        cfg = load_config(config_path)
        try:
            object.__setattr__(cfg, "_config_path", config_path)
        except Exception:
            pass
    except Exception:
        cfg = None

    if cfg is not None:
        project = getattr(cfg, "project", None)
        project_path = getattr(project, "path", None) if project else None
        if project_path:
            db_dir = Path(project_path) / ".dmd"
        else:
            db_dir = Path.cwd() / "data"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(db_dir / "index.db")

        try:
            from dmd.index_store import IndexStore

            store = IndexStore(db_path)
        except Exception:
            store = None

        try:
            from dmd.gateway import Gateway

            gateway = Gateway(cfg)
        except Exception:
            gateway = None
        if gateway is not None:
            try:
                from dmd.stt_health import SttHealthMonitor

                stt_monitor = SttHealthMonitor(gateway, bus=bus)
            except Exception:
                stt_monitor = None
        try:
            from dmd.embedder import Embedder

            emb_cfg = getattr(cfg.models, "embeddings", None)
            embedder = Embedder(model_id=emb_cfg.model_id) if emb_cfg else Embedder()
        except Exception:
            embedder = None

        if store is not None:
            try:
                from dmd.lexicon import build_lexicon

                lexicon_entries = build_lexicon(store.all_entities())
            except Exception:
                lexicon_entries = []

    pool = _make_pool(cfg, bus)
    speaking_tracker: Any = None
    try:
        from dmd.speaking_tracker import SpeakingTracker

        speaking_tracker = SpeakingTracker()
    except Exception:
        speaking_tracker = None
    engine = _make_engine(
        cfg,
        store,
        gateway,
        embedder,
        lexicon_entries,
        pool,
        bus,
        db_dir=db_dir,
        speaking_tracker=speaking_tracker,
    )
    init_runner = _make_init_runner(cfg, store, gateway, embedder, engine=engine)
    status_provider = _build_status_provider(cfg, store, gateway, stt_monitor, engine=engine)

    try:
        from dmd.sources.browser import BrowserAudioSource

        browser_source = BrowserAudioSource()
    except Exception:
        browser_source = None

    app = create_app(
        cfg=cfg,
        engine=engine,
        init_runner=init_runner,
        status_provider=status_provider,
        browser_source=browser_source,
        bus=bus,
        speaking_tracker=speaking_tracker,
    )

    if browser_source is not None and engine is not None:
        try:
            consume = getattr(engine, "consume_source", None)
            if callable(consume):

                async def _browser_consumer() -> None:
                    try:
                        await consume(browser_source)
                    except Exception as exc:
                        import logging

                        logging.getLogger(__name__).warning(
                            "browser consumer ended: %s", exc
                        )

                @app.on_event("startup")
                async def _start_browser_consumer() -> None:
                    import asyncio

                    _t = asyncio.create_task(_browser_consumer())
                    BACKGROUND_TASKS.add(_t)
                    _t.add_done_callback(BACKGROUND_TASKS.discard)

        except Exception:
            pass

    if engine is not None:
        start_mon = getattr(engine, "start_monitor", None)
        stop_mon = getattr(engine, "stop_monitor", None)
        if callable(start_mon):

            @app.on_event("startup")
            async def _start_monitor() -> None:
                try:
                    start_mon()
                except Exception:
                    pass

        if callable(stop_mon):

            @app.on_event("shutdown")
            async def _stop_monitor() -> None:
                try:
                    stop_mon()
                except Exception:
                    pass

    if stt_monitor is not None:

        @app.on_event("startup")
        async def _start_stt_health_monitor() -> None:
            try:
                await stt_monitor.start()
            except Exception:
                pass

        @app.on_event("shutdown")
        async def _stop_stt_health_monitor() -> None:
            try:
                await stt_monitor.stop()
            except Exception:
                pass

    @app.on_event("shutdown")
    async def _stop_engine() -> None:
        if engine is not None:
            try:
                await engine.aclose()
            except Exception:
                pass

    host = (
        os.environ.get(
            "DMD_HOST",
            getattr(getattr(cfg, "server", None), "host", "0.0.0.0") or "0.0.0.0",
        )
        if cfg
        else os.environ.get("DMD_HOST", "0.0.0.0")
    )
    port = int(
        os.environ.get(
            "DMD_PORT", str(getattr(getattr(cfg, "server", None), "port", 8760) or 8760)
        )
        or 8760
    )
    use_https = (
        bool(getattr(getattr(cfg, "server", None), "https_enabled", False))
        if cfg
        else False
    )
    cert_file = (
        getattr(getattr(cfg, "server", None), "cert_file", None) if cfg else None
    )
    key_file = getattr(getattr(cfg, "server", None), "key_file", None) if cfg else None

    try:
        import uvicorn

        if use_https and cert_file and key_file:
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info",
                ssl_certfile=str(cert_file),
                ssl_keyfile=str(key_file),
            )
        else:
            if use_https:
                import logging

                logging.getLogger(__name__).warning(
                    "server.https_enabled true but cert_file/key_file missing — falling back to http"
                )
            uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        if gateway is not None:
            try:
                import asyncio

                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(gateway.aclose())
                finally:
                    loop.close()
            except Exception:
                pass


if __name__ == "__main__":
    import sys

    config_arg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_arg)
