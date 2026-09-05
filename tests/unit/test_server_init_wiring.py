"""Regression: the /api/init runner wiring (live defect, 2026-09-05).

The shipped service never indexed its project: `_make_init_runner` called
`dmd.init_pass.run_init` with keyword names that do not exist
(`path=`/`gateway=` and a `lexicon_entries=` parameter) so every init died
as a TypeError inside a background task, and the only trace was an
`init_progress error:TypeError` bus event nobody looked at. These tests pin
the wiring at the layer where the defect lived:

  * the runner calls run_init with its real signature (project_path/gw) and
    returns the result unchanged;
  * progress callbacks are invoked synchronously (init_pass._cb contract);
  * a completed init hot-swaps the live engine's lexicon
    (SessionEngine.refresh_lexicon) — re-init must not require a restart.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dmd import init_pass
from dmd.config import load_config_dict
from dmd.pipeline import SessionEngine
from dmd.server import _make_init_runner
from dmd.types import LexiconEntry


class _SpyEngine:
    def __init__(self) -> None:
        self.refreshed: list[Any] = []

    def refresh_lexicon(self, entries: list[Any]) -> dict[str, int]:
        self.refreshed.append(entries)
        return {"entries": len(entries), "hotwords": 0}


async def test_init_runner_uses_real_run_init_signature(monkeypatch) -> None:
    """A positional/keyword drift here silently killed every live init."""
    captured: dict[str, Any] = {}
    sentinel = init_pass.InitResult(n_docs=3, n_chunks=5, n_entities=2, lexicon=["L"])

    async def fake_run_init(**kw: Any) -> init_pass.InitResult:
        captured.update(kw)
        return sentinel

    monkeypatch.setattr(init_pass, "run_init", fake_run_init)
    engine = _SpyEngine()
    runner = _make_init_runner(
        cfg="CFG", store="STORE", gateway="GW", embedder="EMB", engine=engine
    )
    stages: list[str] = []
    result = await runner.run_init("/repo/path", progress_cb=stages.append)

    assert captured["project_path"] == "/repo/path"
    assert captured["cfg"] == "CFG"
    assert captured["store"] == "STORE"
    assert captured["gw"] == "GW"
    assert captured["embedder"] == "EMB"
    assert "path" not in captured
    assert "gateway" not in captured
    assert "lexicon_entries" not in captured
    assert result is sentinel
    assert engine.refreshed == [["L"]]


async def test_init_runner_survives_sync_progress_cb(monkeypatch) -> None:
    """init_pass._cb calls progress_cb without awaiting — sync cbs must work."""

    async def fake_run_init(**kw: Any) -> init_pass.InitResult:
        cb = kw["progress_cb"]
        cb("scan_folder")
        cb("embed")
        return init_pass.InitResult()

    monkeypatch.setattr(init_pass, "run_init", fake_run_init)
    runner = _make_init_runner(
        cfg=None, store=None, gateway=None, embedder=None, engine=None
    )
    seen: list[str] = []
    result = await runner.run_init("p", progress_cb=seen.append)
    assert seen == ["scan_folder", "embed"]
    assert result.n_docs == 0


async def test_init_runner_engine_without_refresh_lexicon_is_tolerated(
    monkeypatch,
) -> None:
    """A duck-typed engine lacking refresh_lexicon must not fail the init."""

    async def fake_run_init(**kw: Any) -> init_pass.InitResult:
        return init_pass.InitResult(n_docs=1)

    monkeypatch.setattr(init_pass, "run_init", fake_run_init)

    class _NoRefresh:
        pass

    runner = _make_init_runner(
        cfg=None,
        store=None,
        gateway=None,
        embedder=None,
        engine=_NoRefresh(),
    )
    result = await runner.run_init("p")
    assert result.n_docs == 1


def test_refresh_lexicon_rebuilds_hotword_prompt(tmp_path: Any) -> None:
    """The engine's live prompt/links follow a re-init without a restart."""
    cfg = load_config_dict(
        {
            "project": {"path": str(tmp_path)},
            "models": {
                "synthesis": {"base_url": "http://fake", "model_id": "m"},
                "stt": {"base_url": "http://fake"},
            },
        }
    )
    engine = SessionEngine(
        cfg=cfg,
        store=None,  # type: ignore[arg-type]
        gw=None,  # type: ignore[arg-type]
        entries=[],
        embedder=None,
        pool=None,  # type: ignore[arg-type]
        on_event=lambda e: None,
    )
    assert engine._hotword_prompt == ""
    info = engine.refresh_lexicon(
        [
            LexiconEntry(
                canonical="Ironwood Gate",
                variants=["iron gate"],
                etype="place",
                weight=1.0,
            ),
            LexiconEntry(
                canonical="Brother Alric",
                variants=[],
                etype="npc",
                weight=0.5,
            ),
        ]
    )
    assert info["entries"] == 2
    assert "Ironwood" in engine._hotword_prompt
    assert "Brother Alric" in engine._hotword_prompt
