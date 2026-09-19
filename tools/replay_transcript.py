"""Timed transcript replay: inject a fixture at recorded offsets.

Loads a transcript fixture (``t``/``text``/``type``/``user_id`` entries),
builds the real pipeline composition (Gateway + IndexStore + SessionEngine),
and dispatches each entry through ``handle_utterance`` paced to wall-clock
time — mimicking live play for detection, retrieval, and publishing
observation. Every utterance, trigger verdict, and published card is logged.

Model overrides are CLI flags so live runs never require editing config::

    python tools/replay_transcript.py \\
        tests/regression/golden/cr2e2_3h14m29s_crownsguard.json \\
        --openjev \\
        --synthesis-base-url http://192.168.0.200:8080/v1 \\
        --synthesis-model minicpm5-2b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmd.config import load_config
from dmd.embedder import Embedder
from dmd.gateway import Gateway
from dmd.index_store import IndexStore
from dmd.init_pass import run_init
from dmd.lexicon import build_lexicon
from dmd.orchestrator import JobPool
from dmd.pipeline import SessionEngine
from dmd.sources.transcript_replay import (
    TranscriptReplayer,
    load_transcript_events,
)
from dmd.types import Utterance


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture", help="transcript fixture JSON path")
    p.add_argument("--config", default="config.yaml", help="app config path")
    p.add_argument(
        "--fast",
        action="store_true",
        help="dispatch back-to-back instead of wall-clock paced",
    )
    p.add_argument(
        "--openjev",
        action="store_true",
        help="enable the openjev deploy/wait gate for this run",
    )
    p.add_argument("--openjev-url", default=None, help="openjev-serve base URL")
    p.add_argument("--openjev-threshold", type=float, default=None)
    p.add_argument("--openjev-debounce", type=float, default=None, help="same-kind redeploy window seconds")
    p.add_argument(
        "--directed",
        action="store_true",
        help="use the JEV-routed deterministic worker instead of the agent loop",
    )
    p.add_argument("--max-concurrent", type=int, default=None, help="pool concurrency override")
    p.add_argument("--synthesis-base-url", default=None)
    p.add_argument("--synthesis-model", default=None)
    p.add_argument("--fast-base-url", default=None)
    p.add_argument("--fast-model", default=None)
    p.add_argument("--jsonl", default=None, help="machine log path (JSON lines)")
    return p.parse_args(argv)


class Recorder:
    """Human log to stdout plus optional JSONL; counts the run summary."""

    def __init__(self, jsonl_path: str | None) -> None:
        self._fh = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None
        self.utterances = 0
        self.deploys = 0
        self.waits = 0
        self.debounced = 0
        self.detect_ms: list[float] = []
        self.cards: list[dict[str, Any]] = []

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _write(self, record: dict[str, Any]) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps(record, sort_keys=True) + "\n")

    def utterance(self, t: float, u: Utterance) -> None:
        self.utterances += 1
        print(f"[+{t:6.1f}s] {u.user_id}: {u.text}", flush=True)
        self._write({"ev": "utterance", "t": t, "user_id": u.user_id, "text": u.text})

    def event(self, ev: dict[str, Any]) -> None:
        self._write({"ev": "engine", **ev})
        if ev.get("type") == "turn_latency":
            self.deploys += 1
            self.detect_ms.append(float(ev.get("detect_ms", 0.0)))
            oj = ev.get("openjev") or {}
            extra = f" p={oj.get('p_deploy')} p_tier={oj.get('p_tier')}" if oj else ""
            print(
                f"    -> DEPLOY tier={ev.get('tier')} "
                f"detect_ms={ev.get('detect_ms')}{extra}",
                flush=True,
            )
        elif ev.get("type") == "card":
            card = ev.get("card", {})
            self.cards.append(card)
            print(
                f"    ## CARD kind={card.get('kind')} tier={card.get('tier', '')} "
                f"title={card.get('title')}",
                flush=True,
            )
        elif ev.get("type") == "scene_context":
            print(f"    .. scene: {str(ev.get('text', ''))[:150]}", flush=True)
        elif ev.get("type") == "trigger_verdict":
            if ev.get("debounced"):
                self.debounced += 1
                print(f"    .. debounced p={ev.get('p_deploy')}", flush=True)
            elif not ev.get("deploy"):
                self.waits += 1
                print(f"    .. wait p={ev.get('p_deploy')}", flush=True)

    def card_dropped(self, job: Any, reason: str) -> None:
        print(f"    !! DROPPED kind={getattr(job, 'kind', '?')}: {reason}", flush=True)
        self._write({"ev": "dropped", "kind": str(getattr(job, "kind", "")), "reason": reason})

    def summary(self, wall_s: float) -> None:
        avg = sum(self.detect_ms) / len(self.detect_ms) if self.detect_ms else 0.0
        print(f"--- utterances={self.utterances} deploys={self.deploys} waits={self.waits} "
              f"debounced={self.debounced} cards={len(self.cards)} avg_detect_ms={avg:.1f} wall_s={wall_s:.1f}")


async def _amain(args: argparse.Namespace) -> int:
    events = load_transcript_events(args.fixture)
    print(f"loaded {len(events)} timed entries from {args.fixture}", flush=True)

    cfg = load_config(args.config)
    if args.synthesis_base_url:
        cfg.models.synthesis.base_url = args.synthesis_base_url
    if args.synthesis_model:
        cfg.models.synthesis.model_id = args.synthesis_model
    if args.fast_base_url and cfg.models.fast is not None:
        cfg.models.fast.base_url = args.fast_base_url
    if args.fast_model and cfg.models.fast is not None:
        cfg.models.fast.model_id = args.fast_model
    if args.openjev:
        cfg.openjev.enabled = True
    if args.openjev_url:
        cfg.openjev.base_url = args.openjev_url
    if args.openjev_threshold is not None:
        cfg.openjev.threshold = args.openjev_threshold
    if args.openjev_debounce is not None:
        cfg.openjev.debounce_s = args.openjev_debounce
    if args.directed:
        cfg.openjev.directed_worker = True
    if args.max_concurrent is not None:
        cfg.orchestration.max_concurrent = args.max_concurrent
    print(
        f"trigger={'openjev@' + cfg.openjev.base_url if cfg.openjev.enabled else 'legacy'} "
        f"worker={'directed' if cfg.openjev.directed_worker else 'agent'} "
        f"debounce_s={cfg.openjev.debounce_s} max_concurrent={cfg.orchestration.max_concurrent} "
        f"synthesis={cfg.models.synthesis.model_id}@{cfg.models.synthesis.base_url}",
        flush=True,
    )

    rec = Recorder(args.jsonl)
    tmp = tempfile.mkdtemp(prefix="replay-idx-")
    gateway = Gateway(cfg)
    try:
        store = IndexStore(f"{tmp}/index.db")
        embedder = Embedder()
        campaign = cfg.project.path or "./sample-campaign"
        result = await run_init(campaign, cfg, store, gateway, embedder)
        if result.warnings:
            print(f"init warnings: {result.warnings}", flush=True)
        entries = build_lexicon(store.all_entities())

        async def _on_card(card: Any) -> None:
            rec.event({"type": "card", "card": asdict(card)})

        async def _on_drop(job: Any, reason: str) -> None:
            rec.card_dropped(job, reason)

        pool = JobPool(
            max_concurrent=cfg.orchestration.max_concurrent,
            job_timeout_s=cfg.orchestration.job_timeout_s,
            stale_after_s=cfg.orchestration.stale_after_s,
            on_card=_on_card,
            on_drop=_on_drop,
        )
        engine = SessionEngine(
            cfg=cfg,
            store=store,
            gw=gateway,
            entries=entries,
            embedder=embedder,
            pool=pool,
            on_event=rec.event,
            project_path=campaign,
        )
        try:
            t_start = time.monotonic()

            async def _sink(u: Utterance) -> None:
                rec.utterance(time.monotonic() - t_start, u)
                try:
                    await engine.handle_utterance(u)
                except Exception as exc:  # observation run: never abort on one line
                    print(f"    !! handle_utterance failed: {exc}", flush=True)

            player = TranscriptReplayer(events, _sink, realtime=not args.fast)
            n = await player.run()
            await pool.drain()
            rec.summary(time.monotonic() - t_start)
            return 0 if n == len(events) else 1
        finally:
            await engine.aclose()
            await pool.close()
    finally:
        await gateway.aclose()
        rec.close()


def main() -> None:
    """CLI entry point: parse args and run the replay."""
    raise SystemExit(asyncio.run(_amain(_parse_args())))


if __name__ == "__main__":
    main()
