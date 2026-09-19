"""Session runtime: streaming STT, lexicon post-correction, trigger detection,

synthesis-job submission, and source consumption.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from .agent import AgentResult, WorkerAgent
from .attribution import AttributedSegment, attribute_segments, attribute_whole, group_by_speaker
from .config import AppConfig
from .embedder import Embedder
from .index_store import IndexStore
from .lexicon import correct_text, link_entities
from .jevworker import JevWorker
from .monitor import TranscriptMonitor
from .openjev import Debouncer, OpenjevGate
from .sources.base import AudioSource
from .staging import StagedContext, prefetch_entity, render_staged_block
from .streaming_stt import StreamingSttAdapter
from .triggers import detect_trigger
from .types import Card, Job, LexiconEntry, Priority, Utterance

logger = logging.getLogger(__name__)

# Intake chunk-handler work above this many ms means something awaits in the
# feed loop — the exact SPEC §14 defect (inline STT stall) we must never ship.
_INTAKE_BLOCK_WARN_MS = 50.0


def _lexicon_prompt(entries: list[LexiconEntry], max_chars: int = 500) -> str:
    """Build a compact, weighted hotword prompt for layer-1 STT biasing."""
    seen: set[str] = set()
    parts: list[str] = []
    used = 0
    for e in entries:
        for variant in [e.canonical, *e.variants]:
            v = variant.strip()
            if not v or v.lower() in seen:
                continue
            if not v.replace(" ", "").replace("'", "").replace("-", "").isalnum():
                continue
            seen.add(v.lower())
            cost = len(v) + 1
            if used + cost > max_chars:
                return " ".join(parts)
            parts.append(v)
            used += cost
            break
    return " ".join(parts)


def _jev_base_url(cfg: Any, oj_cfg: Any) -> str:
    """Resolve the JEV gate URL for the configured provider mode.

    Falls back to the remote ``base_url`` when local resolution fails
    (empty local URL): construction must not kill the backend over a
    provider typo — the desktop status endpoint reports the mismatch.
    """
    try:
        from .providers import jev_base_url_for

        return jev_base_url_for(cfg)
    except ValueError:
        logger.warning("JEV provider misconfigured; falling back to remote base_url")
        return str(oj_cfg.base_url)


def _make_job(
    kind: str,
    ctx: dict,
    priority: Priority,
    context_window_s: float,
) -> Job:
    return Job(
        id=uuid.uuid4().hex[:12],
        kind=kind,
        prompt_context=ctx,
        priority=priority,
        t_created=time.monotonic(),
        context_window_s=context_window_s,
    )


class SessionEngine:
    """Owns per-session state: rolling transcript, VAD buffers, lexicon cache,

    embedding, the agentic worker, and the synthesis job pool.

    v2: triggers and manual queries route to a WorkerAgent (an agentic tool
    loop) instead of a fixed embed->retrieve->synthesize pipeline. The fast
    lane verdict is binary — no taxonomy; the openjev gate picks the tier
    (card/ephemeral) per trigger, the legacy path defaults triggers to
    cards. A background TranscriptMonitor proactively surfaces alerts and
    auto-marks cards done.
    """

    def __init__(
        self,
        cfg: AppConfig,
        store: IndexStore,
        gw: Any,
        entries: list[LexiconEntry],
        embedder: Embedder | None,
        pool: Any,
        on_event: Callable[[dict], None],
        project_path: str = "",
        tool_registry: Any | None = None,
        player_state: Any | None = None,
        world_map: str = "",
        speaking_tracker: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.gw = gw
        self.entries = entries
        self.embedder = embedder
        self.pool = pool
        self.on_event = on_event
        self.project_path = project_path
        self.tool_registry = tool_registry
        self.player_state = player_state
        self.speaking_tracker = speaking_tracker
        self._recent: deque[Utterance] = deque(maxlen=30)
        self._hotword_prompt: str = _lexicon_prompt(entries)
        self._active_cards: dict[
            str, Card
        ] = {}  # card_id -> Card (mark-done bookkeeping)
        # card_id -> wall-clock time when a monitor card_done verdict was first
        # seen for it. mark_done is withheld until resolve_grace_s passes.
        self._pending_resolve: dict[str, float] = {}
        self._scene_buffer: deque[dict] = deque(
            maxlen=12
        )  # recent scene notes (for the monitor)
        self._agent = WorkerAgent(
            gw,
            store,
            project_path,
            cfg.agent,
            world_map=world_map,
            embedder=embedder,
            tool_registry=tool_registry,
        )
        self._monitor: TranscriptMonitor | None = None
        # Streaming STT is the only transcription path: audio is pushed
        # straight to the SimulStreaming server and the server's own VAD
        # endpoints segments — finals land ~0.6 s after speech stops.
        # Partial transcripts fire mid-speech for the ephemeral UI. There
        # is no batch queue and no fallback (owner decision).
        self._stt_adapter = StreamingSttAdapter(
            host=cfg.models.stt.stream_host,
            port=cfg.models.stt.stream_port,
            on_partial=self._on_stream_partial,
            on_final=self._on_stream_final,
        )
        self._intake_stats: dict[str, float] = {}
        self._pending_submits: set[asyncio.Task] = set()
        # Session controls (SPEC §15): pause-capture drops incoming audio;
        # OOC keeps transcribing (the event log is the sole truth) but stops
        # fast-lane triggers and proactive monitoring.
        self._capture_paused = False
        self._ooc = False

        # Predictive-retrieval staging ("Predictive RAG", Priority-1 design):
        # the monitor predicts likely-next entities; their excerpts are
        # pre-fetched into a small RAM LRU so an actual turn can pull
        # already-embedded context instead of paying retrieval latency inline.
        # Advisory only — never mutates canonical state — and a miss is a
        # no-op fallback, so these fields never affect correctness.
        self._staging_cfg = getattr(cfg, "staging", None)
        self._staged: StagedContext | None = (
            StagedContext(
                ttl_s=self._staging_cfg.ttl_s,
                max_entries=self._staging_cfg.max_entries,
            )
            if self._staging_cfg is not None
            and getattr(self._staging_cfg, "enabled", False)
            else None
        )
        self._prefetch_tasks: set[asyncio.Task] = set()
        self._prefetch_inflight = 0  # throttle bound for concurrent prefetches
        # Openjev decision gate (binary deploy/wait trigger). Off by default;
        # when disabled the legacy regex + fast-LLM path below is untouched.
        # The base URL follows the JEV provider mode (remote base_url or the
        # local sidecar); synthesis needs no equivalent because Gateway
        # resolves it live through the provider router.
        oj_cfg = getattr(cfg, "openjev", None)
        self._openjev_gate: OpenjevGate | None = (
            OpenjevGate(
                base_url=_jev_base_url(cfg, oj_cfg),
                threshold=oj_cfg.threshold,
                timeout_s=oj_cfg.timeout_s,
                recent_n=oj_cfg.recent_n,
            )
            if oj_cfg is not None and getattr(oj_cfg, "enabled", False)
            else None
        )
        self._openjev_debouncer: Debouncer | None = (
            Debouncer(window_s=getattr(oj_cfg, "debounce_s", 30.0))
            if self._openjev_gate is not None
            else None
        )
        self._jev_worker: JevWorker | None = (
            JevWorker(
                gw,
                store,
                project_path,
                cfg.agent,
                world_map=world_map,
                embedder=embedder,
                tool_registry=tool_registry,
                gate=self._openjev_gate,
                entries=entries,
            )
            if self._openjev_gate is not None
            and getattr(oj_cfg, "directed_worker", False)
            else None
        )

    def _staged_block(self, entities: list[str]) -> tuple[str, list[str]]:
        """Render advisory staged excerpts for ``entities``.

        Returns ``(block, matched)`` where ``block`` is "" (with ``matched``
        empty) on any miss — a pure in-memory read, so a miss adds zero
        latency and leaves the normal path byte-identical.
        """
        if self._staged is None or not entities:
            return "", []
        try:
            excerpts, matched = self._staged.lookup(entities)
        except Exception:
            return "", []
        if not excerpts:
            return "", []
        max_chars = (
            int(getattr(self._staging_cfg, "max_inject_chars", 6000) or 6000)
            if self._staging_cfg is not None
            else 6000
        )
        return render_staged_block(excerpts, max_chars=max_chars), matched

    def staging_stats(self) -> dict[str, Any]:
        """Cache introspection for the session log (hits/misses/size/keys)."""
        if self._staged is None:
            return {"enabled": False}
        snap = self._staged.snapshot()
        snap["enabled"] = True
        snap["prefetch_inflight"] = self._prefetch_inflight
        return snap

    def set_capture_paused(self, paused: bool) -> dict[str, Any]:
        """Pause/resume intake of audio into the pipeline; returns the state."""
        self._capture_paused = bool(paused)
        event = {"type": "capture_state", "paused": self._capture_paused, "t": time.time()}
        self.on_event(event)
        return event

    def set_ooc(self, on: bool) -> dict[str, Any]:
        """Toggle out-of-character mode; returns the state."""
        self._ooc = bool(on)
        event = {"type": "ooc_state", "on": self._ooc, "t": time.time()}
        self.on_event(event)
        return event

    @property
    def capture_paused(self) -> bool:
        return self._capture_paused

    @property
    def ooc(self) -> bool:
        return self._ooc

    def apply_providers(self) -> dict[str, str]:
        """Repoint inference routing after a provider-mode switch.

        Synthesis/fast follow the Gateway's live router automatically;
        the JEV gate holds its URL, so it is repointed here. No restart,
        no rebuild, no model deletion. Returns the effective base URLs.
        """
        from .providers import jev_base_url_for, synthesis_endpoint_for

        eff = synthesis_endpoint_for(self.cfg).base_url
        if self._openjev_gate is not None:
            try:
                self._openjev_gate.set_base_url(jev_base_url_for(self.cfg))
            except ValueError:
                logger.warning("JEV provider switch failed; gate URL unchanged")
        return {
            "synthesis": eff,
            "jev": (
                self._openjev_gate._base_url
                if self._openjev_gate is not None
                else ""
            ),
        }

    def refresh_lexicon(self, entries: list[LexiconEntry]) -> dict[str, Any]:
        """Hot-swap the lexicon after an init pass (no restart needed).

        The STT hotword prompt and entity-linking both derive from the same
        entries list; rebuilding them here is what makes a re-init take
        effect on the live path immediately.
        """
        self.entries = entries
        self._hotword_prompt = _lexicon_prompt(entries)
        return {"entries": len(entries), "hotwords": len(self._hotword_prompt)}

    def intake_stats(self) -> dict[str, float]:
        """Latency counters from the last consume_source run (§14 proof)."""
        return dict(self._intake_stats)

    @property
    def recent_utterances(self) -> list[Utterance]:
        return list(self._recent)

    def _remember(self, u: Utterance) -> None:
        self._recent.append(u)

    async def _on_stream_partial(self, user_id: str, text: str) -> None:
        """Mid-speech hypothesis: ephemeral transcript event, no dispatch."""
        self.on_event(
            {
                "type": "transcript_partial",
                "user_id": user_id,
                "text": text,
                "t": time.time(),
            }
        )

    async def _on_stream_final(
        self, user_id: str, text: str, t_start: float, t_end: float
    ) -> None:
        """Server-VAD-endpointed committed segment: full utterance path.

        §7a JIT attribution against mic-state, lexicon correction,
        transcript event, fast-lane dispatch.
        """
        if not text.strip():
            return
        t_dispatch = time.monotonic()
        groups = self._attribute(user_id, [], text, t_start, t_end)
        await self._publish_and_dispatch(groups)
        post_speech_ms = (time.monotonic() - t_end) * 1000.0
        logger.info(
            "stt-latency(streaming) user=%s post_speech_ms=%.1f",
            user_id,
            post_speech_ms,
        )
        self.on_event(
            {
                "type": "stt_latency",
                "user_id": user_id,
                "path": "streaming",
                "queue_wait_ms": 0.0,
                "stt_ms": round((t_dispatch - t_end) * 1000.0, 1),
                "post_speech_ms": round(post_speech_ms, 1),
                "t": t_dispatch,
            }
        )

    async def _publish_and_dispatch(
        self, groups: list[AttributedSegment]
    ) -> list[Utterance]:
        """Streaming-final tail: lexicon-correct, publish, feed the fast lane."""
        utterances: list[Utterance] = []
        for g in groups:
            corrected, _spans = correct_text(g.text, self.entries)
            if not corrected.strip():
                continue
            event: dict[str, Any] = {
                "type": "transcript",
                "user_id": g.user_id,
                "text": corrected,
                "t": g.t_end,
            }
            if g.name:
                event["name"] = g.name
            self.on_event(event)
            utter = Utterance(
                user_id=g.user_id,
                text=corrected,
                t_start=g.t_start,
                t_end=g.t_end,
                raw_text=g.text,
                name=g.name,
            )
            utterances.append(utter)
            await self.handle_utterance(utter)
        return utterances

    def _attribute(
        self,
        user_id: str,
        segments: list[dict[str, Any]],
        raw: str,
        t_start: float,
        t_end: float,
    ) -> list[AttributedSegment]:
        """Join speaker segments (or the whole window) with speaking events.

        Streaming finals carry no speaker turns today, so segments is
        empty and the whole window is attributed against mic-state; the
        segment join stays for a future server that emits speaker turns.
        """
        tracker = self.speaking_tracker
        if tracker is None:
            return [
                AttributedSegment(
                    user_id=user_id, text=raw, t_start=t_start, t_end=t_end
                )
            ]
        if segments:
            groups = group_by_speaker(
                attribute_segments(tracker, segments, t_start, user_id)
            )
            if groups:
                return groups
        uid, name = attribute_whole(tracker, t_start, t_end, user_id)
        return [
            AttributedSegment(
                user_id=uid, text=raw, t_start=t_start, t_end=t_end, name=name
            )
        ]

    async def _submit_fire(self, job: Job, work: Callable[[], Awaitable[Any]]) -> None:
        """Schedule a pool job without blocking the caller on its completion.

        Cards arrive through the event bus; the caller (an HTTP query handler
        or the streaming-final path) must not sit waiting on a 60-second
        agent run. The `entered` handshake guarantees the job is queued before this
        returns, so ``pool.drain()`` callers never race an un-started submit.
        """
        entered = asyncio.Event()

        async def _go() -> None:
            entered.set()
            try:
                await self.pool.submit(job, work)
            except Exception as exc:  # noqa: BLE001 - job failure is not the submitter's
                logger.warning("pool job %s failed: %s", job.kind, exc)

        task = asyncio.get_running_loop().create_task(_go())
        self._pending_submits.add(task)
        task.add_done_callback(self._pending_submits.discard)
        await entered.wait()

    async def handle_utterance(self, u: Utterance) -> None:
        mentioned_pairs = link_entities(u.text, self.entries)
        seen: set[str] = set()
        mentioned: list[str] = []
        for canonical, _span in mentioned_pairs:
            if canonical not in seen:
                seen.add(canonical)
                mentioned.append(canonical)

        self._remember(u)

        if self._ooc:
            # Out-of-character: the line stays in the rolling transcript (the
            # event log is the sole truth) but must not fire the fast lane.
            return

        t_lane0 = time.monotonic()
        oj_info: dict[str, Any] | None = None
        tier = "card"
        if self._openjev_gate is not None:
            window = [f"{w.name or w.user_id}: {w.text}" for w in self._recent]
            dec = await self._openjev_gate.decide(window)
            is_trigger = dec.deploy
            tier = dec.tier
            detect_ms = dec.latency_s * 1000.0
            oj_info = {
                "p_deploy": round(dec.prob, 4),
                "tier": tier,
                "p_tier": round(dec.tier_prob, 4),
                "error": dec.error,
            }
            debounced = bool(
                is_trigger
                and self._openjev_debouncer is not None
                and self._openjev_debouncer.check(time.monotonic())
            )
            # Every gate verdict is published — waits included — so recall
            # can be tuned on evidence instead of blind. Gate-only event;
            # the legacy path emits nothing here.
            self.on_event(
                {
                    "type": "trigger_verdict",
                    "user_id": u.user_id,
                    "text": u.text[:120],
                    "deploy": bool(is_trigger and not debounced),
                    "p_deploy": round(dec.prob, 4),
                    "debounced": debounced,
                    "error": dec.error,
                    "t": time.time(),
                }
            )
            if debounced:
                return
        else:
            is_trigger = await detect_trigger(self.gw, u.text)
            detect_ms = (time.monotonic() - t_lane0) * 1000.0
        if not is_trigger:
            return

        ctx = {
            "utterance": u.text,
            "entities": mentioned,
            "tier": tier,
            "recent": list(self._recent)[-5:],
        }
        job = _make_job("trigger", ctx, Priority.TRIGGER, context_window_s=120.0)

        async def _work() -> Card:
            return await self._generate_card(ctx)

        await self._submit_fire(job, _work)
        # Fast-lane timing, published so the session log and tools/latency_probe
        # can attribute the transcript -> answer gap: detect_trigger (fast LLM
        # or openjev gate) vs everything after it (context assembly + agent
        # work). The openjev key exists only when the gate is enabled, so the
        # legacy event shape is byte-identical when it is off.
        ev: dict[str, Any] = {
            "type": "turn_latency",
            "user_id": u.user_id,
            "text": u.text[:120],
            "tier": tier,
            "detect_ms": round(detect_ms, 1),
            "lane_ms": round((time.monotonic() - t_lane0) * 1000.0, 1),
            "t": time.time(),
        }
        if oj_info is not None:
            ev["openjev"] = oj_info
        self.on_event(ev)

    async def manual_query(self, text: str) -> None:
        ctx = {
            "utterance": text,
            "entities": [],
            "manual": True,
            "tier": "card",
            "recent": list(self._recent)[-5:],
        }
        job = _make_job("manual_query", ctx, Priority.MANUAL, context_window_s=600.0)

        async def _work() -> Card:
            return await self._generate_card(ctx)

        await self._submit_fire(job, _work)

    def _transcript_text(self) -> str:
        return "\n".join(
            f"{u.name or u.user_id}: {u.text}" for u in self._recent
        )

    def _scene_text(self) -> str:
        return "\n".join(s.get("text", "") for s in self._scene_buffer if s.get("text"))

    def _emit_scene(self, text: str, source: str) -> None:
        """Record and publish a scene-context note (ephemeral tier)."""
        self._scene_buffer.append({"text": text, "t": time.time(), "source": source})
        self.on_event(
            {"type": "scene_context", "text": text, "source": source, "t": time.time()}
        )

    def _task_for_ctx(self, ctx: dict) -> str:
        utterance = ctx.get("utterance", "")
        entities = ctx.get("entities", []) or []
        if ctx.get("manual"):
            return f"The DM asks directly: {utterance}"
        ent = f" Entities mentioned: {', '.join(entities)}." if entities else ""
        return (
            "A live moment needs a DM artifact. Produce what the evidence "
            "supports — table, ruling, briefing, or note. "
            f"Trigger: {utterance}.{ent}"
        )

    async def _generate_card(self, ctx: dict) -> Card | None:
        """Run the agentic worker for a trigger/manual query.

        Card tier returns a Card (the pool's on_card fires). Ephemeral tier
        publishes a scene_context event and returns None.
        """
        tier = ctx.get("tier", "card")
        task = self._task_for_ctx(ctx)
        transcript = self._transcript_text()
        trigger_portion = ctx.get("utterance", "")
        entities = ctx.get("entities", []) or []
        # Predictive-staging injection: pull already-prefetched excerpts for the
        # entities this turn mentions. Pure in-memory read — a miss adds zero
        # latency and leaves the normal path byte-identical.
        staged_block, staged_matched = self._staged_block(entities)
        if staged_block:
            logger.info(
                "staged-inject entities=%s matched=%s chars=%d",
                entities[:8],
                staged_matched[:8],
                len(staged_block),
            )
        worker = self._jev_worker if self._jev_worker is not None else self._agent
        try:
            result = await worker.run(
                task,
                tier,
                trigger_portion=trigger_portion,
                transcript=transcript,
                staged_block=staged_block,
            )
        except Exception as exc:
            result = AgentResult(tier=tier, error=f"{type(exc).__name__}: {exc}")

        if tier == "card":
            c = result.card or {
                "kind": "error",
                "title": "generation failed",
                "body_md": result.error or "(no card produced)",
                "player_ids": [],
                "items": [],
            }
            _meta: dict[str, Any] = {
                "items": list(c.get("items", []) or []),
                "entities": entities,
                "tier": "card",
                "tool_calls": result.tool_calls,
                "error": result.error,
            }
            if staged_matched:
                # Only present when a staged hit actually fed this card; an
                # absent key keeps the common (unstaged) shape unchanged.
                _meta["staged_for"] = staged_matched
            card = Card(
                id=uuid.uuid4().hex[:12],
                kind=str(c.get("kind", "info")),
                title=str(c.get("title", "Note")),
                body_md=str(c.get("body_md", "")),
                t_context=time.time(),
                status="active",
                player_ids=list(c.get("player_ids", []) or []),
                meta=_meta,
            )
            self._active_cards[card.id] = card
            return card
        # ephemeral tier -> scene context (no card event)
        text = result.text or ""
        if result.error or not text.strip():
            # The agent produced nothing (e.g. timed out under load); do not
            # publish an empty scene note — it would decay into a no-op event.
            return None
        self._emit_scene(text, "trigger")
        return None

    # -- card lifecycle (mark-done, never delete) --------------------------
    async def mark_card_done(self, card_id: str) -> bool:
        card = self._active_cards.get(card_id)
        if card is None or card.status == "done":
            return False
        card.status = "done"
        self.on_event({"type": "card_done", "card_id": card_id, "t": time.time()})
        if self.player_state is not None:
            for pid in card.player_ids:
                try:
                    self.player_state.record_card_done(
                        pid,
                        {
                            "id": card.id,
                            "kind": card.kind,
                            "title": card.title,
                            "items": card.meta.get("items", []),
                            "t": time.time(),
                        },
                    )
                except Exception:
                    pass
        return True

    def active_cards(self) -> list[Card]:
        return list(self._active_cards.values())

    # -- transcript monitor (proactive) ------------------------------------
    def start_monitor(self) -> None:
        if self._monitor is not None:
            return

        def _monitor_transcript() -> str:
            # OOC mode (SPEC §15): the proactive monitor stays silent —
            # too short to pass the judge threshold.
            return "" if self._ooc else self._transcript_text()

        self._monitor = TranscriptMonitor(
            self.gw,
            self.cfg.agent,
            get_transcript=_monitor_transcript,
            get_scene=self._scene_text,
            get_cards=lambda: [c.__dict__ for c in self.active_cards()],
            on_action=self._on_monitor_action,
            on_predict=self._on_predict,
        )
        self._monitor.start()

    def stop_monitor(self) -> None:
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None

    async def _on_predict(self, predicted: list[str]) -> None:
        """Monitor verdict predictions -> background prefetch into the cache.

        Never blocks the monitor cadence or the answer path: each entity is
        prefetched on a tracked background task, bounded to at most two
        concurrent embed+search runs, and every failure degrades to a no-op
        miss (best-effort advisory only). OOC silences predictions exactly as
        it silences the monitor's actions.
        """
        if self._staged is None or not predicted or self._ooc:
            return
        if self._prefetch_inflight >= 2:
            return  # throttled; the next cadence tick will retry uncached keys
        st_cfg = self._staging_cfg
        cap = int(getattr(st_cfg, "max_predicted", 6) or 6) if st_cfg is not None else 6
        k = int(getattr(st_cfg, "prefetch_k", 4) or 4) if st_cfg is not None else 4
        self.on_event(
            {
                "type": "staging_predict",
                "entities": predicted[:cap],
                "t": time.time(),
            }
        )
        queued = 0
        for ent in predicted[:cap]:
            if not ent.strip():
                continue
            if self._staged.has(ent):
                continue
            if self._prefetch_inflight >= 2:
                break
            queued += 1
            self._schedule_prefetch(ent, k)
        if queued:
            logger.info("staging-predict entities=%d queued=%d", len(predicted), queued)

    def _schedule_prefetch(self, entity: str, k: int) -> None:
        """Fire one tracked, throttled background prefetch for ``entity``."""
        self._prefetch_inflight += 1

        async def _run() -> None:
            try:
                excerpts = await prefetch_entity(
                    entity, self.embedder, self.store, self.gw, k=k
                )
            except Exception:
                excerpts = []
            finally:
                self._prefetch_inflight -= 1
            if excerpts and self._staged is not None:
                self._staged.put(entity, excerpts)
                logger.info(
                    "staged-prefetch entity=%r excerpts=%d", entity, len(excerpts)
                )

        try:
            task = asyncio.get_running_loop().create_task(_run())
        except RuntimeError:
            self._prefetch_inflight -= 1
            return
        self._prefetch_tasks.add(task)
        task.add_done_callback(self._prefetch_tasks.discard)

    async def _on_monitor_action(self, verdict: dict) -> None:
        action = verdict.get("action")
        if action == "card_done":
            cid = verdict.get("card_id", "")
            if cid and self._active_cards.get(cid) is not None:
                grace = float(
                    getattr(self.cfg.agent, "resolve_grace_s", 0.0) or 0.0
                )
                now = time.time()
                first = self._pending_resolve.get(cid)
                if first is None:
                    # First sighting: record it, withhold the mark.
                    self._pending_resolve[cid] = now
                    logger.info("card_done verdict for %s (grace %.0fs)", cid, grace)
                elif now - first >= grace:
                    # Verdict persisted across the grace window: resolve now.
                    self._pending_resolve.pop(cid, None)
                    await self.mark_card_done(cid)
                # else: still inside the grace window — keep withholding.
            elif cid and cid in self._pending_resolve:
                # Card vanished or was already resolved while pending.
                self._pending_resolve.pop(cid, None)
            return
        if action == "surface":
            tier = verdict.get("tier", "ephemeral")
            if tier == "card":
                ctx = {
                    "utterance": verdict.get("reason", "surface a card"),
                    "entities": [],
                    "kind": "rules",
                    "recent": list(self._recent)[-5:],
                    "source": "monitor",
                }
                job = _make_job(
                    "monitor", ctx, Priority.TRIGGER, context_window_s=120.0
                )

                async def _work() -> Card | None:
                    return await self._generate_card(ctx)

                await self._submit_fire(job, _work)
            else:
                self._emit_scene(verdict.get("text", ""), "monitor")

    async def consume_source(self, source: AudioSource) -> None:
        """Push an AudioSource's PCM to the streaming STT server.

        Audio is assumed to arrive as 16 kHz mono int16 PCM (PcmChunk.sample_rate
        must match cfg.stt_pipeline.sample_rate; no resampling is performed).

        The chunk loop does only the adapter feed; the server's own VAD
        endpoints segments and committed finals arrive via _on_stream_final.
        Intake handler time per chunk is measured and logged so a
        regression to inline awaits (SPEC §14 defect, 2026-09-05 audit) shows
        up immediately in the session log and ``intake_stats()``.
        """
        intake_chunks = 0
        intake_unfed = 0
        intake_work_max_ms = 0.0
        intake_work_total_ms = 0.0
        intake_t0 = time.monotonic()

        try:
            async for chunk in source:
                t_work0 = time.monotonic()
                user_id = chunk.user_id
                if self._capture_paused:
                    # Pause-capture (SPEC §15): audio is dropped, not fed, so
                    # resume never stitches speech across the pause boundary.
                    # The user's server session idles out on its own.
                    continue
                # The adapter never raises; False means the server is
                # unreachable (reconnect is backoff-throttled inside feed).
                # There is no fallback — the stt_health monitor owns the
                # degraded banner — so unfed audio is counted, not queued.
                fed = await self._stt_adapter.feed(user_id, chunk.samples)
                if not fed:
                    intake_unfed += 1
                work_ms = (time.monotonic() - t_work0) * 1000.0
                intake_chunks += 1
                intake_work_total_ms += work_ms
                if work_ms > intake_work_max_ms:
                    intake_work_max_ms = work_ms
                if work_ms > _INTAKE_BLOCK_WARN_MS:
                    logger.warning(
                        "intake stalled: chunk handler took %.1f ms "
                        "(streaming feed must stay a bounded socket write)",
                        work_ms,
                    )
        finally:
            self._intake_stats = {
                "chunks": intake_chunks,
                "unfed": intake_unfed,
                "max_work_ms": round(intake_work_max_ms, 3),
                "total_work_ms": round(intake_work_total_ms, 3),
                "wall_ms": round((time.monotonic() - intake_t0) * 1000.0, 3),
            }
            logger.info(
                "intake-stats chunks=%d unfed=%d max_work_ms=%.3f "
                "total_work_ms=%.3f wall_ms=%.3f",
                intake_chunks,
                intake_unfed,
                intake_work_max_ms,
                intake_work_total_ms,
                self._intake_stats["wall_ms"],
            )
            # End of source: close every stream so the server endpoints any
            # trailing speech and the finals drain before the pool does.
            await self._stt_adapter.close_all()
            drain = getattr(self.pool, "drain", None)
            if drain is not None:
                try:
                    await drain()
                except Exception:
                    pass

    async def aclose(self) -> None:
        """Close streaming sessions and drop any pending work (shutdown)."""
        for task in list(self._prefetch_tasks):
            task.cancel()
        for task in list(self._prefetch_tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._prefetch_tasks.clear()
        await self._stt_adapter.close_all()
        if self._openjev_gate is not None:
            await self._openjev_gate.aclose()
        for task in list(self._pending_submits):
            task.cancel()
        self._pending_submits.clear()
