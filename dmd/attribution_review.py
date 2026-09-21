"""Rolling speaker-attribution review: new context reassesses prior blocks.

Every new finalized transcript block advances a context version and
schedules eligible PRIOR blocks for JEV re-scoring — including blocks
whose initial timing overlap looked confident. A high initial overlap
score never permanently excludes a block.

Two-phase eligibility (configured on ``AttributionReviewConfig``):

* Initial rolling phase: each recent block is reviewed a bounded number
  of times (``initial_reviews``) as later context arrives, whatever its
  timing state. This is how later dialogue corrects earlier identities.
* Stability phase: after the initial passes, only low-confidence or
  unstable (winner-changing) blocks are revisited, and only when new
  context or fresh timing evidence arrives — under bounded age, pass,
  and resource limits. Old stable blocks stop being rescored.

Score distributions are retained per block (bounded ``score_history``)
so confidence changes are observable: scores may rise, fall, oscillate,
change winner, or resolve to ``unknown``/``multiple``. JEV numbers are
labeled uncalibrated conditional option scores, never identity
probabilities. Endpoint failures and invalid distributions invent no
confidence: they are recorded in history and change nothing.

Prompt evidence: every review prompt combines Discord overlap evidence
(coverage and margin quoted, plus a stated timing verdict — the scorer
does not reliably compare magnitudes itself) with deterministic
dialogue-continuity observations — sentence completion and first-person
continuation across the TARGET's boundaries, computed from the
transcript text. Surrounding speaker labels are marked provisional
hypotheses: usable only through continuity with the TARGET, never as
standalone proof. Spell or class stereotypes alone never assign a
speaker.

Scheduling: per-block single-flight review loops with coalescing —
a new context version published while a pass is in flight bumps the
pending target instead of queueing duplicate work, and the loop runs
again at the newest context. Backpressure preserves work (a semaphore
bounds concurrent scorer calls; waiters queue, nothing is silently
dropped) while oldest-first scheduling prevents starvation. Stale or
out-of-order results — a pass whose timing evidence changed mid-flight
or that a newer application superseded — are recorded and discarded,
never applied.

Downstream audit (mutable ``Utterance`` references): the pipeline owns
``Utterance`` objects shared between its rolling transcript
(``_recent``) and its attribution registry (``_attribution_records``);
a revision mutates the stored row in place, so future card prompts and
monitor reads see corrected identities. In-flight card jobs rebuild
their transcript at run time and may observe a correction — accepted
and bounded (revisions only rename speakers, never text). Reviews never
re-enter ``handle_utterance``: no redispatch, no duplicate transcript
lines, no duplicate cards.

Shared-scorer contention is UNVERIFIED: reviews share the pipeline's
``OpenjevGate`` HTTP client with deploy/tier passes and this module
bounds only its own concurrency (``max_pending``). Async scheduling
alone proves nothing about overlap; no gate-side queue instrumentation
exists to measure it. See ``stats()["scorer_contention"]``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .attribution import attribute_whole_evidence

if TYPE_CHECKING:
    from .attribution import AttributedSegment
    from .config import AttributionReviewConfig
    from .openjev import OpenjevGate
    from .speaking_tracker import SpeakingTracker

logger = logging.getLogger(__name__)

#: Label attached to every retained JEV distribution: these numbers are
#: uncalibrated conditional option scores, not identity probabilities.
SCORE_LABEL = (
    "uncalibrated conditional option scores, not identity probability"
)

UNKNOWN_ID = "unknown"
MULTIPLE_ID = "multiple"

#: Bound on tracked transcript blocks (oldest pruned first).
_MAX_TRACKED = 256

#: Cap on the session-participant roster rendered into one prompt.
#: Per-line prompt clips: prompts stay bounded whatever a speaker says.
_MAX_CONTEXT_CHARS = 400
_MAX_TARGET_CHARS = 800


def _clip(text: str, limit: int) -> str:
    """Clip one prompt line to its bound, marking the cut."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


#: Words that, opening an adjacent line, often continue the previous
#: speaker's sentence rather than starting a new turn.
_CONTINUATION_OPENERS = frozenset({
    "and", "but", "so", "then", "also", "plus", "because", "though",
    "although", "while", "which", "who", "whom",
})

_TERMINAL_PUNCT = (".", "!", "?", "…")

#: Winner/runner-up margin (fraction of the window) at or above which the
#: timing leader is stated as a clear lead. Mirrors the timing-certainty
#: floor in ``dmd.attribution``; below it the verdict calls the race tied.
_CLEAR_LEAD_MARGIN = 0.15

#: Best-overlap coverage (fraction of the window) below which even a lone
#: leader is stated as weak evidence. Mirrors the certainty floor.
_MIN_COVERAGE = 0.50

_FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|i'll|i'd|my|me)\b",
                              re.IGNORECASE)


def _ends_open(text: str) -> bool:
    """No terminal punctuation: the sentence may spill past this line."""
    stripped = text.strip().rstrip("\"'“”‘’)")
    return bool(stripped) and not stripped.endswith(_TERMINAL_PUNCT)


def _starts_continuation(text: str) -> str | None:
    """The opening word when a line reads like a sentence continuation."""
    stripped = text.strip().lstrip("\"'“”‘’(")
    if not stripped:
        return None
    first = stripped.split(" ", 1)[0].strip("\"'“”‘’(),:;")
    if not first:
        return None
    if first[0].islower() or first.lower() in _CONTINUATION_OPENERS:
        return first
    return None


def continuation_notes(
    prev: tuple[str, str] | None,
    target: str,
    following: tuple[str, str] | None,
) -> list[str]:
    """Deterministic continuity observations across the TARGET boundaries.

    ``prev``/``following`` are ``(speaker_label, text)`` for the lines
    adjacent to TARGET (None when absent); labels are hypotheses quoted
    for the scorer, never asserted truth. Notes fire on textual shape —
    continuation openers, open (unterminated) sentences, shared
    first-person voice — never on topic or vocabulary.
    """
    notes: list[str] = []
    target_open = _ends_open(target)
    target_first = bool(_FIRST_PERSON_RE.search(target))
    if following is not None:
        label, text = following
        opener = _starts_continuation(text)
        if opener is not None:
            if target_open:
                notes.append(
                    f"The line after TARGET (hypothesis: {label}) begins "
                    f"with '{opener}' and completes the TARGET sentence — "
                    "both lines are likely one speaker's turn."
                )
            else:
                notes.append(
                    f"The line after TARGET (hypothesis: {label}) begins "
                    f"with '{opener}', which often continues the previous "
                    "speaker's sentence — possibly the same speaker's turn."
                )
        if target_first and _FIRST_PERSON_RE.search(text):
            notes.append(
                f"TARGET and the line after it (hypothesis: {label}) both "
                "speak in the first person — consistent with one speaker "
                "continuing."
            )
    if prev is not None:
        label, text = prev
        if _ends_open(text) and _starts_continuation(target) is not None:
            notes.append(
                f"The line before TARGET (hypothesis: {label}) ends openly "
                "and TARGET completes its sentence — both lines are "
                "likely one speaker's turn."
            )
    return notes


def _adjacent(line: dict[str, Any]) -> tuple[str, str]:
    """(Speaker label, text) for a line adjacent to TARGET."""
    who = line.get("name") or line.get("user_id") or "?"
    return str(who), str(line.get("text", ""))


def timing_verdict(
    candidates: list[tuple[str, float]],
    coverage: float,
    margin: float,
    name_of: Callable[[str], str | None] | None = None,
) -> str:
    """One-line stated conclusion from the overlap numbers.

    The scorer does not reliably compare overlap magnitudes itself
    (verified: a 1.1s-vs-0.35s lead lost until stated outright), so the
    conclusion is precomputed here. ``candidates`` is overlap-descending
    ``(user_id, seconds)``; ``coverage``/``margin`` are window fractions.
    Leads are stated plainly, ties and low-coverage windows honestly —
    never stronger than the numbers warrant.
    """
    def _name(uid: str) -> str:
        if name_of is not None:
            try:
                return name_of(uid) or uid
            except Exception:
                pass
        return uid

    if not candidates:
        return (
            "Timing verdict: no member spoke in the TARGET window — "
            "timing supports no one."
        )
    if len(candidates) == 1:
        uid, overlap = candidates[0]
        if coverage < _MIN_COVERAGE:
            return (
                f"Timing verdict: {_name(uid)} (id {uid}) spoke only "
                f"briefly ({overlap:.3f}s, coverage {coverage:.3f} of the "
                "window) — weak timing evidence; follow the dialogue "
                "continuity."
            )
        return (
            f"Timing verdict: {_name(uid)} (id {uid}) is the only member "
            f"who spoke in the TARGET window ({overlap:.3f}s) — prefer "
            f"{_name(uid)} unless dialogue continuity strongly indicates "
            "another member."
        )
    (top, first), (runner, second) = candidates[0], candidates[1]
    if coverage < _MIN_COVERAGE:
        return (
            f"Timing verdict: {_name(top)} (id {top}) leads "
            f"({first:.3f}s vs {second:.3f}s) but covered only "
            f"{coverage:.3f} of the window — weak evidence; follow the "
            "dialogue continuity."
        )
    if margin >= _CLEAR_LEAD_MARGIN:
        return (
            f"Timing verdict: {_name(top)} (id {top}) clearly leads on "
            f"overlap ({first:.3f}s vs {second:.3f}s, margin "
            f"{margin:.3f} of the window). Prefer {_name(top)} unless "
            "dialogue continuity strongly favors another member."
        )
    return (
        f"Timing verdict: {_name(top)} (id {top}) and "
        f"{_name(runner)} (id {runner}) are nearly tied "
        f"({first:.3f}s vs {second:.3f}s) — timing cannot decide; "
        "follow the dialogue evidence."
    )

#: Speaker-scoring task. Deliberately distinct from the locked v39
#: deploy/wait gate wording in ``dmd.openjev`` (which this module never
#: touches): scoring "who spoke" is a separate JEV task from "deploy?".
SPEAKER_QUESTION = (
    "Which option best identifies the speaker of the TARGET line? "
    "Use only the supplied Discord overlap evidence and dialogue context. "
    "Weigh them together: an adjacent line that completes the TARGET's "
    "sentence, continues its first-person action, or directly answers it "
    "usually shares its speaker. Surrounding speaker labels are "
    "hypotheses, not observed truth — use them only through continuity "
    "with the TARGET, never by themselves. Timing is one signal among "
    "several: when overlaps are small, close, or contradictory, prefer "
    "the dialogue evidence. Choose unknown only when neither signal "
    "supports one member, or multiple when overlapping speech makes this "
    "a mixed turn. Do not assign a speaker from spell or class "
    "stereotypes alone."
)


class InvalidScores(ValueError):
    """A scorer response that is not a usable option distribution."""


def validate_scores(
    probs: Any, option_ids: list[str]
) -> dict[str, float]:
    """Validate a complete, finite, in-range option distribution.

    Every offered option must be present exactly once and every value a
    finite float in [0, 1]. Anything else raises :class:`InvalidScores`
    so the caller records the failure instead of inventing confidence.
    """
    if not isinstance(probs, dict):
        raise InvalidScores(f"scores are not an object: {type(probs).__name__}")
    if set(probs.keys()) != set(option_ids) or len(probs) != len(option_ids):
        raise InvalidScores(
            f"scores cover {sorted(probs.keys())}, options are {option_ids}"
        )
    out: dict[str, float] = {}
    for oid in option_ids:
        value = probs[oid]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidScores(f"score for {oid!r} is not a number")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise InvalidScores(f"score for {oid!r} out of range: {value!r}")
        out[oid] = score
    return out


@dataclass
class ReviewRecord:
    """One transcript block's attribution state across rolling reviews."""

    block_id: str
    text: str
    t_start: float
    t_end: float
    source_user_id: str  # capture identity (e.g. browser_mixed), never a guess
    source_name: str | None
    original: dict[str, Any]  # frozen timing evidence at publication
    timing: dict[str, Any]  # refreshed timing evidence (mutable)
    user_id: str  # current inferred identity (mutable)
    name: str | None
    state: str  # timing state or jev outcome (unknown/multiple/contextual_review)
    source: str  # discord_speaking_overlap | jev_contextual_review
    revision: int = 0
    published_context: int = 0
    created_mono: float = 0.0
    timing_serial: int = 0  # bumps on every timing-evidence change
    last_reviewed_context: int | None = None
    last_reviewed_timing_serial: int = -1
    last_applied_context: int | None = None
    initial_done: int = 0
    total_done: int = 0
    jev_applied: bool = False
    score_history: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)


def _timing_key(evidence: dict[str, Any]) -> tuple:
    """Hashable identity of a timing-evidence snapshot for change checks."""
    cands = evidence.get("candidates") or {}
    return (
        evidence.get("user_id"),
        evidence.get("state"),
        tuple(sorted((str(k), round(float(v), 4)) for k, v in cands.items())),
        round(float(evidence.get("coverage", 0.0)), 4),
        round(float(evidence.get("margin", 0.0)), 4),
    )


class RollingAttributionReviewer:
    """Reassess prior transcript blocks as new finalized context arrives."""

    def __init__(
        self,
        cfg: AttributionReviewConfig,
        gate: OpenjevGate | None,
        tracker: SpeakingTracker | None,
        on_revision: Callable[[dict[str, Any]], None],
        context_lines: Callable[[], list[dict[str, Any]]],
    ) -> None:
        self._cfg = cfg
        self._gate = gate
        self._tracker = tracker
        self._on_revision = on_revision
        self._context_lines = context_lines
        self._records: dict[str, ReviewRecord] = {}
        self._context_version = 0
        self._pending: dict[str, int] = {}  # block_id -> newest target context
        self._tasks: dict[str, asyncio.Task] = {}  # block_id -> review loop
        self._sem = asyncio.Semaphore(max(1, int(cfg.max_pending)))
        self._closed = False
        self._attempts = 0
        self._applied = 0
        self._stale = 0
        self._invalid = 0
        self._failed = 0
        self._timing_revisions = 0
        self._max_inflight = 0

    @property
    def context_version(self) -> int:
        """Newest finalized-block context version published so far."""
        return self._context_version

    def stats(self) -> dict[str, Any]:
        """Bounded counters plus the (unverified) contention disclosure."""
        return {
            "context_version": self._context_version,
            "tracked": len(self._records),
            "pending": len(self._pending),
            "in_flight": len(self._tasks),
            "max_in_flight": self._max_inflight,
            "attempts": self._attempts,
            "applied": self._applied,
            "stale_discarded": self._stale,
            "invalid": self._invalid,
            "failed": self._failed,
            "timing_revisions": self._timing_revisions,
            "scorer_contention": (
                "unverified: reviews share the OpenjevGate HTTP client with "
                "deploy/tier passes; only review-side concurrency is bounded "
                f"(max_pending={self._cfg.max_pending}), overlap unmeasured"
            ),
        }

    # -- publication ----------------------------------------------------

    def publish(
        self,
        *,
        block_id: str,
        text: str,
        t_start: float,
        t_end: float,
        source_user_id: str,
        source_name: str | None = None,
        evidence: AttributedSegment,
    ) -> dict[str, Any]:
        """Register a new finalized block; returns its initial attribution.

        Advances the context version, refreshes timing evidence for tracked
        priors (late gateway data corrects rows that predate it), and
        schedules every eligible block — priors and the new block alike —
        for JEV review. Synchronous: scoring itself always runs on bounded
        background tasks, never inline on the transcript-to-card path.
        """
        self._context_version += 1
        ctx = self._context_version
        frozen = {
            "user_id": evidence.user_id,
            "name": evidence.name,
            "candidates": dict(evidence.candidates or {}),
            "coverage": float(evidence.coverage),
            "margin": float(evidence.margin),
            "state": evidence.state,
        }
        record = ReviewRecord(
            block_id=block_id,
            text=text,
            t_start=t_start,
            t_end=t_end,
            source_user_id=source_user_id,
            source_name=source_name,
            original=dict(frozen),
            timing=dict(frozen),
            user_id=evidence.user_id,
            name=evidence.name,
            state=evidence.state,
            source="discord_speaking_overlap",
            published_context=ctx,
            created_mono=time.monotonic(),
        )
        record.history.append({
            "revision": 0, "context": ctx, "user_id": record.user_id,
            "state": record.state, "reason": "initial_timing",
        })
        self._records[block_id] = record
        self._prune()
        self._refresh_timing(exclude={block_id})
        self._schedule_all()
        return {
            "user_id": record.user_id,
            "name": record.name,
            "attribution": self._snapshot(record),
        }

    def _prune(self) -> None:
        """Drop oldest records past the track bound (with their work)."""
        while len(self._records) > _MAX_TRACKED:
            oldest = next(iter(self._records))
            del self._records[oldest]
            self._pending.pop(oldest, None)
            task = self._tasks.pop(oldest, None)
            if task is not None and not task.done():
                task.cancel()

    # -- timing evidence (cheap, synchronous, JEV-independent) -----------

    def _evidence_now(self, record: ReviewRecord) -> dict[str, Any]:
        """Recompute target-window overlap evidence against live tracker."""
        if self._tracker is None:
            return {
                "user_id": record.source_user_id, "name": record.source_name,
                "candidates": {}, "coverage": 0.0, "margin": 0.0,
                "state": "unknown",
            }
        ev = attribute_whole_evidence(
            self._tracker, record.t_start, record.t_end, record.source_user_id
        )
        return {
            "user_id": ev.user_id, "name": ev.name,
            "candidates": dict(ev.candidates or {}),
            "coverage": float(ev.coverage), "margin": float(ev.margin),
            "state": ev.state,
        }

    def _refresh_timing(self, exclude: set[str]) -> None:
        """Apply late gateway evidence to rows without a JEV decision yet.

        Rows already carrying a JEV outcome keep it — JEV decisions are
        only ever revised by later JEV passes, never silently overwritten
        by timing — but their stored evidence still refreshes so the next
        scheduled pass scores against it.
        """
        for bid, record in self._records.items():
            if bid in exclude:
                continue
            fresh = self._evidence_now(record)
            if _timing_key(fresh) == _timing_key(record.timing):
                continue
            record.timing = fresh
            record.timing_serial += 1
            if record.jev_applied:
                continue
            record.user_id = fresh["user_id"]
            record.name = fresh["name"]
            record.state = fresh["state"]
            record.source = "discord_speaking_overlap"
            record.revision += 1
            record.last_applied_context = self._context_version
            record.history.append({
                "revision": record.revision, "context": self._context_version,
                "user_id": record.user_id, "state": record.state,
                "reason": "timing_revision",
            })
            self._trim_history(record)
            self._timing_revisions += 1
            self._emit(record)

    # -- eligibility ----------------------------------------------------

    def _last_scored(self, record: ReviewRecord) -> dict[str, Any] | None:
        """Newest non-stale retained distribution, if any."""
        for entry in reversed(record.score_history):
            if not entry.get("stale"):
                return entry
        return None

    def _is_low(self, record: ReviewRecord) -> bool:
        """A block with no valid scores, an unknown/multiple outcome, or a
        winner below the stability threshold counts as low-confidence."""
        last = self._last_scored(record)
        if last is None:
            return True
        if last.get("winner") in (UNKNOWN_ID, MULTIPLE_ID):
            return True
        try:
            return float(last.get("score", 0.0)) < float(
                self._cfg.low_confidence_below
            )
        except (TypeError, ValueError):
            return True

    def _is_unstable(self, record: ReviewRecord) -> bool:
        """Two or more distinct winners in the recent window is unstable."""
        window = max(1, int(self._cfg.unstable_window))
        winners = {
            e.get("winner")
            for e in record.score_history[-window:]
            if not e.get("stale") and e.get("winner")
        }
        return len(winners) >= 2

    def _wants_pass(self, record: ReviewRecord, target: int) -> bool:
        """Whether ``record`` deserves a JEV pass at context ``target``.

        Initial rolling phase: every recent block gets bounded passes as
        later context arrives, confident timing included. Stability phase:
        only low-confidence or unstable blocks, and only on new input.
        """
        if self._closed or self._gate is None:
            return False
        if not bool(getattr(self._cfg, "enabled", True)):
            return False
        if record.total_done >= int(self._cfg.max_total_reviews):
            return False
        if target - record.published_context > int(self._cfg.max_age_blocks):
            return False
        max_age_s = float(self._cfg.max_age_s)
        if max_age_s > 0 and time.monotonic() - record.created_mono > max_age_s:
            return False
        reviewed = record.last_reviewed_context
        if (
            reviewed is not None
            and target <= reviewed
            and record.timing_serial == record.last_reviewed_timing_serial
        ):
            return False  # same context, same evidence: never rescore
        if record.initial_done < int(self._cfg.initial_reviews):
            return True
        return self._is_low(record) or self._is_unstable(record)

    def _schedule_all(self) -> None:
        """Schedule every eligible block at the newest context, oldest first.

        Pending work coalesces to the newest context version; per-block
        single-flight loops pick it up without duplicate tasks.
        """
        for bid, record in self._records.items():
            if not self._wants_pass(record, self._context_version):
                continue
            if self._pending.get(bid, -1) < self._context_version:
                self._pending[bid] = self._context_version
            self._ensure_task(bid)

    def _ensure_task(self, block_id: str) -> None:
        """Start the review loop for a pending block, at most one per block."""
        task = self._tasks.get(block_id)
        if task is not None and not task.done():
            return
        if task is not None:
            self._tasks.pop(block_id, None)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet; pending work waits for the next publish
        self._tasks[block_id] = loop.create_task(self._review_loop(block_id))
        self._max_inflight = max(self._max_inflight, len(self._tasks))

    # -- review loop -----------------------------------------------------

    async def _review_loop(self, block_id: str) -> None:
        """Single-flight loop: score at the pending target, then re-check.

        When new context lands mid-pass the pending target advances and the
        loop scores again at the newest context instead of dropping the
        update or stacking duplicate tasks.
        """
        try:
            while not self._closed:
                record = self._records.get(block_id)
                target = self._pending.get(block_id)
                if record is None or target is None:
                    break
                if not self._wants_pass(record, target):
                    if self._pending.get(block_id) == target:
                        self._pending.pop(block_id, None)
                    if self._pending.get(block_id) is None:
                        break
                    continue
                await self._attempt(record, target)
                nxt = self._pending.get(block_id)
                if nxt is None or nxt <= target:
                    if nxt == target:
                        self._pending.pop(block_id, None)
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a review must never kill intake
            logger.warning("attribution review loop failed: %s", exc)
        finally:
            if self._tasks.get(block_id) is asyncio.current_task():
                self._tasks.pop(block_id, None)

    async def _attempt(self, record: ReviewRecord, target: int) -> None:
        """Run one bounded JEV pass for ``record`` at context ``target``."""
        member_ids, state_text, question, options = self._build_prompt(record)
        if not member_ids:
            # No plausible session participant to offer: JEV could only
            # return unknown/multiple, which teaches nothing. Leave the
            # block unscheduled until evidence introduces a candidate.
            if self._pending.get(record.block_id) == target:
                self._pending.pop(record.block_id, None)
            return
        row_id = f"speaker:{record.block_id}:c{target}"
        base_serial = record.timing_serial
        try:
            async with self._sem:
                if self._closed:
                    return
                probs = await asyncio.wait_for(
                    self._gate.rank(row_id, state_text, question, options),  # type: ignore[union-attr]
                    timeout=float(self._cfg.review_timeout_s),
                )
        except (asyncio.TimeoutError, TimeoutError, Exception) as exc:
            self._note_attempt(record, target, base_serial)
            record.history.append({
                "revision": record.revision, "context": target,
                "user_id": record.user_id, "state": record.state,
                "reason": "jev_failed", "detail": type(exc).__name__,
            })
            self._trim_history(record)
            self._failed += 1
            logger.info("attribution review failed: %s", type(exc).__name__)
            return
        try:
            valid = validate_scores(probs, [o["id"] for o in options])
        except InvalidScores as exc:
            self._note_attempt(record, target, base_serial)
            record.history.append({
                "revision": record.revision, "context": target,
                "user_id": record.user_id, "state": record.state,
                "reason": "jev_invalid", "detail": str(exc)[:160],
            })
            self._trim_history(record)
            self._invalid += 1
            logger.info("attribution review invalid scores: %s", exc)
            return
        winner = max(valid, key=lambda k: valid[k])
        score = valid[winner]
        if (
            record.timing_serial != base_serial
            or (
                record.last_applied_context is not None
                and target < record.last_applied_context
            )
        ):
            # Stale: newer evidence or a strictly newer application
            # landed mid-pass. The distribution is retained for audit
            # (marked stale) but the identity is never overwritten; a fresh
            # pass is rescheduled. Same-context timing auto-applies do NOT
            # stale a pass built on their evidence — the JEV outcome
            # supersedes them instead.
            self._retain_scores(record, target, valid, winner, score,
                                stale=True)
            record.history.append({
                "revision": record.revision, "context": target,
                "user_id": record.user_id, "state": record.state,
                "reason": "jev_stale", "winner": winner, "score": score,
            })
            self._trim_history(record)
            self._stale += 1
            self._pending[record.block_id] = max(
                self._pending.get(record.block_id, target),
                self._context_version,
            )
            return
        if winner.startswith("member:") and winner not in {
            f"member:{uid}" for uid in member_ids
        }:
            self._note_attempt(record, target, base_serial)
            record.history.append({
                "revision": record.revision, "context": target,
                "user_id": record.user_id, "state": record.state,
                "reason": "jev_unoffered", "winner": winner, "score": score,
            })
            self._trim_history(record)
            self._invalid += 1
            return
        self._note_attempt(record, target, base_serial)
        if winner == UNKNOWN_ID or winner == MULTIPLE_ID:
            # Uncertainty reverts to the observed capture identity — never
            # to an earlier guessed user.
            record.user_id = record.source_user_id
            record.name = record.source_name
            record.state = winner
            reason = "jev_unknown" if winner == UNKNOWN_ID else "jev_multiple"
        else:
            uid = winner.removeprefix("member:")
            record.user_id = uid
            record.name = self._name_of(uid)
            record.state = "contextual_review"
            reason = "jev_review"
        record.source = "jev_contextual_review"
        record.revision += 1
        record.last_applied_context = target
        record.jev_applied = True
        self._retain_scores(record, target, valid, winner, score)
        record.history.append({
            "revision": record.revision, "context": target,
            "user_id": record.user_id, "state": record.state,
            "reason": reason, "winner": winner, "score": score,
        })
        self._trim_history(record)
        self._applied += 1
        self._emit(record)

    def _note_attempt(
        self, record: ReviewRecord, target: int, base_serial: int
    ) -> None:
        """Account one completed scorer attempt (bounds count attempts)."""
        record.last_reviewed_context = target
        record.last_reviewed_timing_serial = base_serial
        if record.initial_done < int(self._cfg.initial_reviews):
            record.initial_done += 1
        record.total_done += 1
        self._attempts += 1

    def _retain_scores(
        self,
        record: ReviewRecord,
        target: int,
        valid: dict[str, float],
        winner: str,
        score: float,
        stale: bool = False,
    ) -> None:
        """Preserve a valid distribution (bounded); stale ones stay marked."""
        record.score_history.append({
            "context": target, "revision": record.revision,
            "scores": dict(valid), "winner": winner, "score": score,
            "stale": stale,
            "label": SCORE_LABEL,
        })
        while len(record.score_history) > max(1, int(self._cfg.history_len)):
            record.score_history.pop(0)

    def _trim_history(self, record: ReviewRecord) -> None:
        """Keep the bounded audit trail at its configured length."""
        while len(record.history) > max(1, int(self._cfg.history_len)):
            record.history.pop(0)

    # -- prompt ----------------------------------------------------------

    def _name_of(self, user_id: str) -> str | None:
        """Display name for a member id, when the tracker knows one."""
        get_name = getattr(self._tracker, "name_of", None)
        if not callable(get_name):
            return None
        try:
            return get_name(user_id)
        except Exception:
            return None

    def _plausible_candidates(
        self, record: ReviewRecord
    ) -> list[tuple[str, float]]:
        """Return only members observed speaking during this target window.

        The Discord speaking signal is the candidate gate. Adjacent speakers
        and the broader session roster are context, never attribution options.
        An empty target window therefore offers only ``unknown``/``multiple``.
        """
        target = {
            str(k): float(v)
            for k, v in (record.timing.get("candidates") or {}).items()
            if float(v) > 0
        }
        ordered = sorted(target, key=lambda u: target[u], reverse=True)
        cap = max(0, int(self._cfg.max_candidates))
        return [(uid, target.get(uid, 0.0)) for uid in ordered[:cap]]

    def _build_prompt(
        self, record: ReviewRecord
    ) -> tuple[list[str], str, str, list[dict[str, str]]]:
        """Build one TARGET-unambiguous speaker-scoring prompt.

        Returns ``(member_ids, state, question, options)``. Preceding and
        following dialogue carry provisional labels within the configured
        window (null means all retained lines); Discord timing evidence
        is quoted with coverage and margin, and deterministic continuity
        observations name the adjacent lines' hypotheses.
        """
        try:
            lines = list(self._context_lines() or [])
        except Exception:
            lines = []
        idx = next(
            (i for i, ln in enumerate(lines)
             if ln.get("id") == record.block_id),
            None,
        )
        recent_n = self._cfg.recent_n
        following_n = self._cfg.following_n
        if idx is None:
            preceding = lines if recent_n is None else lines[-recent_n:]
            following = []
        else:
            preceding = (
                lines[:idx] if recent_n is None
                else lines[max(0, idx - recent_n):idx]
            )
            following = (
                lines[idx + 1:] if following_n is None
                else lines[idx + 1: idx + 1 + following_n]
            )

        def _label(ln: dict[str, Any]) -> str:
            who = ln.get("name") or ln.get("user_id") or "?"
            clipped = _clip(str(ln.get("text", "")), _MAX_CONTEXT_CHARS)
            return f"[provisional {who}] {clipped}"

        candidates = self._plausible_candidates(record)
        member_ids = [uid for uid, _ in candidates]
        roster = [f"- {self._name_of(uid) or uid} (id {uid})"
                  for uid, _ in candidates]
        coverage = float(record.timing.get("coverage", 0.0))
        margin = float(record.timing.get("margin", 0.0))
        parts = [
            "You are identifying the speaker of one transcript line. "
            "Surrounding speaker labels are HYPOTHESES, not observed "
            "truth: use them only through dialogue continuity with the "
            "TARGET (sentence completion, first-person continuation, "
            "direct reply), never as standalone proof.",
            "",
            "Members speaking in the TARGET window "
            "(attribution candidates):",
            *(roster or ["- (none in this window)"]),
            "",
            "Discord speaking overlap for the TARGET window "
            f"[{record.t_start:.2f}-{record.t_end:.2f}]:",
        ]
        if candidates:
            for uid, overlap in candidates:
                parts.append(
                    f"- {self._name_of(uid) or uid} (id {uid}): "
                    f"{overlap:.3f}s overlap"
                )
        else:
            parts.append("- (no speaking overlap recorded)")
        parts.extend([
            f"Window evidence: best coverage {coverage:.3f} of the window, "
            f"margin over runner-up {margin:.3f}. Larger overlap favors "
            "that member; small or close overlaps may mislead — weigh "
            "dialogue continuity alongside timing.",
            "",
            "Preceding context (provisional labels):",
            *([_label(ln) for ln in preceding] or ["- (none)"]),
            "",
            "TARGET LINE (identify this speaker):",
            f"[TARGET id={record.block_id} "
            f"{record.t_start:.2f}-{record.t_end:.2f}] "
            f"{_clip(record.text, _MAX_TARGET_CHARS)}",
            "",
            "Following context (provisional labels):",
            *([_label(ln) for ln in following] or ["- (none yet)"]),
            "",
            "Dialogue continuity (textual observations):",
            *(
                [f"- {note}" for note in continuation_notes(
                    _adjacent(preceding[-1]) if preceding else None,
                    record.text,
                    _adjacent(following[0]) if following else None,
                )] or ["- (no continuation signal)"]
            ),
            "",
            # Closing position is deliberate: this configuration (verdict
            # stated last, ahead of the question) is what verified live.
            timing_verdict(candidates, coverage, margin, self._name_of),
        ])
        options: list[dict[str, str]] = []
        for uid, overlap in candidates:
            options.append({
                "id": f"member:{uid}",
                "description": (
                    f"{self._name_of(uid) or uid}; Discord speaking overlap "
                    f"{overlap:.3f}s in the TARGET window."
                ),
            })
        options.extend([
            {"id": UNKNOWN_ID, "description": (
                "Neither timing nor dialogue continuity supports one "
                "member.")},
            {"id": MULTIPLE_ID, "description": (
                "Overlapping speech makes this a mixed turn.")},
        ])
        return member_ids, "\n".join(parts), SPEAKER_QUESTION, options

    # -- snapshots, lifecycle --------------------------------------------

    def _low_confidence(self, record: ReviewRecord) -> bool:
        """Winner below ``min_prob`` (or no timing certainty yet) is low."""
        last = self._last_scored(record)
        if last is None:
            return record.state != "certain"
        try:
            return float(last.get("score", 0.0)) < float(self._cfg.min_prob)
        except (TypeError, ValueError):
            return True

    def _snapshot(self, record: ReviewRecord) -> dict[str, Any]:
        """Serializable attribution audit for events and stored rows."""
        last = self._last_scored(record)
        snap: dict[str, Any] = {
            "state": record.state,
            "coverage": round(float(record.timing.get("coverage", 0.0)), 3),
            "margin": round(float(record.timing.get("margin", 0.0)), 3),
            "candidates": dict(record.timing.get("candidates") or {}),
            "source": record.source,
            "revision": record.revision,
            "block_id": record.block_id,
            "source_user_id": record.source_user_id,
            "original": {
                "user_id": record.original.get("user_id"),
                "name": record.original.get("name"),
                "candidates": dict(record.original.get("candidates") or {}),
                "coverage": record.original.get("coverage"),
                "margin": record.original.get("margin"),
                "state": record.original.get("state"),
            },
            "context_version": (
                record.last_reviewed_context
                if record.last_reviewed_context is not None
                else record.published_context
            ),
            "score_history": [dict(e) for e in record.score_history],
            "history": [dict(e) for e in record.history],
            # Every speaker label here is an inference — even a
            # timing-certain one stays subject to rolling review.
            "provisional": True,
            "low_confidence": self._low_confidence(record),
        }
        if last is not None:
            snap["review_scores"] = dict(last["scores"])
            snap["review_scores_label"] = SCORE_LABEL
        return snap

    def _emit(self, record: ReviewRecord) -> None:
        """Deliver one revision outcome to the pipeline (never raises)."""
        try:
            self._on_revision({
                "block_id": record.block_id,
                "user_id": record.user_id,
                "name": record.name,
                "attribution": self._snapshot(record),
            })
        except Exception as exc:  # noqa: BLE001 - revisions must not break intake
            logger.warning("attribution revision callback failed: %s", exc)

    async def drain(self, timeout_s: float = 5.0) -> bool:
        """Wait until scheduled reviews settle; False on timeout."""
        for bid in list(self._pending):
            self._ensure_task(bid)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while self._tasks:
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.01)
        return True

    async def aclose(self) -> None:
        """Cancel pending reviews and wait for orderly shutdown."""
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        self._pending.clear()
