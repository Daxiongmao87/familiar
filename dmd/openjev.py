"""Openjev decision gate: binary deploy/wait trigger behind ``/score``.

One forward pass reads the A-P option logits; no tokens are generated, so
this path pays no reasoning-token tax. ``decide`` answers only "does this
moment demand a worker?" — no taxonomy anywhere. When it fires, a second
pass picks the output tier (card/note); the artifact's shape is decided
downstream from evidence.

Fail-closed: any transport, timeout, or shape error resolves to ``wait``.
A missed artifact costs a turn; a spurious deploy costs GPU and DM noise.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Production baseline jev-gate-v39. LOCKED: do not paraphrase, reorder,
# or otherwise modify the question or option descriptions without
# rerunning the regression suite (tests/regression/test_jev_gate_golden.py).
# Option order is load-bearing (deploy FIRST): testing found a
# substantial positional effect.
GATE_VERSION = "jev-gate-v39"
DEPLOY_QUESTION = "Is there useful information work created by this exchange?"
DEPLOY_OPTIONS = [
    {
        "id": "deploy",
        "description": (
            "Yes. The exchange created a reason to retrieve, surface, clarify, "
            "or retain information that can materially help the DM handle the "
            "current situation or preserve continuity."
        ),
    },
    {
        "id": "wait",
        "description": (
            "No. The exchange created no meaningful information work. Additional "
            "context would be unnecessary noise, or the moment is incomplete, "
            "purely descriptive, sensory, scene-setting, or merely atmospheric."
        ),
    },
]

TIER_QUESTION = "Does this moment need a durable card or a brief note?"
TIER_OPTIONS = [
    {
        "id": "card",
        "description": (
            "A durable reference the DM keeps: a table, a ruling, a "
            "briefing worth re-reading."
        ),
    },
    {
        "id": "ephemeral",
        "description": (
            "A brief scene note: glanced at once, then gone. Nothing "
            "the DM will need again."
        ),
    },
]
TIER_IDS = tuple(o["id"] for o in TIER_OPTIONS)

# Relevance is judged AFTER both legs retrieve: the verdict compares
# retrieved evidence, never presumed locations. Provisional wording
# (trial-tested, not yet locked like the v39 gate prompt).
REL_QUESTION = "Which retrieved evidence is more relevant to the DM's current need?"
REL_OPTIONS = [
    {
        "id": "offline",
        "description": (
            "The campaign-notes evidence answers the need; the web results "
            "add nothing relevant."
        ),
    },
    {
        "id": "online",
        "description": (
            "The web evidence answers the need; the campaign notes add "
            "nothing relevant."
        ),
    },
    {
        "id": "both",
        "description": (
            "Both contribute relevant evidence the DM needs."
        ),
    },
]
REL_IDS = tuple(o["id"] for o in REL_OPTIONS)

TERM_RANK_QUESTION = (
    "Which span is the most useful search term for finding "
    "DM-supporting information about this exchange?"
)


class OpenjevError(RuntimeError):
    """Raised for openjev transport or response-shape problems."""


@dataclass(slots=True)
class OpenjevDecision:
    """Outcome of one gate evaluation over a transcript window."""

    deploy: bool
    prob: float  # P(deploy); uncalibrated conditional score, not confidence
    tier: str = "ephemeral"  # card | ephemeral (meaningful only on deploy)
    tier_prob: float = 0.0  # P(tier); 0 when the tier pass did not run
    latency_s: float = 0.0  # both scoring passes, when the tier pass ran
    error: str | None = None  # set when the gate failed closed to wait


@dataclass(slots=True)
class RouteVerdict:
    """Which retrieved evidence is relevant: offline | online | both."""

    route: str
    probs: dict[str, float] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None  # set; the verdict already failed over to both


class Debouncer:
    """Suppress redeploys inside a window (duplicate beats).

    One clock, no categories: any deploy holds the gate for ``window_s``.
    Fresh windows always pass.
    """

    def __init__(self, window_s: float = 30.0) -> None:
        self._window_s = max(0.0, float(window_s))
        self._last: float | None = None

    @property
    def window_s(self) -> float:
        """The suppression window in seconds (0 disables)."""
        return self._window_s

    def check(self, now: float) -> bool:
        """Return True when a deploy at ``now`` must be dropped as a repeat."""
        if self._window_s <= 0:
            return False
        if self._last is not None and now - self._last < self._window_s:
            return True
        self._last = now
        return False


class OpenjevGate:
    """Binary deploy/wait gate over an openjev-serve ``/score`` endpoint."""

    def __init__(
        self,
        base_url: str,
        threshold: float = 0.5,
        timeout_s: float = 3.0,
        recent_n: int = 8,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._threshold = float(threshold)
        self._timeout_s = float(timeout_s)
        self._recent_n = max(1, int(recent_n))
        self._client = client
        self._owns_client = client is None

    def set_base_url(self, base_url: str) -> None:
        """Repoint the gate at another /score endpoint (provider switch).

        Remote and local JEV speak the identical wire protocol, so only
        the base URL changes; threshold, timeout, and window are kept.
        """
        self._base_url = base_url.rstrip("/")

    async def aclose(self) -> None:
        """Release the HTTP client when this gate owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _client_or_new(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=self._timeout_s,
                    write=self._timeout_s,
                    pool=5.0,
                )
            )
            self._owns_client = True
        return self._client

    async def _score(
        self,
        row_id: str,
        state: str,
        question: str,
        options: list[dict[str, str]],
    ) -> tuple[dict[str, float], float]:
        """POST one decision row; return ({option_id: prob}, latency_s)."""
        row = {
            "id": row_id,
            "state": state,
            "question": question,
            "options": options,
        }
        t0 = time.monotonic()
        try:
            resp = await self._client_or_new().post(
                f"{self._base_url}/score",
                json=row,
                timeout=self._timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise OpenjevError(f"score timed out after {self._timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise OpenjevError(f"score transport failed: {exc}") from exc
        latency_s = time.monotonic() - t0
        if resp.status_code != 200:
            raise OpenjevError(f"score HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
            ids = list(body["option_ids"])
            probs = [float(p) for p in body["probabilities"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise OpenjevError(f"score bad shape: {exc}") from exc
        want = [o["id"] for o in options]
        if ids != want or len(probs) != len(want):
            raise OpenjevError(f"score option mismatch: {ids} != {want}")
        return dict(zip(ids, probs, strict=True)), latency_s

    async def decide(self, lines: list[str]) -> OpenjevDecision:
        """Evaluate the gate over recent transcript lines (current last).

        Fails closed to ``wait`` on any error; the error is recorded on
        the decision and logged, never raised.
        """
        window = [ln for ln in lines[-self._recent_n :] if ln.strip()]
        if not window:
            return OpenjevDecision(False, 0.0)
        state = "\n".join(window)
        tag = f"g{time.monotonic_ns()}"
        try:
            probs, dt = await self._score(tag, state, DEPLOY_QUESTION, DEPLOY_OPTIONS)
            p_deploy = probs["deploy"]
            if p_deploy < self._threshold:
                return OpenjevDecision(False, p_deploy, latency_s=dt)
            try:
                tiers, dt2 = await self._score(
                    f"{tag}t", state, TIER_QUESTION, TIER_OPTIONS
                )
            except OpenjevError as exc:
                logger.warning("openjev tier failed over to card: %s", exc)
                return OpenjevDecision(True, p_deploy, "card", 0.0, dt, error=str(exc))
            tier = max(TIER_IDS, key=lambda k: tiers[k])
            return OpenjevDecision(True, p_deploy, tier, tiers[tier], dt + dt2)
        except OpenjevError as exc:
            logger.warning("openjev gate failed closed to wait: %s", exc)
            return OpenjevDecision(False, 0.0, error=str(exc))

    async def rank(
        self, row_id: str, state: str, question: str, options: list[dict[str, str]]
    ) -> dict[str, float]:
        """Score one open pass; return {option_id: prob}. Raises OpenjevError."""
        probs, _ = await self._score(row_id, state, question, options)
        return probs

    async def relevance(self, state: str) -> RouteVerdict:
        """Judge retrieved evidence offline / online / both.

        Called AFTER both legs retrieve; compares evidence, never
        presumed locations. Fails over to ``both`` on any error:
        synthesizing from everything costs tokens, dropping the side
        that holds the answer costs the card.
        """
        tag = f"r{time.monotonic_ns()}"
        try:
            probs, dt = await self._score(tag, state, REL_QUESTION, REL_OPTIONS)
            return RouteVerdict(max(REL_IDS, key=lambda k: probs[k]), probs, dt)
        except OpenjevError as exc:
            logger.warning("openjev relevance failed over to both: %s", exc)
            return RouteVerdict("both", {}, 0.0, error=str(exc))

    async def health(self) -> dict[str, Any]:
        """GET the server /health document (raises OpenjevError on failure)."""
        try:
            resp = await self._client_or_new().get(
                f"{self._base_url}/health", timeout=self._timeout_s
            )
        except httpx.HTTPError as exc:
            raise OpenjevError(f"health failed: {exc}") from exc
        if resp.status_code != 200:
            raise OpenjevError(f"health HTTP {resp.status_code}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise OpenjevError(f"health bad shape: {exc}") from exc
        if not isinstance(body, dict):
            raise OpenjevError("health bad shape: not an object")
        return body
