"""§7a per-speaker attribution: join transcript windows to named speakers.

The live capture is one mixed audio stream (DAVE per-user RTP is Phase 0 and
stays deferred by design), so speaker identity is recovered by a two-source
join — the owner-ordered design:

  * the streaming STT server endpoints the mixed audio into transcript
    windows (whole-utterance; no speaker turns today);
  * the Discord gateway's ``member_speaking_state_update`` events feed a
    :class:`~dmd.speaking_tracker.SpeakingTracker` of per-user speaking
    windows on the session's monotonic clock.

Attributing one window means picking the user whose speaking window
overlaps it the most. Windows with no overlapping speaking window keep
the source's fallback identity, so attribution can only refine identity,
never invent it. The segment-level join (``attribute_segments``) stays
for a future server that emits speaker turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MIN_SEG_OVERLAP_S = 0.15
_MIN_COVERAGE = 0.50
_MIN_MARGIN = 0.15


@dataclass
class AttributedSegment:
    """One diarized segment resolved to a named speaker."""

    user_id: str
    text: str
    t_start: float
    t_end: float
    speaker_label: str | None = None  # pyannote label (SPEAKER_00 …)
    name: str | None = None  # display name from the tracker, if known
    candidates: dict[str, float] | None = None
    coverage: float = 0.0
    margin: float = 0.0
    state: str = "unknown"  # certain | ambiguous | unknown


def attribute_segments(
    tracker: Any,
    segments: list[dict[str, Any]],
    t_start: float,
    fallback_user_id: str,
) -> list[AttributedSegment]:
    """Join pyannote segment windows x the tracker's speaking windows.

    ``segments`` are gateway-shaped: ``{"speaker", "start", "end", "text"}``
    with clip-relative seconds; ``t_start`` is the utterance's absolute
    (monotonic) start. A segment is named when the best-overlapping user
    covers at least ``_MIN_SEG_OVERLAP_S`` of it; otherwise it keeps
    ``fallback_user_id``.
    """
    out: list[AttributedSegment] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        a = t_start + float(seg.get("start", 0.0) or 0.0)
        b = t_start + float(seg.get("end", 0.0) or 0.0)
        if b < a:
            a, b = b, a
        user_id, name = _best_speaker(tracker, a, b, seg_dur=max(b - a, 0.0))
        out.append(
            AttributedSegment(
                user_id=user_id or fallback_user_id,
                text=text,
                t_start=a,
                t_end=b,
                speaker_label=seg.get("speaker"),
                name=name,
            )
        )
    return out


def _best_speaker(
    tracker: Any, a: float, b: float, seg_dur: float
) -> tuple[str | None, str | None]:
    if tracker is None:
        return None, None
    try:
        overlaps = tracker.overlaps_during(a, b)
    except Exception:
        return None, None
    if not overlaps:
        return None, None
    uid = max(overlaps, key=lambda u: overlaps[u])
    need = min(_MIN_SEG_OVERLAP_S, seg_dur) if seg_dur else _MIN_SEG_OVERLAP_S
    if overlaps[uid] < need:
        return None, None
    name = None
    get_name = getattr(tracker, "name_of", None)
    if callable(get_name):
        try:
            name = get_name(uid)
        except Exception:
            name = None
    return uid, name


def group_by_speaker(
    attributed: list[AttributedSegment],
) -> list[AttributedSegment]:
    """Merge consecutive segments attributed to the same speaker.

    One VAD utterance may contain crosstalk; each speaker change starts a new
    grouped segment so the transcript line carries exactly one identity.
    """
    out: list[AttributedSegment] = []
    for seg in attributed:
        if out and out[-1].user_id == seg.user_id:
            last = out[-1]
            last.text = f"{last.text} {seg.text}".strip()
            last.t_end = seg.t_end
            if seg.name and not last.name:
                last.name = seg.name
            continue
        out.append(
            AttributedSegment(
                user_id=seg.user_id,
                text=seg.text,
                t_start=seg.t_start,
                t_end=seg.t_end,
                speaker_label=seg.speaker_label,
                name=seg.name,
            )
        )
    return out


def attribute_whole(
    tracker: Any,
    t_start: float,
    t_end: float,
    fallback_user_id: str,
) -> tuple[str, str | None]:
    """Coarse fallback (no diarized segments): the dominant speaker across the
    whole utterance window, or the source's own identity."""
    if tracker is None:
        return fallback_user_id, None
    try:
        overlaps = tracker.overlaps_during(t_start, t_end)
    except Exception:
        return fallback_user_id, None
    if not overlaps:
        return fallback_user_id, None
    uid = max(overlaps, key=lambda u: overlaps[u])
    if overlaps[uid] < min(_MIN_SEG_OVERLAP_S, max(t_end - t_start, 0.0)):
        return fallback_user_id, None
    name = None
    get_name = getattr(tracker, "name_of", None)
    if callable(get_name):
        try:
            name = get_name(uid)
        except Exception:
            name = None
    return uid, name


def attribute_whole_evidence(
    tracker: Any, t_start: float, t_end: float, fallback_user_id: str
) -> AttributedSegment:
    """Return an attribution with auditable overlap evidence.

    Coverage and margin are evidence measurements, not calibrated model
    probabilities. An ambiguous or unsupported window keeps the capture
    identity rather than inventing a Discord member.
    """
    duration = max(t_end - t_start, 0.0)
    if tracker is None or duration <= 0:
        return AttributedSegment(fallback_user_id, "", t_start, t_end)
    try:
        overlaps = tracker.overlaps_during(t_start, t_end)
    except Exception:
        overlaps = {}
    candidates = {
        str(uid): round(max(0.0, value), 4) for uid, value in overlaps.items()
    }
    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return AttributedSegment(
            fallback_user_id, "", t_start, t_end, candidates=candidates, state="unknown"
        )
    uid, winner = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    coverage = min(1.0, winner / duration)
    margin = max(0.0, (winner - runner_up) / duration)
    if winner < min(_MIN_SEG_OVERLAP_S, duration) or coverage < _MIN_COVERAGE:
        state = "unknown"
    elif len(ranked) > 1 and margin < _MIN_MARGIN:
        state = "ambiguous"
    else:
        state = "certain"
    if state != "certain":
        return AttributedSegment(
            fallback_user_id, "", t_start, t_end, candidates=candidates,
            coverage=coverage, margin=margin, state=state,
        )
    name = None
    get_name = getattr(tracker, "name_of", None)
    if callable(get_name):
        try:
            name = get_name(uid)
        except Exception:
            name = None
    return AttributedSegment(
        uid, "", t_start, t_end, name=name, candidates=candidates,
        coverage=coverage, margin=margin, state=state,
    )
