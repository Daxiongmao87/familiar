"""§7a per-speaker attribution: join diarized segments to named speakers.

The live capture is one mixed audio stream (DAVE per-user RTP is Phase 0 and
stays deferred by design), so speaker identity is recovered by a two-source
join — the owner-ordered design:

  * pyannote (WhisperX ``diarize=true``) segments the mixed audio into
    anonymous speaker windows relative to the clip start;
  * the Discord gateway's ``member_speaking_state_update`` events feed a
    :class:`~dmd.speaking_tracker.SpeakingTracker` of per-user speaking
    windows on the session's monotonic clock.

Attributing one diarized segment means mapping its clip-relative window to
absolute time (utterance start + segment offset) and picking the user whose
speaking window overlaps it the most. Segments with no overlapping speaking
window keep the source's fallback identity, so attribution can only refine
identity, never invent it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MIN_SEG_OVERLAP_S = 0.15


@dataclass
class AttributedSegment:
    """One diarized segment resolved to a named speaker."""

    user_id: str
    text: str
    t_start: float
    t_end: float
    speaker_label: str | None = None  # pyannote label (SPEAKER_00 …)
    name: str | None = None  # display name from the tracker, if known


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
