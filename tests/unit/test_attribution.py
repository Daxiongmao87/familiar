"""§7a attribution: join pyannote segment windows x speaking windows.

The mixed-capture reality (DAVE per-user RTP deferred): speaker identity is
recovered by intersecting diarized segment windows (clip-relative seconds
mapped through the utterance's absolute start) with the Discord gateway
SpeakingTracker's per-user speaking windows.
"""

from __future__ import annotations

from typing import Any

from dmd.attribution import (
    AttributedSegment,
    attribute_segments,
    attribute_whole,
    attribute_whole_evidence,
    group_by_speaker,
)
from dmd.speaking_tracker import SpeakingTracker


def _tracker_with_windows() -> SpeakingTracker:
    tr = SpeakingTracker()
    # user 111 speaks [0, 3); user 222 speaks [3, 6) on the monotonic clock.
    tr.on_speaking("111", True, t=0.0, name="Kael")
    tr.on_speaking("111", False, t=3.0)
    tr.on_speaking("222", True, t=3.0, name="Mira")
    tr.on_speaking("222", False, t=6.0)
    return tr


def test_segment_named_by_max_overlapping_window() -> None:
    tr = _tracker_with_windows()
    segments = [
        {"speaker": "SPEAKER_00", "start": 0.2, "end": 2.5, "text": "I search the corpse"},
        {"speaker": "SPEAKER_01", "start": 2.6, "end": 5.5, "text": "mind the blood"},
    ]
    out = attribute_segments(tr, segments, t_start=0.0, fallback_user_id="browser_mixed")
    assert [s.user_id for s in out] == ["111", "222"]
    assert [s.name for s in out] == ["Kael", "Mira"]
    assert out[0].t_start == 0.2 and out[1].t_end == 5.5
    assert out[0].speaker_label == "SPEAKER_00"


def test_segment_without_overlap_keeps_source_identity() -> None:
    tr = _tracker_with_windows()
    segments = [{"speaker": "SPEAKER_00", "start": 8.0, "end": 9.0, "text": "stray"}]
    out = attribute_segments(tr, segments, t_start=0.0, fallback_user_id="browser_mixed")
    assert out[0].user_id == "browser_mixed"
    assert out[0].name is None


def test_bystander_blip_does_not_steal_a_segment() -> None:
    """A speaker overlapping <15% of a segment (and nothing else overlapping)
    must not win — attribution only refines on real signal."""
    tr = SpeakingTracker()
    tr.on_speaking("333", True, t=20.0)
    tr.on_speaking("333", False, t=20.05)
    segments = [{"speaker": "SPEAKER_00", "start": 20.0, "end": 20.2, "text": "mostly quiet"}]
    out = attribute_segments(tr, segments, t_start=0.0, fallback_user_id="browser_mixed")
    assert out[0].user_id == "browser_mixed"
    # ...but a blip inside a tiny segment where it is the only speaker still
    # attributes (its window covers the whole clip).
    segments = [{"speaker": "SPEAKER_00", "start": 20.0, "end": 20.05, "text": "short"}]
    out = attribute_segments(tr, segments, t_start=0.0, fallback_user_id="browser_mixed")
    assert out[0].user_id == "333"


def test_crosstalk_grouping_splits_by_speaker_and_merges_consecutive() -> None:
    attributed = [
        AttributedSegment(user_id="a", text="one", t_start=0.0, t_end=1.0),
        AttributedSegment(user_id="a", text="two", t_start=1.0, t_end=2.0),
        AttributedSegment(user_id="b", text="three", t_start=2.0, t_end=3.0),
        AttributedSegment(user_id="a", text="four", t_start=3.0, t_end=4.0),
    ]
    groups = group_by_speaker(attributed)
    assert [(g.user_id, g.text, g.t_start, g.t_end) for g in groups] == [
        ("a", "one two", 0.0, 2.0),
        ("b", "three", 2.0, 3.0),
        ("a", "four", 3.0, 4.0),
    ]


def test_attribute_whole_picks_dominant_speaker() -> None:
    tr = _tracker_with_windows()
    uid, name = attribute_whole(tr, 3.5, 5.5, "browser_mixed")
    assert uid == "222"
    assert name == "Mira"


def test_attribute_whole_no_signal_falls_back() -> None:
    tr = _tracker_with_windows()
    uid, name = attribute_whole(tr, 42.0, 43.0, "browser_mixed")
    assert uid == "browser_mixed"
    assert name is None


def test_evidence_reports_certain_assignment_without_probability_claim() -> None:
    tr = _tracker_with_windows()
    result = attribute_whole_evidence(tr, 3.5, 5.5, "browser_mixed")
    assert result.user_id == "222"
    assert result.state == "certain"
    assert result.coverage == 1.0
    assert result.margin == 1.0
    assert result.candidates == {"222": 2.0}


def test_evidence_preserves_mixed_capture_on_near_tied_crosstalk() -> None:
    tr = SpeakingTracker()
    tr.on_speaking("sam", True, t=0)
    tr.on_speaking("sam", False, t=0.7)
    tr.on_speaking("matt", True, t=0.1)
    tr.on_speaking("matt", False, t=0.8)
    result = attribute_whole_evidence(tr, 0, 1, "browser_mixed")
    assert result.user_id == "browser_mixed"
    assert result.state == "ambiguous"
    assert result.candidates == {"sam": 0.7, "matt": 0.7}


def test_attribution_survives_absent_tracker() -> None:
    segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "x"}]
    out = attribute_segments(None, segments, 0.0, "src")  # type: ignore[arg-type]
    assert out[0].user_id == "src"
