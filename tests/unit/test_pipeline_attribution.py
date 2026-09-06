"""§7a/§4: the pipeline publishes attributed, named transcript lines.

Mixed capture (browser source) tags every VAD utterance with an anonymous
source id; when a SpeakingTracker is wired, the pipeline must replace it with
the Discord user behind each diarized segment (per-speaker attribution), or —
without diarized segments — the dominant speaker across the utterance window.
"""

from __future__ import annotations

from typing import Any

from dmd.config import load_config_dict
from dmd.pipeline import SessionEngine
from dmd.speaking_tracker import SpeakingTracker

SOURCE_ID = "browser_mixed"


class _DiarGw:
    """Gateway stub: whisperx-style text + speaker-labeled segments."""

    def __init__(self, text: str, segments: list[dict[str, Any]]) -> None:
        self._text = text
        self._segments = segments
        self.calls: list[dict] = []

    async def transcribe_diarized(
        self, wav: bytes, prompt: str | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        self.calls.append({"wav_len": len(wav)})
        return self._text, self._segments

    async def transcribe(self, wav: bytes, prompt: str | None = None) -> str:
        self.calls.append({"wav_len": len(wav)})
        return self._text


class _NoPool:
    async def submit(self, job: Any, work: Any) -> None:
        return None


def _engine(gw: Any, tracker: Any, events: Any) -> SessionEngine:
    cfg = load_config_dict(
        {
            "project": {"path": "/nonexistent"},
            "models": {
                "synthesis": {"base_url": "http://fake", "model_id": "m"},
                "stt": {"base_url": "http://fake", "dialect": "whisperx"},
            },
        }
    )
    return SessionEngine(
        cfg=cfg,
        store=None,  # type: ignore[arg-type]
        gw=gw,
        entries=[],
        embedder=None,
        pool=_NoPool(),
        on_event=events.append,
        project_path="",
        speaking_tracker=tracker,
    )


def _tracker() -> SpeakingTracker:
    tr = SpeakingTracker()
    tr.on_speaking("111", True, t=10.0, name="Kael")
    tr.on_speaking("111", False, t=13.0)
    tr.on_speaking("222", True, t=13.0, name="Mira")
    tr.on_speaking("222", False, t=16.0)
    return tr


async def test_diarized_segments_become_named_speaker_lines() -> None:
    segments = [
        {"speaker": "SPEAKER_00", "start": 0.1, "end": 2.8, "text": "I search the body"},
        {"speaker": "SPEAKER_01", "start": 3.1, "end": 5.9, "text": "careful with the trap"},
    ]
    events: list[dict] = []
    gw = _DiarGw("I search the body careful with the trap", segments)
    engine = _engine(gw, _tracker(), events)
    # diarize is OFF by default now (owner decision 2026-09-05 — dead on the
    # mixed stream); this test pins the legacy segment-join path explicitly.
    engine.cfg.models.stt.diarize = True

    utts = await engine.transcribe_pcm(SOURCE_ID, b"\x01\x02" * 100, 10.0, 16.0)

    assert [u.user_id for u in utts] == ["111", "222"]
    assert [u.name for u in utts] == ["Kael", "Mira"]
    transcripts = [e for e in events if e["type"] == "transcript"]
    assert [t["user_id"] for t in transcripts] == ["111", "222"]
    assert [t.get("name") for t in transcripts] == ["Kael", "Mira"]


async def test_no_segments_falls_back_to_dominant_speaker() -> None:
    events: list[dict] = []
    gw = _DiarGw("a single voice line", [])
    engine = _engine(gw, _tracker(), events)

    utts = await engine.transcribe_pcm(SOURCE_ID, b"\x01\x02" * 100, 12.0, 16.0)

    assert len(utts) == 1
    assert utts[0].user_id == "222"  # 4 s of 222 vs 1 s of 111 across [12,16]
    assert events[0]["user_id"] == "222"


async def test_without_tracker_source_identity_is_kept() -> None:
    """Replay/per-user sources must not be re-attributed (no gateway events)."""
    events: list[dict] = []
    gw = _DiarGw("plain line", [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "plain line"},
    ])
    engine = _engine(gw, None, events)
    utts = await engine.transcribe_pcm("alice", b"\x01\x02" * 100, 5.0, 6.0)
    assert [u.user_id for u in utts] == ["alice"]
    assert events[0]["type"] == "transcript"
    assert events[0]["user_id"] == "alice"
    assert "name" not in events[0]


async def test_diarized_gw_used_only_when_diarize_enabled() -> None:
    segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "hi"}]
    events: list[dict] = []
    gw = _DiarGw("hi", segments)
    engine = _engine(gw, _tracker(), events)
    engine.cfg.models.stt.diarize = False
    await engine.transcribe_pcm(SOURCE_ID, b"\x01\x02" * 100, 10.5, 11.5)
    assert gw.calls and "wav_len" in gw.calls[0]
    # diarize off -> plain transcribe() path, whole-window attribution still applies
    transcript = events[0]
    assert transcript["user_id"] == "111"


async def test_empty_segment_text_produces_no_line() -> None:
    events: list[dict] = []
    gw = _DiarGw("", [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "   "},
    ])
    engine = _engine(gw, _tracker(), events)
    utts = await engine.transcribe_pcm(SOURCE_ID, b"\x01\x02" * 100, 10.0, 11.0)
    assert utts == []
    assert events == []


async def test_diarize_requested_only_when_tracker_has_windows() -> None:
    """§14/§7a live finding: pyannote costs ~2 s/utterance on the real
    endpoint; it must run only when there are speaking windows to join."""
    segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "hi"}]
    events: list[dict] = []

    # Empty tracker -> plain transcribe(), no diarized call.
    gw = _DiarGw("hi", segments)
    engine = _engine(gw, SpeakingTracker(), events)
    assert engine._attribution_active() is False
    await engine.transcribe_pcm(SOURCE_ID, b"\x01\x02" * 100, 5.0, 6.0)
    assert len(gw.calls) == 1  # transcribe() only

    # Tracker with speaking history -> diarized path engages.
    tr = SpeakingTracker()
    tr.on_speaking("111", True, t=5.0, name="Kael")
    tr.on_speaking("111", False, t=6.0)
    gw2 = _DiarGw("hi", segments)
    engine2 = _engine(gw2, tr, events)
    assert engine2._attribution_active() is True
    utts = await engine2.transcribe_pcm(SOURCE_ID, b"\x01\x02" * 100, 5.0, 6.0)
    assert len(gw2.calls) == 1
    assert utts[0].user_id == "111"  # attributed, from diarized segments
