"""§7a/§4: the pipeline publishes attributed, named transcript lines.

Mixed capture (browser source) tags every streaming final with an anonymous
source id; when a SpeakingTracker is wired, the pipeline must replace it with
the Discord user dominating the final's window (whole-window attribution —
streaming finals carry no speaker turns).
"""

from __future__ import annotations

from dmd.config import load_config_dict
from dmd.pipeline import SessionEngine
from dmd.speaking_tracker import SpeakingTracker

SOURCE_ID = "browser_mixed"


class _NoPool:
    async def submit(self, job, work) -> None:
        return None


def _engine(tracker, events) -> SessionEngine:
    cfg = load_config_dict(
        {
            "project": {"path": "/nonexistent"},
            "models": {
                "synthesis": {"base_url": "http://fake", "model_id": "m"},
                "stt": {},
            },
        }
    )
    return SessionEngine(
        cfg=cfg,
        store=None,  # type: ignore[arg-type]
        gw=object(),  # STT never touches the gateway (streaming-only)
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


async def test_streaming_finals_become_named_speaker_lines() -> None:
    events: list[dict] = []
    engine = _engine(_tracker(), events)

    await engine._on_stream_final("111-src", "I search the body", 10.0, 13.0)
    await engine._on_stream_final("222-src", "careful with the trap", 13.0, 16.0)

    transcripts = [e for e in events if e["type"] == "transcript"]
    assert [t["user_id"] for t in transcripts] == ["111", "222"]
    assert [t.get("name") for t in transcripts] == ["Kael", "Mira"]
    await engine.aclose()


async def test_final_falls_back_to_dominant_speaker() -> None:
    events: list[dict] = []
    engine = _engine(_tracker(), events)

    await engine._on_stream_final(SOURCE_ID, "a single voice line", 12.0, 16.0)

    transcripts = [e for e in events if e["type"] == "transcript"]
    assert len(transcripts) == 1
    assert transcripts[0]["user_id"] == "222"  # 4 s of 222 vs 1 s of 111
    await engine.aclose()


async def test_without_tracker_source_identity_is_kept() -> None:
    """Replay/per-user sources must not be re-attributed (no gateway events)."""
    events: list[dict] = []
    engine = _engine(None, events)
    await engine._on_stream_final("alice", "plain line", 5.0, 6.0)
    assert len(events) >= 1
    assert events[0]["type"] == "transcript"
    assert events[0]["user_id"] == "alice"
    assert "name" not in events[0]
    await engine.aclose()


async def test_empty_tracker_keeps_source_identity() -> None:
    """A tracker with no speaking windows cannot refine identity."""
    events: list[dict] = []
    engine = _engine(SpeakingTracker(), events)
    await engine._on_stream_final(SOURCE_ID, "hi", 5.0, 6.0)
    transcripts = [e for e in events if e["type"] == "transcript"]
    assert [t["user_id"] for t in transcripts] == [SOURCE_ID]
    await engine.aclose()


async def test_empty_final_text_produces_no_line() -> None:
    events: list[dict] = []
    engine = _engine(_tracker(), events)
    await engine._on_stream_final(SOURCE_ID, "   ", 10.0, 11.0)
    assert events == []
    await engine.aclose()
