"""Behavior tests for opt-in desktop diagnostic and transcript persistence."""

from __future__ import annotations

import json

from dmd.server import _log_debug_event


def test_debug_final_transcript_is_saved_as_private_jsonl(tmp_path, monkeypatch):
    """A finalized transcript becomes one structured, private record."""
    target = tmp_path / "logs" / "familiar-transcript.jsonl"
    monkeypatch.setenv("DMD_DEBUG", "1")
    monkeypatch.setenv("DMD_TRANSCRIPT_LOG", str(target))

    _log_debug_event(
        {
            "type": "transcript",
            "t": 123.5,
            "user_id": "188660400722673664",
            "name": "Patrick",
            "text": "Does anyone have the rules for grapple?",
        }
    )

    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "type": "transcript",
            "id": None,
            "t": 123.5,
            "user_id": "188660400722673664",
            "name": "Patrick",
            "text": "Does anyone have the rules for grapple?",
            "attribution": None,
        }
    ]
    assert target.stat().st_mode & 0o777 == 0o600


def test_debug_partial_transcript_is_not_saved(tmp_path, monkeypatch):
    """Unstable partial hypotheses never pollute the saved transcript."""
    target = tmp_path / "familiar-transcript.jsonl"
    monkeypatch.setenv("DMD_DEBUG", "1")
    monkeypatch.setenv("DMD_TRANSCRIPT_LOG", str(target))

    _log_debug_event(
        {"type": "transcript_partial", "user_id": "speaker", "text": "grap"}
    )

    assert not target.exists()


def test_debug_transcript_revision_is_saved_for_reconstruction(tmp_path, monkeypatch):
    """A late attribution correction follows its stable transcript ID."""
    target = tmp_path / "familiar-transcript.jsonl"
    monkeypatch.setenv("DMD_DEBUG", "1")
    monkeypatch.setenv("DMD_TRANSCRIPT_LOG", str(target))
    _log_debug_event({
        "type": "transcript_revision", "id": "line-1", "user_id": "sam",
        "name": "Sam", "attribution": {"state": "contextual_review"},
    })
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["type"] == "transcript_revision"
    assert record["id"] == "line-1"
    assert record["attribution"]["state"] == "contextual_review"


def test_transcript_is_not_saved_outside_debug_mode(tmp_path, monkeypatch):
    """Normal launches remain non-persistent even if a stale path exists."""
    target = tmp_path / "familiar-transcript.jsonl"
    monkeypatch.delenv("DMD_DEBUG", raising=False)
    monkeypatch.setenv("DMD_TRANSCRIPT_LOG", str(target))

    _log_debug_event(
        {"type": "transcript", "user_id": "speaker", "text": "private words"}
    )

    assert not target.exists()
