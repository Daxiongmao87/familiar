"""Timed transcript replay: fixture loading and wall-clock pacing."""

from __future__ import annotations

import json

import pytest

from dmd.sources.transcript_replay import (
    TranscriptEvent,
    TranscriptReplayer,
    load_transcript_events,
)
from dmd.types import Utterance


def _write(tmp_path, entries) -> str:
    p = tmp_path / "fx.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return str(p)


def test_load_sorts_and_skips_non_transcript(tmp_path) -> None:
    path = _write(
        tmp_path,
        [
            {"t": 5.0, "text": "b", "type": "transcript", "user_id": "sam"},
            {"t": 1.0, "text": "a", "type": "transcript", "user_id": "dm"},
            {"t": 3.0, "type": "card", "card": {}},
        ],
    )
    events = load_transcript_events(path)
    assert [(e.t, e.user_id, e.text) for e in events] == [
        (1.0, "dm", "a"),
        (5.0, "sam", "b"),
    ]


def test_load_rejects_bad_fixtures(tmp_path) -> None:
    with pytest.raises(ValueError):
        load_transcript_events(_write(tmp_path, {"t": 0}))
    with pytest.raises(ValueError):
        load_transcript_events(_write(tmp_path, [{"t": 0, "text": "x"}]))
    with pytest.raises(ValueError):
        load_transcript_events(_write(tmp_path, [{"t": 0, "user_id": "u", "text": "  "}]))


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    async def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


async def test_realtime_paces_entries_to_recorded_offsets() -> None:
    events = [
        TranscriptEvent(t=10.0, user_id="dm", text="a"),
        TranscriptEvent(t=12.5, user_id="sam", text="b"),
        TranscriptEvent(t=13.0, user_id="dm", text="c"),
    ]
    clock = _Clock()
    got: list[Utterance] = []

    async def sink(u: Utterance) -> None:
        got.append(u)

    n = await TranscriptReplayer(events, sink, realtime=True, sleep=clock.sleep, clock=lambda: clock.now).run()
    assert n == 3
    assert clock.sleeps == pytest.approx([2.5, 0.5])
    assert [u.text for u in got] == ["a", "b", "c"]
    assert got[0].t_start < got[1].t_start < got[2].t_start


async def test_realtime_never_sleeps_negative_on_slow_sink() -> None:
    events = [
        TranscriptEvent(t=0.0, user_id="dm", text="a"),
        TranscriptEvent(t=0.5, user_id="sam", text="b"),
    ]
    clock = _Clock()
    clock.now = 0.0

    async def slow_sink(_u: Utterance) -> None:
        clock.now += 10.0  # sink overruns the next offset: catch up, don't rewind

    n = await TranscriptReplayer(
        events, slow_sink, realtime=True, sleep=clock.sleep, clock=lambda: clock.now
    ).run()
    assert n == 2
    assert clock.sleeps == []


async def test_fast_mode_dispatches_without_sleeping() -> None:
    events = [
        TranscriptEvent(t=0.0, user_id="dm", text="a"),
        TranscriptEvent(t=179.0, user_id="sam", text="b"),
    ]
    clock = _Clock()
    got: list[Utterance] = []

    async def sink(u: Utterance) -> None:
        got.append(u)

    n = await TranscriptReplayer(events, sink, realtime=False, sleep=clock.sleep, clock=lambda: clock.now).run()
    assert n == 2
    assert clock.sleeps == []
    assert [u.user_id for u in got] == ["dm", "sam"]


async def test_empty_fixture_delivers_nothing() -> None:
    async def sink(_u: Utterance) -> None:  # pragma: no cover
        raise AssertionError("no dispatch expected")

    assert await TranscriptReplayer([], sink).run() == 0


def test_cr_fixture_loads_with_expected_shape() -> None:
    events = load_transcript_events(
        "tests/regression/golden/cr2e2_3h14m29s_crownsguard.json"
    )
    assert len(events) == 45
    assert events[0].t == 0.0
    assert events[-1].t == pytest.approx(179.0)
    assert len({e.user_id for e in events}) == 7
