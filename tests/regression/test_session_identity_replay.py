"""Replay real bot callbacks and transcript attribution without live services."""

import asyncio
from types import SimpleNamespace
import sys
import wave

import pytest

from dmd.speaking_tracker import SpeakingTracker
from dmd.voice_presence import VoicePresence
from dmd.pipeline import SessionEngine
from dmd.config import load_config_dict
from tests.session_replay import ReplayEvent, SessionReplay, word_error_rate


class Clock:
    """Virtual monotonic clock, preserving gaps without wall-clock waits."""

    def __init__(self):
        self.now = 100.0

    async def sleep(self, seconds):
        self.now += seconds


async def test_real_discord_callback_to_attributed_transcript(monkeypatch):
    """Uncached member, SSRC remap, overlap and unresolved stream replay."""
    clock = Clock()
    monkeypatch.setattr("dmd.speaking_tracker.time", SimpleNamespace(monotonic=lambda: clock.now))
    members = {111: SimpleNamespace(id=111, display_name="Sam"),
               222: SimpleNamespace(id=222, display_name="Matt")}
    mapping = {}
    guild = SimpleNamespace(voice_client=SimpleNamespace(_ssrc_to_id=mapping),
                            get_member=members.get)

    class Client:
        def __init__(self, **kwargs):
            pass

        def event(self, callback):
            setattr(self, callback.__name__, callback)
            return callback

        def get_guild(self, uid):
            return guild

        async def start(self, token):
            pass

    monkeypatch.setitem(sys.modules, "discord", SimpleNamespace(
        Client=Client, Intents=SimpleNamespace(none=lambda: SimpleNamespace())))
    tracker = SpeakingTracker()
    bot = VoicePresence("fixture", 1, 111, tracker)
    await bot.start()
    published = []
    cfg = load_config_dict({"models": {
        "synthesis": {"base_url": "http://unused", "model_id": "fixture"}, "stt": {}}})
    engine = SessionEngine(cfg, None, None, [], None, None, published.append,
                           speaking_tracker=tracker)
    engine.set_ooc(True)  # Keep real transcription/attribution; exclude generation.

    async def discord_event(p):
        if p.get("uid"):
            mapping[p["ssrc"]] = p["uid"]
        else:
            mapping.pop(p["ssrc"], None)
        await bot._client.on_member_speaking_state_update(None, p["ssrc"], p["state"])

    def speaking(at, uid, ssrc, state):
        return ReplayEvent(at, "discord", dict(uid=uid, ssrc=ssrc, state=state))

    def final(at, start, end, text):
        return ReplayEvent(at, "final", dict(start=start, end=end, text=text))

    events = [speaking(1, 111, 1502, 1), speaking(2, 111, 1502, 0),
              final(2.5, 1, 2, "Is there a guard?"),
              speaking(3, 222, 1502, 1), speaking(4, 111, 900, 1),
              speaking(4.5, 111, 900, 0), speaking(5, 222, 1502, 0),
              final(5.5, 3, 5, "There are two guards."),
              speaking(6, None, 999, 1),
              final(7, 6, 6.5, "An unidentified voice.")]
    replay = SessionReplay(events, discord_event, engine._on_stream_final,
                           clock=lambda: clock.now, sleep=clock.sleep)
    try:
        await replay.run()
        transcripts = [p for p in published if p["type"] == "transcript"]
        assert [p["user_id"] for p in transcripts] == ["111", "222", "browser_mixed"]
        assert [p.get("name") for p in transcripts[:2]] == ["Sam", "Matt"]
        assert tracker.snapshot()["active"] == {}
        assert clock.now == 107
    finally:
        await engine.aclose()
        await bot._task


async def test_pcm_replay_preserves_samples_and_clock(tmp_path):
    """Real WAV decoding reaches AudioSource unchanged, with aligned events."""
    path = tmp_path / "audio.wav"
    pcm = b"\x01\x00" * 6400
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)
    clock = Clock()
    seen = []

    async def sink(payload):
        seen.append(clock.now)

    replay = SessionReplay([ReplayEvent(.1, "discord", {})], sink, sink,
                           wav_path=str(path), clock=lambda: clock.now, sleep=clock.sleep)
    await replay.run()
    chunks = [chunk async for chunk in replay]
    assert b"".join(c.samples for c in chunks) == pcm
    assert [c.t_mono for c in chunks] == [100, 100.2]
    assert seen == [100.1]


async def test_late_speaking_evidence_revises_ui_without_second_dispatch():
    """A later gateway window revises only an uncertain transcript row."""
    tracker = SpeakingTracker()
    published = []
    cfg = load_config_dict({"models": {
        "synthesis": {"base_url": "http://unused", "model_id": "fixture"}, "stt": {}}})
    engine = SessionEngine(cfg, None, None, [], None, None, published.append,
                           speaking_tracker=tracker)
    engine.set_ooc(True)
    try:
        await engine._on_stream_final("browser_mixed", "First line", 1.0, 2.0)
        first = next(event for event in published if event["type"] == "transcript")
        assert first["attribution"]["state"] == "unknown"

        tracker.on_speaking("111", True, t=1.0, name="Sam")
        tracker.on_speaking("111", False, t=2.0)
        await engine._on_stream_final("browser_mixed", "Second line", 3.0, 4.0)

        revisions = [event for event in published if event["type"] == "transcript_revision"]
        assert len(revisions) == 1
        assert revisions[0]["id"] == first["id"]
        assert revisions[0]["user_id"] == "111"
        assert revisions[0]["attribution"]["revision"] == 1
    finally:
        await engine.aclose()


async def test_jev_review_revises_only_ambiguous_row(monkeypatch):
    """JEV review is bounded UI work and cannot re-enter card dispatch."""
    tracker = SpeakingTracker()
    tracker.on_speaking("111", True, t=1.0, name="Sam")
    tracker.on_speaking("111", False, t=1.7)
    tracker.on_speaking("222", True, t=1.1, name="Matt")
    tracker.on_speaking("222", False, t=1.8)
    published = []
    cfg = load_config_dict({"models": {
        "synthesis": {"base_url": "http://unused", "model_id": "fixture"}, "stt": {}}})
    engine = SessionEngine(cfg, None, None, [], None, None, published.append,
                           speaking_tracker=tracker)
    engine.set_ooc(True)

    async def rank(*_args):
        return {"member:111": .8, "member:222": .1, "unknown": .05, "multiple": .05}

    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final("browser_mixed", "I cast a spell.", 1.0, 2.0)
        # Bounded drain, not a single yield: the rolling reviewer serializes
        # passes through a semaphore + timeout, so the old fire-and-forget
        # timing assumption (one sleep(0)) no longer holds. Expectation kept.
        drain = getattr(engine, "drain_attribution_reviews", None)
        if callable(drain):
            await drain(5.0)
        else:
            await asyncio.sleep(0)
        revisions = [event for event in published if event["type"] == "transcript_revision"]
        assert len(revisions) == 1
        assert revisions[0]["user_id"] == "111"
        assert revisions[0]["attribution"]["source"] == "jev_contextual_review"
        assert revisions[0]["attribution"]["review_scores"]["member:111"] == .8
    finally:
        await engine.aclose()


def test_word_error_measurement_and_mode_separation():
    """Substitution/deletion errors count; audio cannot inject oracle text."""
    assert word_error_rate("Two guards stand here.", "Two guard here") == .5
    with pytest.raises(ValueError, match="real STT"):
        SessionReplay([ReplayEvent(1, "final", dict(start=0, end=1, text="x"))],
                      None, None, wav_path="unused.wav")
