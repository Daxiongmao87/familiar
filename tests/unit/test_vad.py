"""Unit tests for dmd.vad: UtteranceSegmenter boundary detection."""

from __future__ import annotations

import struct

import pytest

from dmd.vad import UtteranceSegmenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pcm(duration_s: float, amplitude: int, sample_rate: int = 16000) -> bytes:
    """Synthesize constant-amplitude int16 LE mono PCM (no numpy needed)."""
    n_samples = int(duration_s * sample_rate)
    # `*values` splat avoids the struct "pack expected N items" trap with a list.
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _speech(duration_s: float, sample_rate: int = 16000) -> bytes:
    """Constant-1000 int16 -> RMS == 1000, well above the 250 energy_floor."""
    return _pcm(duration_s, 1000, sample_rate)


def _silence(duration_s: float, sample_rate: int = 16000) -> bytes:
    """Constant-0 int16 -> RMS == 0, below the 250 energy_floor."""
    return _pcm(duration_s, 0, sample_rate)


# ---------------------------------------------------------------------------
# Defaults / construction
# ---------------------------------------------------------------------------

def test_segmenter_default_max_bytes_is_60s_at_16khz():
    """The 60s cap is 60 * sample_rate * 2 bytes for int16 mono."""
    seg = UtteranceSegmenter(sample_rate=16000)
    assert seg._max_bytes == 60 * 16000 * 2  # 1,920,000


def test_segmenter_keeps_configured_thresholds():
    """Constructor params are stored verbatim on the instance."""
    seg = UtteranceSegmenter(
        sample_rate=8000, silence_ms=500, min_utterance_ms=250, energy_floor=123,
    )
    assert seg.sample_rate == 8000
    assert seg.silence_ms == 500
    assert seg.min_utterance_ms == 250
    assert seg.energy_floor == 123
    # Cap recomputed from sample_rate.
    assert seg._max_bytes == 60 * 8000 * 2


# ---------------------------------------------------------------------------
# Emits after speech then silence
# ---------------------------------------------------------------------------

def test_emits_utterance_after_speech_then_silence():
    """Speech followed by >= silence_ms of silence emits exactly one Utterance."""
    seg = UtteranceSegmenter(silence_ms=700, min_utterance_ms=400, energy_floor=250)
    out_after_speech = seg.feed("alice", _speech(0.5), t_mono=0.0)
    out_after_silence = seg.feed("alice", _silence(0.8), t_mono=0.5)

    # Speech alone must not emit; the silence must close the utterance.
    assert out_after_speech == []
    assert len(out_after_silence) == 1

    u = out_after_silence[0]
    assert u.user_id == "alice"
    assert u.t_start == 0.0
    # t_end is the last feed's t_mono (the silence feed).
    assert u.t_end == 0.5
    # text/raw_text are populated post-STT, not by the VAD.
    assert u.text == ""
    assert u.raw_text is None


def test_no_emit_when_silence_below_threshold():
    """Silence shorter than silence_ms must NOT trigger emission."""
    seg = UtteranceSegmenter(silence_ms=700, min_utterance_ms=400, energy_floor=250)
    seg.feed("alice", _speech(0.5), t_mono=0.0)
    out = seg.feed("alice", _silence(0.3), t_mono=0.5)  # 300ms < 700ms
    assert out == []
    # And a subsequent short silence still doesn't emit.
    out2 = seg.feed("alice", _silence(0.3), t_mono=0.8)  # cumulative 600ms < 700ms
    assert out2 == []


# ---------------------------------------------------------------------------
# No emission below min_utterance_ms (feed flow)
# ---------------------------------------------------------------------------

def test_no_emission_below_min_utterance_ms_after_silence_threshold():
    """When silence threshold is met but total buffer < min_utterance_ms, no Utterance is emitted.

    Uses custom params so the silence threshold can be crossed while the buffer is
    still shorter than min_utterance_ms (impossible at defaults because silence_ms
    alone already exceeds min_utterance_ms).
    """
    seg = UtteranceSegmenter(
        sample_rate=16000, silence_ms=200, min_utterance_ms=600, energy_floor=250,
    )
    # 100ms speech, then 250ms silence -> pcm=350ms, silence_accum=250ms.
    # 250 >= 200 (silence threshold) BUT 350 < 600 (min utterance).
    seg.feed("alice", _speech(0.10), t_mono=0.0)
    out = seg.feed("alice", _silence(0.25), t_mono=0.10)
    assert out == []
    # And the state must have been reset, so further feeds start fresh.
    # Feed only silence now: the (reset) state has speech_active=False, so silence
    # is ignored entirely and nothing emits.
    out2 = seg.feed("alice", _silence(1.0), t_mono=0.35)
    assert out2 == []


# ---------------------------------------------------------------------------
# flush_user
# ---------------------------------------------------------------------------

def test_flush_user_emits_pending_utterance_above_min():
    """flush_user emits a pending utterance whose duration >= min_utterance_ms."""
    seg = UtteranceSegmenter(silence_ms=700, min_utterance_ms=400, energy_floor=250)
    seg.feed("alice", _speech(0.5), t_mono=0.0)
    out = seg.flush_user("alice")
    assert len(out) == 1
    u = out[0]
    assert u.user_id == "alice"
    assert u.t_start == 0.0
    assert u.t_end == 0.0  # last_t was set during the speech feed


def test_flush_user_below_min_does_not_emit():
    """flush_user does NOT emit if pending audio is shorter than min_utterance_ms."""
    seg = UtteranceSegmenter(silence_ms=700, min_utterance_ms=400, energy_floor=250)
    seg.feed("alice", _speech(0.1), t_mono=0.0)  # 100ms < 400ms
    assert seg.flush_user("alice") == []


def test_flush_user_no_state_is_noop():
    """flush_user on an unknown user is a safe no-op (no exception)."""
    seg = UtteranceSegmenter()
    assert seg.flush_user("never-seen") == []


def test_flush_user_after_silence_emit_is_noop():
    """Once a silence-triggered emission has fired and reset state, flush is a no-op."""
    seg = UtteranceSegmenter(silence_ms=700, min_utterance_ms=400, energy_floor=250)
    seg.feed("alice", _speech(0.5), t_mono=0.0)
    seg.feed("alice", _silence(0.8), t_mono=0.5)  # emits + resets
    assert seg.flush_user("alice") == []


# ---------------------------------------------------------------------------
# 60s cap force-emits
# ---------------------------------------------------------------------------

def test_60s_cap_force_emits_continuous_speech():
    """Feeding > 60s of speech-marked audio (in many small chunks) must force an emit.

    The emit is gated by `len(state.pcm) >= self._max_bytes`, NOT by the
    silence threshold, so we never feed silence here.
    """
    sr = 16000
    seg = UtteranceSegmenter(sample_rate=sr, silence_ms=700, min_utterance_ms=400, energy_floor=250)
    # 1-second chunks of constant-1000 speech keep each call's overhead low.
    one_second_speech = _speech(1.0, sample_rate=sr)

    emitted: list = []
    t = 0.0
    feeds = 0
    for _ in range(120):  # 120s upper bound; cap is hit well before this
        out = seg.feed("alice", one_second_speech, t_mono=t)
        feeds += 1
        if out:
            emitted.extend(out)
            break
        t += 1.0
    else:
        pytest.fail("60s cap never triggered an emission")

    assert len(emitted) == 1
    u = emitted[0]
    assert u.user_id == "alice"
    # t_start was the t_mono of the FIRST speech chunk, t_end is the t_mono of the
    # most recent feed (the one whose pcm.extend pushed the buffer past 60s).
    assert u.t_start == 0.0
    # The cap fires on the 60th 1-second feed (indices 0..59), t_mono 0.0..59.0.
    assert feeds == 60
    assert u.t_end == 59.0


def test_60s_cap_resets_state_for_subsequent_silence():
    """After the cap emits, the segmenter should be in a fresh state.

    Specifically: speech_active should be False, so an immediate silence feed
    does NOT append to the buffer or trigger another emission.
    """
    sr = 16000
    seg = UtteranceSegmenter(sample_rate=sr, silence_ms=700, min_utterance_ms=400, energy_floor=250)
    one_second_speech = _speech(1.0, sample_rate=sr)

    # Drive the cap with 60 seconds of speech.
    t = 0.0
    for _ in range(120):
        out = seg.feed("alice", one_second_speech, t_mono=t)
        if out:
            break
        t += 1.0
    else:
        pytest.fail("60s cap never triggered")

    # Now feed silence. With a clean reset, this should be a no-op
    # (speech_active=False -> early return).
    out2 = seg.feed("alice", _silence(1.0), t_mono=t + 1.0)
    assert out2 == []


# ---------------------------------------------------------------------------
# Per-user isolation
# ---------------------------------------------------------------------------

def test_per_user_isolation_interleaved_feeds():
    """Two users fed interleaved speech + silence emit independently, tagged correctly."""
    seg = UtteranceSegmenter(silence_ms=700, min_utterance_ms=400, energy_floor=250)
    speech_500 = _speech(0.5)
    silence_800 = _silence(0.8)

    # Speech from both users (their states are independent).
    assert seg.feed("alice", speech_500, t_mono=0.0) == []
    assert seg.feed("bob", speech_500, t_mono=0.5) == []

    # Alice's silence closes her utterance, Bob's buffer is unaffected.
    out_alice = seg.feed("alice", silence_800, t_mono=1.0)
    out_bob_intermediate = seg.feed("bob", _silence(0.3), t_mono=1.5)
    assert out_bob_intermediate == []  # only 300ms silence, below 700ms threshold

    # Now Bob's silence crosses the threshold.
    out_bob = seg.feed("bob", silence_800, t_mono=1.8)

    assert len(out_alice) == 1
    assert out_alice[0].user_id == "alice"
    assert out_alice[0].t_start == 0.0
    assert out_alice[0].t_end == 1.0

    assert len(out_bob) == 1
    assert out_bob[0].user_id == "bob"
    assert out_bob[0].t_start == 0.5
    assert out_bob[0].t_end == 1.8


def test_per_user_isolation_one_user_does_not_emit_other():
    """Emitting for one user does not affect the other user's accumulated buffer."""
    seg = UtteranceSegmenter(silence_ms=700, min_utterance_ms=400, energy_floor=250)
    seg.feed("alice", _speech(0.5), t_mono=0.0)
    seg.feed("bob", _speech(0.5), t_mono=0.0)

    # Only feed Alice's silence: Bob's buffer must not emit, and the next Alice feed
    # must start clean.
    out_alice = seg.feed("alice", _silence(0.8), t_mono=0.5)
    assert len(out_alice) == 1
    assert out_alice[0].user_id == "alice"

    # Alice has been reset; another silence feed alone must not emit.
    assert seg.feed("alice", _silence(0.8), t_mono=1.3) == []

    # Bob's buffer is untouched: he still has 500ms of speech and zero silence.
    # Feeding Bob's silence now must produce Bob's utterance.
    out_bob = seg.feed("bob", _silence(0.8), t_mono=0.5)
    assert len(out_bob) == 1
    assert out_bob[0].user_id == "bob"
