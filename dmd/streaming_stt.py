"""Streaming STT client for the SimulStreaming whisper server (TCP).

Protocol, verified against the vendored server
(services/stt_server/whisper_online_server.py: upstream
ufal/whisper_streaming @ 6da90b44 plus a JSON/finals patch, run with
--vac so VAD utterance ends become is_final lines):

- Client connects and streams continuous raw s16le mono 16 kHz PCM
  bytes — no handshake at all (reference client:
  ``arecord -f S16_LE -c1 -r 16000 -t raw | nc localhost $PORT``).
- Server replies with newline-delimited JSON per processing iteration:
  ``{"text": str, "start": float, "end": float, "is_final": bool,
  "emission_time": float}`` — ``is_final: false`` are mid-speech
  partials; ``is_final: true`` is the VAD-ended committed segment.
  Empty ``{}`` lines (VAD-only ticks) carry no text and are ignored.
- The server accepts ONE client connection at a time (``listen(1)``).
  For a small table group we hold a session per active speaker
  sequentially; ``StreamingSttAdapter`` opens a user session on first
  audio and closes it on mic-off / silence, and callers keep the
  server free by feeding quiet users no bytes.

Contract with the pipeline:

- ``feed(user_id, pcm16_bytes)`` — push 16 kHz mono int16 audio.
- ``on_partial(user_id, text)`` — provisional mid-speech hypothesis.
- ``on_final(user_id, text, t_start, t_end)`` — committed transcript.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class StreamingSttAdapter:
    """Per-user streaming transcription against whisper_online_server.

    The server's socket protocol is: client streams raw 16-bit PCM bytes
    from byte zero (no handshake — the reference client is
    ``arecord ... | nc host port``); the server replies with newline JSON
    per confirmed increment: {"text","start","end","is_final"}.

    We run ONE connection per user, held open while that user's Discord mic
    light is on. On mic-off (or silence timeout) we close the socket; any
    trailing ``is_final`` response is drained first with a short timeout.
    """

    # Minimum seconds between reconnect attempts per user while the server
    # is unreachable (streaming is the only STT path — no batch fallback).
    RETRY_S = 2.0

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 43007,
        *,
        on_partial: Callable[[str, str], Awaitable[None]] | None = None,
        on_final: Callable[[str, str, float, float], Awaitable[None]]
        | None = None,
        drain_timeout_s: float = 2.5,
        idle_close_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.on_partial = on_partial
        self.on_final = on_final
        self.drain_timeout_s = drain_timeout_s
        # A session with no audio for this long is closed. The server's own
        # VAD endpoints segments at ~0.85 s of silence (finals already
        # fired), so this only reaps idle sockets between turns.
        self.idle_close_s = idle_close_s
        self._sessions: dict[str, _UserStream] = {}
        self._last_fail: dict[str, float] = {}

    async def feed(self, user_id: str, pcm16: bytes) -> bool:
        """Append PCM for a user; opens a stream lazily on first bytes.

        Never raises: a streaming outage must not stall audio intake. A
        failed session is dropped and reopened on later audio, at most
        once per ``RETRY_S`` per user so a dead server costs one cheap
        refused connect every couple of seconds instead of one per chunk.
        Returns True when a live session holds the user's audio.
        """
        now = time.monotonic()
        stale = [
            uid
            for uid, s in self._sessions.items()
            if (now - s.last_activity) > self.idle_close_s
        ]
        for uid in stale:
            await self.close_user(uid)
        sess = self._sessions.get(user_id)
        if sess is None:
            if now - self._last_fail.get(user_id, 0.0) < self.RETRY_S:
                return False
            sess = _UserStream(user_id, self)
            try:
                await sess.start()
            except OSError:
                self._sessions.pop(user_id, None)
                self._last_fail[user_id] = now
                return False
            self._last_fail.pop(user_id, None)
            self._sessions[user_id] = sess
        await sess.push(pcm16)
        return not sess._closed

    def has_session(self, user_id: str) -> bool:
        """True while a live server session holds this user's audio."""
        sess = self._sessions.get(user_id)
        return sess is not None and not sess._closed

    async def close_user(self, user_id: str) -> None:
        """Mic went quiet/off — finish and drop that user's stream."""
        sess = self._sessions.pop(user_id, None)
        if sess is not None:
            await sess.finish()

    async def close_all(self) -> None:
        for uid in list(self._sessions):
            await self.close_user(uid)


class _UserStream:
    """One TCP session to the streaming server for one user."""

    # how often to flush buffered PCM to the socket (seconds of audio)
    FLUSH_S = 0.32

    def __init__(self, user_id: str, owner: StreamingSttAdapter) -> None:
        self.user_id = user_id
        self.owner = owner
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._buf = bytearray()
        self._bytes_sent = 0
        self._t0 = time.monotonic()
        self._seg_start = self._t0
        self.last_activity = self._t0  # updated on every push (idle reaping)
        self._pump: asyncio.Task | None = None
        self._acc = ""  # accumulated increments of the current VAD segment
        self._closed = False

    async def start(self) -> None:
        try:
            self._reader, self._writer = await asyncio.open_connection(
                self.owner.host, self.owner.port
            )
        except OSError as exc:
            logger.warning("streaming stt connect failed: %s", exc)
            self._closed = True
            raise
        # No handshake: the reference client pipes raw PCM straight into
        # netcat (README "arecord ... | nc localhost $PORT"). The server
        # consumes the stream as audio from byte zero.
        self._pump = asyncio.create_task(self._read_loop())

    async def push(self, pcm16: bytes) -> None:
        """Stream raw PCM to the server.

        Server protocol (verified in whisper_server.py): the client streams
        a continuous flow of s16le 16 kHz mono bytes from byte zero — no
        handshake lines. Writes are bounded: if the server has stalled
        (busy with another connection, wedged, or dead), we drop this user
        session rather than block the audio intake path.
        """
        if self._closed or self._writer is None:
            return
        self.last_activity = time.monotonic()
        self._buf.extend(pcm16)
        need = int(self.FLUSH_S * SAMPLING_RATE) * 2
        if len(self._buf) >= need:
            chunk = bytes(self._buf[: len(self._buf) - len(self._buf) % 2])
            self._buf.clear()
            try:
                self._writer.write(chunk)
                await asyncio.wait_for(self._writer.drain(), timeout=2.0)
                self._bytes_sent += len(chunk)
            except (OSError, asyncio.TimeoutError):
                await self._die()

    async def _read_loop(self) -> None:
        """Release disconnected sessions so the next audio chunk reconnects."""
        try:
            await self._read_messages()
        finally:
            if not self._closed:
                await self._die()

    async def _read_messages(self) -> None:
        """Publish incremental text and VAD boundaries from the server."""
        assert self._reader is not None
        while not self._closed:
            try:
                line = await self._reader.readline()
            except (asyncio.IncompleteReadError, ConnectionResetError):
                break
            if not line:
                break
            try:
                msg = json.loads(line.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            text = msg.get("text", "")
            if not text and not msg.get("is_final"):
                continue  # empty VAD-only tick
            # Audio-clock mapping: msg["start"]/["end"] are seconds of
            # stream audio; feeding is paced in real time, so wall(speech
            # end) ~= session t0 + end. This keeps post_speech_ms honest
            # (arrival of the final - actual end of speech), which is the
            # §14 budget number.
            a_end = msg.get("end")
            a_start = msg.get("start")
            if not self._acc and isinstance(a_start, (int, float)):
                self._seg_start = self._t0 + float(a_start)
            t_end = (
                self._t0 + float(a_end)
                if isinstance(a_end, (int, float))
                else time.monotonic()
            )
            # Emissions are append-only confirmed increments (greedy
            # decoding): partial lines extend the hypothesis, the is_final
            # line carries the last remainder of the utterance. The
            # server's increments already carry their own leading space.
            self._acc = (self._acc + text).strip()
            if msg.get("is_final"):
                whole = self._acc
                self._acc = ""
                if whole:
                    await self._fire_final(whole, self._seg_start, t_end)
                self._seg_start = t_end
            else:
                await self._fire_partial(self._acc)

    async def _fire_partial(self, text: str) -> None:
        if self.owner.on_partial:
            try:
                await self.owner.on_partial(self.user_id, text)
            except Exception:
                logger.exception("on_partial failed")

    async def _fire_final(self, text: str, t0: float, t1: float) -> None:
        if self.owner.on_final:
            try:
                await self.owner.on_final(self.user_id, text, t0, t1)
            except Exception:
                logger.exception("on_final failed")

    async def finish(self) -> None:
        """Flush remainder, drain finals briefly, close."""
        if self._closed:
            return
        self._closed = True
        # Cancel the pump BEFORE draining: the server does not flush on
        # disconnect (finish() is commented out in whisper_server.py), so
        # the last `is_final` arrives only while the socket stays open and
        # VAD sees ~0.85 s of silence. Two concurrent readline() callers on
        # one StreamReader would corrupt each other — cancel and await the
        # unwind so the pump is really out of readline() before we drain.
        if self._pump:
            self._pump.cancel()
            try:
                await self._pump
            except asyncio.CancelledError:
                pass
        if self._buf and self._writer is not None:
            try:
                tail = bytes(self._buf)
                self._buf.clear()
                self._writer.write(tail)
                await self._writer.drain()
            except OSError:
                pass
        if self._writer is not None and self._writer.can_write_eof():
            try:
                self._writer.write_eof()
            except OSError:
                pass
        # drain any trailing final response with a bounded timeout
        if self._reader is not None:
            try:
                await asyncio.wait_for(
                    self._drain_tail(), timeout=self.owner.drain_timeout_s
                )
            except asyncio.TimeoutError:
                pass
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass

    async def _drain_tail(self) -> None:
        """Read until a final arrives (empty VAD ticks in between).

        Uses the same increment accumulation as the pump loop, which the
        caller has already cancelled before we run.
        """
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                return
            try:
                msg = json.loads(line.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            text = msg.get("text", "")
            if not text and not msg.get("is_final"):
                continue
            # Same audio-clock mapping as the pump loop so drain-path
            # latency telemetry stays honest (t_end := now would zero it).
            a_end = msg.get("end")
            a_start = msg.get("start")
            if not self._acc and isinstance(a_start, (int, float)):
                self._seg_start = self._t0 + float(a_start)
            t_end = (
                self._t0 + float(a_end)
                if isinstance(a_end, (int, float))
                else time.monotonic()
            )
            self._acc = (self._acc + text).strip()
            if msg.get("is_final"):
                whole = self._acc
                self._acc = ""
                if whole:
                    await self._fire_final(whole, self._seg_start, t_end)
                return

    async def _die(self) -> None:
        """Mark dead and evict so the next ``feed`` for this user reopens."""
        self._closed = True
        if self.owner._sessions.get(self.user_id) is self:
            self.owner._sessions.pop(self.user_id, None)
        if self._writer is not None:
            self._writer.close()
        if self._pump and self._pump is not asyncio.current_task():
            self._pump.cancel()


async def probe_server(host: str = "127.0.0.1", port: int = 43007) -> bool:
    try:
        r, w = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=1.0
        )
        w.close()
        return True
    except (OSError, asyncio.TimeoutError):
        return False
