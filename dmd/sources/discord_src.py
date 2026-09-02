"""Discord voice audio source (Phase 0 spike plumbing).

Connects to one voice channel, receives per-user RTP streams via py-cord's
sink-based recording interface, and emits decoded mono int16 PCM at 16 kHz
downstream as ``PcmChunk`` objects on the inherited queue.

Library status (as of py-cord 2.8.1, verified by introspection):
    - ``voice.start_recording`` in py-cord 2.7+ instantiates an
      ``AudioReader`` which is the documented streaming sink.
    - The reader calls ``sink.write(data, user)`` where ``data`` is a
      ``VoiceData`` (``packet``, ``source``, ``pcm``) carrying one decoded
      20 ms frame at 48 kHz stereo int16 inside ``data.pcm``.
    - py-cord prints a runtime warning that voice reception is broken under
      Discord's DAVE (E2EE) protocol and points to issue #3139. The sink
      path here still works for cleartext sessions and is what the Phase 0
      spike will use to gate the DAVE decision (see SPEC 4.1).
    - ``discord.voice`` import is gated by ``PyNaCl`` and ``davey`` (issue
      ``MissingVoiceDependenciesError``); that gate is detected and re-raised
      as an actionable RuntimeError at run-time.
    - The optional ``is_opus()`` attribute py-cord's ``PacketDecoder`` calls
      on its sink is filled in here (``return False``) so we get decoded PCM.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Any

import numpy as np

from dmd.sources.base import AudioSource
from dmd.types import PcmChunk

if TYPE_CHECKING:
    import discord

logger = logging.getLogger(__name__)


try:
    import discord

    HAVE_DISCORD = True
    DISCORD_VERSION = getattr(discord, "__version__", "unknown")
except Exception as _exc:  # pragma: no cover - only fires when py-cord absent
    HAVE_DISCORD = False
    DISCORD_VERSION = "not-installed"
    discord = None  # type: ignore[assignment]
    _DISCORD_IMPORT_ERROR: Exception | None = _exc
else:
    _DISCORD_IMPORT_ERROR = None

try:
    import discord.sinks.errors as _sink_errors  # noqa: F401  (re-exported below)
    from discord.sinks import Sink as _SinkBase
    from discord.sinks.ogg import OGGSink

    HAVE_DISCORD_SINKS = True
except Exception as _exc:  # pragma: no cover
    HAVE_DISCORD_SINKS = False
    _SinkBase = None  # type: ignore[assignment]
    OGGSink = None  # type: ignore[assignment]
    _DISCORD_SINKS_ERROR: Exception | None = _exc
else:
    _DISCORD_SINKS_ERROR = None


def _voice_dep_status() -> tuple[bool, list[str]]:
    """Probe whether py-cord's voice module is importable.

    Returns ``(ok, missing)``. ``missing`` lists the native packages py-cord
    reports as uninstalled. ``ok`` is True only if voice is fully usable.
    """
    if not HAVE_DISCORD:
        return False, ["discord (py-cord)"]
    try:
        import discord.voice as _voice  # noqa: F401
    except Exception as exc:
        name = type(exc).__name__
        msg = str(exc)
        if "PyNaCl" in msg or "pynacl" in msg.lower():
            return False, ["PyNaCl"]
        if "davey" in msg.lower():
            return False, ["davey"]
        return False, [f"discord.voice ({name}: {msg})"]
    return True, []


def _require_voice_dependencies() -> None:
    """Raise RuntimeError if voice is unusable. The message names the dep."""
    ok, missing = _voice_dep_status()
    if not ok:
        joined = ", ".join(missing)
        raise RuntimeError(
            "DiscordSource needs the py-cord voice stack but the following "
            f"are unavailable: {joined}. "
            "Install with: pip install 'py-cord[voice]' "
            "(requires PyNaCl + davey). "
            "Voice reception in py-cord 2.8 is also flagged broken under "
            "DAVE E2EE (see Pycord-Development/pycord#3139) — the Phase 0 "
            "spike results feed SPEC 4.1's library fallback ladder."
        )


def downmix_resample(
    samples: np.ndarray,
    src_rate: int = 48000,
    dst_rate: int = 16000,
) -> bytes:
    """Stereo int16 -> mono int16 LE PCM at ``dst_rate``.

    ``samples`` is accepted as either ``(N,)`` mono or ``(N, 2)`` stereo at
    ``src_rate`` Hz. Channel mean for stereo uses int32 promotion to avoid
    int16 overflow. Resampling is linear interpolation. Result is int16
    little-endian (``.tobytes()``) suitable for ``soundfile``/``wave``.
    """
    if not isinstance(samples, np.ndarray):
        raise TypeError(f"samples must be a numpy.ndarray, got {type(samples).__name__}")
    if samples.dtype != np.int16:
        samples = samples.astype(np.int16, copy=False)
    if samples.ndim == 1:
        mono = samples
    elif samples.ndim == 2:
        if samples.shape[1] != 2:
            raise ValueError(
                f"only stereo (N, 2) or mono (N,) supported; got shape {samples.shape!r}"
            )
        promoted = samples.astype(np.int32, copy=False)
        mono = (promoted.sum(axis=1) // samples.shape[1]).astype(np.int16)
    else:
        raise ValueError(f"unsamples.ndim={samples.ndim}; expected 1-D or 2-D")
    if mono.size == 0:
        return b""
    if src_rate == dst_rate:
        return mono.tobytes()
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"sample rates must be positive (got {src_rate}, {dst_rate})")
    n_src = mono.shape[0]
    n_dst = round(n_src * dst_rate / src_rate)
    if n_dst <= 0:
        return b""
    x_dst = np.linspace(0.0, float(n_src - 1), num=n_dst, endpoint=True)
    base_idx = np.floor(x_dst).astype(np.int64)
    frac = (x_dst - base_idx.astype(np.float64)).astype(np.float32)
    next_idx = np.minimum(base_idx + 1, n_src - 1)
    y0 = mono[base_idx].astype(np.int32)
    y1 = mono[next_idx].astype(np.int32)
    interpolated = y0 + ((y1 - y0).astype(np.float32) * frac).astype(np.int32)
    clipped = np.clip(interpolated, -32768, 32767).astype(np.int16)
    return clipped.tobytes()


@dataclass(slots=True)
class _PerUserBuffer:
    """Per-speaker PCM accumulator used by ``StreamSink``."""

    pcm_buf: bytearray = field(default_factory=bytearray)
    samples_emitted: int = 0
    packets_seen: int = 0
    decode_failures: int = 0
    last_packet_t: float | None = None
    largest_silence_s: float = 0.0


class _StreamSink(_SinkBase if _SinkBase is not None else object):  # type: ignore[misc,valid-type]
    """Streaming py-cord sink."""

    __sink_listeners__: ClassVar[list] = []  # type: ignore[assignment]

    def walk_children(self, with_self: bool = False):  # type: ignore[override]
        if with_self:
            yield self
        return
        yield  # make generator

    @property
    def root(self):  # type: ignore[override]
        return self
    """Streaming py-cord sink.

    Subclasses ``discord.sinks.Sink`` so it slots into ``voice.start_recording``.
    On each inbound packet it decodes (already-decoded) stereo int16 PCM at
    48 kHz, accumulates per-user, and on the configured chunk boundary emits a
    mono 16 kHz ``PcmChunk`` into the source's ``emit`` queue.

    The base ``cleanup()`` would normally fire on stop and write per-user files;
    we override it to drain remaining buffers as ``PcmChunk``s and skip file
    writes.
    """

    CHUNK_FRAMES = 5

    def __init__(
        self,
        *,
        emit: Callable[[PcmChunk], None],
        on_packet: Callable[[str, int, bool], None] | None = None,
    ) -> None:
        if not HAVE_DISCORD_SINKS or _SinkBase is None:
            raise RuntimeError(
                "discord.sinks unavailable; install py-cord and ensure "
                "discord.sinks.Sink imports cleanly. Underlying error: "
                f"{_DISCORD_SINKS_ERROR!r}"
            )
        _SinkBase.__init__(self)
        self._emit = emit
        self._on_packet = on_packet
        self._buffers: dict[int, _PerUserBuffer] = {}
        self._bytes_per_frame = 960 * 2 * 2
        self.finished = False

    @property
    def buffers(self) -> dict[int, _PerUserBuffer]:
        return self._buffers

    def is_opus(self) -> bool:
        return False

    def _user_key(self, user: Any) -> int:
        user_id = getattr(user, "id", None)
        if user_id is None:
            return 0
        return int(user_id)

    def _emit_chunk(self, user_key: int, user_obj: Any, buf: _PerUserBuffer) -> None:
        raw = bytes(buf.pcm_buf)
        buf.pcm_buf = bytearray()
        if not raw:
            return
        if len(raw) % 4 != 0:
            pad = 4 - (len(raw) % 4)
            raw = raw + b"\x00" * pad
        np_samples = np.frombuffer(raw, dtype=np.int16)
        if np_samples.size < 2:
            return
        if np_samples.size % 2 == 1:
            np_samples = np_samples[:-1]
        stereo = np_samples.reshape(-1, 2)
        pcm16 = downmix_resample(stereo, src_rate=48000, dst_rate=16000)
        if not pcm16:
            return
        buf.samples_emitted += len(pcm16) // 2
        self._emit(
            PcmChunk(
                user_id=str(user_key),
                samples=pcm16,
                sample_rate=16000,
            )
        )

    def write(self, data: Any, user: Any) -> None:
        if self.finished:
            return
        user_key = self._user_key(user)
        buf = self._buffers.get(user_key)
        if buf is None:
            buf = _PerUserBuffer()
            self._buffers[user_key] = buf
        pcm = getattr(data, "pcm", None)
        if not pcm:
            buf.decode_failures += 1
            if self._on_packet is not None:
                try:
                    self._on_packet(str(user_key), 0, True)
                except Exception:
                    logger.exception("on_packet callback failed")
            return
        now = time.monotonic()
        if buf.last_packet_t is not None:
            gap = now - buf.last_packet_t
            buf.largest_silence_s = max(buf.largest_silence_s, gap)
        buf.last_packet_t = now
        buf.packets_seen += 1
        buf.pcm_buf.extend(pcm)
        if self._on_packet is not None:
            try:
                self._on_packet(str(user_key), len(pcm), False)
            except Exception:
                logger.exception("on_packet callback failed")
        if len(buf.pcm_buf) >= self.CHUNK_FRAMES * self._bytes_per_frame:
            self._emit_chunk(user_key, user, buf)

    def cleanup(self) -> None:
        for user_key, buf in list(self._buffers.items()):
            self._emit_chunk(user_key, None, buf)
        self.finished = True


class _SpikeVoiceClient:
    """VoiceClient subclass that passes ``self_deaf`` / ``self_mute`` through.

    Constructed with the desired flags stored on instance attrs; ``connect``
    forwards them to the underlying ``VoiceClient.connect``. We import the
    concrete client class lazily inside ``__init__`` because importing
    ``discord.voice`` at module top would prevent the package from importing on
    machines where PyNaCl / davey are absent.
    """

    def __init__(self, bot_client: Any, channel: Any, *, self_deaf: bool = False, self_mute: bool = True) -> None:
        try:
            from discord.voice import VoiceClient as _VC
        except Exception as exc:
            raise RuntimeError(
                "Could not import discord.voice.VoiceClient to build the "
                "self_deaf/self_mute forwarding subclass. Underlying error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.__class__ = _VC  # type: ignore[assignment]
        _VC.__init__(self, bot_client, channel)
        self._spike_self_deaf = self_deaf
        self._spike_self_mute = self_mute

    async def connect(self, *, reconnect: bool = True, timeout: float = 30.0) -> None:  # type: ignore[override]
        from discord.voice.client import VoiceClient as _PlainVC

        sig = inspect.signature(_PlainVC.connect)
        params = sig.parameters
        if "self_deaf" not in params or "self_mute" not in params:
            raise RuntimeError(
                f"discord.voice.VoiceClient.connect does not accept "
                f"self_deaf/self_mute kwargs in this build. Detected parameters: "
                f"{list(params)!r}"
            )
        await _PlainVC.connect(
            self,
            reconnect=reconnect,
            timeout=timeout,
            self_deaf=self._spike_self_deaf,
            self_mute=self._spike_self_mute,
        )


def _tap_intents() -> Any:
    """Intents for the voice tap: guilds + voice_states (both non-privileged).

    voice_states is required to see who is sitting in which voice channel,
    which powers guild-scoped auto-join. No privileged intents are needed.
    """
    intents = discord.Intents.none()
    intents.guilds = True
    intents.voice_states = True
    return intents


def _channel_factory(intents: Any) -> Any:
    """Build a ``discord.Client`` with empty intents. Voice reception works

    on the media socket independently of gateway intents.
    """
    if not HAVE_DISCORD:
        raise RuntimeError(
            "discord (py-cord) is not installed; cannot construct a Client. "
            "Run: pip install py-cord"
        )
    return discord.Client(intents=intents)


class DiscordSource(AudioSource):
    """AudioSource that taps one Discord voice channel via py-cord.

    Parameters
    ----------
    token:
        Bot token. Not logged, not echoed; pass via config or env.
    channel_id:
        Snowflake of the voice channel to join.
    self_mute:
        Whether to join muted. Default ``True`` (don't send back to the channel).
    self_deaf:
        Whether to join deafened. Default ``False`` (must be False to RECEIVE).
        Passing ``True`` raises at connect time (the bot can never receive
        while self-deafened; SPEC 4.1 mandates ``selfDeaf: false``).

    Notes
    -----
    The source emits ``PcmChunk(user_id="<snowflake>", samples=<int16 LE mono>,
    sample_rate=16000)`` for every ~100 ms of audio received per speaker.
    Per-user stats (packet count, decode failure count, largest silence gap)
    are tracked on the internal sink and are NOT exposed on the public
    surface — consumers needing them in a Phase-0 spike context can read
    them via the (private) ``_sink.buffers`` mapping, while production
    consumers will subscribe only to the chunk stream.

    """

    def __init__(
        self,
        token: str,
        channel_id: int | str | None = None,
        self_mute: bool = True,
        self_deaf: bool = False,
        guild_id: int | str | None = None,
        dm_user_id: int | str | None = None,
    ) -> None:
        super().__init__()
        if not token:
            raise ValueError("DiscordSource: token must be a non-empty string")
        if channel_id is None and guild_id is None:
            raise ValueError(
                "DiscordSource: provide channel_id or guild_id — with guild_id "
                "the source auto-joins the only occupied voice channel, or follows "
                "the DM when dm_user_id is set."
            )
        if channel_id is not None and not isinstance(channel_id, int):
            try:
                channel_id = int(str(channel_id))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"channel_id must be int-coercible, got {channel_id!r}") from exc
        if guild_id is not None and not isinstance(guild_id, int):
            try:
                guild_id = int(str(guild_id))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"guild_id must be int-coercible, got {guild_id!r}") from exc
        if dm_user_id is not None and not isinstance(dm_user_id, int):
            try:
                dm_user_id = int(str(dm_user_id))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"dm_user_id must be int-coercible, got {dm_user_id!r}") from exc
        if self_deaf:
            raise ValueError(
                "DiscordSource: self_deaf=True is forbidden — a deafened bot "
                "cannot receive audio. SPEC 4.1 requires selfDeaf=false."
            )
        self._token = token
        self._channel_id = channel_id
        self._guild_id = guild_id
        self._dm_user_id = dm_user_id
        self._self_mute = bool(self_mute)
        self._self_deaf = bool(self_deaf)
        self._stop_event = asyncio.Event()
        self._sink: _StreamSink | None = None
        self._stats_on_packet: list[tuple[str, int, bool]] = []
        self.resolved_channel: Any | None = None

    @property
    def sink(self) -> _StreamSink | None:
        """Internal sink exposing per-user stats for the spike harness only."""
        return self._sink

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        self._queue.put_nowait(None)

    def _resolve_voice_channel(self, client: Any) -> Any:
        """Pick the target voice channel.

        channel_id set  -> that exact channel (get then fetch).
        else guild_id   -> the guild's voice channels; exactly one occupied
                           (non-bot member) auto-joins; zero or many raise
                           with an explicit listing so the caller can pin one.
        """
        if self._channel_id is not None:
            channel = client.get_channel(self._channel_id)
            if channel is None:
                raise RuntimeError(
                    f"Channel {self._channel_id} not visible to the bot. "
                    "Confirm it is invited to the guild and has View + Connect."
                )
            self.resolved_channel = channel
            return channel

        guild = client.get_guild(self._guild_id)
        if guild is None:
            visible = ", ".join(f"{g.name}({g.id})" for g in client.guilds) or "(none)"
            raise RuntimeError(
                f"Guild {self._guild_id} not visible to the bot. Visible guilds: {visible}. "
                "Confirm the invite and View Channels permission."
            )
        voice_channels = list(getattr(guild, "voice_channels", []) or [])
        if not voice_channels:
            raise RuntimeError(f"Guild {guild.name} ({guild.id}) has no voice channels.")

        me = getattr(guild, "me", None)

        def _connectable(vc: Any) -> bool:
            perms = vc.permissions_for(me) if me is not None else None
            if perms is None:
                return True
            return bool(getattr(perms, "connect", False))

        # DM-follow: if dm_user_id is set, join their current voice channel
        if self._dm_user_id is not None:
            vs = getattr(guild, "_voice_states", {}) or {}
            dm_vs = vs.get(int(self._dm_user_id))
            dm_channel = getattr(dm_vs, "channel", None) if dm_vs is not None else None
            if dm_channel is not None:
                if _connectable(dm_channel):
                    logger.info(
                        "following DM %s into voice channel %s (%s)",
                        self._dm_user_id,
                        dm_channel.name,
                        dm_channel.id,
                    )
                    self.resolved_channel = dm_channel
                    return dm_channel
                raise RuntimeError(
                    f"DM is in {dm_channel.name} ({dm_channel.id}) but Familiar lacks Connect there. "
                    "Grant View + Connect on that channel."
                )
            logger.info(
                "DM %s not in voice (no voice_state); falling through to auto-join",
                self._dm_user_id,
            )
            raise RuntimeError(
                f"DM {self._dm_user_id} is not in voice — Familiar is idle, waiting for them to join. "
                "Have the DM enter a voice channel, or pin one with --channel."
            )

        joinable = [vc for vc in voice_channels if _connectable(vc)]
        blocked = [vc for vc in voice_channels if not _connectable(vc)]
        occupied = [
            vc
            for vc in joinable
            if any(not getattr(m, "bot", False) for m in getattr(vc, "members", []) or [])
        ]
        blocked_occupied = [
            vc
            for vc in blocked
            if any(not getattr(m, "bot", False) for m in getattr(vc, "members", []) or [])
        ]
        if len(occupied) == 1:
            logger.info(
                "auto-joined occupied voice channel %s (%s) in guild %s",
                occupied[0].name,
                occupied[0].id,
                guild.name,
            )
            self.resolved_channel = occupied[0]
            return occupied[0]
        listing = "; ".join(
            f"{vc.name}({vc.id}, {len(getattr(vc, 'members', []) or [])} connected)"
            for vc in (occupied or joinable)
        )
        if not occupied:
            hint = ""
            if blocked_occupied:
                denied = ", ".join(vc.name for vc in blocked_occupied)
                hint = (
                    f" Note: occupied but Connect-denied for me: {denied} — grant "
                    "Connect there or join an allowed channel."
                )
            raise RuntimeError(
                f"No joinable occupied voice channel in {guild.name}. "
                f"Joinable: {listing or '(none)'}.{hint} "
                "Join a channel first, or pin one with --channel."
            )
        raise RuntimeError(
            f"Multiple occupied joinable voice channels in {guild.name}: {listing}. "
            "Pin one with --channel <id>."
        )

    async def run(self) -> None:
        """Connect, start recording, drain until stopped.

        Drives the source until ``stop()`` is awaited, then performs a clean
        disconnect. Raises ``RuntimeError`` with a precise message when the
        platform stack is missing or py-cord's voice/receive surface
        disagrees with what this module assumes.
        """
        if not HAVE_DISCORD:
            raise RuntimeError(
                f"discord (py-cord) is not importable: {_DISCORD_IMPORT_ERROR!r}. "
                "Run: pip install py-cord"
            )
        if not HAVE_DISCORD_SINKS:
            raise RuntimeError(
                f"discord.sinks not importable: {_DISCORD_SINKS_ERROR!r}"
            )
        _require_voice_dependencies()
        try:
            import ctypes.util as _ctypes_util

            import discord.opus as _opus

            if not _opus.is_loaded():
                candidate = _ctypes_util.find_library("opus") or "libopus.so.0"
                try:
                    _opus.load_opus(candidate)
                except Exception as exc:
                    logger.warning("discord.opus.load_opus(%r) failed: %r", candidate, exc)
            if not _opus.is_loaded():
                raise RuntimeError(
                    "discord.opus native library could not be loaded — incoming "
                    "Opus frames cannot be decoded to PCM. Install libopus "
                    "(e.g. apt install libopus0) and retry."
                )
            logger.info("discord.opus loaded: %s", getattr(_opus, "_lib", None) is not None)
        except Exception as exc:
            raise RuntimeError(
                f"discord.opus could not be inspected: {type(exc).__name__}: {exc}"
            ) from exc

        client = _channel_factory(_tap_intents())
        ready_event = asyncio.Event()
        runtime_error: list[BaseException] = []

        @client.event
        async def on_ready() -> None:
            logger.info("discord client on_ready: user=%s", client.user)
            ready_event.set()

        @client.event
        async def on_resumed() -> None:
            logger.info("discord gateway resumed")

        @client.event
        async def on_disconnect() -> None:
            logger.warning("discord gateway disconnected")

        @client.event
        async def on_error(event: str, *args: Any, **kwargs: Any) -> None:
            logger.exception("discord client error in event %s", event)

        async def on_voice_state_update(member: Any, before: Any, after: Any) -> None:
            if getattr(member, "guild", None) is None:
                return
            if getattr(member, "id", None) == getattr(client.user, "id", None):
                logger.info(
                    "self voice state change: deaf=%s mute=%s channel=%s",
                    getattr(after, "self_deaf", None),
                    getattr(after, "self_mute", None),
                    getattr(after, "channel", None),
                )

        client.event(on_voice_state_update)

        async def _on_done(exc: BaseException | None) -> None:
            if exc is not None:
                logger.warning("sink finished_callback received exception: %r", exc)
            else:
                logger.info("sink finished_callback returned cleanly")

        client_task: asyncio.Task[None] | None = None
        voice: Any = None
        try:
            client_task = asyncio.create_task(
                client.start(self._token, reconnect=True),
                name="discord-src-client-start",
            )
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=60.0)
            except asyncio.TimeoutError as exc:
                client_task.cancel()
                raise RuntimeError(
                    "discord client did not become ready within 60s; bot "
                    "token invalid or gateway unreachable"
                ) from exc
            logger.info("discord.__version__=%s; voice deps ok", DISCORD_VERSION)

            channel = self._resolve_voice_channel(client)

            from discord.voice import VoiceClient as _VC

            voice = _VC(client, channel)
            key_id, _ = channel._get_voice_client_key()
            state = channel._state
            if state._get_voice_client(key_id):
                from discord.errors import ClientException

                raise ClientException("Already connected to a voice channel.")
            state._add_voice_client(key_id, voice)
            await voice.connect(
                reconnect=True,
                timeout=60.0,
                self_deaf=self._self_deaf,
                self_mute=self._self_mute,
            )

            try:
                actual_self_deaf = bool(getattr(voice, "self_deaf", None))
                actual_self_mute = bool(getattr(voice, "self_mute", None))
                logger.info(
                    "voice connected: self_deaf=%s self_mute=%s (requested deaf=%s mute=%s)",
                    actual_self_deaf,
                    actual_self_mute,
                    self._self_deaf,
                    self._self_mute,
                )
                if actual_self_deaf:
                    raise RuntimeError(
                        "voice.self_deaf resolved to True after connect; bot "
                        "will not receive audio. Disconnecting."
                    )
            except AttributeError:
                logger.info("voice client does not expose self_deaf attribute; skipping check")

            def _on_packet(user_id: str, nbytes: int, failed: bool) -> None:
                self._stats_on_packet.append((user_id, nbytes, failed))

            sink = _StreamSink(emit=self.emit, on_packet=_on_packet)
            self._sink = sink
            await asyncio.sleep(2.0)
            logger.info(
                "pre-recording check: is_connected=%s secret_key_present=%s",
                voice.is_connected() if hasattr(voice, "is_connected") else "unknown",
                bool(getattr(voice, "secret_key", None)),
            )
            try:
                voice.start_recording(sink, _on_done)
                logger.info("start_recording returned; is_recording=%s", voice.is_recording() if hasattr(voice, "is_recording") else "unknown")
            except TypeError as exc:
                raise RuntimeError(
                    f"voice.start_recording signature mismatch in this py-cord "
                    f"build ({DISCORD_VERSION}): {exc}. Expected "
                    f"start_recording(sink, callback)."
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"voice.start_recording failed: {type(exc).__name__}: {exc}"
                ) from exc

            self._running = True
            logger.info(
                "DiscordSource connected and recording; awaiting stop() "
                "(duration and stop controls are managed by the caller)"
            )
            try:
                await self._stop_event.wait()
            finally:
                logger.info("DiscordSource stop requested")
        finally:
            try:
                if voice is not None:
                    try:
                        voice.stop_recording()
                    except Exception as exc:
                        logger.warning("voice.stop_recording failed: %r", exc)
                    try:
                        await voice.disconnect(force=True)
                    except Exception as exc:
                        logger.warning("voice.disconnect failed: %r", exc)
            finally:
                try:
                    if self._sink is not None:
                        try:
                            self._sink.cleanup()
                        except Exception as exc:
                            logger.warning("sink.cleanup raised: %r", exc)
                finally:
                    try:
                        if not client.is_closed():
                            await client.close()
                    except Exception as exc:
                        logger.warning("client.close failed: %r", exc)
                    if client_task is not None:
                        if client_task.done():
                            with_suppress = client_task.exception()
                            if with_suppress is not None:
                                logger.debug("client_task ended: %r", with_suppress)
                        else:
                            try:
                                await asyncio.wait_for(
                                    asyncio.shield(client_task), timeout=5.0
                                )
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                client_task.cancel()
                                try:
                                    await asyncio.wait_for(
                                        asyncio.shield(client_task), timeout=5.0
                                    )
                                except Exception:
                                    pass
                    if runtime_error:
                        err = runtime_error[0]
                        logger.error("client_task ended with exception: %r", err)
            self.emit(None)
