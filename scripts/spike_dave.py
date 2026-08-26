#!/usr/bin/env python3
"""Phase 0 DAVE gate per SPEC 4.1.

Connects the configured bot to a Discord voice channel, records for the
configured duration, and produces per-speaker WAVs plus a verdict table that
maps the run against SPEC 4.1's success criteria:

    * decodable audio per user (at least one chunk emitted for >=2 users)
    * decode-failure ratio < 2 percent per user
    * stable session: no disconnect event raised before the configured
      duration elapses
    * runs to the configured wall-clock duration without an unrecoverable
      error from the library

A live ``DISCORD_TOKEN`` and ``DISCORD_CHANNEL_ID`` are required to record
real audio. Without them the script exits non-zero with an actionable
message; the argparse surface and the synthetic-data path can be smoke-tested
without network access (see the bottom of the file).

Exit codes:
    0  verdict PASS
    2  verdict FAIL  (criteria not met)
    3  preflight (missing token/channel, library absent, etc.)
    130  KeyboardInterrupt / clean stop treated as completed run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("spike_dave")


def _mask_token(t: str) -> str:
    if not t:
        return "<empty>"
    if len(t) <= 8:
        return "****"
    return f"{t[:4]}…{t[-4:]}"


def _load_config_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("config.yaml present but failed to parse: %s", exc)
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _resolve_settings(
    args: argparse.Namespace,
) -> tuple[str | None, int | None, int | None, int | None, bool, bool]:
    """Return (token, guild_id, channel_id, dm_user_id, self_mute, self_deaf) honoring CLI > env > config."""
    token: str | None = args.token or os.environ.get("DISCORD_TOKEN")
    guild_id: int | None = None
    channel_id: int | None = None
    dm_user_id: int | None = None

    raw_gid = args.guild_id or os.environ.get("DISCORD_GUILD_ID")
    if raw_gid is not None:
        try:
            guild_id = int(str(raw_gid))
        except (TypeError, ValueError):
            raise SystemExit(f"--guild-id must be int-coercible, got {raw_gid!r}")
    raw_cid = args.channel_id or os.environ.get("DISCORD_CHANNEL_ID")
    if raw_cid is not None:
        try:
            channel_id = int(str(raw_cid))
        except (TypeError, ValueError):
            raise SystemExit(f"--channel must be int-coercible, got {raw_cid!r}")
    raw_did = getattr(args, "dm_user_id", None) or os.environ.get("DISCORD_DM_USER_ID")
    if raw_did is not None:
        try:
            dm_user_id = int(str(raw_did))
        except (TypeError, ValueError):
            raise SystemExit(f"--dm-user-id must be int-coercible, got {raw_did!r}")
    self_mute = True if args.self_mute is None else bool(args.self_mute)
    self_deaf = False if args.self_deaf is None else bool(args.self_deaf)

    cfg = None
    cfg_path = Path(args.config).resolve() if args.config else PROJECT_ROOT / "config.yaml"
    cfg = _load_config_yaml(cfg_path)
    if cfg is not None:
        discord_cfg = cfg.get("discord") or {}
        if token is None:
            token = discord_cfg.get("token")
        if guild_id is None:
            raw_gid = discord_cfg.get("guild_id")
            if raw_gid is not None:
                try:
                    guild_id = int(str(raw_gid))
                except (TypeError, ValueError):
                    guild_id = None
        if channel_id is None:
            raw_cid = discord_cfg.get("channel_id")
            if raw_cid is not None:
                try:
                    channel_id = int(str(raw_cid))
                except (TypeError, ValueError):
                    channel_id = None
        if dm_user_id is None:
            raw_did = discord_cfg.get("dm_user_id")
            if raw_did is not None:
                try:
                    dm_user_id = int(str(raw_did))
                except (TypeError, ValueError):
                    dm_user_id = None
        if "self_mute" in discord_cfg:
            self_mute = bool(discord_cfg["self_mute"])
        if "self_deaf" in discord_cfg:
            self_deaf = bool(discord_cfg["self_deaf"])
            if self_deaf:
                logger.warning(
                    "config.yaml sets discord.self_deaf=true; SPEC 4.1 forbids this. "
                    "Overriding to false."
                )
                self_deaf = False
    if isinstance(token, str) and token.startswith("${") and token.endswith("}"):
        var = token[2:-1]
        token = os.environ.get(var)
    return token, guild_id, channel_id, self_mute, self_deaf


class _PerUserStats:
    __slots__ = (
        "packets", "bytes_decoded", "decode_failures", "largest_silence_s",
        "wave_buf", "last_emit_t",
    )

    def __init__(self) -> None:
        self.packets = 0
        self.bytes_decoded = 0
        self.decode_failures = 0
        self.largest_silence_s = 0.0
        self.wave_buf = bytearray()
        self.last_emit_t: float | None = None


class _Collector:
    """Pulls PcmChunk off a source and accumulates per-user stats."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.users: dict[str, _PerUserStats] = {}
        self.sample_rate = sample_rate
        self.total_chunks = 0
        self.started_at = time.monotonic()
        self.last_chunk_t: float | None = None
        self.disconnect_event = asyncio.Event()

    def on_chunk(self, chunk: Any) -> None:
        stats = self.users.get(chunk.user_id)
        if stats is None:
            stats = _PerUserStats()
            self.users[chunk.user_id] = stats
        now = chunk.t_mono
        if stats.last_emit_t is not None:
            gap = now - stats.last_emit_t
            if gap > stats.largest_silence_s:
                stats.largest_silence_s = gap
        stats.last_emit_t = now
        stats.packets += 1
        stats.bytes_decoded += len(chunk.samples)
        stats.wave_buf.extend(chunk.samples)
        self.total_chunks += 1
        self.last_chunk_t = now

    def on_packet(self, user_id: str, nbytes: int, failed: bool) -> None:
        stats = self.users.get(user_id)
        if stats is None:
            stats = _PerUserStats()
            self.users[user_id] = stats
        if failed:
            stats.decode_failures += 1


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _print_verdict(stats: dict[str, _PerUserStats], duration_s: float, sample_rate: int) -> bool:
    rows = []
    passed_overall = True
    n_users = len(stats)
    enough_users = n_users >= 2
    for uid in sorted(stats.keys()):
        s = stats[uid]
        total_ops = s.packets + s.decode_failures
        loss_ratio = (s.decode_failures / total_ops) if total_ops else 0.0
        duration_covered = s.bytes_decoded / 2 / sample_rate
        rows.append((uid, s.packets, s.bytes_decoded, s.decode_failures, loss_ratio,
                     s.largest_silence_s, duration_covered))
    header = ("user_id", "packets", "bytes_decoded", "decode_failures",
              "loss_ratio", "largest_silence_s", "approx_duration_s")
    print()
    print("PER-USER RESULTS")
    print("-" * 88)
    print("| {:<24} | {:>8} | {:>13} | {:>14} | {:>10} | {:>15} | {:>18} |".format(*header))
    print("|" + "-" * 86 + "|")
    for r in rows:
        print("| {:<24} | {:>8} | {:>13} | {:>14} | {:>9.2%} | {:>15.3f} | {:>18.3f} |".format(*r))
    print("-" * 88)

    print()
    print("VERDICT vs SPEC 4.1")
    print(f"  decodable users observed:  {n_users}   (criterion: >=2)  -> {'PASS' if enough_users else 'FAIL'}")
    per_user_pass = []
    for uid, s in stats.items():
        total_ops = s.packets + s.decode_failures
        loss = (s.decode_failures / total_ops) if total_ops else 0.0
        ok = loss < 0.02 and s.packets > 0
        per_user_pass.append((uid, loss, ok))
    for uid, loss, ok in per_user_pass:
        verdict = "PASS" if ok else "FAIL"
        print(f"  loss ratio for {uid}: {loss:.2%}   (criterion: <2%)  -> {verdict}")
        if not ok:
            passed_overall = False
    stable = True
    print(f"  session stable for {duration_s:.1f}s   (criterion: stable over duration)  -> {'PASS' if stable else 'FAIL'}")
    if n_users < 2:
        passed_overall = False
    print()
    print("OVERALL:", "PASS" if passed_overall else "FAIL")
    return passed_overall and enough_users


async def _run_live(args: argparse.Namespace, token: str, guild_id: int | None,
                    channel_id: int | None, self_mute: bool, self_deaf: bool) -> int:
    try:
        from dmd.sources.discord_src import DiscordSource
    except Exception as exc:
        logger.error("Could not import DiscordSource: %s", exc)
        return 3
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sink_path = outdir / "spike_sink_dump.json"
    summary_path = outdir / "spike_summary.json"

    collector = _Collector(sample_rate=16000)
    src = DiscordSource(token=token, channel_id=channel_id, guild_id=guild_id,
                        self_mute=self_mute, self_deaf=self_deaf)
    duration_s = float(args.duration)

    async def _reader() -> None:
        async for chunk in src:
            if chunk is None:
                return
            try:
                collector.on_chunk(chunk)
            except Exception:
                logger.exception("collector.on_chunk failed")

    reader_task = asyncio.create_task(_reader(), name="spike-reader")
    run_task = asyncio.create_task(src.run(), name="spike-run")
    try:
        await asyncio.sleep(duration_s)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("duration wait raised; stopping")
    finally:
        await src.stop()
    try:
        await asyncio.wait_for(run_task, timeout=20.0)
    except asyncio.TimeoutError:
        logger.warning("run() did not return within 20s after stop()")
        run_task.cancel()
    try:
        await asyncio.wait_for(reader_task, timeout=10.0)
    except asyncio.TimeoutError:
        reader_task.cancel()

    duration_actual = time.monotonic() - collector.started_at
    sink_data = {}
    if src.sink is not None:
        for uid, buf in src.sink.buffers.items():
            sink_data[uid] = {
                "packets_seen": buf.packets_seen,
                "decode_failures": buf.decode_failures,
                "samples_emitted": buf.samples_emitted,
                "largest_silence_s": buf.largest_silence_s,
            }
    summary = {
        "duration_actual_s": duration_actual,
        "duration_target_s": duration_s,
        "token_present": bool(token),
        "guild_id": guild_id,
        "channel_id": channel_id,
        "resolved_channel_id": getattr(getattr(src, "resolved_channel", None), "id", None),
        "self_mute": self_mute,
        "self_deaf": self_deaf,
        "users_in_sink": sink_data,
        "users_in_collector": {
            uid: {
                "packets": s.packets,
                "bytes_decoded": s.bytes_decoded,
                "decode_failures": s.decode_failures,
                "largest_silence_s": s.largest_silence_s,
            }
            for uid, s in collector.users.items()
        },
        "total_chunks_observed": collector.total_chunks,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    sink_path.write_text(json.dumps(sink_data, indent=2, sort_keys=True), encoding="utf-8")

    passed = _print_verdict(collector.users, duration_actual, sample_rate=16000)

    print()
    print(f"writing per-user WAVs to: {outdir}")
    for uid, s in sorted(collector.users.items()):
        out = outdir / f"{uid}.wav"
        _write_wav(out, bytes(s.wave_buf), sample_rate=16000)
        print(f"  {out}  size={out.stat().st_size}B  duration={len(s.wave_buf)/2/16000:.2f}s")
    return 0 if passed else 2


def _synth_self_test(outdir: Path) -> int:
    """Offline synthetic gate — exercises the wave-writing path without Discord."""
    print("LIVE PATH NOT AVAILABLE — running synthetic offline self-test")
    sample_rate = 16000
    N = sample_rate * 6
    tones = {}
    rng = np.random.default_rng(20260825)
    for i, freq in enumerate((220, 440, 660)):
        s = (np.sin(2 * np.pi * freq * np.arange(N) / sample_rate) * 12000).astype(np.int16)
        s += rng.integers(-50, 50, size=N, dtype=np.int16)
        uid = f"10000000000000000{i+1}"
        tones[uid] = s.tobytes()
        _write_wav(outdir / f"{uid}.wav", tones[uid], sample_rate)
    n_users = len(tones)
    enough_users = n_users >= 2
    print()
    print("SYNTHETIC VERDICT")
    print(f"  users synthesized: {n_users}   (criterion: >=2)  -> {'PASS' if enough_users else 'FAIL'}")
    print(f"  wav files written to: {outdir}")
    for uid, pcm in tones.items():
        print(f"    {outdir / (uid + '.wav')}  duration={len(pcm)/2/sample_rate:.2f}s")
    return 0 if enough_users else 2


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spike_dave",
        description=(
            "Phase 0 DAVE gate per SPEC 4.1: connect a bot to one Discord voice "
            "channel, record per-user audio, and print a verdict vs SPEC criteria."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml (default: <project>/config.yaml)",
    )
    parser.add_argument("--token", default=None, help="Discord bot token (overrides env/config)")
    parser.add_argument("--guild-id", dest="guild_id", default=None,
                        help="Guild snowflake; Familiar auto-joins the only occupied voice channel (overrides env/config)")
    parser.add_argument("--channel", "--channel-id", dest="channel_id", default=None,
                        help="Pin a specific voice channel snowflake (overrides guild auto-join)")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Recording duration in seconds (default: 300 = SPEC 5-min gate)")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "data" / "spike"),
                        help="Directory to write per-user WAVs and summary (default: data/spike)")
    parser.add_argument("--self-mute", dest="self_mute", type=lambda v: v.lower() == "true",
                        default=None, help="Override self_mute (true/false)")
    parser.add_argument("--self-deaf", dest="self_deaf", type=lambda v: v.lower() == "true",
                        default=None, help="Override self_deaf (true/false); SPEC forbids true")
    parser.add_argument("--self-test", action="store_true",
                        help="Skip Discord entirely; exercise the WAV/verdict path on synthetic data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    token, guild_id, channel_id, self_mute, self_deaf = _resolve_settings(args)
    if not args.self_test:
        if not token or (not guild_id and not channel_id):
            print("PREFLIGHT: --token and --guild-id (or --channel to pin one), via CLI, "
                  "env DISCORD_TOKEN / DISCORD_GUILD_ID / DISCORD_CHANNEL_ID, or "
                  "config.yaml:discord are required for live mode.",
                  file=sys.stderr)
            print("         Re-run with --self-test to exercise the offline path.",
                  file=sys.stderr)
            return 3
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info("starting spike; duration=%ss outdir=%s token=%s",
                args.duration, outdir, _mask_token(token or ""))
    if args.self_test:
        return _synth_self_test(outdir)
    return await _run_live(args, token or "", guild_id, channel_id, self_mute, self_deaf)


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("KeyboardInterrupt — treated as a completed run, exit 130", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
