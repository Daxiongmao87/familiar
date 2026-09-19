#!/usr/bin/env python3
"""Voice->transcript->answer latency probe for the live dmd service.

Priority-1 owner requirement (2026-09-05): measure, with logged evidence,
the two end-to-end budgets —

  A) speech-stop -> transcript visible on the event bus   (target <= 5 s)
  B) transcript committed -> answer (card / scene note)   (target <= 15 s)

The probe drives the REAL intake path: synthesized (or supplied) speech is
streamed as 100 ms PCM chunks over `wss://host/ws/audio` into the running
service's BrowserAudioSource, exactly like the browser capture UI does, and
every relevant event (`transcript`, `stt_latency`, `turn_latency`, `card`,
`scene_context`) is timestamped as it arrives on `wss://host/ws`.

Reported stages per utterance (see the brief's chain):

  vad_endpoint      speech-stop anchor + server silence hangover
                    (stt_pipeline.silence_ms read live from /api/config)
  audio_ready       last speech chunk handed to the WebSocket
  whisperx_*        direct POST to the configured STT endpoint (queue/decode
                    split from the server's own `timings` field) — component
                    measurement, diarize on and off
  utterance_committed   `transcript` event arrival (voice->transcript A)
  agent_context_ready   `turn_latency` event (fast-lane trigger classified,
                    job queued; carries detect_ms / lane_ms)
  first_token       n/a — the gateway is non-streaming (one POST per model
                    call), so no first-token exists on the wire; stated, not
                    faked
  full_answer       first `card` / `scene_context` event attributable to the
                    utterance (transcript->answer B)

Usage:
  python tools/latency_probe.py [--base https://127.0.0.1:8760] [--out PATH]
                                [--reps N] [--stt-only] [--e2e-only]
  # custom audio instead of espeak (16 kHz mono int16 wav preferred):
  python tools/latency_probe.py --wav short=/path/a.wav --expect-text short=loot

Honest measurement, no claims: every number printed and written to the JSON
artifact is a logged wall-clock delta from this run. The correlation between
an answer event and its utterance is time-window based (first answer event
after that utterance's commit); the 30 s proactive monitor can in principle
emit an unrelated scene note inside the window — flagged in the JSON as
`correlation: temporal-first`.

Requires: `websockets`, `httpx` (both already project deps), `numpy`, and
`espeak` on PATH for synthesis. Output artifact defaults to
`data/latency/probe-<utc>.json` (gitignored).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import ssl
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

try:
    import websockets
except ImportError:  # pragma: no cover - hard requirement of this tool
    print("websockets is required: pip install websockets", file=sys.stderr)
    raise

SR = 16000
CHUNK_S = 0.100
LEAD_SILENCE_S = 0.6
TAIL_SILENCE_S = 2.5
ANSWER_TIMEOUT_S = 40.0
TRANSCRIBE_TIMEOUT_S = 30.0

# Three utterance lengths (~3 s / ~8 s / ~15 s of speech) built from the
# sample campaign's real entities so the fast-lane classifier and the agent
# have true triggers (loot) and true retrieval targets.
UTTERANCES: list[dict[str, str]] = [
    {
        "name": "short",
        "text": "We loot Brother Alric's body.",
        "expect": "loot alric body",
    },
    {
        "name": "medium",
        "text": (
            "Kael searches the sunken chapel altar for anything the "
            "bell keeper left behind."
        ),
        "expect": "chapel altar searches kael",
    },
    {
        "name": "long",
        "text": (
            "Before we touch the altar, Mira wants to know what the "
            "confession pouch is worth and whether the resonance stone is "
            "still under the flagstone, so we search the body and every "
            "loose stone in the chapel carefully."
        ),
        "expect": "confession pouch resonance stone searches body",
    },
]


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower())}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


# --------------------------------------------------------------------------
# audio synthesis / loading
# --------------------------------------------------------------------------


def _resample_to16k_mono_int16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n_ch = w.getnchannels()
        sw = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise SystemExit(f"{path}: only 16-bit wav supported (got width {sw})")
    x = np.frombuffer(raw, dtype="<i2")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    x = x.astype(np.float64)
    if rate != SR:
        tgt = max(1, int(round(len(x) * SR / rate)))
        x = np.interp(np.linspace(0, len(x) - 1, tgt), np.arange(len(x)), x)
    return np.clip(np.round(x), -32768, 32767).astype("<i2")


def synth_espeak(text: str) -> np.ndarray:
    """Render `text` to int16 16 kHz mono via espeak (robotic but whisper-legible)."""
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "t.wav")
        subprocess.run(
            ["espeak", "-s", "150", "-p", "55", "-a", "120", "-w", p, text],
            check=True,
            capture_output=True,
        )
        return _resample_to16k_mono_int16(p)


def load_wav(path: str) -> np.ndarray:
    return _resample_to16k_mono_int16(path)


def wav_bytes(pcm: np.ndarray) -> bytes:
    """Wrap PCM in a minimal WAV container (what the gateway sends whisperx)."""
    data = pcm.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1, SR, SR * 2, 2, 16,
        b"data", len(data),
    )
    return header + data


# --------------------------------------------------------------------------
# stage records
# --------------------------------------------------------------------------


@dataclass
class UtteranceResult:
    name: str
    target_text: str
    speech_s: float = 0.0
    t_audio_last_sent: float = 0.0  # epoch when last speech chunk went out
    t_transcript: float = 0.0
    transcript_text: str = ""
    t_turn_latency: float = 0.0
    turn_latency: dict[str, Any] = field(default_factory=dict)
    t_answer: float = 0.0
    answer_kind: str = ""  # "card" | "scene_context"
    answer_title: str = ""
    stt_latency: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def voice_to_transcript_s(self) -> float:
        return self.t_transcript - self.t_audio_last_sent if self.t_transcript else -1.0

    @property
    def transcript_to_answer_s(self) -> float:
        return self.t_answer - self.t_transcript if (self.t_answer and self.t_transcript) else -1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_text": self.target_text,
            "speech_s": round(self.speech_s, 2),
            "voice_to_transcript_s": round(self.voice_to_transcript_s, 3),
            "transcript_to_answer_s": round(self.transcript_to_answer_s, 3),
            "transcript_text": self.transcript_text,
            "stt_latency_event": self.stt_latency,
            "turn_latency_event": self.turn_latency,
            "answer": {"kind": self.answer_kind, "title": self.answer_title},
            "first_token": "n/a (gateway is non-streaming)",
            "extra": self.extra,
        }


# --------------------------------------------------------------------------
# live-path probe
# --------------------------------------------------------------------------


class LivePathProbe:
    """Streams one utterance through /ws/audio and times the event chain."""

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.ws_base = ("wss://" if base.startswith("https") else "ws://") + self.base.split("://", 1)[1]
        self.events: list[dict[str, Any]] = []
        self._listener: asyncio.Task | None = None

    async def __aenter__(self) -> "LivePathProbe":
        self._ctx = _ssl_ctx()
        self._ev_ws = await websockets.connect(
            f"{self.ws_base}/ws", ssl=self._ctx, max_queue=None
        )
        self._listener = asyncio.create_task(self._listen())
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._listener:
            self._listener.cancel()
        await self._ev_ws.close()

    async def _listen(self) -> None:
        async for raw in self._ev_ws:
            try:
                ev = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            ev["_recv"] = time.time()
            ev.setdefault("t", ev["_recv"])
            self.events.append(ev)

    async def stream_utterance(self, pcm: np.ndarray) -> dict[str, float]:
        """Push [silence, speech, silence] paced at real time; return anchors."""
        t_send: dict[str, float] = {}
        async with websockets.connect(
            f"{self.ws_base}/ws/audio", ssl=self._ctx, max_queue=None
        ) as audio_ws:
            n = int(SR * CHUNK_S)
            silent = (b"\x00\x00" * n)
            t0 = time.monotonic()
            for k in range(1, int(LEAD_SILENCE_S / CHUNK_S) + 1):
                await audio_ws.send(silent)
                await asyncio.sleep(max(0.0, CHUNK_S * k - (time.monotonic() - t0)))
            t0 = time.monotonic()
            k = 0
            for i in range(0, len(pcm), n):
                await audio_ws.send(pcm[i : i + n].tobytes())
                k += 1
                await asyncio.sleep(max(0.0, CHUNK_S * k - (time.monotonic() - t0)))
            t_send["speech_stop"] = time.time()
            for _ in range(int(TAIL_SILENCE_S / CHUNK_S)):
                await audio_ws.send(silent)
                await asyncio.sleep(CHUNK_S)
        return t_send

    async def await_chain(
        self, expect_text: str, since: float, timeout_s: float = ANSWER_TIMEOUT_S
    ) -> UtteranceResult:
        """Collect transcript -> turn_latency -> answer events for one utterance."""
        res = UtteranceResult(name="", target_text=expect_text)
        deadline = time.monotonic() + timeout_s
        answered = False
        # NOTE: event `t` fields mix clocks in the pipeline (transcript/stt
        # carry monotonic seconds, turn/scene/card carry epoch) — this probe
        # therefore windows exclusively on `_recv` (epoch stamped at receipt;
        # same machine, so deltas against `since` are exact).
        while time.monotonic() < deadline:
            for ev in self.events:
                if ev.get("_seen") or float(ev.get("_recv", 0)) < since - 1.0:
                    continue
                typ = ev.get("type")
                if typ == "transcript" and not res.t_transcript:
                    if _overlap(str(ev.get("text", "")), expect_text) >= 0.4:
                        ev["_seen"] = True
                        res.t_transcript = float(ev.get("_recv"))
                        res.transcript_text = str(ev.get("text", ""))
                elif typ == "turn_latency" and not res.t_turn_latency:
                    if _overlap(str(ev.get("text", "")), expect_text) >= 0.4:
                        ev["_seen"] = True
                        res.t_turn_latency = float(ev.get("_recv"))
                        res.turn_latency = {
                            k: ev.get(k)
                            for k in ("kind", "tier", "detect_ms", "lane_ms", "t")
                        }
                elif typ in ("card", "scene_context") and not answered:
                    if res.t_transcript and float(ev.get("_recv")) >= res.t_transcript:
                        ev["_seen"] = True
                        answered = True
                        res.t_answer = float(ev.get("_recv"))
                        res.answer_kind = typ
                        if typ == "card":
                            card = ev.get("card") or {}
                            res.answer_title = str(card.get("title", ""))[:120]
                        else:
                            res.answer_title = str(ev.get("text", ""))[:120]
            stt = [
                e
                for e in self.events
                if e.get("type") == "stt_latency"
                and float(e.get("_recv", 0)) >= since
            ]
            if stt:
                res.stt_latency = {
                    k: stt[-1].get(k)
                    for k in ("queue_wait_ms", "stt_ms", "post_speech_ms")
                }
            if res.t_transcript and answered:
                break
            await asyncio.sleep(0.2)
        return res


# --------------------------------------------------------------------------
# whisperx component probe (queue/decode split + diarize cost)
# --------------------------------------------------------------------------


async def probe_whisperx(stt_base: str, pcm: np.ndarray, diarize: bool) -> dict[str, Any]:
    url = stt_base.rstrip("/") + "/transcribe"
    body = wav_bytes(pcm)
    async with httpx.AsyncClient(timeout=TRANSCRIBE_TIMEOUT_S) as client:
        t0 = time.monotonic()
        r = await client.post(
            url,
            params={"diarize": "true" if diarize else "false", "align": "false"},
            content=body,
            headers={"Content-Type": "audio/wav"},
        )
        wall = time.monotonic() - t0
    d = r.json() if r.status_code == 200 else {}
    text = d.get("text") or " ".join(
        s.get("text", "") for s in d.get("segments", []) if isinstance(s, dict)
    )
    return {
        "http": r.status_code,
        "wall_s": round(wall, 3),
        "timings": d.get("timings"),
        "text": (text or "").strip()[:160],
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    async with httpx.AsyncClient(verify=False, timeout=10.0) as hc:
        try:
            r = await hc.get(f"{args.base}/api/config")
            cfg = r.json().get("config") or {}
        except Exception as exc:
            print(f"cannot read /api/config from {args.base}: {exc}", file=sys.stderr)
            return 2
    stt_pipeline = cfg.get("stt_pipeline") or {}
    stt_cfg = ((cfg.get("models") or {}).get("stt")) or {}
    silence_ms = int(stt_pipeline.get("silence_ms", 700))
    stt_base = stt_cfg.get("base_url") or "http://127.0.0.1:8123"
    diarize_cfg = bool(stt_cfg.get("diarize", True))

    print(f"service: {args.base}  silence_ms={silence_ms}  "
          f"stt={stt_base} (diarize_cfg={diarize_cfg})")

    # Resolve the three utterances (espeak by default, --wav overrides).
    wavs: dict[str, tuple[np.ndarray, str, str]] = {}
    overrides = dict(
        kv.split("=", 1) for kv in (args.wav or []) if "=" in kv
    )
    expect_over = dict(
        kv.split("=", 1) for kv in (args.expect_text or []) if "=" in kv
    )
    for u in UTTERANCES:
        name = u["name"]
        if name in overrides:
            pcm = load_wav(overrides[name])
            text = f"(supplied wav {overrides[name]})"
        else:
            pcm = synth_espeak(u["text"])
            text = u["text"]
        wavs[name] = (pcm, text, expect_over.get(name, u["expect"]))
        print(f"  synth {name}: {len(pcm)/SR:.1f}s of speech — {text[:60]}")

    results: list[UtteranceResult] = []
    component: dict[str, Any] = {}

    if not args.e2e_only:
        # Component: whisperx direct (warm), diarize off then on.
        for name, (pcm, text, _exp) in wavs.items():
            component[name] = {
                "whisperx_diarize_false": await probe_whisperx(stt_base, pcm, False),
                "whisperx_diarize_true": await probe_whisperx(stt_base, pcm, True),
            }
            t = component[name]["whisperx_diarize_false"]
            print(f"  whisperx {name}: wall {t['wall_s']}s timings {t['timings']}")

    if not args.stt_only:
        async with LivePathProbe(args.base) as probe:
            for name, (pcm, text, expect) in wavs.items():
                since = time.time()
                anchors = await probe.stream_utterance(pcm)
                res = await probe.await_chain(expect, since=since)
                res.name = name
                res.speech_s = len(pcm) / SR
                res.t_audio_last_sent = anchors["speech_stop"]
                res.extra.update(
                    {
                        "vad_endpoint_est_s_after_speech_stop": silence_ms / 1000.0,
                        "correlation": "temporal-first",
                        "espeak_target": text,
                    }
                )
                vt = res.voice_to_transcript_s
                ta = res.transcript_to_answer_s
                print(
                    f"\n== {name}: speech {res.speech_s:.1f}s"
                    f"\n   voice->transcript : {vt:.2f}s (target 5s)"
                    f"\n   transcript->answer: {ta:.2f}s (target 15s)"
                    f"\n   transcript        : {res.transcript_text[:90]!r}"
                    f"\n   stt_latency evt   : {res.stt_latency}"
                    f"\n   turn_latency evt  : {res.turn_latency}"
                    f"\n   answer            : {res.answer_kind} "
                    f"{res.answer_title[:70]!r}"
                )
                results.append(res)
                await asyncio.sleep(2.0)

    artifact = {
        "id": uuid.uuid4().hex[:10],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base": args.base,
        "config_snapshot": {
            "silence_ms": silence_ms,
            "stt_base": stt_base,
            "stt_diarize_cfg": diarize_cfg,
        },
        "utterances": [r.as_dict() for r in results],
        "whisperx_component": component,
        "notes": [
            "first_token unmeasurable: gateway.chat is non-streaming",
            "answer correlation is temporal-first (first card/scene event after the utterance's transcript)",
        ],
    }
    out = Path(args.out) if args.out else Path("data/latency") / (
        f"probe-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    # console summary vs targets
    ok_a = bool(results) and all(0 <= r.voice_to_transcript_s <= 5.0 for r in results)
    ok_b = bool(results) and all(0 <= r.transcript_to_answer_s <= 15.0 for r in results)
    print("TARGET A voice->transcript <=5s :", "MET" if ok_a else "NOT MET" if results else "n/a")
    print("TARGET B transcript->answer <=15s:", "MET" if ok_b else "NOT MET" if results else "n/a")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default="https://127.0.0.1:8760")
    p.add_argument("--out", default="")
    p.add_argument("--reps", type=int, default=1, help="repeat the full sweep N times")
    p.add_argument("--stt-only", action="store_true", help="component measurement only")
    p.add_argument("--e2e-only", action="store_true", help="skip the direct whisperx probe")
    p.add_argument("--wav", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--expect-text", action="append", default=[], metavar="NAME=KEYWORDS")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args(sys.argv[1:]))))
