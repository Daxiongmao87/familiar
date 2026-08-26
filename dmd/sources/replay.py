"""Deterministic file-replay AudioSource.

Loads a wav file per user at construction time, decodes to int16 mono PCM at
the target sample rate, and emits 100 ms chunks. With ``realtime=False`` the
chunks are emitted back-to-back (useful for unit tests and the replay
harness); with ``realtime=True`` the source sleeps between chunks so playback
matches wall-clock time. ``PcmChunk.t_mono`` always reflects the logical
playback position, not the actual emit time, so downstream VAD operates on a
stable clock.
"""

from __future__ import annotations

import asyncio
import time
import wave

import numpy as np

from ..types import PcmChunk
from .base import AudioSource


def _decode_to_int16(path: str) -> tuple[np.ndarray, int]:
    """Decode any supported wav to mono int16 numpy array + source sample rate."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        samples_u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
        samples = (samples_u8 - 128) * 256
    elif sampwidth == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    elif sampwidth == 3:
        raise ValueError(
            f"24-bit wav not supported by wav_to_pcm16k: {path!r} (sampwidth=3). "
            "Re-encode to 16-bit or float and retry."
        )
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.int64)
        samples = (samples >> 16).astype(np.int32)
    else:
        raise ValueError(
            f"unsupported sample width {sampwidth} bytes in wav: {path!r}"
        )

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    elif n_channels == 1:
        samples = samples.reshape(-1)
    else:
        raise ValueError(f"wav has zero channels: {path!r}")

    return samples.astype(np.int16), int(src_rate)


def _linear_resample(mono: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    if src_rate == target_rate or len(mono) == 0:
        return mono.astype(np.int16, copy=False)
    duration = len(mono) / float(src_rate)
    target_len = max(1, int(round(duration * target_rate)))
    src_x = np.arange(len(mono), dtype=np.float64)
    tgt_x = np.linspace(0, len(mono) - 1, num=target_len, dtype=np.float64)
    resampled = np.interp(tgt_x, src_x, mono.astype(np.float64))
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16)


def wav_to_pcm16k(path: str, target_rate: int = 16000) -> bytes:
    """Read a wav file and return int16 LE mono PCM at the target sample rate."""
    mono, src_rate = _decode_to_int16(path)
    resampled = _linear_resample(mono, src_rate, target_rate)
    return resampled.astype("<i2").tobytes()


class ReplaySource(AudioSource):
    """Replay one wav per user. Chunks are 100 ms; trailing 0.7 s silence is
    emitted at the end of every track so the VAD finalizes the last utterance.
    """

    def __init__(
        self,
        tracks: dict[str, str],
        sample_rate: int = 16000,
        realtime: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.realtime = bool(realtime)
        self._chunk_samples = max(1, self.sample_rate // 10)
        self._tracks: list[tuple[str, np.ndarray]] = []
        for user_id, path in tracks.items():
            pcm_bytes = wav_to_pcm16k(path, target_rate=self.sample_rate)
            self._tracks.append((user_id, np.frombuffer(pcm_bytes, dtype="<i2")))

    async def run(self) -> None:
        base = time.monotonic()
        cumulative = 0.0
        chunk_seconds = self._chunk_samples / float(self.sample_rate)
        silence_seconds = 0.7

        for user_id, pcm in self._tracks:
            if len(pcm) > 0:
                cumulative = await self._emit_track(
                    user_id, pcm, base, cumulative, chunk_seconds
                )
            cumulative = await self._emit_silence(
                user_id, silence_seconds, base, cumulative, chunk_seconds
            )
        self.emit(None)

    async def _emit_track(
        self,
        user_id: str,
        pcm: np.ndarray,
        base: float,
        cumulative: float,
        chunk_seconds: float,
    ) -> float:
        n = len(pcm)
        for start in range(0, n, self._chunk_samples):
            end = min(start + self._chunk_samples, n)
            samples = pcm[start:end].tobytes()
            self.emit(
                PcmChunk(
                    user_id=user_id,
                    samples=samples,
                    sample_rate=self.sample_rate,
                    t_mono=base + cumulative,
                )
            )
            cumulative += chunk_seconds
            if self.realtime:
                await asyncio.sleep(chunk_seconds)
        return cumulative

    async def _emit_silence(
        self,
        user_id: str,
        seconds: float,
        base: float,
        cumulative: float,
        chunk_seconds: float,
    ) -> float:
        remaining = float(seconds)
        while remaining > 1e-9:
            this = min(chunk_seconds, remaining)
            n = max(1, int(round(this * self.sample_rate)))
            self.emit(
                PcmChunk(
                    user_id=user_id,
                    samples=b"\x00\x00" * n,
                    sample_rate=self.sample_rate,
                    t_mono=base + cumulative,
                )
            )
            cumulative += this
            remaining -= this
            if self.realtime:
                await asyncio.sleep(this)
        return cumulative
