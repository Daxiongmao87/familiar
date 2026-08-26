"""Dependency-free RMS-based utterance boundary detector over int16 mono PCM."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .types import Utterance


@dataclass
class _UserState:
    pcm: bytearray = field(default_factory=bytearray)
    t_start: float = 0.0
    speech_active: bool = False
    silence_ms_accum: float = 0.0
    last_t: float = 0.0


class UtteranceSegmenter:
    """Per-user rolling RMS-based utterance boundary detector."""

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_ms: int = 700,
        min_utterance_ms: int = 400,
        energy_floor: int = 250,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_ms = silence_ms
        self.min_utterance_ms = min_utterance_ms
        self.energy_floor = energy_floor
        self._max_bytes = 60 * sample_rate * 2
        self._state: dict[str, _UserState] = {}

    @staticmethod
    def _rms(pcm: bytes) -> float:
        n = len(pcm) // 2
        if n == 0:
            return 0.0
        samples = struct.unpack(f"<{n}h", pcm)
        total = 0
        for s in samples:
            total += s * s
        return (total / n) ** 0.5

    def _duration_ms(self, state: _UserState) -> float:
        return (len(state.pcm) / 2 / self.sample_rate) * 1000.0

    def feed(self, user_id: str, pcm: bytes, t_mono: float) -> list[Utterance]:
        state = self._state.setdefault(user_id, _UserState())
        chunk_ms = (len(pcm) / 2 / self.sample_rate) * 1000.0
        rms = self._rms(pcm)
        state.last_t = t_mono
        emitted: list[Utterance] = []

        if rms >= self.energy_floor:
            if not state.speech_active:
                state.speech_active = True
                state.t_start = t_mono
            state.silence_ms_accum = 0.0
            state.pcm.extend(pcm)
        else:
            if state.speech_active:
                state.silence_ms_accum += chunk_ms
                state.pcm.extend(pcm)
            else:
                return emitted

        if len(state.pcm) >= self._max_bytes:
            u = self._build(user_id, state)
            if u is not None:
                emitted.append(u)
            state.pcm.clear()
            state.t_start = t_mono
            state.silence_ms_accum = 0.0
            state.speech_active = False
            return emitted

        if state.speech_active and state.silence_ms_accum >= self.silence_ms:
            if self._duration_ms(state) >= self.min_utterance_ms:
                u = self._build(user_id, state)
                if u is not None:
                    emitted.append(u)
            self._reset(state)
        return emitted

    def flush_user(self, user_id: str) -> list[Utterance]:
        state = self._state.get(user_id)
        if state is None or not state.speech_active or not state.pcm:
            if state is not None:
                self._reset(state)
            return []
        emitted: list[Utterance] = []
        if self._duration_ms(state) >= self.min_utterance_ms:
            u = self._build(user_id, state)
            if u is not None:
                emitted.append(u)
        self._reset(state)
        return emitted

    @staticmethod
    def _build(user_id: str, state: _UserState) -> Utterance | None:
        if not state.pcm:
            return None
        return Utterance(
            user_id=user_id,
            text="",
            t_start=state.t_start,
            t_end=state.last_t,
        )

    @staticmethod
    def _reset(state: _UserState) -> None:
        state.pcm.clear()
        state.speech_active = False
        state.silence_ms_accum = 0.0
        state.t_start = 0.0