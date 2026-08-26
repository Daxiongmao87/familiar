"""Session runtime: STT over VAD, lexicon post-correction, trigger detection,
synthesis-job submission, and source consumption.
"""

from __future__ import annotations

import struct
import time
import uuid
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from .config import AppConfig
from .embedder import Embedder
from .index_store import IndexStore
from .lexicon import correct_text, link_entities
from .orchestrator import JobPool
from .sources.base import AudioSource
from .triggers import detect_trigger
from .types import Card, Job, LexiconEntry, Priority, Retrieved, Utterance
from .vad import UtteranceSegmenter

CARD_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["skill_table", "lore", "rules", "info"]},
        "title": {"type": "string"},
        "body_md": {"type": "string"},
    },
    "required": ["kind", "title", "body_md"],
    "additionalProperties": False,
}

_SYNTHESIS_SYSTEM = (
    "You are a Dungeon Master's live assistant. You answer from the supplied lore "
    "excerpts only; never invent names, places, or rules. Produce concise, "
    "table-first output. For loot/search/examine intents, return a skill-check "
    "table with a DC column and one row per relevant skill (Investigation, "
    "Perception, Stealth, etc.). For lore questions, give a short briefing. For "
    "rules questions, cite the rule. Output is strict JSON."
)


def _wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw int16 mono PCM into a minimal 16-bit PCM WAV container."""
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm


def _lexicon_prompt(entries: list[LexiconEntry], max_chars: int = 500) -> str:
    """Build a compact, weighted hotword prompt for layer-1 STT biasing."""
    seen: set[str] = set()
    parts: list[str] = []
    used = 0
    for e in entries:
        for variant in [e.canonical, *e.variants]:
            v = variant.strip()
            if not v or v.lower() in seen:
                continue
            if not v.replace(" ", "").replace("'", "").replace("-", "").isalnum():
                continue
            seen.add(v.lower())
            cost = len(v) + 1
            if used + cost > max_chars:
                return " ".join(parts)
            parts.append(v)
            used += cost
            break
    return " ".join(parts)


def _hits_to_excerpts(hits: list[Retrieved]) -> str:
    blocks: list[str] = []
    for i, h in enumerate(hits, 1):
        text = h.text.strip()
        if not text:
            continue
        src = h.source or h.doc_id
        blocks.append(f"[{i}] ({src})\n```\n{text}\n```")
    return "\n\n".join(blocks)


def _build_messages(
    ctx: dict,
    excerpts: str,
    user_id: str,
    entities: list[str],
) -> list[dict]:
    utterance = ctx.get("utterance", "")
    kind = ctx.get("kind", "other")
    recent = ctx.get("recent", [])

    recent_lines = []
    for r in recent[-5:]:
        if isinstance(r, Utterance):
            recent_lines.append(f"- {r.user_id}: {r.text}")
        else:
            recent_lines.append(f"- {r}")
    recent_block = "\n".join(recent_lines) if recent_lines else "(none)"

    user = (
        f"Speaker: {user_id}\n"
        f"Intent kind: {kind}\n"
        f"Entities mentioned: {', '.join(entities) if entities else '(none)'}\n"
        f"Recent utterances:\n{recent_block}\n\n"
        f"Current utterance:\n{utterance}\n\n"
        f"Lore excerpts (use ONLY these; cite numbers in brackets):\n{excerpts or '(no excerpts retrieved)'}\n\n"
        "If the intent hints loot, search, or examine, produce a skill-check table "
        "with a DC column and one row per relevant skill. Otherwise produce a concise "
        "briefing in markdown. Return JSON matching the schema."
    )
    return [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {"role": "user", "content": user},
    ]


def _make_job(
    kind: str,
    ctx: dict,
    priority: Priority,
    context_window_s: float,
) -> Job:
    return Job(
        id=uuid.uuid4().hex[:12],
        kind=kind,
        prompt_context=ctx,
        priority=priority,
        t_created=time.monotonic(),
        context_window_s=context_window_s,
    )


class SessionEngine:
    """Owns per-session state: rolling transcript, VAD buffers, lexicon cache,
    embedding, retrieval, and the synthesis job pool.
    """

    def __init__(
        self,
        cfg: AppConfig,
        store: IndexStore,
        gw: Any,
        entries: list[LexiconEntry],
        embedder: Optional[Embedder],
        pool: Any,
        on_event: Callable[[dict], None],
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.gw = gw
        self.entries = entries
        self.embedder = embedder
        self.pool = pool
        self.on_event = on_event
        self._recent: deque[Utterance] = deque(maxlen=30)
        self._hotword_prompt: str = _lexicon_prompt(entries)

    @property
    def recent_utterances(self) -> list[Utterance]:
        return list(self._recent)

    def _remember(self, u: Utterance) -> None:
        self._recent.append(u)

    async def _dispatch_utterance(
        self, user_id: str, audio: bytes, u: Utterance
    ) -> None:
        try:
            produced = await self.transcribe_pcm(user_id, audio, u.t_start, u.t_end)
        except Exception as exc:
            self.on_event(
                {
                    "type": "transcript_error",
                    "user_id": user_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "t": u.t_end,
                }
            )
            return
        if produced is not None:
            await self.handle_utterance(produced)

    async def transcribe_pcm(
        self,
        user_id: str,
        pcm: bytes,
        t_start: float,
        t_end: float,
    ) -> Utterance | None:
        if not pcm:
            return None
        wav = _wav_bytes(pcm, sample_rate=self.cfg.stt_pipeline.sample_rate)
        prompt = self._hotword_prompt or None
        try:
            raw = await self.gw.transcribe(wav, prompt=prompt)
        except Exception as exc:
            self.on_event(
                {
                    "type": "transcript_error",
                    "user_id": user_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "t": t_end,
                }
            )
            return None
        corrected, _spans = correct_text(raw or "", self.entries)
        self.on_event(
            {
                "type": "transcript",
                "user_id": user_id,
                "text": corrected,
                "t": t_end,
            }
        )
        return Utterance(
            user_id=user_id,
            text=corrected,
            t_start=t_start,
            t_end=t_end,
            raw_text=raw or "",
        )

    async def handle_utterance(self, u: Utterance) -> None:
        mentioned_pairs = link_entities(u.text, self.entries)
        seen: set[str] = set()
        mentioned: list[str] = []
        for canonical, _span in mentioned_pairs:
            if canonical not in seen:
                seen.add(canonical)
                mentioned.append(canonical)

        self._remember(u)

        is_trigger, kind = await detect_trigger(self.gw, u.text)
        if not is_trigger:
            return

        ctx = {
            "utterance": u.text,
            "entities": mentioned,
            "kind": kind,
            "recent": list(self._recent)[-5:],
        }
        job = _make_job("trigger", ctx, Priority.TRIGGER, context_window_s=120.0)

        async def _work() -> Card:
            return await self._generate_card(ctx)

        await self.pool.submit(job, _work)

    async def manual_query(self, text: str) -> None:
        ctx = {
            "utterance": text,
            "entities": [],
            "kind": "manual",
            "recent": list(self._recent)[-5:],
        }
        job = _make_job("manual_query", ctx, Priority.MANUAL, context_window_s=600.0)

        async def _work() -> Card:
            return await self._generate_card(ctx)

        await self.pool.submit(job, _work)

    async def _generate_card(self, ctx: dict) -> Card:
        utterance = ctx.get("utterance", "")
        entities = ctx.get("entities", []) or []
        q: Optional[np.ndarray] = None
        if self.embedder is not None and utterance:
            try:
                vecs = self.embedder.embed([utterance])
                if vecs.ndim == 2 and vecs.shape[0] >= 1:
                    q = vecs[0]
            except Exception:
                q = None
        try:
            hits = self.store.search(embedding=q, query_text=utterance, k=8)
        except Exception:
            hits = []
        excerpts = _hits_to_excerpts(hits)

        user_id = "manual" if ctx.get("kind") == "manual" else "session"
        messages = _build_messages(ctx, excerpts, user_id, entities)

        try:
            result = await self.gw.chat(
                "synthesis",
                messages,
                json_schema=CARD_SCHEMA,
                temperature=0.3,
            )
        except Exception as exc:
            return Card(
                id=uuid.uuid4().hex[:12],
                kind="error",
                title="generation failed",
                body_md=str(exc),
                t_context=time.time(),
                meta={"sources": [h.source for h in hits], "entities": entities},
            )

        card_kind = "info"
        card_title = ""
        card_body = ""
        if isinstance(result, dict):
            card_kind = str(result.get("kind") or "info")
            card_title = str(result.get("title") or "")
            card_body = str(result.get("body_md") or "")
        elif isinstance(result, str):
            card_body = result

        return Card(
            id=uuid.uuid4().hex[:12],
            kind=card_kind,
            title=card_title,
            body_md=card_body,
            t_context=time.time(),
            meta={
                "sources": [h.source for h in hits],
                "entities": entities,
            },
        )

    async def consume_source(self, source: AudioSource) -> None:
        """Stream an AudioSource through the VAD, transcribing each utterance.

        Audio is assumed to arrive as 16 kHz mono int16 PCM (PcmChunk.sample_rate
        must match cfg.stt_pipeline.sample_rate; no resampling is performed).
        """
        sp = self.cfg.stt_pipeline
        segmenter = UtteranceSegmenter(
            sample_rate=sp.sample_rate,
            silence_ms=sp.silence_ms,
            min_utterance_ms=sp.min_utterance_ms,
        )
        buffers: dict[str, bytearray] = {}
        offsets: dict[str, int] = {}

        try:
            async for chunk in source:
                user_id = chunk.user_id
                buf = buffers.setdefault(user_id, bytearray())
                offsets.setdefault(user_id, 0)
                buf.extend(chunk.samples)

                utts = segmenter.feed(user_id, chunk.samples, chunk.t_mono)
                for u in utts:
                    cur_len = len(buf)
                    audio = bytes(buf[offsets[user_id] : cur_len])
                    offsets[user_id] = cur_len
                    if not audio:
                        continue
                    await self._dispatch_utterance(user_id, audio, u)
        finally:
            for user_id, buf in buffers.items():
                cur_len = len(buf)
                tail = bytes(buf[offsets[user_id] : cur_len])
                if not tail:
                    segmenter.flush_user(user_id)
                    continue
                flush_utts = segmenter.flush_user(user_id)
                if not flush_utts:
                    continue
                u = flush_utts[0]
                await self._dispatch_utterance(user_id, tail, u)
            drain = getattr(self.pool, "drain", None)
            if drain is not None:
                try:
                    await drain()
                except Exception:
                    pass
