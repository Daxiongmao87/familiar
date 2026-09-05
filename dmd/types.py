"""Shared data types crossing module boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(slots=True)
class PcmChunk:
    """A block of decoded mono PCM audio attributed to one speaker."""

    user_id: str
    samples: bytes  # int16 LE mono
    sample_rate: int
    t_mono: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class Utterance:
    """One speaker turn segmented by VAD, post-STT."""

    user_id: str
    text: str
    t_start: float
    t_end: float
    raw_text: str | None = None  # pre-correction transcript if available
    name: str | None = None  # display name from the speaking tracker (§7a)


class Priority(int, Enum):
    """Scene priority levels for the live fast-lane trigger router."""
    AMBIENT = 0
    TRIGGER = 1
    MANUAL = 2


@dataclass(slots=True)
class Job:
    """A discrete unit of synthesis work."""

    id: str
    kind: str  # "manual_query" | "trigger" | "ambient" | "enrich"
    prompt_context: dict[str, Any]
    priority: Priority = Priority.TRIGGER
    t_created: float = field(default_factory=time.monotonic)
    context_window_s: float = 120.0  # staleness horizon


@dataclass(slots=True)
class Card:
    """A generated artifact pushed to the UI."""

    id: str
    kind: str  # "skill_table" | "lore" | "rules" | "info" | "error" | "transcript_notice"
    title: str
    body_md: str
    t_context: float  # session-time this card answers
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # "active" | "done"  (mark-done, never delete)
    player_ids: list[str] = field(default_factory=list)  # players this card concerns


@dataclass(slots=True)
class Entity:
    """Canonical entity extracted from the repo."""

    canonical: str
    aliases: list[str] = field(default_factory=list)
    etype: str = "unknown"  # character | place | item | faction | spell | concept
    weight: float = 1.0
    source_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LexiconEntry:
    """A lexicon entry: canonical form, variants, and matched weight."""
    canonical: str
    variants: list[str]
    etype: str
    weight: float


@dataclass(slots=True)
class ToolSpec:
    """A repo-provided integration registered after a successful probe."""

    name: str
    command: list[str]
    description: str
    output_schema_hint: str  # JSON schema or observed shape description
    timeout_s: float
    cache_ttl_s: float = 300.0


@dataclass(slots=True)
class Retrieved:
    """A retrieved document plus its text snippet."""
    doc_id: str
    source: str
    score: float
    text: str
