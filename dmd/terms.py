"""Zero-LLM search-term collection: RAKE phrases fused with lexicon spans.

Candidates come from two programmatic sources — statistical phrases
(RAKE word scores with a C-value length boost and corpus IDF) and
``link_entities`` matched spans — then filter to retrieval-worthy
terms. No model calls anywhere in this module; ranking of the collected
terms happens downstream in a single JEV pass.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from .lexicon import link_entities
from .types import LexiconEntry

_STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with about into through
    during before after above below to from up down in out on off over under
    again further once here there all any both each few more most other some
    such no nor not only own same so than too very can will just should now
    i me my we our you your he him his she her it its they them their what
    which who whom this that these those am is are was were be been being
    have has had having do does did doing would could ought i'm you're we're
    don't can't let yeah yes oh okay sure never mind like got make roll go
    going hold say of as there's that's how gonna wanna
    """.split()
)
_NUMBER_WORDS = frozenset(
    "one two three four five six seven eight nine ten first second third".split()
)

_SPEAKER_RE = re.compile(r"^[A-Za-z_]+:\s*")
_WORD_RE = re.compile(r"[A-Za-z']+")


def strip_speakers(lines: Sequence[str]) -> list[str]:
    """Remove leading ``user:`` tags; tags must never become terms."""
    return [_SPEAKER_RE.sub("", ln) for ln in lines]


def rake_candidates(text: str, cap: int = 8) -> list[str]:
    """Statistical keyphrases: RAKE scores with a C-value length boost.

    Phrases break on stopwords AND sentence punctuation, so spans never
    merge across sentence boundaries.
    """
    tokens = re.findall(r"[A-Za-z']+|[.!?;:]", text.lower())
    phrases: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if not _WORD_RE.fullmatch(tok) or tok in _STOPWORDS or len(tok) < 2:
            if current:
                phrases.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        phrases.append(current)
    joined = [" ".join(p) for p in phrases if p]
    if not joined:
        return []
    freq: dict[str, int] = {}
    degree: dict[str, int] = {}
    for p in joined:
        ws = p.split()
        for w in ws:
            freq[w] = freq.get(w, 0) + 1
            degree[w] = degree.get(w, 0) + len(ws)
    scored: dict[str, float] = {}
    uniq = set(joined)
    for p in uniq:
        ws = p.split()
        rake = sum(degree[w] / freq[w] for w in ws)
        nested = sum(1 for q in uniq if q != p and p in q)
        cval = math.log2(len(ws) + 1) * (joined.count(p) - nested * 0.5)
        scored[p] = rake * (1 + cval)
    ranked = sorted(scored, key=scored.get, reverse=True)  # type: ignore[arg-type]
    kept: list[str] = []
    for p in ranked:
        if " " not in p and any(p in k.split() for k in kept):
            continue
        kept.append(p)
        if len(kept) >= cap:
            break
    return kept


def collect_terms(
    lines: Sequence[str],
    entries: Sequence[LexiconEntry],
    cap: int = 12,
) -> list[str]:
    """Fuse RAKE phrases with lexicon matched spans; filter to terms.

    Lexicon contributions use the matched surface span (what FTS can
    actually match), never the canonical. Drops fragments: short,
    numeric, number-word, and stopword-only candidates.
    """
    clean = strip_speakers(lines)
    text = "\n".join(clean)
    cands: dict[str, None] = {}
    for p in rake_candidates(text):
        cands.setdefault(p.lower(), None)
    for _canon, (start, end) in link_entities(text, entries):
        span = text[start:end].strip().lower()
        if span:
            cands.setdefault(span, None)
    out = []
    for c in cands:
        if len(c) < 2 or c.isdigit() or c in _NUMBER_WORDS:
            continue
        if all(w in _STOPWORDS for w in c.split()):
            continue
        out.append(c)
    return out[:cap]


def mass_cutoff(
    probs: list[float],
    floor: float = 0.15,
    mass: float = 0.85,
    max_keep: int = 4,
) -> list[int]:
    """Keep top probabilities while each clears ``floor`` and kept mass
    stays under ``mass``; always keeps the top-1. Returns kept indices."""
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    kept: list[int] = []
    total = 0.0
    for i in order:
        if kept and (probs[i] < floor or total >= mass or len(kept) >= max_keep):
            break
        kept.append(i)
        total += probs[i]
    return kept
