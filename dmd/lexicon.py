"""Lexicon construction, fuzzy token correction, and entity linking."""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Sequence

from .types import Entity, LexiconEntry

COMMON_WORDS: frozenset[str] = frozenset({
    "a", "about", "above", "across", "after", "against", "along", "already",
    "also", "although", "always", "am", "among", "an", "and", "another",
    "any", "anybody", "anything", "anywhere", "are", "around", "as", "ask",
    "asked", "at", "away", "back", "be", "became", "because", "become",
    "been", "before", "behind", "being", "below", "beneath", "beside",
    "best", "better", "between", "beyond", "big", "both", "bring", "build",
    "but", "by", "call", "came", "can", "cannot", "could", "carry", "catch",
    "certain", "change", "clean", "clear", "close", "cold", "come", "consider",
    "contain", "continue", "cool", "create", "cut", "dark", "day", "days",
    "dead", "deep", "describe", "destroy", "did", "die", "different", "do",
    "does", "doing", "done", "down", "draw", "drive", "drop", "during",
    "each", "early", "easy", "either", "empty", "end", "enough", "even",
    "ever", "every", "everybody", "everyone", "everything", "explain",
    "face", "fact", "fall", "far", "fast", "feel", "felt", "few", "fight",
    "find", "fine", "finish", "first", "five", "fly", "follow", "for",
    "forget", "forward", "four", "free", "friend", "from", "full", "further",
    "gave", "get", "give", "go", "going", "gone", "good", "got", "great",
    "had", "hand", "happen", "hard", "has", "have", "having", "he", "hear",
    "heard", "help", "her", "here", "hers", "high", "him", "his", "hit",
    "hold", "home", "hope", "hot", "how", "however", "i", "if", "important",
    "in", "include", "indeed", "inside", "instead", "into", "is", "it",
    "its", "just", "keep", "kept", "kind", "knew", "know", "large", "last",
    "late", "later", "least", "leave", "led", "less", "let", "like", "likely",
    "little", "live", "long", "look", "lose", "lost", "lot", "made", "make",
    "many", "may", "maybe", "me", "mean", "meet", "might", "mine", "more",
    "most", "much", "must", "my", "near", "nearly", "necessary", "need",
    "neither", "never", "new", "next", "nice", "nine", "no", "nobody",
    "none", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often",
    "on", "once", "one", "only", "open", "or", "other", "our", "ours",
    "out", "outside", "over", "own", "past", "people", "perhaps", "place",
    "play", "point", "possible", "present", "pretty", "probably", "provide",
    "pull", "push", "put", "quite", "rather", "reach", "read", "ready",
    "real", "really", "remain", "remember", "reply", "represent", "rest",
    "return", "right", "rise", "run", "said", "same", "saw", "say", "says",
    "second", "see", "seem", "seen", "sell", "send", "set", "seven",
    "several", "shall", "she", "should", "show", "side", "since", "six",
    "slow", "small", "so", "some", "somebody", "someone", "something",
    "sometimes", "somewhere", "soon", "sort", "stand", "start", "stay",
    "still", "stop", "such", "sure", "take", "talk", "tell", "ten", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "thing", "think", "third", "this", "those", "though",
    "three", "through", "throughout", "thus", "till", "time", "to", "today",
    "together", "told", "too", "took", "toward", "towards", "try", "turn",
    "two", "under", "underneath", "unless", "unlike", "until", "up", "upon",
    "us", "use", "usually", "very", "want", "was", "watch", "way", "we",
    "well", "went", "were", "what", "when", "where", "whether", "which",
    "while", "who", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "won", "work", "would", "write", "yes", "yet",
    "you", "your", "yours", "yourself",
})


_VOWELS = frozenset("aeiou")
_TOKEN_RE = re.compile(r"\b[\w'\u2019\-]+\b")
_META_INITIAL_PREFIXES = ("kn", "gn", "pn", "wr", "ps")
_META_DIGRAPHS = (
    ("sch", "sk"),
    ("sh", "x"),
    ("ch", "x"),
    ("ph", "f"),
    ("th", "0"),
    ("gh", ""),
    ("ck", "k"),
    ("qu", "kw"),
    ("zh", "j"),
)


def _metaphone(word: str) -> str:
    """Compact metaphone that groups obvious rhymes together."""
    if not word:
        return ""
    s = "".join(c for c in word.lower() if c.isalpha())
    if not s:
        return ""
    if len(s) > 2 and s.startswith(_META_INITIAL_PREFIXES):
        s = s[1:]
    if len(s) > 1 and s.startswith("x"):
        s = "s" + s[1:]
    if len(s) > 2 and s.startswith("wh"):
        s = "w" + s[2:]
    for src, dst in _META_DIGRAPHS:
        s = s.replace(src, dst)
    out: list[str] = []
    for i, c in enumerate(s):
        if c in _VOWELS or c == "h":
            if i == 0:
                out.append(c)
        elif c == "y":
            out.append("i")
        else:
            out.append(c)
    s = "".join(out)
    if len(s) > 1 and s.endswith("e"):
        s = s[:-1]
    return s.upper()


def _is_bad_variant(value: str) -> bool:
    if len(value) < 4:
        return True
    if value.isdigit():
        return True
    if value.lower() in COMMON_WORDS:
        return True
    return False


def _apply_case(template: str, target: str) -> str:
    if not target:
        return target
    if len(template) > 1 and template.isupper():
        return target.upper()
    if template and template[0].isupper():
        rest_lower = template[1:].islower() if len(template) > 1 else True
        if rest_lower:
            return target[0].upper() + target[1:].lower()
    return target.lower()


def build_lexicon(entities: Iterable[Entity]) -> list[LexiconEntry]:
    """Build a normalized, deduplicated lexicon from entities."""
    entity_list = list(entities)
    if not entity_list:
        return []

    max_weight = max((e.weight for e in entity_list), default=0.0) or 1.0

    canonicals_lower: dict[str, str] = {}
    for e in entity_list:
        key = e.canonical.lower()
        if key not in canonicals_lower:
            canonicals_lower[key] = e.canonical

    entries: list[LexiconEntry] = []
    for e in entity_list:
        seen_lower: set[str] = set()
        variants: list[str] = []
        if not _is_bad_variant(e.canonical):
            variants.append(e.canonical)
            seen_lower.add(e.canonical.lower())
        for alias in e.aliases:
            alias_lower = alias.lower()
            if alias_lower in canonicals_lower:
                continue
            if alias_lower in seen_lower:
                continue
            if _is_bad_variant(alias):
                continue
            variants.append(alias)
            seen_lower.add(alias_lower)
        if not variants:
            continue
        normalized = e.weight / max_weight if max_weight > 0 else 0.0
        entries.append(
            LexiconEntry(
                canonical=e.canonical,
                variants=variants,
                etype=e.etype,
                weight=normalized,
            )
        )

    entries.sort(key=lambda x: (-x.weight, x.canonical.lower()))
    return entries


def correct_text(
    text: str,
    entries: Sequence[LexiconEntry],
    fuzzy_threshold: float = 0.82,
) -> tuple[str, list[tuple[int, int, str]]]:
    """Fuzzy-correct tokens to canonical entity names."""
    exact_lookup: dict[str, str] = {}
    candidates: list[tuple[str, str, str]] = []
    for entry in entries:
        all_variants = [entry.canonical, *entry.variants]
        for v in all_variants:
            key = v.lower()
            exact_lookup.setdefault(key, entry.canonical)
            if len(v) >= 4:
                candidates.append((key, v, entry.canonical))

    if not text or not candidates:
        return text, []

    candidate_meta = {key: _metaphone(key) for key, _, _ in candidates}

    tokens = [(m.start(), m.end(), m.group(0)) for m in _TOKEN_RE.finditer(text)]

    replacements: list[tuple[int, int, str]] = []
    for start, end, tok in tokens:
        if tok.lower() in exact_lookup:
            continue
        best_score = 0.0
        best_canonical = ""
        tok_meta = _metaphone(tok)
        for key, _, canonical in candidates:
            if abs(len(tok) - len(key)) > 3:
                continue
            ratio = difflib.SequenceMatcher(None, tok.lower(), key).ratio()
            score = ratio
            if tok_meta and candidate_meta[key] and tok_meta == candidate_meta[key]:
                score = max(score, 1.0)
            if score > best_score:
                best_score = score
                best_canonical = canonical

        if best_score >= fuzzy_threshold and best_canonical:
            replacements.append((start, end, best_canonical))

    if not replacements:
        return text, []

    out_parts: list[str] = []
    last = 0
    spans: list[tuple[int, int, str]] = []
    for start, end, canonical in replacements:
        out_parts.append(text[last:start])
        original = text[start:end]
        out_parts.append(_apply_case(original, canonical))
        spans.append((start, end, canonical))
        last = end
    out_parts.append(text[last:])
    return "".join(out_parts), spans


def link_entities(
    text: str,
    entries: Sequence[LexiconEntry],
) -> list[tuple[str, tuple[int, int]]]:
    """Find non-overlapping left-to-right longest matches of canonicals/aliases."""
    raw_matches: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int]] = set()
    for entry in entries:
        for variant in [entry.canonical, *entry.variants]:
            if not variant:
                continue
            pattern = re.escape(variant)
            for m in re.finditer(pattern, text, re.IGNORECASE):
                key = (m.start(), m.end())
                if key in seen:
                    continue
                seen.add(key)
                raw_matches.append((m.start(), m.end(), m.group(0), entry.canonical))

    raw_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    result: list[tuple[str, tuple[int, int]]] = []
    occupied_end = -1
    for start, end, _matched, canonical in raw_matches:
        if start < occupied_end:
            continue
        result.append((canonical, (start, end)))
        occupied_end = end
    return result