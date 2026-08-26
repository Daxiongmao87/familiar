"""Unit tests for dmd.lexicon: build_lexicon, correct_text, link_entities."""

from __future__ import annotations

import pytest

from dmd.lexicon import (
    COMMON_WORDS,
    build_lexicon,
    correct_text,
    link_entities,
)
from dmd.types import Entity, LexiconEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(canonical: str, variants: list[str] | None = None, weight: float = 1.0,
           etype: str = "character") -> LexiconEntry:
    return LexiconEntry(
        canonical=canonical,
        variants=variants if variants is not None else [canonical],
        etype=etype,
        weight=weight,
    )


# ===========================================================================
# build_lexicon
# ===========================================================================

# --- dedupe ---------------------------------------------------------------

def test_build_lexicon_dedupes_repeated_aliases():
    """An alias listed more than once in one entity appears only once in variants."""
    entities = [
        Entity(canonical="Frodo", aliases=["Frodo", "Frodo", "Frodo"], etype="character", weight=1.0),
    ]
    entries = build_lexicon(entities)
    assert len(entries) == 1
    frodo_lower = [v.lower() for v in entries[0].variants]
    assert frodo_lower.count("frodo") == 1


def test_build_lexicon_dedupes_alias_equal_to_canonical():
    """If an alias matches the canonical (case-insensitive), it is not duplicated in variants."""
    entities = [
        Entity(canonical="Gandalf", aliases=["gandalf", "Mithrandir"], etype="character", weight=1.0),
    ]
    entries = build_lexicon(entities)
    assert len(entries) == 1
    gandalf_lower = [v.lower() for v in entries[0].variants]
    assert gandalf_lower.count("gandalf") == 1
    assert "mithrandir" in gandalf_lower


def test_build_lexicon_empty_input_returns_empty():
    """No entities -> no entries (and no crash on max(empty))."""
    assert build_lexicon([]) == []


# --- canonical wins -------------------------------------------------------

def test_canonical_wins_over_other_entitys_alias():
    """An alias that matches another entity's canonical (case-insensitive) is dropped."""
    entities = [
        Entity(canonical="Frodo", aliases=[], etype="character", weight=1.0),
        Entity(canonical="Gandalf", aliases=["Frodo", "Mithrandir"], etype="character", weight=1.0),
    ]
    entries = build_lexicon(entities)
    frodo = [e for e in entries if e.canonical == "Frodo"]
    gandalf = [e for e in entries if e.canonical == "Gandalf"]
    assert len(frodo) == 1
    assert len(gandalf) == 1
    # Frodo must exist exactly once and only as a canonical (no separate "frodo" duplicate).
    canonicals_seen = {e.canonical for e in entries}
    assert canonicals_seen == {"Frodo", "Gandalf"}
    # Gandalf must not have inherited Frodo as a variant.
    assert "Frodo" not in gandalf[0].variants
    assert "Mithrandir" in gandalf[0].variants


def test_canonical_wins_case_insensitive_on_alias():
    """Alias conflict check is case-insensitive ('frodo' aliases block each other)."""
    entities = [
        Entity(canonical="Frodo", aliases=[], etype="character", weight=1.0),
        Entity(canonical="Gandalf", aliases=["FRODO"], etype="character", weight=1.0),
    ]
    entries = build_lexicon(entities)
    gandalf = [e for e in entries if e.canonical == "Gandalf"][0]
    # "FRODO" aliases "frodo" canonical -> dropped.
    assert all(v.lower() != "frodo" for v in gandalf.variants)


# --- common-word filtering ------------------------------------------------

def test_common_word_alias_is_filtered():
    """An alias that is a member of COMMON_WORDS is dropped from variants."""
    entities = [
        Entity(canonical="Mordor", aliases=["the", "Awesome"], etype="place", weight=1.0),
    ]
    entries = build_lexicon(entities)
    assert len(entries) == 1
    variants_lower = [v.lower() for v in entries[0].variants]
    assert "the" not in variants_lower
    assert "mordor" in variants_lower
    assert "awesome" in variants_lower


def test_short_alias_is_filtered():
    """Aliases shorter than 4 characters are dropped (too short to be a name)."""
    entities = [
        Entity(canonical="Mordor", aliases=["ok", "Mordor"], etype="place", weight=1.0),
    ]
    entries = build_lexicon(entities)
    variants_lower = [v.lower() for v in entries[0].variants]
    assert "ok" not in variants_lower
    assert "mordor" in variants_lower


def test_all_bad_variants_drops_entity():
    """If every variant of an entity is bad, the entity itself is omitted from the lexicon."""
    entities = [
        Entity(canonical="the", aliases=[], etype="concept", weight=1.0),
    ]
    entries = build_lexicon(entities)
    assert entries == []


def test_digit_alias_is_filtered():
    """Pure-digit aliases are dropped."""
    entities = [
        Entity(canonical="Mordor", aliases=["1234", "Realm"], etype="place", weight=1.0),
    ]
    entries = build_lexicon(entities)
    variants_lower = [v.lower() for v in entries[0].variants]
    assert "1234" not in variants_lower
    assert "realm" in variants_lower


# --- weight normalization -------------------------------------------------

def test_weight_normalization_against_max():
    """Each entry's weight is e.weight / max(e.weight for e in entities)."""
    entities = [
        Entity(canonical="Frodo", aliases=[], etype="character", weight=1.0),
        Entity(canonical="Gandalf", aliases=[], etype="character", weight=2.0),
        Entity(canonical="Aragorn", aliases=[], etype="character", weight=4.0),
    ]
    entries = build_lexicon(entities)
    weights = {e.canonical: e.weight for e in entries}
    assert weights["Aragorn"] == pytest.approx(1.0)
    assert weights["Gandalf"] == pytest.approx(0.5)
    assert weights["Frodo"] == pytest.approx(0.25)


def test_weight_normalization_equal_weights_yield_one():
    """All-equal weights normalize to 1.0 (max_weight is itself)."""
    entities = [
        Entity(canonical="A", aliases=[], etype="x", weight=3.0),
        Entity(canonical="B", aliases=[], etype="x", weight=3.0),
    ]
    entries = build_lexicon(entities)
    assert all(e.weight == pytest.approx(1.0) for e in entries)


def test_entries_sorted_by_weight_desc_then_canonical():
    """Entries are sorted by (-weight, canonical.lower()) for deterministic ordering."""
    entities = [
        Entity(canonical="Frodo", aliases=[], etype="character", weight=1.0),
        Entity(canonical="Aragorn", aliases=[], etype="character", weight=3.0),
        Entity(canonical="Gandalf", aliases=[], etype="character", weight=2.0),
    ]
    entries = build_lexicon(entities)
    # Expected order: Aragorn (3) > Gandalf (2) > Frodo (1)
    assert [e.canonical for e in entries] == ["Aragorn", "Gandalf", "Frodo"]


# --- determinism ----------------------------------------------------------

def test_build_lexicon_determinism():
    """Same input -> byte-identical output (no hidden ordering or randomness)."""
    entities = [
        Entity(canonical="Gandalf", aliases=["Mithrandir", "Olorin"], etype="character", weight=2.0),
        Entity(canonical="Frodo", aliases=["Mr. Frodo"], etype="character", weight=1.0),
        Entity(canonical="Aragorn", aliases=[], etype="character", weight=3.0),
    ]
    first = build_lexicon(entities)
    second = build_lexicon(entities)
    assert first == second
    # Also verify the explicit sort order is stable.
    assert [e.canonical for e in first] == ["Aragorn", "Gandalf", "Frodo"]


# ===========================================================================
# correct_text
# ===========================================================================

def test_correct_text_exact_match_untouched():
    """A token that exactly matches a canonical/variant is left as-is, with empty spans."""
    entries = [_entry("Gandalf", ["Mithrandir"])]
    text = "Gandalf is here"
    out, spans = correct_text(text, entries)
    assert out == text
    assert spans == []


def test_correct_text_fuzzy_typo_corrected():
    """A close-but-imperfect typo is corrected to the canonical."""
    entries = [_entry("Gandalf", [])]
    out, spans = correct_text("Gandalff is here", entries)
    assert out == "Gandalf is here"
    assert len(spans) == 1
    assert spans[0][2] == "Gandalf"


def test_correct_text_metaphone_only_match():
    """Two words with the same metaphone but distant string similarity still match."""
    # 'knight' and 'nite' share metaphone "NT" but SequenceMatcher.ratio() = 0.6 (below 0.82).
    # Without metaphone boost this would NOT be corrected; with the boost it must be.
    entries = [_entry("Knight", [])]
    out, spans = correct_text("the nite attacked", entries)
    # 'nite' is lowercase, so the corrected token is also lowercase
    # (case preservation is covered by its own tests).
    assert out == "the knight attacked"
    assert len(spans) == 1
    # The span's canonical is always the entry's canonical, never the variant.
    assert spans[0][2] == "Knight"


def test_correct_text_casing_preservation_uppercase():
    """An all-uppercase token yields an all-uppercase correction."""
    entries = [_entry("Gandalf", [])]
    out, _ = correct_text("GANDALFF rides", entries)
    assert out == "GANDALF rides"


def test_correct_text_casing_preservation_titlecase():
    """A title-cased token yields a title-cased correction."""
    entries = [_entry("Gandalf", [])]
    out, _ = correct_text("Gandlaf rides", entries)
    assert out == "Gandalf rides"


def test_correct_text_casing_preservation_lowercase():
    """A lowercase token yields a lowercase correction."""
    entries = [_entry("Gandalf", [])]
    out, _ = correct_text("gandalff rides", entries)
    assert out == "gandalf rides"


def test_correct_text_spans_index_original_text():
    """Spans (start, end, canonical) index into the ORIGINAL text, not the output."""
    entries = [_entry("Aragorn", [])]
    text = "hello Aragornn world"
    out, spans = correct_text(text, entries)
    assert out == "hello Aragorn world"
    assert len(spans) == 1
    start, end, canonical = spans[0]
    # Slice into the original text to verify alignment.
    assert text[start:end] == "Aragornn"
    assert canonical == "Aragorn"
    # And the corrected output should be text with [start:end] replaced.
    assert out == text[:start] + "Aragorn" + text[end:]


def test_correct_text_no_correction_of_common_words():
    """Tokens that are COMMON_WORDS are not 'corrected' to entity canonicals."""
    # build_lexicon already filters common-word aliases; here we also test that
    # a common word in the TEXT is not turned into a canonical even if it could
    # hypothetically fuzzy-match.
    entries = build_lexicon([Entity(canonical="Mordor", aliases=["the"], etype="place", weight=1.0)])
    # The "the" alias must have been filtered out.
    mordor = entries[0]
    assert "the" not in [v.lower() for v in mordor.variants]
    # And a sentence containing "the" is untouched.
    text = "the dark land of Mordor"
    out, spans = correct_text(text, entries)
    assert out == text
    assert spans == []


def test_correct_text_no_change_when_below_threshold():
    """A token whose best fuzzy score is below 0.82 is left alone."""
    entries = [_entry("Gandalf", [])]
    # "xyzzy" is far from "Gandalf" by both ratio and metaphone.
    out, spans = correct_text("xyzzy plover", entries)
    assert out == "xyzzy plover"
    assert spans == []


def test_correct_text_empty_input():
    """Empty text and empty entries are both safe no-ops."""
    assert correct_text("", [_entry("Gandalf", [])]) == ("", [])
    assert correct_text("hello there", []) == ("hello there", [])


# ===========================================================================
# link_entities
# ===========================================================================

def test_link_entities_longest_match_wins():
    """When both 'Aragorn' and 'Aragorn II' match, the longer one is selected."""
    entries = [
        _entry("Aragorn", []),
        _entry("Aragorn II", []),
    ]
    text = "I saw Aragorn II in the city"
    matches = link_entities(text, entries)
    assert matches == [("Aragorn II", (6, 16))]


def test_link_entities_non_overlapping():
    """A shorter match starting inside an already-claimed span is skipped."""
    entries = [
        _entry("Frodo", []),
        _entry("Baggins", []),
    ]
    text = "Frodo Baggins went home"
    matches = link_entities(text, entries)
    # Both are non-overlapping AND outside each other, so both should be returned.
    assert matches == [("Frodo", (0, 5)), ("Baggins", (6, 13))]


def test_link_entities_skips_overlapping_shorter():
    """A shorter match whose span is fully inside a longer one's span is dropped."""
    entries = [
        _entry("Frodo", []),
        _entry("Frodo Baggins", []),
    ]
    text = "Frodo Baggins went home"
    matches = link_entities(text, entries)
    # The longer one (0..13) wins; the shorter (0..5) is dropped because it overlaps.
    assert matches == [("Frodo Baggins", (0, 13))]


def test_link_entities_canonical_labeling_via_alias():
    """A match against an alias is still labeled with the entry's canonical."""
    entries = [
        _entry("Gandalf", ["Mr. Gandalf", "Mithrandir"]),
    ]
    text = "I met Mr. Gandalf today"
    matches = link_entities(text, entries)
    assert len(matches) == 1
    canonical, (start, end) = matches[0]
    assert canonical == "Gandalf"  # not "Mr. Gandalf"
    assert text[start:end].lower() == "mr. gandalf"
    assert (start, end) == (6, 17)


def test_link_entities_case_insensitive():
    """Matching is case-insensitive but the reported span uses original text casing."""
    entries = [_entry("Gandalf", [])]
    text = "hello GANDALF world"
    matches = link_entities(text, entries)
    assert matches == [("Gandalf", (6, 13))]
    assert text[6:13] == "GANDALF"  # original casing preserved in the span slice


def test_link_entities_no_match_returns_empty():
    """No entries or no overlap -> empty result."""
    assert link_entities("no names here", []) == []
    assert link_entities("no names here", [_entry("Gandalf", [])]) == []


def test_link_entities_left_to_right_order():
    """Multiple non-overlapping matches are emitted in left-to-right text order."""
    entries = [
        _entry("Frodo", []),
        _entry("Gandalf", []),
        _entry("Aragorn", []),
    ]
    text = "Aragorn, Gandalf, and Frodo traveled together"
    matches = link_entities(text, entries)
    canonicals_in_order = [c for c, _ in matches]
    assert canonicals_in_order == ["Aragorn", "Gandalf", "Frodo"]
    # Sanity: every span is on a non-overlapping, increasing segment.
    ends = [e for _, (_, e) in matches]
    assert ends == sorted(ends)
    assert len(set(ends)) == len(ends)  # no two end at the same point
