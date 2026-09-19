"""Unit tests for dmd.terms: zero-LLM term collection and mass cutoff."""

from __future__ import annotations

from dmd.terms import collect_terms, mass_cutoff, rake_candidates, strip_speakers
from dmd.types import LexiconEntry


def _entry(canonical: str, variants: list[str]) -> LexiconEntry:
    return LexiconEntry(canonical=canonical, variants=variants,
                        etype="character", weight=1.0)


def test_strip_speakers_removes_user_tags() -> None:
    assert strip_speakers(["dm: Make a check.", "sam: ok"]) == ["Make a check.", "ok"]


def test_rake_finds_content_phrases() -> None:
    cands = rake_candidates("Make a deception check. Deception? I got a fourteen.")
    assert "deception check" in cands
    assert "fourteen" in cands
    assert all(" " not in c or len(c.split()) > 1 for c in cands)


def test_rake_empty_on_stopwords_only() -> None:
    assert rake_candidates("oh no, I never... yeah sure") == []


def test_collect_fuses_lexicon_spans() -> None:
    entries = [_entry("Nott", ["Nott"])]
    terms = collect_terms(["liam: take Nott down"], entries)
    assert "nott" in terms


def test_collect_uses_matched_span_not_canonical() -> None:
    entries = [_entry("guard", ["guard", "guards"])]
    terms = collect_terms(["sam: two guards here"], entries)
    assert "guards" in terms
    assert "guard" not in terms or "guards" in terms


def test_collect_drops_speakers_numbers_and_fragments() -> None:
    terms = collect_terms(["sam: two plus 14 x y"], [])
    assert "sam" not in terms
    assert "two" not in terms
    assert "14" not in terms
    assert "x" not in terms
    assert "y" not in terms


def test_collect_caps_output() -> None:
    lines = [f"alpha beta gamma delta epsilon zeta eta theta number {i}" for i in range(3)]
    assert len(collect_terms(lines, [], cap=3)) <= 3


def test_mass_cutoff_keeps_top1_and_mass() -> None:
    assert mass_cutoff([0.9, 0.05, 0.05]) == [0]
    assert mass_cutoff([0.5, 0.3, 0.2]) == [0, 1, 2]
    assert mass_cutoff([0.4, 0.3, 0.3]) == [0, 1, 2]
    assert mass_cutoff([0.86, 0.1, 0.04]) == [0]


def test_mass_cutoff_respects_floor_and_max() -> None:
    assert mass_cutoff([0.5, 0.14, 0.14]) == [0]
    assert mass_cutoff([0.3, 0.3, 0.2, 0.1, 0.1], max_keep=2) == [0, 1]
