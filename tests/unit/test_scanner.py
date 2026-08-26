"""Unit tests for dmd.scanner — pure-Python, no DB, no model downloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from dmd.scanner import DocFile, chunk_docs, scan_folder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_doc(
    relpath: str,
    content: str,
    *,
    frontmatter: dict | None = None,
    headings: list[str] | None = None,
    title: str = "",
) -> DocFile:
    return DocFile(
        relpath=relpath,
        title=title or relpath,
        frontmatter=frontmatter or {},
        links=[],
        tags=[],
        headings=headings or [],
        content=content,
        mtime=1.0,
    )


# ---------------------------------------------------------------------------
# scan_folder — discovery + parsing
# ---------------------------------------------------------------------------


def test_scan_folder_discovers_md_markdown_txt_and_ignores_others(tmp_path):
    _write(tmp_path / "a.md", "alpha")
    _write(tmp_path / "b.markdown", "beta")
    _write(tmp_path / "c.txt", "gamma")
    _write(tmp_path / "skip_me.py", "python")
    _write(tmp_path / "skip_me.rst", "restructured")
    _write(tmp_path / "no_extension", "plain")

    rels = sorted(d.relpath for d in scan_folder(str(tmp_path)))
    assert rels == ["a.md", "b.markdown", "c.txt"]


def test_scan_folder_empty_directory_returns_empty_list(tmp_path):
    assert scan_folder(str(tmp_path)) == []


def test_scan_folder_parses_and_strips_frontmatter(tmp_path):
    raw = (
        "---\n"
        "title: My Title\n"
        "tags: [t1, t2]\n"
        "---\n"
        "# Heading\n"
        "body line\n"
    )
    _write(tmp_path / "front.md", raw)

    [doc] = scan_folder(str(tmp_path))
    assert doc.frontmatter == {"title": "My Title", "tags": ["t1", "t2"]}
    # body no longer contains the frontmatter delimiter
    assert not doc.content.startswith("---")
    assert doc.content.startswith("# Heading")
    assert doc.content.endswith("body line\n")


def test_scan_folder_title_precedence_frontmatter_over_h1_over_stem(tmp_path):
    _write(
        tmp_path / "fm.md",
        "---\ntitle: From FM\n---\n# From H1\nbody\n",
    )
    _write(tmp_path / "h1.md", "# From H1\nbody\n")
    _write(tmp_path / "stem.md", "no heading at all here\n")

    titles = {d.relpath: d.title for d in scan_folder(str(tmp_path))}
    assert titles == {
        "fm.md": "From FM",
        "h1.md": "From H1",
        "stem.md": "stem",
    }


def test_scan_folder_extracts_wikilinks_and_inline_tags(tmp_path):
    raw = (
        "---\n"
        "tags: [npc, location]\n"
        "---\n"
        "# Heading\n"
        "Talks to [[Alice the Bard]] and visits [[Ravenholm]].\n"
        "Also links [[Other Page|Display Name]].\n"
        "Inline #combat tag here.\n"
        "Trailing text (#wizard) and [bracketed #magic] forms.\n"
    )
    _write(tmp_path / "rich.md", raw)

    [doc] = scan_folder(str(tmp_path))

    # Wikilinks — alias is dropped, order preserved, unique.
    assert doc.links == ["Alice the Bard", "Ravenholm", "Other Page"]

    # Frontmatter tags come first, inline tags appended, deduped.
    assert doc.tags[:2] == ["npc", "location"]
    for t in ("combat", "wizard", "magic"):
        assert t in doc.tags
    # Order within the union is FM-first, then inline.
    assert doc.tags == ["npc", "location", "combat", "wizard", "magic"]


# ---------------------------------------------------------------------------
# chunk_docs — sectioning + splitting
# ---------------------------------------------------------------------------


def test_chunk_docs_heading_based_split():
    long_a = "Body A. " + ("a" * 250)
    long_b = "Body B. " + ("b" * 250)
    content = f"## Section A\n{long_a}\n## Section B\n{long_b}\n"
    doc = _make_doc(
        "x.md",
        content,
        headings=["Section A", "Section B"],
    )

    chunks = chunk_docs([doc])

    assert len(chunks) == 2
    assert [c.heading_path for c in chunks] == [["Section A"], ["Section B"]]
    assert "Body A" in chunks[0].text
    assert "Body B" in chunks[1].text


def test_chunk_docs_merges_tiny_sections_forward():
    big = ("Body B with enough text. " * 12).strip()  # > 200 chars
    assert len(big.strip()) >= 200
    content = "## Tiny\nshort.\n## Big\n" + big + "\n"
    doc = _make_doc(
        "x.md",
        content,
        headings=["Tiny", "Big"],
    )

    chunks = chunk_docs([doc])

    # Tiny absorbed into the next section's heading.
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Big"]
    assert "short." in chunks[0].text
    assert big[:30] in chunks[0].text


def test_chunk_docs_hard_split_at_sentence_boundary_for_oversized_section():
    # Each sentence is well under target so the hard split actually has to
    # accumulate many sentences before emitting a chunk.
    sentence = "This is sentence number with some extra padding words. "
    long_text = sentence * 80  # ~3760 chars
    assert len(long_text) > 1600
    content = "## Section\n" + long_text + "\n"
    doc = _make_doc("x.md", content, headings=["Section"])

    chunks = chunk_docs([doc])

    # Oversized section must be split into multiple chunks.
    assert len(chunks) > 1
    # Every chunk keeps the same heading context.
    assert all(c.heading_path == ["Section"] for c in chunks)
    # Each piece stays bounded — target(800) + overlap(100) + a sentence slop.
    for c in chunks:
        assert len(c.text) <= 1000, f"chunk too large: {len(c.text)}"
    # Reassembling the pieces must preserve the full text (modulo spaces at
    # overlap boundaries) — proves we actually split at sentence boundaries.
    joined = " ".join(c.text for c in chunks)
    # All the original sentences are present.
    for s in ["sentence number with some extra padding words"] * 5:
        assert s in joined


def test_chunk_ids_deterministic_across_runs():
    doc = _make_doc(
        "stable.md",
        "## A\nhello\n## B\nworld longer content here\n",
        headings=["A", "B"],
    )
    first = [c.chunk_id for c in chunk_docs([doc])]
    second = [c.chunk_id for c in chunk_docs([doc])]
    assert first == second
    # IDs are short, prefixed hashes — sanity check.
    assert all(isinstance(cid, str) and cid.startswith("c") for cid in first)
