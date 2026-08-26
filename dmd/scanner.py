"""Deterministic markdown / text repo scanning and chunking."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import List

import yaml

_MARKDOWN_EXTS = (".md", ".markdown", ".txt")
_FRONT_RE = re.compile(r"\A---[ \t]*\r?\n(.*?\r?\n)---[ \t]*\r?\n", re.DOTALL)
_ATX_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\n|]+?)(?:\|[^\[\]\n]+?)?\]\]")
_INLINE_TAG_RE = re.compile(r"(?:^|[\s(\[])#([A-Za-z][A-Za-z0-9_\-/]+)")


@dataclass(slots=True)
class DocFile:
    relpath: str
    title: str
    frontmatter: dict
    links: List[str]
    tags: List[str]
    headings: List[str]
    content: str
    mtime: float


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    heading_path: List[str]
    pos: int


def _hash_chunk(doc_path: str, heading_path: List[str], pos: int, text: str) -> str:
    h = hashlib.sha1()
    h.update(doc_path.encode("utf-8"))
    h.update(b"\x00")
    h.update("\x00".join(heading_path).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(pos).encode("utf-8"))
    h.update(b"\x00")
    h.update(text[:80].encode("utf-8"))
    return "c" + h.hexdigest()[:15]


def _split_sections(content: str) -> List[tuple[List[str], str]]:
    lines = content.splitlines(keepends=True)
    sections: List[tuple[List[str], str]] = []
    stack: List[tuple[int, str]] = []
    buf: List[str] = []
    emitted_pre = False

    def flush() -> None:
        nonlocal emitted_pre
        text = "".join(buf).strip("\n")
        buf.clear()
        if not text.strip():
            return
        path = [t for _, t in stack]
        if not path and not emitted_pre:
            sections.append(([], text))
            emitted_pre = True
        elif path:
            sections.append((list(path), text))

    for line in lines:
        m = _ATX_RE.match(line.rstrip("\n"))
        if m and len(m.group(1)) >= 2:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _hard_split(text: str, target: int, overlap: int) -> List[str]:
    if len(text) <= target:
        return [text]
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_re.split(text)
    if len(sentences) == 1:
        return [text[i : i + target] for i in range(0, len(text), target)]
    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
            continue
        if len(cur) + 1 + len(s) <= target:
            cur = cur + " " + s
        else:
            chunks.append(cur)
            tail_words = cur.split()
            keep = max(1, overlap // 5)
            tail = " ".join(tail_words[-keep:]) if len(tail_words) > keep else cur
            cur = (tail + " " + s).strip()
    if cur:
        chunks.append(cur)
    return chunks


def scan_folder(root: str) -> List[DocFile]:
    """Walk root for *.md/*.txt/*.markdown and parse each into a DocFile."""
    docs: List[DocFile] = []
    root = os.path.abspath(root)
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext not in _MARKDOWN_EXTS:
                continue
            full = os.path.join(dirpath, name)
            relpath = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                with open(full, "r", encoding="utf-8") as f:
                    raw = f.read()
                mtime = os.stat(full).st_mtime
            except OSError:
                continue

            fm: dict = {}
            body = raw
            m = _FRONT_RE.match(raw)
            if m:
                try:
                    parsed = yaml.safe_load(m.group(1))
                    if isinstance(parsed, dict):
                        fm = parsed
                except yaml.YAMLError:
                    fm = {}
                body = raw[m.end():]

            tags: List[str] = []
            fm_tags = fm.get("tags")
            if isinstance(fm_tags, list):
                for t in fm_tags:
                    if isinstance(t, str):
                        t = t.strip()
                        if t and t not in tags:
                            tags.append(t)
            elif isinstance(fm_tags, str):
                t = fm_tags.strip()
                if t and t not in tags:
                    tags.append(t)
            for t in _INLINE_TAG_RE.findall(body):
                if t not in tags:
                    tags.append(t)

            links = _WIKILINK_RE.findall(body)
            seen: set = set()
            unique_links: List[str] = []
            for ln in links:
                if ln not in seen:
                    seen.add(ln)
                    unique_links.append(ln)

            headings: List[str] = []
            for line in body.splitlines():
                mh = _ATX_RE.match(line)
                if mh:
                    headings.append(mh.group(2).strip())

            title = ""
            fm_title = fm.get("title")
            if isinstance(fm_title, str) and fm_title.strip():
                title = fm_title.strip()
            if not title:
                for line in body.splitlines():
                    mh = _ATX_RE.match(line)
                    if mh and len(mh.group(1)) == 1:
                        title = mh.group(2).strip()
                        break
            if not title:
                title = os.path.splitext(os.path.basename(relpath))[0]

            docs.append(
                DocFile(
                    relpath=relpath,
                    title=title,
                    frontmatter=fm,
                    links=unique_links,
                    tags=tags,
                    headings=headings,
                    content=body,
                    mtime=mtime,
                )
            )
    docs.sort(key=lambda d: d.relpath)
    return docs


def chunk_docs(
    docs: List[DocFile], target_chars: int = 800, overlap: int = 100
) -> List[Chunk]:
    """Split each doc into chunks, preferring markdown heading boundaries."""
    out: List[Chunk] = []
    for doc in docs:
        sections = _split_sections(doc.content)
        merged: List[tuple[List[str], str]] = []
        i = 0
        while i < len(sections):
            path, text = sections[i]
            if len(text.strip()) < 200 and i + 1 < len(sections):
                npath, ntext = sections[i + 1]
                merged.append((npath, text + "\n\n" + ntext))
                i += 2
            else:
                merged.append((path, text))
                i += 1

        pos = 0
        for path, text in merged:
            if len(text) > 1600:
                for piece in _hard_split(text, target_chars, overlap):
                    cid = _hash_chunk(doc.relpath, path, pos, piece)
                    out.append(
                        Chunk(
                            chunk_id=cid,
                            doc_id=doc.relpath,
                            text=piece,
                            heading_path=list(path),
                            pos=pos,
                        )
                    )
                    pos += 1
            else:
                cid = _hash_chunk(doc.relpath, path, pos, text)
                out.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=doc.relpath,
                        text=text,
                        heading_path=list(path),
                        pos=pos,
                    )
                )
                pos += 1
    return out