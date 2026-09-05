"""SPEC §5 / §16 doctrine guard: no baked game rules in product code.

The DM (LLM) never mutates canonical state and no rulebook is embedded: the
assistant synthesizes skill tables, rulings, and lore from the campaign repo
at runtime. This static guard fails the moment dice notation, a hardcoded
DC/AC/slot table, or any other rule constant is pasted into shipped code.

CSS hex colors are stripped first so `--bg-soft: #1d2230` cannot read as 1d2230.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE_SUFFIXES = {".py", ".js", ".css", ".html"}
SKIP_DIRS = {"__pycache__", "node_modules"}

# Numeric game-rule constructs: dice notation, DC/AC/THAC0 constants, and
# named rulebook mechanics with attached values.
RULE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d*d\d+\b"),
    re.compile(r"\b(?:DC|AC|CR)\s*[:=]?\s*\d+"),
    re.compile(r"\b(proficiency|saving throw|spell slot|hit dice|armor class)\b[^`\n]{0,40}\d", re.IGNORECASE),
]


def _source_files() -> list[Path]:
    out: list[Path] = []
    for base in (ROOT / "dmd", ROOT / "web"):
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix not in SOURCE_SUFFIXES:
                continue
            if SKIP_DIRS & set(p.parts):
                continue
            out.append(p)
    return out


def _strip_css_hex(text: str) -> str:
    return re.sub(r"#[0-9a-fA-F]{3,8}\b", "", text)


def test_no_baked_rule_values_in_product_code() -> None:
    offenders: list[str] = []
    for path in _source_files():
        text = _strip_css_hex(path.read_text(encoding="utf-8", errors="replace"))
        for pat in RULE_PATTERNS:
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line}: {m.group(0)!r}")
    assert not offenders, "baked game-rule constants in shipped code:\n" + "\n".join(offenders)


def test_prompt_templates_ask_for_rules_without_embedding_them() -> None:
    """The shipped prompts may *name* rule artifacts (DC, skill) but must not
    carry any numeric rule table — the guard above enforces this; this test
    pins the audit intent at the prompt layer specifically."""
    from dmd.agent import _SYSTEM_PROMPT
    from dmd.pipeline import _make_job  # noqa: F401  (import = shipped surface)
    from dmd.triggers import _CLASSIFIER_SYSTEM

    for prompt in (_SYSTEM_PROMPT, _CLASSIFIER_SYSTEM):
        assert not re.search(r"\d+d\d+|\bDC\s*\d", prompt), (
            "a shipped prompt embeds a rule value"
        )
