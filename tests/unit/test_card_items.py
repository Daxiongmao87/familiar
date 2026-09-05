"""Item normalization: skill-check tables written into body_md must become
addressable structured items with dc_find (SPEC §9; owner-verified 2026-09-05:
loot cards carried tables in body_md but items:[] — the monitor could not
match "we find the silver dagger" to the card that listed it).
"""

from __future__ import annotations

from dmd.agent import _items_from_table


def test_items_from_clean_table() -> None:
    md = (
        "| Find | Skill / DC | Qty | Value | Notes |\n"
        "|---|---|---|---|---|\n"
        "| Hidden pouch | Perception DC 14 | 1 | 150 gp | In his coat |\n"
        "| Silver dagger | Investigation DC 18 | 1 | 75 gp | |\n"
    )
    items = _items_from_table(md)
    assert len(items) == 2
    assert items[0]["name"] == "Hidden pouch"
    assert items[0]["dc_find"] == 14
    assert items[0]["quantity"] == 1
    assert items[1]["name"] == "Silver dagger"
    assert items[1]["dc_find"] == 18


def test_items_from_malformed_leading_pipe() -> None:
    # ling-tiny over-formats tables with a doubled leading pipe; the extractor
    # must tolerate the blank first cell.
    md = (
        "| | Find | Skill / DC | Qty | Value | Notes | |\n"
        "| |---|---|---|---|---|---| |\n"
        "| | Pouch | Perception DC 12 | 2 | 10 gp | | |\n"
    )
    items = _items_from_table(md)
    assert len(items) == 1
    assert items[0]["name"] == "Pouch"
    assert items[0]["dc_find"] == 12
    assert items[0]["quantity"] == 2


def test_items_from_no_table_is_empty() -> None:
    assert _items_from_table("No table here\njust prose") == []
    assert _items_from_table("") == []


def test_items_dc_none_when_no_number() -> None:
    md = (
        "| Find | Skill | Qty |\n"
        "|---|---|---|\n"
        "| Tracks | Survival | 1 |\n"
    )
    items = _items_from_table(md)
    assert items == [] or items[0]["dc_find"] is None
