"""Vendor the local-JEV scorer from the inspected ../openjev checkout.

The local JEV sidecar must reproduce openjev scoring EXACTLY (same chat
template, same single-forward letter-slot softmax, same /score + /health
wire protocol). Instead of reimplementing it, the build copies the
inspected modules verbatim and records their provenance.

Discipline: ``OPENJEV_REV`` pins the sibling commit this integration was
verified against. Vendoring refuses a sibling at any other revision —
an openjev change must be re-inspected (and the golden suite re-run)
before the pin is deliberately bumped here.

Usage:
    python3 services/jev_sidecar/vendor.py          # copy + provenance
    python3 services/jev_sidecar/vendor.py --check  # verify only (CI)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OPENJEV_REV = "b4782a6c953f05c6255706d7a219f4e032af5b58"
VENDOR_FILES = (
    "src/openjev_phase1/__init__.py",
    "src/openjev_phase1/core.py",
    "src/openjev_phase1/direct.py",
    "src/openjev_phase1/server.py",
    "LICENSE",
)

HERE = Path(__file__).resolve().parent
DEST = HERE / "_vendored"


def _sibling() -> Path:
    return HERE.parent.parent.parent / "openjev"


def _head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vendor() -> dict:
    """Copy pinned modules; return the provenance record."""
    sibling = _sibling()
    if not (sibling / ".git").exists():
        raise SystemExit(
            f"sibling openjev checkout not found at {sibling}\n"
            "Clone it next to familiar (it is a build-time dependency)."
        )
    head = _head(sibling)
    if head != OPENJEV_REV:
        raise SystemExit(
            f"sibling openjev is at {head[:12]}, want pinned {OPENJEV_REV[:12]}.\n"
            "Re-inspect its scoring path, re-run the JEV golden suite, then "
            "bump OPENJEV_REV in services/jev_sidecar/vendor.py deliberately."
        )
    DEST.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for rel in VENDOR_FILES:
        src = sibling / rel
        name = Path(rel).name
        if rel.startswith("src/openjev_phase1/") and name != "LICENSE":
            target = DEST / "openjev_phase1" / name
        else:
            target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
        files[str(target.relative_to(DEST))] = _sha256(target)
    provenance = {
        "source": "../openjev",
        "revision": OPENJEV_REV,
        "files": files,
        "vendored_at": datetime.now(timezone.utc).isoformat(),
        "license": "MIT (see _vendored/LICENSE)",
    }
    (DEST / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def check() -> None:
    """Verify the vendored tree matches the pinned sibling (no writes)."""
    prov_path = DEST / "PROVENANCE.json"
    if not prov_path.exists():
        raise SystemExit("no vendored tree; run vendor.py first")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    if prov.get("revision") != OPENJEV_REV:
        raise SystemExit(
            f"vendored {prov.get('revision')} != pinned {OPENJEV_REV}; re-vendor"
        )
    sibling = _sibling()
    for rel in VENDOR_FILES:
        name = Path(rel).name
        target = (
            DEST / "openjev_phase1" / name
            if rel.startswith("src/openjev_phase1/") and name != "LICENSE"
            else DEST / name
        )
        key = str(target.relative_to(DEST))
        if not target.exists() or _sha256(target) != prov["files"].get(key):
            raise SystemExit(f"vendored file drifted: {key}")
        if (sibling / rel).exists() and _head(sibling) == OPENJEV_REV:
            if (sibling / rel).read_bytes() != target.read_bytes():
                raise SystemExit(f"vendored file differs from sibling: {key}")
    print(f"vendored tree OK ({len(prov['files'])} files @ {OPENJEV_REV[:12]})")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check()
    else:
        prov = vendor()
        print(f"vendored {len(prov['files'])} files @ {OPENJEV_REV[:12]} -> {DEST}")


if __name__ == "__main__":
    main()
