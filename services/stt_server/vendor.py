"""Verify the vendored (patched) streaming STT server.

Stock upstream (ufal/whisper_streaming @ STT_SERVER_REV) sends plain-text
"beg end text" lines with no utterance-final signal, which
dmd/streaming_stt.py cannot use. The committed whisper_online_server.py
is that file plus a minimal JSON/finals patch (see its header). There is
nothing to copy at build time — this script only verifies the committed
file stays intact and reports its size for the manifest:

    python3 services/stt_server/vendor.py          # verify + report size
    python3 services/stt_server/vendor.py --check  # verify only (CI)
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

STT_SERVER_REV = "6da90b44b7e50d79695e68166d2a2c7609c75abb"
PRISTINE_SHA256 = "d89178d8a57c646ab46f76e9d4c6957e82c6e75fffb75438516ad81983101101"
SERVER_FILE = "whisper_online_server.py"

HERE = Path(__file__).resolve().parent


def check() -> Path:
    """Verify markers + compilability; return the server file path."""
    target = HERE / SERVER_FILE
    text = target.read_text(encoding="utf-8")
    for marker in (STT_SERVER_REV, PRISTINE_SHA256, "FAMILIAR PATCH", "--vac"):
        if marker not in text:
            raise SystemExit(f"{target.name}: missing marker {marker!r}")
    py_compile.compile(str(target), doraise=True)
    return target


def main(argv: list[str]) -> None:
    target = check()
    size = target.stat().st_size
    if "--check" not in argv:
        print(f"{target.name}: ok ({size} bytes, upstream {STT_SERVER_REV[:12]})")


if __name__ == "__main__":
    main(sys.argv[1:])
