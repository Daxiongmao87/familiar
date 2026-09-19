"""Local JEV sidecar: openjev-serve wire protocol over pinned AWQ weights.

Thin CLI over the vendored openjev modules (services/jev_sidecar/_vendored/,
see vendor.py): identical /score + /health behavior to the remote scorer,
serving the downloaded local weights. No reimplementation, no drift.

Requires one visible CUDA GPU (AWQ kernels), like upstream openjev.
CPU-only systems must use the remote JEV provider.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendored"))

from openjev_phase1.core import load_causal_model
from openjev_phase1.direct import score as direct_score
from openjev_phase1.server import WARMUP_ROW, build_server


def main() -> None:
    """Serve /score + /health from local weights (openjev-serve parity)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8299)
    parser.add_argument("--model", required=True, help="local AWQ weights dir")
    parser.add_argument("--revision", required=True, help="weights revision label")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    print(f"loading {args.model} (rev {args.revision[:12]})...", flush=True)
    model, tokenizer, metadata = load_causal_model(args.model, args.revision)

    def scorer(row: dict) -> dict:
        return direct_score(model, tokenizer, row, metadata, args.max_tokens)

    warm = scorer(WARMUP_ROW)
    print(f"warmup done in {warm['total_seconds']:.1f}s", flush=True)
    server = build_server(args.host, args.port, scorer, {"model": metadata})
    print(f"listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
