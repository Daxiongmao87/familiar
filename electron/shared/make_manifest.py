"""Generate electron/shared/manifest.json from verified upstream pins.

Every pin below was verified live against its authoritative source
(HuggingFace model API / GitHub API) on 2026-09-19. To update a
component: change the pin, re-run this script, commit the regenerated
manifest. The installer treats ``manifest_version`` bumps as migrations
and only downloads files whose size or revision changed.

Usage: python3 electron/shared/make_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_VERSION = 1

HF = "https://huggingface.co"

# whisper_streaming @ HEAD 2026-09-19 (GitHub API, tree sizes in bytes).
STT_SERVER_REV = "6da90b44b7e50d79695e68166d2a2c7609c75abb"
STT_SERVER_FILES = {
    "whisper_online.py": 38837,
    "silero_vad_iterator.py": 5899,
    "line_packet.py": 3201,
}
# Stock whisper_online_server.py sends plain text with no finals signal, so
# Familiar ships a patched JSON/finals variant instead (vendored, copied
# from app resources at install; size read from the committed file).
STT_SERVER_VENDORED = "services/stt_server/whisper_online_server.py"

# faster-whisper large-v3-turbo: canonical repo id after the
# mobiuslabsgmbh -> dropbox-dash move (HF API 307, followed 2026-09-19).
STT_MODEL_REPO = "dropbox-dash/faster-whisper-large-v3-turbo"
STT_MODEL_REV = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
STT_MODEL_FILES = {
    "model.bin": 1617884929,
    "config.json": 2263,
    "tokenizer.json": 2710337,
    "vocabulary.json": 1068114,
    "preprocessor_config.json": 340,
}

# Community MLC build (no official mlc-ai MiniCPM5 build exists as of
# 2026-09-19): model_type llama, q4f16_1, own webgpu wasm lib included.
SYNTH_MODEL_REPO = "ozhyhinas/MiniCPM5-2B-q4f16_1-MLC"
SYNTH_MODEL_REV = "602e6da83b3c3db304d34439fe968fd65f223e07"
SYNTH_MODEL_FILES = {
    "libs/MiniCPM5-2B-q4f16_1-MLC-webgpu.wasm": 6562687,
    "mlc-chat-config.json": 2158,
    "params_shard_0.bin": 133693440,
    "params_shard_1.bin": 133693440,
    "params_shard_2.bin": 33423360,
    "params_shard_3.bin": 28311552,
    "params_shard_4.bin": 28311552,
    "params_shard_5.bin": 28311552,
    "params_shard_6.bin": 28311552,
    "params_shard_7.bin": 28311552,
    "params_shard_8.bin": 28311552,
    "params_shard_9.bin": 28311552,
    "params_shard_10.bin": 28311552,
    "params_shard_11.bin": 28311552,
    "params_shard_12.bin": 28311552,
    "params_shard_13.bin": 28311552,
    "params_shard_14.bin": 28311552,
    "params_shard_15.bin": 28311552,
    "params_shard_16.bin": 28311552,
    "params_shard_17.bin": 28311552,
    "params_shard_18.bin": 28311552,
    "params_shard_19.bin": 28311552,
    "params_shard_20.bin": 28311552,
    "params_shard_21.bin": 28311552,
    "params_shard_22.bin": 28311552,
    "params_shard_23.bin": 28311552,
    "params_shard_24.bin": 28311552,
    "params_shard_25.bin": 28311552,
    "params_shard_26.bin": 28311552,
    "params_shard_27.bin": 28311552,
    "params_shard_28.bin": 28311552,
    "params_shard_29.bin": 28311552,
    "params_shard_30.bin": 28311552,
    "params_shard_31.bin": 28311552,
    "params_shard_32.bin": 28311552,
    "params_shard_33.bin": 28311552,
    "params_shard_34.bin": 31850496,
    "params_shard_35.bin": 32440320,
    "params_shard_36.bin": 32440320,
    "params_shard_37.bin": 32440320,
    "params_shard_38.bin": 32440320,
    "params_shard_39.bin": 33030144,
    "params_shard_40.bin": 33030144,
    "params_shard_41.bin": 9785344,
    "release-manifest.json": 6690,
    "tensor-cache.json": 171390,
    "tokenizer.json": 9894271,
    "tokenizer_config.json": 94391,
}

# Production JEV scorer: identical repo+revision to the golden fixture
# tests/regression/golden/jev_gate_v39.json (verified: HEAD == pin).
JEV_MODEL_REPO = "QuantTrio/Qwen3.5-4B-AWQ"
JEV_MODEL_REV = "32c292e3a73afe1138518180b1b6d2868c980ee2"
JEV_MODEL_FILES = {
    "model-00001-of-00003.safetensors": 2980701464,
    "model-00002-of-00003.safetensors": 2991143600,
    "model-00003-of-00003.safetensors": 99642624,
    "model.safetensors.index.json": 84925,
    "config.json": 3032,
    "tokenizer.json": 12807982,
    "tokenizer_config.json": 16710,
    "vocab.json": 6722759,
    "merges.txt": 3353259,
    "chat_template.jinja": 7756,
    "generation_config.json": 303,
    "configuration.json": 73,
    "preprocessor_config.json": 390,
    "video_preprocessor_config.json": 385,
}


def _hf_files(repo: str, rev: str, files: dict[str, int]) -> list[dict]:
    return [
        {
            "path": path,
            "url": f"{HF}/{repo}/resolve/{rev}/{path}",
            "size": size,
        }
        for path, size in sorted(files.items())
    ]


def _vendored_server_size() -> int:
    """Byte size of the committed patched server (manifest size check)."""
    root = Path(__file__).resolve().parent.parent.parent
    return (root / STT_SERVER_VENDORED).stat().st_size


def build_manifest() -> dict:
    """Build the manifest dict from the pins above."""
    ws_base = (
        "https://raw.githubusercontent.com/ufal/whisper_streaming"
        f"/{STT_SERVER_REV}"
    )
    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_by": "electron/shared/make_manifest.py",
        "components": [
            {
                "id": "stt-server",
                "label": "Speech recognition server",
                "kind": "files",
                "version": STT_SERVER_REV[:12],
                "source": f"github:ufal/whisper_streaming@{STT_SERVER_REV}",
                "install_dir": "stt/server",
                "requirements": "Bundled Python + faster-whisper env (stt-runtime).",
                "files": sorted(
                    [
                        {
                            "path": path,
                            "url": f"{ws_base}/{path}",
                            "size": size,
                        }
                        for path, size in STT_SERVER_FILES.items()
                    ]
                    + [
                        {
                            "path": "whisper_online_server.py",
                            "vendored": "stt_server/whisper_online_server.py",
                            "size": _vendored_server_size(),
                        }
                    ],
                    key=lambda f: f["path"],
                ),
            },
            {
                "id": "stt-runtime",
                "label": "Speech recognition runtime",
                "kind": "pip-env",
                # Full import closure of whisper_online.py at the pinned rev:
                # faster-whisper (backend) + librosa/soundfile (audio IO) +
                # torch/numpy (VAC path; torch/numpy aligned with the
                # openjev-verified stack). Verified install 2026-09-19.
                "version": "stt-pins-1",
                "source": "pypi",
                "install_dir": "stt/venv",
                "requirements": "NVIDIA CUDA GPU; network for pip; ~7 GB disk.",
                "packages": [
                    "faster-whisper==1.2.1",
                    "librosa==1.0.0",
                    "soundfile==0.14.0",
                    "numpy==2.2.6",
                    "torch==2.10.0",
                ],
            },
            {
                "id": "stt-model",
                "label": "Speech recognition model",
                "kind": "files",
                "version": STT_MODEL_REV[:12],
                "source": f"hf:{STT_MODEL_REPO}@{STT_MODEL_REV}",
                "install_dir": "stt/model",
                "requirements": "~1.7 GB disk.",
                "files": _hf_files(STT_MODEL_REPO, STT_MODEL_REV, STT_MODEL_FILES),
            },
            {
                "id": "synth-model",
                "label": "MiniCPM5-2B local synthesis model",
                "kind": "files",
                "version": SYNTH_MODEL_REV[:12],
                "source": f"hf:{SYNTH_MODEL_REPO}@{SYNTH_MODEL_REV}",
                "install_dir": "synth/minicpm5-2b-mlc",
                "requirements": "WebGPU with shader-f16; ~1.5 GB disk.",
                "files": _hf_files(SYNTH_MODEL_REPO, SYNTH_MODEL_REV, SYNTH_MODEL_FILES),
            },
            {
                "id": "jev-model",
                "label": "Qwen 4B AWQ JEV model",
                "kind": "files",
                "version": JEV_MODEL_REV[:12],
                "revision": JEV_MODEL_REV,
                "source": f"hf:{JEV_MODEL_REPO}@{JEV_MODEL_REV}",
                "install_dir": "jev/model",
                "requirements": "NVIDIA CUDA GPU; ~6.1 GB disk.",
                "files": _hf_files(JEV_MODEL_REPO, JEV_MODEL_REV, JEV_MODEL_FILES),
            },
            {
                "id": "jev-runtime",
                "label": "OpenJEV runtime",
                "kind": "pip-env",
                "version": "openjev-pins-1",
                "source": "pypi",
                "install_dir": "jev/venv",
                "requirements": "NVIDIA CUDA GPU; ~8 GB disk for torch/CUDA wheels.",
                "packages": [
                    "torch==2.10.0",
                    "transformers==5.17.0",
                    "accelerate==1.15.0",
                    "gptqmodel==7.5.0",
                    "safetensors==0.8.0",
                    "huggingface-hub==1.31.0",
                    "tokenizers==0.23.2",
                    "numpy==2.2.6",
                    "sentencepiece==0.2.1",
                    "protobuf==7.36.1",
                ],
            },
        ],
    }


def main() -> None:
    """Write manifest.json next to this script."""
    out = Path(__file__).resolve().parent / "manifest.json"
    out.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
