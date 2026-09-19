#!/usr/bin/env bash
# Build Familiar-<version>.AppImage reproducibly.
#
# Prerequisites (one-time, documented in docs/DESKTOP.md):
#   - Node 20+ and npm
#   - Python 3.12 with the project .venv installed (requirements.txt + PyInstaller)
#   - sibling ../openjev checkout at the pinned revision (JEV sidecar vendor)
#   - FUSE tooling for AppImage runtime test (optional): libfuse2
#
# Usage: packaging/build_appimage.sh [--skip-backend] [--skip-python]
# Output: electron/dist/Familiar-<version>.AppImage
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO/packaging/appimage-work"
VERSION="$(node -p "require('$REPO/electron/package.json').version")"
SKIP_BACKEND=0
SKIP_PYTHON=0
for arg in "$@"; do
  case "$arg" in
    --skip-backend) SKIP_BACKEND=1 ;;
    --skip-python) SKIP_PYTHON=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

PYTHON_TAG="20260901"
PYTHON_FILE="cpython-3.12.14+${PYTHON_TAG}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_TAG}/${PYTHON_FILE//+/%2B}"

step() { echo "==> $*"; }
fail() { echo "BUILD FAILED: $*" >&2; exit 1; }

cd "$REPO"

step "1/6 vendoring JEV sidecar (pinned ../openjev) + STT server (patched)"
python3 services/jev_sidecar/vendor.py
python3 services/jev_sidecar/vendor.py --check
python3 services/stt_server/vendor.py --check

step "2/6 installing electron deps + running unit tests"
(
  cd electron
  npm ci --no-audit --no-fund
  npm test
  npm run bundle:webllm
)

step "3/6 running python unit tests"
if ! .venv/bin/python -m pytest tests/unit -q -p no:warnings 2>&1 | tail -n 2; then
  echo "unit tests failed once; retrying after load settles (timing flakes)" >&2
  sleep 10
  .venv/bin/python -m pytest tests/unit -q -p no:warnings 2>&1 | tail -n 2 \
    || fail "python unit tests failed twice"
fi

if [ "$SKIP_BACKEND" = "1" ]; then
  step "4/6 SKIPPED backend bundle (--skip-backend)"
else
  step "4/6 bundling Python backend with PyInstaller"
  # Hermetic build venv: the dev .venv may leak user-site packages
  # (torch etc.) into the bundle via over-broad hooks, so the bundle is
  # always built from a fresh venv containing exactly requirements.txt.
  if [ ! -x "$WORK/build-venv/bin/python" ]; then
    python3 -m venv "$WORK/build-venv" || fail "cannot create build venv"
  fi
  "$WORK/build-venv/bin/python" -m pip install -q -r "$REPO/requirements.txt" \
    "pyinstaller>=6" || fail "cannot install build deps"
  rm -rf "$WORK/backend"
  mkdir -p "$WORK/backend"
  (cd "$WORK/backend" && "$WORK/build-venv/bin/python" -m PyInstaller \
    --distpath "$WORK/backend/dist" --workpath "$WORK/backend/build" \
    --clean "$REPO/packaging/backend.spec")
  [ -x "$WORK/backend/dist/familiar-backend/familiar-backend" ] \
    || fail "backend bundle missing"
  [ -f "$WORK/backend/dist/familiar-backend/_internal/web/index.html" ] \
    || fail "backend bundle missing web UI"
fi

if [ "$SKIP_PYTHON" = "1" ]; then
  step "5/6 SKIPPED standalone python (--skip-python)"
else
  step "5/6 staging standalone CPython for sidecar venvs"
  rm -rf "$WORK/python" "$WORK/py.tgz"
  mkdir -p "$WORK/python"
  curl -fL -o "$WORK/py.tgz" "$PYTHON_URL" || fail "python download failed"
  tar xzf "$WORK/py.tgz" -C "$WORK/python" --strip-components=1
  "$WORK/python/bin/python3" --version || fail "staged python broken"
  "$WORK/python/bin/python3" -m venv --without-pip "$WORK/venv-probe" \
    || fail "staged python cannot create venvs"
  rm -rf "$WORK/venv-probe"
fi

step "6/6 building AppImage with electron-builder"
(
  cd electron
  npx electron-builder --linux AppImage --config "$REPO/packaging/electron-builder.yml"
)

APPIMAGE="$REPO/electron/dist/Familiar-${VERSION}.AppImage"
[ -f "$APPIMAGE" ] || fail "AppImage not produced"
chmod +x "$APPIMAGE"
echo "OK: $APPIMAGE ($(du -h "$APPIMAGE" | cut -f1))"
