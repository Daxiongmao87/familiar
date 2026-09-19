#!/usr/bin/env bash
# Bundled SearXNG for the familiar project (owner directive 2026-09-05).
#
# Runs the project's own SearXNG instance (installed into .venv-searxng from
# .searxng-src) bound to 127.0.0.1:8888. The dmd worker agents search through
# this instance (dmd/config.py SearchConfig.endpoint) with DDG scrape as the
# automatic fallback.
#
# Usage:
#   tools/searxng.sh start     launch in background, pidfile in .searxng/
#   tools/searxng.sh stop      stop the instance
#   tools/searxng.sh status    report pid + health
#   tools/searxng.sh log       tail the instance log

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
VENV="$ROOT/.venv-searxng"
SRC="$ROOT/.searxng-src"
RUN_DIR="$ROOT/.searxng"
PIDFILE="$RUN_DIR/searxng.pid"
LOGFILE="$RUN_DIR/searxng.log"
PORT="${SEARXNG_PORT:-8888}"

if [[ ! -x "$VENV/bin/searxng-run" ]]; then
  echo "error: .venv-searxng not built — run: python3 -m venv .venv-searxng && .venv-searxng/bin/pip install -e .searxng-src/" >&2
  exit 2
fi

# Project-owned settings: SearXNG looks for SEARXNG_SETTINGS_PATH or the
# searxng/settings.yml beside the source. We layer a minimal override file
# that pins localhost-only binding + JSON API + disabled limiter (safe for a
# loopback-only instance used by bots).
if [[ ! -f "$RUN_DIR/settings.yml" ]]; then
  mkdir -p "$RUN_DIR"
  cat > "$RUN_DIR/settings.yml" <<'YAML'
use_default_settings: true
server:
  secret_key: "familiar-bundled-searxng-local-only-2026"
  bind_address: "127.0.0.1"
  port: 8888
  limiter: false
  public_instance: false
search:
  formats:
    - html
    - json
YAML
fi

start() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running pid $(cat "$PIDFILE")"
    exit 0
  fi
  mkdir -p "$RUN_DIR"
  export SEARXNG_SETTINGS_PATH="$RUN_DIR/settings.yml"
  nohup "$VENV/bin/searxng-run" >> "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  echo "started pid $(cat "$PIDFILE")"
  for _ in $(seq 1 30); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/search?q=test&format=json" 2>/dev/null; then
      echo "health OK at http://127.0.0.1:$PORT"
      exit 0
    fi
    sleep 1
  done
  echo "warning: instance up but health check not confirmed; log: $LOGFILE" >&2
}

stop() {
  if [[ ! -f "$PIDFILE" ]]; then
    echo "not running"
    exit 0
  fi
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "stopped"
}

status() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "running pid $(cat "$PIDFILE")"
    curl -sf -o /dev/null -w "health: HTTP %{http_code}\n" "http://127.0.0.1:$PORT/search?q=test&format=json" 2>/dev/null || echo "health: DOWN"
  else
    echo "not running"
    exit 1
  fi
}

log() { tail -n 60 "$LOGFILE"; }

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  log) log ;;
  *) echo "usage: $0 {start|stop|status|log}" >&2; exit 2 ;;
esac
