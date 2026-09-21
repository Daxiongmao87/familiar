/** Opt-in, bounded, redacted diagnostic logging for the desktop shell.
 *
 * Debug mode is enabled with ``--familiar-debug`` (or ``FAMILIAR_DEBUG=1``).
 * Logs live under Electron's per-user data directory so a headless support
 * session can retrieve them without access to the GUI.
 */
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const MAX_BYTES = 5 * 1024 * 1024;

/** Return whether persistent diagnostics were explicitly requested. */
function debugRequested(argv, env) {
  const args = argv || process.argv;
  const vars = env || process.env;
  return args.includes('--familiar-debug') || vars.FAMILIAR_DEBUG === '1';
}

/** Remove common credential forms before any text reaches disk. */
function redact(value) {
  let text = typeof value === 'string' ? value : JSON.stringify(value);
  if (text === undefined) text = String(value);
  text = text.replace(/\bBearer\s+[^\s"']+/gi, 'Bearer [REDACTED]');
  text = text.replace(
    /((?:api[_-]?key|authorization|password|secret|token)["']?\s*[:=]\s*["']?)[^\s,"'}]+/gi,
    '$1[REDACTED]',
  );
  return text;
}

/** Rotate one bounded backup before appending a new diagnostic session. */
function rotate(logFile, maxBytes) {
  try {
    if (!fs.existsSync(logFile) || fs.statSync(logFile).size < maxBytes) return;
    const backup = `${logFile}.1`;
    fs.rmSync(backup, { force: true });
    fs.renameSync(logFile, backup);
  } catch { /* diagnostics must never prevent startup */ }
}

/** Create the no-op or persistent desktop diagnostic sink. */
function createDebugLogger(opts) {
  const enabled = !!opts.enabled;
  const logDir = path.join(opts.userData, 'logs');
  const logFile = path.join(logDir, 'familiar-debug.log');
  if (!enabled) {
    return { enabled: false, logFile, log: () => {} };
  }
  try {
    fs.mkdirSync(logDir, { recursive: true, mode: 0o700 });
    rotate(logFile, opts.maxBytes || MAX_BYTES);
    if (fs.existsSync(logFile)) fs.chmodSync(logFile, 0o600);
  } catch { /* append below remains best-effort */ }

  function log(source, event, detail) {
    try {
      rotate(logFile, opts.maxBytes || MAX_BYTES);
      const suffix = detail === undefined ? '' : ` ${redact(detail)}`;
      fs.appendFileSync(
        logFile,
        `${new Date().toISOString()} [${redact(source)}] ${redact(event)}${suffix}\n`,
        { encoding: 'utf-8', mode: 0o600 },
      );
    } catch { /* diagnostics must never prevent startup */ }
  }

  log('electron', 'debug_started', { pid: process.pid, argv: process.argv.slice(1) });
  return { enabled: true, logFile, log };
}

module.exports = { MAX_BYTES, createDebugLogger, debugRequested, redact, rotate };
