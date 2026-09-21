/** Tests for bounded, opt-in, redacted desktop diagnostic logging. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { createDebugLogger, debugRequested, redact } = require('../debug_log.js');

test('debug mode requires an explicit flag or environment opt-in', () => {
  assert.equal(debugRequested(['Familiar'], {}), false);
  assert.equal(debugRequested(['Familiar', '--debug'], {}), false);
  assert.equal(debugRequested(['Familiar', '--familiar-debug'], {}), true);
  assert.equal(debugRequested(['Familiar'], { FAMILIAR_DEBUG: '1' }), true);
});

test('redaction removes bearer and key-shaped credential values', () => {
  const input = 'Authorization: Bearer abc.def token="discord.secret" api_key=sk-live';
  const out = redact(input);
  assert.doesNotMatch(out, /abc\.def|discord\.secret|sk-live/);
  assert.match(out, /\[REDACTED\]/);
});

test('enabled logger persists diagnostics and rotates a bounded backup', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'familiar-debug-'));
  const logDir = path.join(root, 'logs');
  fs.mkdirSync(logDir);
  const logFile = path.join(logDir, 'familiar-debug.log');
  const oldLog = 'old-log-that-must-rotate'.repeat(30);
  fs.writeFileSync(logFile, oldLog);

  const logger = createDebugLogger({ enabled: true, userData: root, maxBytes: 512 });
  logger.log('backend', 'stderr', 'token=do-not-write status failed');

  assert.equal(logger.logFile, logFile);
  assert.equal(fs.readFileSync(`${logFile}.1`, 'utf-8'), oldLog);
  const current = fs.readFileSync(logFile, 'utf-8');
  assert.match(current, /debug_started/);
  assert.match(current, /status failed/);
  assert.doesNotMatch(current, /do-not-write/);
  fs.rmSync(root, { recursive: true, force: true });
});

test('disabled logger writes no files', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'familiar-debug-off-'));
  const logger = createDebugLogger({ enabled: false, userData: root });
  logger.log('electron', 'ignored');
  assert.equal(fs.existsSync(path.join(root, 'logs')), false);
  fs.rmSync(root, { recursive: true, force: true });
});
