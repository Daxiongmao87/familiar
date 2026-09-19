/** Supervisor tests: exit reporting, restart budget, clean shutdown. */
'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { supervise } = require('../supervise.js');

function events() {
  const seen = [];
  return { seen, onEvent: (ev) => seen.push(ev) };
}

function waitFor(seen, type, timeoutMs) {
  const t0 = Date.now();
  return new Promise((resolve, reject) => {
    (function poll() {
      if (seen.some((e) => e.type === type)) return resolve(true);
      if (Date.now() - t0 > (timeoutMs || 15000)) return reject(new Error(`no ${type}`));
      setTimeout(poll, 50);
    })();
  });
}

test('unexpected exit is reported and the child restarts', async () => {
  const ev = events();
  const sup = supervise({
    name: 'flaky', command: process.execPath, args: ['-e', 'process.exit(3)'],
    onEvent: ev.onEvent,
  });
  sup.start();
  await waitFor(ev.seen, 'exited', 10000);
  const exited = ev.seen.find((e) => e.type === 'exited');
  assert.equal(exited.code, 3);
  assert.equal(exited.name, 'flaky');
  await waitFor(ev.seen, 'restarting', 10000);
  const t0 = Date.now();
  while (sup.state().starts < 2 && Date.now() - t0 < 10000) {
    await new Promise((r) => setTimeout(r, 50));
  }
  assert.ok(sup.state().starts >= 2);
  await sup.stop();
});

test('hot-loop exits give up with a manual-restart report', async () => {
  const ev = events();
  const sup = supervise({
    name: 'looper', command: process.execPath, args: ['-e', 'process.exit(1)'],
    onEvent: ev.onEvent,
  });
  sup.start();
  await waitFor(ev.seen, 'restart_giveup', 30000);
  const giveup = ev.seen.find((e) => e.type === 'restart_giveup');
  assert.match(giveup.detail, /manual restart/);
  await sup.stop();
});

test('stop() terminates a running child (no orphan)', async () => {
  const ev = events();
  const sup = supervise({
    name: 'sleeper', command: process.execPath,
    args: ['-e', 'setInterval(() => {}, 1000)'],
    onEvent: ev.onEvent,
  });
  sup.start();
  await new Promise((r) => setTimeout(r, 400));
  const pid = sup.state().pid;
  assert.ok(pid);
  await sup.stop();
  assert.equal(sup.state().running, false);
  assert.throws(() => process.kill(pid, 0), /ESRCH/,
    'child process must be gone after stop()');
});

test('restart() respawns a dead child and clears the budget', async () => {
  const ev = events();
  const sup = supervise({
    name: 'once', command: process.execPath, args: ['-e', 'process.exit(7)'],
    autoRestart: false, onEvent: ev.onEvent,
  });
  sup.start();
  await waitFor(ev.seen, 'exited', 10000);
  assert.equal(sup.state().running, false);
  sup.restart();
  await new Promise((r) => setTimeout(r, 600));
  assert.ok(sup.state().starts >= 2);
  await sup.stop();
});

test('missing executable reports spawn_error', async () => {
  const ev = events();
  const sup = supervise({
    name: 'missing', command: '/nonexistent/familiar-test-binary-xyz',
    autoRestart: false, onEvent: ev.onEvent,
  });
  sup.start();
  await waitFor(ev.seen, 'spawn_error', 10000);
  await sup.stop();
});
