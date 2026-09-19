/** Generic child-process supervisor: spawn, exit reporting, clean shutdown.
 *
 * One policy for the Python backend and both sidecars: unexpected exits
 * are reported (never silently respawned in a hot loop — max 3 restarts
 * inside 60 s, then the app surfaces the failure with a manual restart
 * affordance). Normal app exit kills the whole tree (SIGTERM, SIGKILL
 * after a grace period) so no inference/STT server is ever orphaned.
 */
'use strict';

const { spawn } = require('node:child_process');

const RESTART_WINDOW_MS = 60_000;
const MAX_RESTARTS = 3;
const KILL_GRACE_MS = 5000;

/**
 * Supervise one child process.
 * @param {object} opts
 * @param {string} opts.name label for logs/events.
 * @param {string} opts.command executable.
 * @param {string[]} [opts.args]
 * @param {object} [opts.env]
 * @param {string} [opts.cwd] working dir for the child.
 * @param {object} [opts.onEvent] sink: ({type,name,...}) => void.
 * @param {boolean} [opts.autoRestart=true]
 * @returns {{start:()=>void, stop:()=>Promise<void>, state:()=>object}}
 */
function supervise(opts) {
  const { name, command } = opts;
  const args = opts.args || [];
  const onEvent = opts.onEvent || (() => {});
  const autoRestart = opts.autoRestart !== false;
  let child = null;
  let stopping = false;
  let restarts = [];
  let startCount = 0;

  function emit(type, extra) {
    onEvent(Object.assign({ type, name }, extra || {}));
  }

  function start() {
    if (child || stopping) return;
    startCount += 1;
    emit('starting', { command, args, attempt: startCount });
    child = spawn(command, args, {
      env: Object.assign({}, process.env, opts.env || {}),
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: process.platform !== 'win32',
      cwd: opts.cwd || undefined,
    });
    const startedAt = Date.now();
    if (child.stdout) child.stdout.on('data', (d) => emit('stdout', { text: d.toString().slice(0, 2000) }));
    if (child.stderr) child.stderr.on('data', (d) => emit('stderr', { text: d.toString().slice(0, 2000) }));
    child.on('error', (err) => emit('spawn_error', { error: String(err) }));
    child.on('exit', (code, signal) => {
      const livedMs = Date.now() - startedAt;
      child = null;
      if (stopping) {
        emit('stopped', { code, signal });
        return;
      }
      emit('exited', { code, signal, livedMs });
      if (!autoRestart) return;
      const now = Date.now();
      restarts = restarts.filter((t) => now - t < RESTART_WINDOW_MS).concat([now]);
      if (restarts.length > MAX_RESTARTS) {
        emit('restart_giveup', { detail: `exited ${restarts.length} times in 60 s; manual restart required` });
        return;
      }
      emit('restarting', { inMs: 1000 });
      setTimeout(() => { if (!stopping) start(); }, 1000);
    });
  }

  async function stop() {
    stopping = true;
    if (!child) return;
    const proc = child;
    emit('stopping', { pid: proc.pid });
    try {
      if (proc.pid && process.platform !== 'win32') {
        process.kill(-proc.pid, 'SIGTERM'); // whole group: no orphans
      } else {
        proc.kill('SIGTERM');
      }
    } catch { /* already gone */ }
    const exited = await new Promise((resolve) => {
      const timer = setTimeout(() => resolve(false), KILL_GRACE_MS);
      proc.on('exit', () => { clearTimeout(timer); resolve(true); });
    });
    if (!exited) {
      try {
        if (proc.pid && process.platform !== 'win32') process.kill(-proc.pid, 'SIGKILL');
        else proc.kill('SIGKILL');
      } catch { /* already gone */ }
    }
    child = null;
  }

  function state() {
    return { name, running: !!child, pid: child ? child.pid : null, starts: startCount };
  }

  function restart() {
    // Manual restart: kill the child and let the exit handler respawn it
    // (counts toward the restart budget like any other unexpected exit).
    // If the supervisor gave up earlier, clear the budget and start fresh.
    restarts = [];
    if (child) {
      emit('restart_requested', { pid: child.pid });
      try {
        child.kill('SIGTERM');
      } catch { /* already gone; exit handler still fires */ }
    } else if (!stopping) {
      start();
    }
  }

  return { start, stop, state, restart };
}

module.exports = { supervise };
