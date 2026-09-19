/** STT and JEV sidecar launchers (pip-env creation + server argv).
 *
 * Both sidecars run from per-user virtualenvs created at first run from
 * the bundled standalone CPython (dev: system python3). Packages are the
 * manifest pins — never unpinned latest — so installs reproduce.
 *
 * - STT: pinned whisper_online_server.py + faster-whisper large-v3-turbo
 *   model dir; serves the TCP protocol dmd/streaming_stt.py speaks.
 * - JEV: vendored openjev-serve-compatible scorer (services/jev_sidecar/,
 *   copied from ../openjev at build) + pinned AWQ weights; serves the
 *   identical /score + /health protocol on jev_local_url's port.
 */
'use strict';

const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

/**
 * Locate a base interpreter for sidecar venvs.
 * @param {string} resourcesPath Electron process.resourcesPath.
 * @returns {string} python executable path.
 */
function basePython(resourcesPath) {
  if (process.env.FAMILIAR_DEV === '1') return 'python3';
  const bundled = path.join(resourcesPath, 'python', 'bin', 'python3');
  if (fs.existsSync(bundled)) return bundled;
  return 'python3'; // last resort: PATH python3 (reported in setup log)
}

/**
 * Create (or reuse) a venv and pip-install pinned packages.
 * Progress and failures stream through onLog for the setup UI.
 * @param {string} python base interpreter.
 * @param {string} venvDir destination venv dir.
 * @param {string[]} packages pip pins, e.g. ["torch==2.10.0"].
 * @param {(line:string)=>void} onLog
 */
function ensureVenv(python, venvDir, packages, onLog) {
  const log = onLog || (() => {});
  const venvPy = path.join(venvDir, 'bin', 'python');
  if (!fs.existsSync(venvPy)) {
    log(`creating venv ${venvDir}`);
    const made = spawnSync(python, ['-m', 'venv', venvDir], { encoding: 'utf-8' });
    if (made.status !== 0) {
      throw new Error(`venv creation failed: ${(made.stderr || '').slice(0, 500)}`);
    }
  }
  const marker = path.join(venvDir, 'familiar-pins.txt');
  const want = `${packages.join('\n')}\n`;
  if (fs.existsSync(marker) && fs.readFileSync(marker, 'utf-8') === want) {
    log(`venv up to date (${packages.length} pins)`);
    return venvPy;
  }
  log(`pip install ${packages.length} pinned packages…`);
  const pip = spawnSync(
    venvPy, ['-m', 'pip', 'install', '--disable-pip-version-check', ...packages],
    { encoding: 'utf-8', maxBuffer: 8 * 1024 * 1024 },
  );
  if (pip.status !== 0) {
    throw new Error(`pip install failed: ${(pip.stderr || pip.stdout || '').slice(0, 800)}`);
  }
  fs.writeFileSync(marker, want, 'utf-8');
  return venvPy;
}

/**
 * STT server argv: whisper_online_server.py from the pinned download.
 * @param {string} venvPy sidecar venv python.
 * @param {string} serverDir downloaded stt/server dir.
 * @param {string} modelDir downloaded stt/model dir.
 * @param {number} port TCP port (matches config stream_port).
 * @returns {{command:string,args:string[]}}
 */
function sttArgv(venvPy, serverDir, modelDir, port) {
  return {
    command: venvPy,
    args: [path.join(serverDir, 'whisper_online_server.py'),
      '--host', '127.0.0.1', '--port', String(port),
      '--model', 'large-v3-turbo', '--model_dir', modelDir,
      '--backend', 'faster-whisper'],
  };
}

/**
 * JEV sidecar argv: vendored scorer + pinned AWQ weights.
 * @param {string} venvPy sidecar venv python.
 * @param {string} sidecarDir vendored jev_sidecar code dir.
 * @param {string} modelDir downloaded jev/model dir.
 * @param {number} port serve port (matches desktop.jev_local_url).
 * @param {string} revision weights revision label (manifest pin, for /health).
 * @returns {{command:string,args:string[],env:Object}}
 */
function jevArgv(venvPy, sidecarDir, modelDir, port, revision) {
  return {
    command: venvPy,
    args: [path.join(sidecarDir, 'serve.py'),
      '--host', '127.0.0.1', '--port', String(port),
      '--model', modelDir,
      '--revision', revision || 'local-pinned-awq'],
    env: { CUDA_VISIBLE_DEVICES: process.env.CUDA_VISIBLE_DEVICES || '0' },
  };
}

/**
 * Wait for an HTTP health endpoint to answer 200.
 * @param {string} url e.g. http://127.0.0.1:8299/health.
 * @param {number} timeoutMs total budget.
 * @returns {Promise<boolean>}
 */
async function waitForHttp(url, timeoutMs) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return true;
    } catch { /* keep waiting */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

/**
 * Wait for a TCP port to accept (STT server has no HTTP health).
 * @param {string} host
 * @param {number} port
 * @param {number} timeoutMs total budget.
 * @returns {Promise<boolean>}
 */
async function waitForTcp(host, port, timeoutMs) {
  const net = require('node:net');
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const ok = await new Promise((resolve) => {
      const sock = net.connect(port, host);
      sock.on('connect', () => { sock.destroy(); resolve(true); });
      sock.on('error', () => resolve(false));
      setTimeout(() => { sock.destroy(); resolve(false); }, 1000);
    });
    if (ok) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

module.exports = { basePython, ensureVenv, sttArgv, jevArgv, waitForHttp, waitForTcp };
