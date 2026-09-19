/** Bundled-Python-backend resolution and user config seeding.
 *
 * Packaged AppImage: the backend is a PyInstaller one-dir bundle under
 * `<resources>/backend/` exposing `familiar-backend` (+ a `config.example.yaml`
 * seed). Dev (`FAMILIAR_DEV=1`): the repo checkout's `.venv` python runs
 * `dmd/server.py` so the shell can be developed without a full build.
 *
 * The backend config lives OUTSIDE the AppImage at
 * `<userData>/familiar-config.yaml`, seeded once from the packaged example
 * (example only — a user's real config.yaml, with secrets, is never bundled
 * and never overwritten once seeded).
 */
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const BACKEND_PORT_DEFAULT = 8760;

/**
 * Resolve how to launch the backend.
 * @param {object} opts
 * @param {string} opts.resourcesPath Electron process.resourcesPath.
 * @param {string} opts.userData app.getPath('userData').
 * @param {string} opts.repoRoot familiar checkout (dev only).
 * @returns {{command:string,args:string[],configPath:string,backendHost:string,backendPort:number,baseUrl:string,env:Object}}
 */
function resolveBackend(opts) {
  const userData = opts.userData;
  const configPath = path.join(userData, 'familiar-config.yaml');
  const backendHost = '127.0.0.1';
  const backendPort = parseInt(process.env.FAMILIAR_BACKEND_PORT || '', 10)
    || BACKEND_PORT_DEFAULT;
  const baseUrl = `http://${backendHost}:${backendPort}`;
  const commonEnv = {
    DMD_HOST: backendHost,
    DMD_PORT: String(backendPort),
    // Keep fastembed/HF caches inside the per-user data dir (offline reuse).
    HF_HOME: path.join(userData, 'hf-cache'),
    FAMILIAR_DESKTOP: '1',
  };
  if (process.env.FAMILIAR_DEV === '1') {
    const root = opts.repoRoot;
    const venvPy = path.join(root, '.venv', 'bin', 'python');
    const command = fs.existsSync(venvPy) ? venvPy : 'python3';
    return {
      command, configPath, backendHost, backendPort, baseUrl,
      args: [path.join(root, 'dmd', 'server.py'), configPath],
      env: Object.assign({}, commonEnv, {
        PYTHONPATH: root + (process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : ''),
      }),
    };
  }
  const backendDir = path.join(opts.resourcesPath, 'backend');
  return {
    command: path.join(backendDir, 'familiar-backend'),
    configPath, backendHost, backendPort, baseUrl,
    args: [configPath],
    env: commonEnv,
  };
}

/**
 * Seed the user config from the packaged example when absent. Never
 * overwrites an existing file (it may hold secrets and tuned URLs).
 * Forces desktop.enabled=true on the seeded copy.
 * @param {string} examplePath packaged config.example.yaml.
 * @param {string} configPath user config destination.
 * @returns {'seeded'|'exists'}
 */
function ensureUserConfig(examplePath, configPath) {
  if (fs.existsSync(configPath)) return 'exists';
  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  const text = fs.readFileSync(examplePath, 'utf-8');
  fs.writeFileSync(configPath, enableDesktop(text), 'utf-8');
  return 'seeded';
}

/**
 * Flip a template-shaped config into desktop mode (line surgery; the
 * desktop block may carry trailing comments). Exactly one desktop block
 * exists afterwards — never a duplicate.
 * @param {string} text config file text.
 * @returns {string} text with `desktop.enabled: true`.
 */
function enableDesktop(text) {
  const lines = text.split('\n');
  const di = lines.findIndex((l) => /^desktop:\s*(#.*)?$/.test(l));
  if (di < 0) {
    return `${text.replace(/\n*$/, '\n')}\ndesktop:\n  enabled: true\n`;
  }
  for (let i = di + 1; i < lines.length && /^\s/.test(lines[i]) && lines[i].trim() !== ''; i++) {
    if (/^\s+enabled:\s*false/.test(lines[i])) {
      lines[i] = lines[i].replace(/enabled:\s*false/, 'enabled: true');
      return lines.join('\n');
    }
    if (/^\s+enabled:\s*true/.test(lines[i])) return lines.join('\n');
  }
  lines.splice(di + 1, 0, '  enabled: true');
  return lines.join('\n');
}

module.exports = { resolveBackend, ensureUserConfig, enableDesktop };
