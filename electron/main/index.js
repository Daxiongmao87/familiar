/** Familiar desktop shell: lifecycle, windows, supervision, boot flow.
 *
 * Startup order (after first-run setup completes):
 * 1. Electron starts; user config is seeded if absent.
 * 2. Installed-component state is verified against the manifest.
 * 3. The bundled Python backend is started and supervised.
 * 4. Required local services start (only those the providers need).
 * 5. Health checks run (backend /api/status, sidecars, bridge).
 * 6. The main UI appears when ready, or a degraded state with retry.
 *
 * Security defaults: contextIsolation on, no Node in renderers, navigation
 * locked to the backend origin (main window) or local files (setup/infer),
 * all privileged work behind validated IPC handlers (see ipc.js).
 */
'use strict';

const path = require('node:path');

const { app, BrowserWindow, desktopCapturer, session } = require('electron');
const { installCaptureHandler } = require('./capture');

const { ensureUserConfig, resolveBackend } = require('./backend');
const { startBridge } = require('./bridge');
const { createDebugLogger, debugRequested } = require('./debug_log');
const { detectCuda, verdict } = require('./hardware');
const { registerIpc } = require('./ipc');
const { basePython, ensureVenv, jevArgv, sttArgv, waitForHttp, waitForTcp } = require('./sidecars');
const { supervise } = require('./supervise');
const { downloadFile } = require('../installer/downloader');
const { allInstalled, installVendoredFile, loadState, markFileDone, migrateState, planInstall, saveState } = require('../installer/state');

const SETUP_W = 780;
const SETUP_H = 780;

const ctx = {
  windows: {}, supervisor: {}, bridge: null, bridgeStatus: null,
  manifest: null, layoutMod: null, dirs: null, state: null,
  backend: null, hardware: null, cuda: null, hwVerdict: null, forcedRemote: [],
  installLog: [],
  debug: { enabled: false, logFile: '', log: () => {} },
  providers: { synthesis: 'remote', jev: 'remote', stt: 'local' },
};

function repoRoot() {
  return path.resolve(__dirname, '..', '..');
}

function resources() {
  return process.env.FAMILIAR_DEV === '1' ? repoRoot() : process.resourcesPath;
}

function exampleConfigPath() {
  if (process.env.FAMILIAR_DEV === '1') return path.join(repoRoot(), 'config.example.yaml');
  return path.join(process.resourcesPath, 'config.example.yaml');
}

function createWindow(name, opts) {
  const win = new BrowserWindow(Object.assign({
    width: 1280, height: 900,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      preload: path.join(__dirname, '..', 'preload', name === 'main' ? 'app.js' : `${name}.js`),
    },
  }, opts || {}));
  win.on('closed', () => { delete ctx.windows[name]; });
  win.on('unresponsive', () => ctx.debug.log('renderer', 'unresponsive', { name }));
  win.webContents.on('render-process-gone', (_ev, detail) => {
    ctx.debug.log('renderer', 'process_gone', { name, detail });
  });
  win.webContents.on('did-fail-load', (_ev, code, description, url) => {
    ctx.debug.log('renderer', 'load_failed', { name, code, description, url });
  });
  // The hidden inference window intentionally remains alive behind the UI.
  // Closing the visible main window must still begin the full shutdown path.
  if (name === 'main' || name === 'setup') win.on('closed', () => { app.quit(); });
  ctx.windows[name] = win;
  return win;
}

function lockNavigation(win, allowed) {
  const allow = allowed.map((prefix) => prefix.toLowerCase());
  win.webContents.on('will-navigate', (ev, url) => {
    if (!allow.some((p) => url.toLowerCase().startsWith(p))) ev.preventDefault();
  });
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
}

function send(channel, payload) {
  for (const win of Object.values(ctx.windows)) {
    if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
  }
}

function logSupervisor(ev) {
  ctx.debug.log(`service:${ev.name}`, ev.type, ev);
  ctx.installLog.push(`[${ev.name}] ${ev.type}${ev.text ? `: ${ev.text.slice(0, 200)}` : ''}`);
  if (ctx.installLog.length > 300) ctx.installLog.shift();
  if (['exited', 'restart_giveup', 'spawn_error', 'restarting'].includes(ev.type)) {
    send('service:event', { name: ev.name, type: ev.type,
      detail: ev.detail || ev.error || `code=${ev.code}` });
  }
}

async function bootServices(onProgress) {
  const { backend } = ctx;
  const progress = onProgress || (() => {});
  ctx.debug.log('electron', 'boot_services', { providers: ctx.providers });
  // Backend first: the UI and health checks hang off it.
  const be = supervise({
    name: 'backend', command: backend.command, args: backend.args,
    env: backend.env, onEvent: logSupervisor,
    cwd: require('electron').app.getPath('userData'),
  });
  ctx.supervisor.backend = be;
  be.start();
  progress({ stage: 'backend', detail: 'starting Python backend…' });
  const ok = await waitForHttp(`${backend.baseUrl}/api/status`, 60_000);
  if (!ok) throw new Error('Python backend did not answer /api/status in 60 s');
  progress({ stage: 'backend', detail: 'backend healthy' });
  // STT sidecar when STT mode is local and assets exist.
  if (ctx.providers.stt === 'local' && serviceWanted('stt')) {
    await startStt(progress);
  }
  // JEV sidecar when the JEV provider is local.
  if (ctx.providers.jev === 'local') {
    await startJev(progress);
  }
  // Local synthesis: load the resident WebLLM model (progress to the UI).
  if (ctx.providers.synthesis === 'local') {
    await loadLocalModel(progress);
  }
}

function serviceWanted(kind) {
  const ids = kind === 'stt' ? ['stt-server', 'stt-model'] : ['jev-model'];
  return allInstalled(ctx.manifest, ctx.dirs.componentDir, ids);
}

async function startStt(progress) {
  const venvPy = venvPython('stt/venv');
  const { command, args } = sttArgv(venvPy,
    ctx.dirs.componentDir('stt/server'), ctx.dirs.componentDir('stt/model'), 43007);
  const stt = supervise({ name: 'stt', command, args, onEvent: logSupervisor });
  ctx.supervisor.stt = stt;
  stt.start();
  progress({ stage: 'stt', detail: 'starting streaming STT server…' });
  const ok = await waitForTcp('127.0.0.1', 43007, 120_000);
  if (!ok) throw new Error('STT server did not open port 43007 in 120 s');
  progress({ stage: 'stt', detail: 'STT server listening' });
}

async function startJev(progress) {
  const venvPy = venvPython('jev/venv');
  const sidecarDir = process.env.FAMILIAR_DEV === '1'
    ? path.join(repoRoot(), 'services', 'jev_sidecar')
    : path.join(process.resourcesPath, 'jev_sidecar');
  const jevComp = ctx.manifest.components.find((c) => c.id === 'jev-model');
  const { command, args, env } = jevArgv(venvPy, sidecarDir,
    ctx.dirs.componentDir('jev/model'), 8299, jevComp && jevComp.revision);
  const jev = supervise({ name: 'jev', command, args, env, onEvent: logSupervisor });
  ctx.supervisor.jev = jev;
  jev.start();
  progress({ stage: 'jev', detail: 'loading JEV scorer (first load is slow)…' });
  const ok = await waitForHttp('http://127.0.0.1:8299/health', 300_000);
  if (!ok) throw new Error('JEV sidecar did not answer /health in 300 s');
  progress({ stage: 'jev', detail: 'JEV scorer ready' });
}

function venvPython(installDir) {
  return path.join(ctx.dirs.componentDir(installDir), 'bin', 'python');
}

async function loadLocalModel(progress) {
  const win = ctx.windows.infer;
  if (!win || win.isDestroyed()) throw new Error('inference window unavailable');
  progress({ stage: 'synth', detail: 'loading MiniCPM5-2B into WebGPU…' });
  const bridgePort = bridgePortFor();
  const modelUrl = `http://127.0.0.1:${bridgePort}/local-model/resolve/local`;
  const synth = ctx.manifest.components.find((c) => c.id === 'synth-model');
  const wasm = (synth.files || []).find((f) => f.path.endsWith('.wasm'));
  if (!wasm) throw new Error('manifest synth-model has no .wasm lib');
  const wasmUrl = `${modelUrl}/${wasm.path}`;
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('model load timed out (300 s)')), 300_000);
    const { ipcMain } = require('electron');
    const onProgress = (_ev, p) => progress({ stage: 'synth', detail: p.text || 'loading…', frac: p.frac });
    const onDone = (_ev, r) => {
      clearTimeout(timer);
      ipcMain.removeListener('infer:load-progress', onProgress);
      if (r && r.ok) resolve();
      else reject(new Error((r && r.error) || 'model load failed'));
    };
    ipcMain.on('infer:load-progress', onProgress);
    ipcMain.once('infer:load-done', onDone);
    win.webContents.send('infer:load', { modelUrl, wasmUrl });
  });
  progress({ stage: 'synth', detail: 'local model resident' });
}

function bridgePortFor() {
  return parseInt(process.env.FAMILIAR_BRIDGE_PORT || '', 10) || 8791;
}

async function startBridgeServer() {
  const { ipcMain } = require('electron');
  const pending = new Map();
  let seq = 0;
  ipcMain.on('infer:chat-done', (_ev, msg) => {
    const entry = pending.get(msg && msg.id);
    if (entry) {
      pending.delete(msg.id);
      clearTimeout(entry.timer);
      if (msg.ok) entry.resolve({ content: msg.content, usage: msg.usage || null });
      else entry.reject(new Error(msg.error || 'inference failed'));
    }
  });
  const modelDir = ctx.dirs.componentDir('synth/minicpm5-2b-mlc');
  ctx.bridgeStatus = { modelLoaded: false, webgpu: ctx.hardware };
  ipcMain.on('infer:status', (_ev, st) => {
    ctx.bridgeStatus = { modelLoaded: !!(st && st.modelLoaded), webgpu: ctx.hardware };
  });
  ctx.bridge = await startBridge({
    host: '127.0.0.1',
    port: bridgePortFor(),
    modelDir,
    status: () => ctx.bridgeStatus,
    infer: (payload, timeoutMs) => new Promise((resolve, reject) => {
      const win = ctx.windows.infer;
      if (!win || win.isDestroyed()) {
        reject(new Error('inference window not ready'));
        return;
      }
      const id = `c${Date.now()}-${(seq += 1)}`;
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error('inference timed out'));
      }, timeoutMs);
      pending.set(id, { resolve, reject, timer });
      win.webContents.send('infer:chat', Object.assign({ id }, payload));
    }),
  });
}

async function runInstall(wantedIds, progress) {
  const plan = planInstall(ctx.manifest, ctx.state, ctx.dirs.componentDir, wantedIds);
  let done = plan.doneBytes;
  const total = plan.totalBytes;
  progress({ overall: total ? done / total : 1, detail: 'starting downloads…' });
  for (const item of plan.files) {
    const rel = `${item.component.id}/${item.file.path}`;
    if (item.file.vendored) {
      progress({ overall: total ? done / total : 1, detail: `installing ${rel}` });
      installVendoredFile(item.file.vendored, item.dest, item.file.size, {
        dev: process.env.FAMILIAR_DEV === '1',
        resourcesPath: process.resourcesPath,
        repoRoot: repoRoot(),
      });
      progress({ overall: total ? (done + (item.file.size || 0)) / total : 1,
        detail: `installing ${rel}`, file: rel,
        fileDone: item.file.size || 0, fileTotal: item.file.size });
      done += item.file.size || 0;
      markFileDone(ctx.state, item.component, item.file.path, item.file.size);
      saveState(ctx.dirs.stateFile, ctx.state);
      continue;
    }
    progress({ overall: total ? done / total : 1, detail: `downloading ${rel}` });
    await downloadFile(item.file.url, item.dest, {
      expectedSize: item.file.size,
      onProgress: (d) => {
        const base = done;
        progress({ overall: total ? (base + d) / total : 1,
          detail: `downloading ${rel}`, file: rel, fileDone: d, fileTotal: item.file.size });
      },
    });
    done += item.file.size || 0;
    markFileDone(ctx.state, item.component, item.file.path, item.file.size);
    saveState(ctx.dirs.stateFile, ctx.state);
  }
  // Pip envs for the wanted file-components that need them.
  const need = new Set(wantedIds);
  const py = basePython(process.resourcesPath);
  if (need.has('stt-model') || need.has('stt-server')) {
    const comp = ctx.manifest.components.find((c) => c.id === 'stt-runtime');
    progress({ overall: 1, detail: 'installing STT runtime (pip)…' });
    ensureVenv(py, ctx.dirs.componentDir('stt/venv'), comp.packages,
      (line) => progress({ overall: 1, detail: line }));
  }
  if (need.has('jev-model')) {
    const comp = ctx.manifest.components.find((c) => c.id === 'jev-runtime');
    progress({ overall: 1, detail: 'installing OpenJEV runtime (pip)…' });
    ensureVenv(py, ctx.dirs.componentDir('jev/venv'), comp.packages,
      (line) => progress({ overall: 1, detail: line }));
  }
  progress({ overall: 1, detail: 'verifying…' });
  return true;
}

function wantedFromProviders() {
  const ids = [];
  // Forced-remote STT is excluded: the credentials step flips it to a
  // remote endpoint on save, so its local assets must not download.
  if (ctx.providers.stt === 'local' && !sttMustBeRemote()) {
    ids.push('stt-server', 'stt-model');
  }
  if (ctx.providers.synthesis === 'local') ids.push('synth-model');
  if (ctx.providers.jev === 'local') ids.push('jev-model');
  return ids;
}

/** True when STT is set to local streaming on CUDA-less hardware. */
function sttMustBeRemote() {
  return ctx.providers.stt === 'local' && !!ctx.hwVerdict && !ctx.hwVerdict.sttLocal.ok;
}

/**
 * Force endpoint-only for providers this machine cannot run locally.
 * Writes the forced modes to config (remote URLs/keys are kept, models
 * stay cached) and records what changed so setup can explain it.
 */
function forceEndpointOnly() {
  const v = ctx.hwVerdict;
  if (!v) return;
  const { writeProviders } = require('./providers_file');
  const forced = [];
  const flip = {};
  if (ctx.providers.synthesis === 'local' && !v.synthesisLocal.ok) {
    flip.synthesis = 'remote';
    forced.push({ provider: 'synthesis', reason: v.synthesisLocal.reason });
  }
  if (ctx.providers.jev === 'local' && !v.jevLocal.ok) {
    flip.jev = 'remote';
    forced.push({ provider: 'jev', reason: v.jevLocal.reason });
  }
  if (Object.keys(flip).length > 0) {
    writeProviders(ctx.backend.configPath, flip);
    ctx.providers = readProvidersFromConfig(ctx.backend.configPath);
  }
  // STT has no provider flag: forcing means the credentials step must
  // collect a remote STT endpoint (it cannot stay on local streaming).
  if (ctx.providers.stt === 'local' && !v.sttLocal.ok) {
    forced.push({ provider: 'stt', reason: v.sttLocal.reason });
  }
  ctx.forcedRemote = forced;
  if (forced.length > 0) {
    ctx.installLog.push(`hardware force: ${forced.map((f) => `${f.provider} (${f.reason})`).join('; ')}`);
  }
}

function readProvidersFromConfig(configPath) {
  // Minimal YAML read without a dependency: provider lines are stable
  // `key: value` pairs in the committed template (see scanProviders).
  return scanProviders(configPath);
}

const { scanProviders } = require('./providers_file');

async function probeHardware() {
  const win = ctx.windows.infer;
  if (!win || win.isDestroyed()) return { supported: false, reason: 'no window' };
  return new Promise((resolve) => {
    const { ipcMain } = require('electron');
    const timer = setTimeout(() => resolve({ supported: false, reason: 'probe timeout' }), 15000);
    ipcMain.once('infer:hardware', (_ev, hw) => {
      clearTimeout(timer);
      resolve(hw || { supported: false, reason: 'empty probe' });
    });
    win.webContents.send('infer:probe-hardware');
  });
}

async function ready() {
  const gotLock = app.requestSingleInstanceLock();
  if (!gotLock) {
    app.quit();
    return;
  }
  const userData = app.getPath('userData');
  ctx.debug = createDebugLogger({
    enabled: debugRequested(process.argv, process.env), userData,
  });
  ctx.debug.log('electron', 'ready', { userData, resources: resources() });
  const { layout } = require('../shared/paths');
  ctx.layoutMod = layout;
  ctx.dirs = layout(userData);
  ctx.manifest = require('../shared/manifest.json');
  const loadedState = loadState(ctx.dirs.stateFile);
  const firstRun = loadedState.manifest_version === null
    && Object.keys(loadedState.components || {}).length === 0;
  ctx.state = migrateState(ctx.manifest, loadedState);
  saveState(ctx.dirs.stateFile, ctx.state);

  ctx.backend = resolveBackend({ resourcesPath: resources(), userData, repoRoot: repoRoot() });
  if (ctx.debug.enabled) {
    const transcriptLog = path.join(userData, 'logs', 'familiar-transcript.jsonl');
    ctx.backend.env.DMD_DEBUG = '1';
    ctx.backend.env.DMD_TRANSCRIPT_LOG = transcriptLog;
    ctx.debug.log('electron', 'diagnostic_paths', {
      debug_log: ctx.debug.logFile, transcript_log: transcriptLog,
    });
  }
  ensureUserConfig(exampleConfigPath(), ctx.backend.configPath);
  ctx.providers = readProvidersFromConfig(ctx.backend.configPath);
  ctx.debug.log('electron', 'providers_loaded', ctx.providers);

  // Hidden inference window first: hardware probe + future WebLLM host.
  const infer = createWindow('infer',
    { show: false, width: 400, height: 300, title: 'Familiar inference' });
  lockNavigation(infer, ['file://']);
  await infer.loadFile(path.join(__dirname, '..', 'inference', 'host.html'));
  ctx.hardware = await probeHardware();
  ctx.cuda = await detectCuda();
  ctx.hwVerdict = verdict(ctx.hardware, ctx.cuda);
  ctx.debug.log('electron', 'hardware_verdict', ctx.hwVerdict);
  forceEndpointOnly();

  registerIpc(ctx, { runInstall, wantedFromProviders, bootServices, scanProviders,
    bootAndShowMain, showMainWindow, sttMustBeRemote });
  await startBridgeServer();

  const wanted = wantedFromProviders();
  // Setup shows on first launch (providers + credentials + install),
  // when downloads are missing, or when hardware forced a provider flip.
  const ready = (wanted.length === 0
    || allInstalled(ctx.manifest, ctx.dirs.componentDir, wanted))
    && ctx.forcedRemote.length === 0 && !firstRun;
  if (ready) {
    try {
      await bootAndShowMain(null);
    } catch (err) {
      // Fast-path boot failure lands in setup for recovery (provider
      // switching, retry) instead of a silent quit.
      const win = showSetup(wanted);
      const msg = String((err && err.message) || err);
      win.webContents.on('did-finish-load', () => {
        win.webContents.send('setup:error', { message: msg,
          canOpenAnyway: !!ctx.supervisor.backend });
      });
    }
  } else {
    showSetup(wanted);
  }
}

function showSetup(wanted) {
  const win = createWindow('setup', {
    width: SETUP_W, height: SETUP_H, resizable: false, title: 'Setting up Familiar',
  });
  lockNavigation(win, ['file://']);
  win.loadFile(path.join(__dirname, '..', 'renderer', 'setup', 'index.html'));
  win.webContents.on('did-finish-load', () => {
    win.webContents.send('setup:state', setupState(wanted));
  });
  return win;
}

/** Full setup-window state: providers, hardware, plan, saved values. */
function setupState(wanted) {
  const { scanSetup } = require('./providers_file');
  const plan = planInstall(ctx.manifest, ctx.state, ctx.dirs.componentDir, wanted);
  return {
    providers: ctx.providers,
    hardware: ctx.hardware,
    cuda: ctx.cuda,
    hwVerdict: ctx.hwVerdict,
    forcedRemote: ctx.forcedRemote,
    saved: scanSetup(ctx.backend.configPath),
    wanted,
    totalBytes: plan.totalBytes,
    doneBytes: plan.doneBytes,
    files: plan.files.length,
    components: ctx.manifest.components.filter((c) => wanted.includes(c.id)).map((c) => ({
      id: c.id, label: c.label,
      bytes: (c.files || []).reduce((a, f) => a + (f.size || 0), 0),
    })),
  };
}

async function bootAndShowMain(setupWin) {
  const progress = (p) => {
    ctx.debug.log('boot', p.stage || 'progress', p);
    if (setupWin && !setupWin.isDestroyed()) setupWin.webContents.send('setup:boot', p);
  };
  if (!ctx.booted) {
    try {
      await bootServices(progress);
    } catch (err) {
      ctx.debug.log('boot', 'failed', { error: String(err && (err.stack || err)) });
      if (setupWin && !setupWin.isDestroyed()) {
        const be = ctx.supervisor.backend;
        setupWin.webContents.send('setup:error',
          { message: String((err && err.message) || err),
            canOpenAnyway: !!(be && be.state().running) });
        return;
      }
      throw err;
    }
    ctx.booted = true;
  }
  showMainWindow();
  if (setupWin && !setupWin.isDestroyed()) setupWin.close();
}

function showMainWindow() {
  if (ctx.windows.main && !ctx.windows.main.isDestroyed()) {
    ctx.windows.main.focus();
    return ctx.windows.main;
  }
  const main = createWindow('main', { width: 1280, height: 900, title: 'Familiar' });
  installCaptureHandler(session.defaultSession, desktopCapturer, () => ctx.windows.main);
  lockNavigation(main, [ctx.backend.baseUrl]);
  main.loadURL(`${ctx.backend.baseUrl}/`);
  ctx.debug.log('electron', 'main_window_loading', { url: ctx.backend.baseUrl });
  return main;
}

async function shutdown() {
  ctx.debug.log('electron', 'shutdown_started');
  send('app:quitting', {});
  const order = ['jev', 'stt', 'backend'];
  for (const name of order) {
    try {
      if (ctx.supervisor[name]) await ctx.supervisor[name].stop();
    } catch { /* best effort */ }
  }
  if (ctx.bridge) {
    try {
      await new Promise((resolve) => ctx.bridge.close(resolve));
    } catch { /* best effort */ }
  }
}

if (require.main === module || process.env.FAMILIAR_ELECTRON_MAIN) {
  if (process.platform === 'linux') {
    const features = app.commandLine.getSwitchValue('enable-features');
    app.commandLine.appendSwitch('enable-features',
      [features, 'PulseaudioLoopbackForScreenShare'].filter(Boolean).join(','));
  }
  app.whenReady().then(() => ready().catch((err) => {
    ctx.debug.log('electron', 'fatal_startup', { error: String(err && (err.stack || err)) });
    // eslint-disable-next-line no-console
    console.error('fatal startup error:', err);
    app.quit();
  }));
  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
  });
  app.on('before-quit', (ev) => {
    ev.preventDefault();
    shutdown().finally(() => app.exit(0));
  });
}

module.exports = {
  ctx, ready, shutdown, bootAndShowMain, showMainWindow, showSetup, wantedFromProviders,
  readProvidersFromConfig, scanProviders, runInstall,
};
