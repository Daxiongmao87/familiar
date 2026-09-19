/** Validated IPC surface between renderers and the main process.
 *
 * Every handler validates its payload shape before acting; filesystem and
 * process handles never cross the boundary — renderers get progress
 * events and narrowly-scoped actions only.
 */
'use strict';

const { BrowserWindow, desktopCapturer, ipcMain } = require('electron');

const { planInstall } = require('../installer/state');
const { scanSetup, writeProviders, writeSetupConfig } = require('./providers_file');
const { checkVerdict, validateProviderModes, validateSaveSetup } = require('./setup_validate');

/**
 * Register all IPC handlers.
 * @param {object} ctx shared main-process context (see main/index.js).
 * @param {object} fns {runInstall, wantedFromProviders, bootServices, scanProviders}.
 */
function registerIpc(ctx, fns) {
  // --- setup window -------------------------------------------------
  ipcMain.handle('setup:get-plan', () => {
    const wanted = fns.wantedFromProviders();
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
    };
  });

  ipcMain.handle('setup:set-providers', (_ev, modes) => {
    validateProviderModes(modes);
    checkVerdict(modes, ctx.hwVerdict);
    writeProviders(ctx.backend.configPath, modes);
    ctx.providers = fns.scanProviders(ctx.backend.configPath);
    const wanted = fns.wantedFromProviders();
    const plan = planInstall(ctx.manifest, ctx.state, ctx.dirs.componentDir, wanted);
    return { providers: ctx.providers, wanted,
      totalBytes: plan.totalBytes, doneBytes: plan.doneBytes };
  });

  ipcMain.handle('setup:save-setup', (_ev, setup) => {
    // Step-2 save: credentials + remote endpoints + STT mode (rules in
    // setup_validate.js). Discord is soft: absent/empty skips.
    const saved = scanSetup(ctx.backend.configPath);
    validateSaveSetup(setup, ctx.providers, saved, ctx.hwVerdict, fns.sttMustBeRemote());
    writeSetupConfig(ctx.backend.configPath, setup);
    ctx.providers = fns.scanProviders(ctx.backend.configPath);
    ctx.forcedRemote = [];
    const wanted = fns.wantedFromProviders();
    const plan = planInstall(ctx.manifest, ctx.state, ctx.dirs.componentDir, wanted);
    return { providers: ctx.providers, saved: scanSetup(ctx.backend.configPath),
      wanted, totalBytes: plan.totalBytes, doneBytes: plan.doneBytes,
      files: plan.files.length };
  });

  ipcMain.handle('setup:start', async (ev) => {
    const win = BrowserWindow.fromWebContents(ev.sender);
    const progress = (p) => {
      if (win && !win.isDestroyed()) win.webContents.send('setup:progress', p);
    };
    try {
      const wanted = fns.wantedFromProviders();
      if (wanted.length > 0) {
        await fns.runInstall(wanted, progress);
      }
      return { ok: true };
    } catch (err) {
      return { ok: false, error: String((err && err.message) || err).slice(0, 800) };
    }
  });

  ipcMain.handle('setup:boot', async (ev) => {
    // Boot supervised services, then swap the setup window for the UI.
    // bootAndShowMain reports boot failures back on setup:error itself.
    const win = BrowserWindow.fromWebContents(ev.sender);
    await fns.bootAndShowMain(win || null);
    return { ok: true };
  });

  ipcMain.handle('setup:open-anyway', async () => {
    // Degraded entry: only when the backend itself is up.
    const be = ctx.supervisor.backend;
    if (!be || !be.state().running) {
      throw new Error('backend is not running');
    }
    fns.showMainWindow();
    return { ok: true };
  });

  // --- main window --------------------------------------------------
  ipcMain.handle('app:services', () => {
    const out = {};
    for (const [name, sup] of Object.entries(ctx.supervisor)) {
      out[name] = sup.state();
    }
    return out;
  });

  ipcMain.handle('app:restart-service', (_ev, payload) => {
    const name = payload && payload.name;
    if (!['backend', 'stt', 'jev'].includes(name)) throw new Error('unknown service');
    const sup = ctx.supervisor[name];
    if (!sup) throw new Error(`service not started: ${name}`);
    sup.restart();
    return { ok: true };
  });

  ipcMain.handle('app:desktop-sources', async () => {
    // Screen picker support for a future capture-source UI. Thumbnail
    // intentionally tiny: ids only, no framebuffer exfiltration channel.
    const sources = await desktopCapturer.getSources({
      types: ['screen'], thumbnailSize: { width: 0, height: 0 },
    });
    return sources.map((s) => ({ id: s.id, name: s.name }));
  });
}

module.exports = { registerIpc };
