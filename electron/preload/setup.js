/** Preload for the setup window: install actions + progress events only. */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('familiarSetup', {
  getPlan: () => ipcRenderer.invoke('setup:get-plan'),
  setProviders: (modes) => ipcRenderer.invoke('setup:set-providers', modes),
  saveSetup: (setup) => ipcRenderer.invoke('setup:save-setup', setup),
  startInstall: () => ipcRenderer.invoke('setup:start'),
  boot: () => ipcRenderer.invoke('setup:boot'),
  openAnyway: () => ipcRenderer.invoke('setup:open-anyway'),
  onState: (fn) => ipcRenderer.on('setup:state', (_ev, v) => fn(v)),
  onProgress: (fn) => ipcRenderer.on('setup:progress', (_ev, v) => fn(v)),
  onBoot: (fn) => ipcRenderer.on('setup:boot', (_ev, v) => fn(v)),
  onError: (fn) => ipcRenderer.on('setup:error', (_ev, v) => fn(v)),
});
