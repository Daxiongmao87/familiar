/** Preload for the hidden inference window: chat/load/probe channels. */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('familiarInfer', {
  onChat: (fn) => ipcRenderer.on('infer:chat', (_ev, v) => fn(v)),
  chatDone: (msg) => ipcRenderer.send('infer:chat-done', msg),
  onLoad: (fn) => ipcRenderer.on('infer:load', (_ev, v) => fn(v)),
  loadProgress: (p) => ipcRenderer.send('infer:load-progress', p),
  loadDone: (r) => ipcRenderer.send('infer:load-done', r),
  onProbeHardware: (fn) => ipcRenderer.on('infer:probe-hardware', () => fn()),
  hardware: (hw) => ipcRenderer.send('infer:hardware', hw),
  status: (st) => ipcRenderer.send('infer:status', st),
});
