/** Preload for the main Familiar UI window.
 *
 * Narrowly scoped: an Electron marker (so the UI can adapt capture hints),
 * service supervision state + restart, and screen-source enumeration for a
 * future capture picker. No filesystem or process access. The Familiar UI
 * itself is unchanged — it talks to the backend over HTTP/WebSocket as usual.
 */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('familiarDesktop', {
  isElectron: true,
  getServices: () => ipcRenderer.invoke('app:services'),
  restartService: (name) => ipcRenderer.invoke('app:restart-service', { name }),
  getDesktopSources: () => ipcRenderer.invoke('app:desktop-sources'),
  onServiceEvent: (fn) => ipcRenderer.on('service:event', (_ev, v) => fn(v)),
});
