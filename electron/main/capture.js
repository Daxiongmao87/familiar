/** Desktop capture for the trusted Familiar window and its Capture button. */
'use strict';

/** Register Electron's required display-media broker for the main UI. */
function installCaptureHandler(session, desktopCapturer, getWindow) {
  session.setDisplayMediaRequestHandler(async (request, callback) => {
    const win = getWindow();
    if (!win || win.isDestroyed() || request.frame !== win.webContents.mainFrame) {
      callback({});
      return;
    }
    try {
      const sources = await desktopCapturer.getSources({
        types: ['screen'], thumbnailSize: { width: 0, height: 0 },
      });
      if (!sources.length) { callback({}); return; }
      callback({ video: sources[0], audio: request.audioRequested ? 'loopback' : undefined });
    } catch {
      callback({});
    }
  });
}

module.exports = { installCaptureHandler };
