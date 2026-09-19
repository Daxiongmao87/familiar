/** Electron display capture must deliver Discord/system audio to Familiar. */
'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { installCaptureHandler } = require('../capture');

test('trusted main frame receives screen video and loopback audio', async () => {
  let handler;
  const frame = {};
  const source = { id: 'screen:0:0' };
  installCaptureHandler(
    { setDisplayMediaRequestHandler: (fn) => { handler = fn; } },
    { getSources: async () => [source] },
    () => ({ isDestroyed: () => false, webContents: { mainFrame: frame } }),
  );
  const result = await new Promise((resolve) => {
    handler({ frame, audioRequested: true }, resolve);
  });
  assert.deepEqual(result, { video: source, audio: 'loopback' });
});

test('capture requests from any other frame are denied', async () => {
  let handler;
  const frame = {};
  installCaptureHandler(
    { setDisplayMediaRequestHandler: (fn) => { handler = fn; } },
    { getSources: async () => [{ id: 'screen:0:0' }] },
    () => ({ isDestroyed: () => false, webContents: { mainFrame: frame } }),
  );
  const result = await new Promise((resolve) => {
    handler({ frame: {}, audioRequested: true }, resolve);
  });
  assert.deepEqual(result, {});
});
