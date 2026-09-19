/** Preload surface tests: allowlisted bridge, no privileged leakage.
 *
 * The preload scripts are the only code crossing the renderer/main
 * boundary. These tests read them as text and pin the exposed surface:
 * exact channel names, contextIsolation-safe patterns only, and no
 * filesystem/process/shell access reachable from rendered content.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const PRELOAD = path.resolve(__dirname, '..', '..', 'preload');

function src(name) {
  return fs.readFileSync(path.join(PRELOAD, name), 'utf-8');
}

const BANNED = [
  'child_process', 'execSync', 'spawnSync', 'shell', 'BrowserWindow',
  'require(\'fs', 'require("fs', 'readFileSync', 'writeFileSync',
  'process.env', 'nodeIntegration',
];

for (const file of ['setup.js', 'infer.js', 'app.js']) {
  test(`${file} uses contextBridge and leaks no privileged APIs`, () => {
    const text = src(file);
    assert.ok(text.includes('contextBridge.exposeInMainWorld'),
      'must use the context bridge');
    assert.ok(text.includes('ipcRenderer'), 'must use ipcRenderer');
    for (const banned of BANNED) {
      assert.ok(!text.includes(banned), `${file} must not contain ${banned}`);
    }
  });
}

test('setup.js exposes exactly the installer surface', () => {
  const text = src('setup.js');
  for (const fn of ['getPlan', 'setProviders', 'saveSetup', 'startInstall', 'boot',
    'openAnyway', 'onState', 'onProgress', 'onBoot', 'onError']) {
    assert.ok(text.includes(fn), `missing ${fn}`);
  }
  for (const ch of ['setup:get-plan', 'setup:set-providers', 'setup:save-setup',
    'setup:start', 'setup:boot', 'setup:open-anyway']) {
    assert.ok(text.includes(ch), `missing channel ${ch}`);
  }
});

test('infer.js exposes exactly the inference channels', () => {
  const text = src('infer.js');
  for (const ch of ['infer:chat', 'infer:chat-done', 'infer:load',
    'infer:load-progress', 'infer:load-done', 'infer:probe-hardware',
    'infer:hardware', 'infer:status']) {
    assert.ok(text.includes(ch), `missing channel ${ch}`);
  }
});

test('app.js exposes the narrow desktop helper (capture-ready)', () => {
  const text = src('app.js');
  assert.ok(text.includes('familiarDesktop'), 'exposes window.familiarDesktop');
  for (const fn of ['getServices', 'restartService', 'getDesktopSources',
    'onServiceEvent', 'isElectron']) {
    assert.ok(text.includes(fn), `missing ${fn}`);
  }
});
