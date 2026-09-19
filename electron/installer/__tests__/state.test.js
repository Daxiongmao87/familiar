/** Install-state tests: planning, atomicity, migration, already-installed. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
  allInstalled, installVendoredFile, loadState, markFileDone, migrateState, planInstall, saveState,
} = require('../state.js');

const MANIFEST_V1 = {
  manifest_version: 1,
  components: [
    {
      id: 'a', kind: 'files', version: 'aaa', install_dir: 'a',
      files: [
        { path: 'f1.bin', url: 'http://x/f1', size: 10 },
        { path: 'f2.bin', url: 'http://x/f2', size: 20 },
      ],
    },
    {
      id: 'b', kind: 'files', version: 'bbb', install_dir: 'b',
      files: [{ path: 'g.bin', url: 'http://x/g', size: 5 }],
    },
    { id: 'rt', kind: 'pip-env', version: 'p1', install_dir: 'rt' },
  ],
};

function setup() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fam-st-'));
  return { dir, stateFile: path.join(dir, 'install-state.json'),
    dirFor: (sub) => path.join(dir, sub) };
}

test('fresh state plans every file; pip-envs excluded', () => {
  const { dirFor } = setup();
  const plan = planInstall(MANIFEST_V1, loadState('/nonexistent/x.json'), dirFor);
  assert.equal(plan.files.length, 3);
  assert.equal(plan.totalBytes, 35);
  assert.equal(plan.doneBytes, 0);
});

test('files already on disk with matching size are skipped', () => {
  const { dirFor } = setup();
  fs.mkdirSync(dirFor('a'), { recursive: true });
  fs.writeFileSync(path.join(dirFor('a'), 'f1.bin'), Buffer.alloc(10));
  const plan = planInstall(MANIFEST_V1, loadState('/nonexistent/x.json'), dirFor);
  assert.equal(plan.files.length, 2);
  assert.equal(plan.doneBytes, 10);
  assert.ok(!plan.files.some((f) => f.file.path === 'f1.bin'));
});

test('wrong-size files re-download (corruption After crash)', () => {
  const { dirFor } = setup();
  fs.mkdirSync(dirFor('b'), { recursive: true });
  fs.writeFileSync(path.join(dirFor('b'), 'g.bin'), Buffer.alloc(3)); // torn
  const plan = planInstall(MANIFEST_V1, loadState('/nonexistent/x.json'), dirFor);
  assert.ok(plan.files.some((f) => f.file.path === 'g.bin'));
});

test('wantedIds restricts the plan to selected components', () => {
  const { dirFor } = setup();
  const plan = planInstall(MANIFEST_V1, loadState('/nonexistent/x.json'), dirFor, ['b']);
  assert.deepEqual(plan.files.map((f) => f.file.path), ['g.bin']);
});

test('markFileDone completes the component only when whole', () => {
  const state = loadState('/nonexistent/x.json');
  markFileDone(state, MANIFEST_V1.components[0], 'f1.bin', 10);
  assert.equal(state.components.a.status, 'partial');
  markFileDone(state, MANIFEST_V1.components[0], 'f2.bin', 20);
  assert.equal(state.components.a.status, 'done');
});

test('save/load round-trips atomically; corrupt file recovers fresh', () => {
  const { stateFile } = setup();
  const state = { manifest_version: 1, components: { a: { version: 'aaa' } } };
  saveState(stateFile, state);
  assert.deepEqual(loadState(stateFile), state);
  assert.ok(!fs.existsSync(`${stateFile}.tmp`), 'no temp left behind');
  fs.writeFileSync(stateFile, '{torn json');
  assert.deepEqual(loadState(stateFile), { manifest_version: null, components: {} });
});

test('migration keeps matching files, drops changed components', () => {
  const state = {
    manifest_version: 1,
    components: {
      a: { version: 'aaa', status: 'done',
        files: { 'f1.bin': { size: 10 }, 'f2.bin': { size: 20 } } },
      b: { version: 'bbb', status: 'done', files: { 'g.bin': { size: 5 } } },
    },
  };
  const v2 = JSON.parse(JSON.stringify(MANIFEST_V1));
  v2.manifest_version = 2;
  v2.components[0].version = 'aaa2'; // a changed -> drop
  v2.components[1].files[0].size = 6; // b same version, file changed -> drop file
  const out = migrateState(v2, state);
  assert.equal(out.manifest_version, 2);
  assert.ok(!('a' in out.components));
  assert.deepEqual(out.components.b.files, {});
});

test('allInstalled is true only when every wanted file verifies', () => {
  const { dirFor } = setup();
  assert.equal(allInstalled(MANIFEST_V1, dirFor, ['b']), false);
  fs.mkdirSync(dirFor('b'), { recursive: true });
  fs.writeFileSync(path.join(dirFor('b'), 'g.bin'), Buffer.alloc(5));
  assert.equal(allInstalled(MANIFEST_V1, dirFor, ['b']), true);
  assert.equal(allInstalled(MANIFEST_V1, dirFor, ['a', 'b']), false);
});

test('installVendoredFile copies from resources and enforces size', () => {
  const { dirFor } = setup();
  const resDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fam-res-'));
  fs.mkdirSync(path.join(resDir, 'stt_server'), { recursive: true });
  fs.writeFileSync(path.join(resDir, 'stt_server', 'srv.py'), Buffer.alloc(11));
  const dest = path.join(dirFor('stt/server'), 'srv.py');
  installVendoredFile('stt_server/srv.py', dest, 11,
    { dev: false, resourcesPath: resDir, repoRoot: '/none' });
  assert.equal(fs.statSync(dest).size, 11);
  assert.throws(() => installVendoredFile('stt_server/srv.py', dest, 12,
    { dev: false, resourcesPath: resDir, repoRoot: '/none' }), /size/);
});

test('installVendoredFile in dev reads from the repo services tree', () => {
  const { dirFor } = setup();
  const repo = fs.mkdtempSync(path.join(os.tmpdir(), 'fam-repo-'));
  fs.mkdirSync(path.join(repo, 'services', 'stt_server'), { recursive: true });
  fs.writeFileSync(path.join(repo, 'services', 'stt_server', 'srv.py'), Buffer.alloc(7));
  const dest = path.join(dirFor('stt/server'), 'srv.py');
  installVendoredFile('stt_server/srv.py', dest, 7,
    { dev: true, resourcesPath: '/none', repoRoot: repo });
  assert.equal(fs.statSync(dest).size, 7);
});
