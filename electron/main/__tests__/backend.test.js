/** backend tests: user-config seeding and dev backend resolution. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { enableDesktop, ensureUserConfig, resolveBackend } = require('../backend.js');

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');

test('ensureUserConfig seeds from the example and enables desktop', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fam-be-'));
  const dest = path.join(dir, 'sub', 'familiar-config.yaml');
  const example = path.join(REPO_ROOT, 'config.example.yaml');
  assert.equal(ensureUserConfig(example, dest), 'seeded');
  const text = fs.readFileSync(dest, 'utf-8');
  assert.equal(text.match(/^desktop:/gm).length, 1, 'exactly one desktop block');
  assert.ok(/enabled:\s*true/.test(text.split(/^desktop:/m)[1].split(/^\S/m)[0]));
});

test('enableDesktop tolerates trailing comments and missing blocks', () => {
  const withComment = 'a: 1\ndesktop:  # trailing comment\n  enabled: false  # why\n  x: 1\n';
  const out = enableDesktop(withComment);
  assert.equal(out.match(/^desktop:/gm).length, 1);
  assert.ok(out.includes('enabled: true  # why'));
  assert.equal(enableDesktop('a: 1\n'), 'a: 1\n\ndesktop:\n  enabled: true\n');
});

test('ensureUserConfig never overwrites an existing config', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fam-be-'));
  const dest = path.join(dir, 'familiar-config.yaml');
  fs.writeFileSync(dest, 'user: secret-stuff\n');
  assert.equal(ensureUserConfig(path.join(REPO_ROOT, 'config.example.yaml'), dest), 'exists');
  assert.equal(fs.readFileSync(dest, 'utf-8'), 'user: secret-stuff\n');
});

test('resolveBackend dev mode points at the checkout server', () => {
  process.env.FAMILIAR_DEV = '1';
  try {
    const out = resolveBackend({ resourcesPath: '/x', userData: '/tmp/u', repoRoot: REPO_ROOT });
    assert.ok(out.args[out.args.length - 1].endsWith('familiar-config.yaml'));
    assert.ok(out.args[0].endsWith(path.join('dmd', 'server.py')));
    assert.equal(out.baseUrl, 'http://127.0.0.1:8760');
    assert.ok((out.env.PYTHONPATH || '').includes(REPO_ROOT));
  } finally {
    delete process.env.FAMILIAR_DEV;
  }
});

test('resolveBackend packaged mode points at the bundled binary', () => {
  const out = resolveBackend(
    { resourcesPath: '/res', userData: '/tmp/u', repoRoot: REPO_ROOT });
  assert.equal(out.command, path.join('/res', 'backend', 'familiar-backend'));
  assert.equal(out.configPath, path.join('/tmp/u', 'familiar-config.yaml'));
});
