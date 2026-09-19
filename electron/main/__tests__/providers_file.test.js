/** providers_file tests: scan + write round-trip on template-shaped YAML. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { scanProviders, scanSetup, writeProviders, writeSetupConfig } = require('../providers_file.js');

const TEMPLATE = `server:
  port: 8760

models:
  synthesis:
    provider: remote
    base_url: http://remote:8080/v1
    model_id: minicpm5-2b
  stt:
    stream_host: 127.0.0.1
    stream_port: 43007
  embeddings:
    provider: local
    model_id: BAAI/bge-small-en-v1.5

openjev:
  enabled: true
  provider: remote
  base_url: http://127.0.0.1:8199

desktop:
  enabled: true
`;

function tmpConfig(text) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fam-cfg-'));
  const file = path.join(dir, 'familiar-config.yaml');
  fs.writeFileSync(file, text || TEMPLATE);
  return file;
}

test('scan reads synthesis/jev providers and loopback STT', () => {
  const out = scanProviders(tmpConfig());
  assert.deepEqual(out, { synthesis: 'remote', jev: 'remote', stt: 'local' });
  const remote = tmpConfig(TEMPLATE.replace('stream_host: 127.0.0.1', 'stream_host: 192.168.0.9'));
  assert.equal(scanProviders(remote).stt, 'remote');
});

test('scan ignores the embeddings provider flag', () => {
  const file = tmpConfig(TEMPLATE.replace('provider: local\n    model_id: BAAI', 'provider: endpoint\n    model_id: BAAI'));
  assert.equal(scanProviders(file).synthesis, 'remote');
});

test('scan defaults safely on missing file or sections', () => {
  assert.deepEqual(scanProviders('/nonexistent/x.yaml'),
    { synthesis: 'remote', jev: 'remote', stt: 'remote' });
  assert.deepEqual(scanProviders(tmpConfig('server:\n  port: 1\n')),
    { synthesis: 'remote', jev: 'remote', stt: 'remote' });
});

test('write flips modes and scan reads them back', () => {
  const file = tmpConfig();
  writeProviders(file, { synthesis: 'local', jev: 'local' });
  assert.deepEqual(scanProviders(file),
    { synthesis: 'local', jev: 'local', stt: 'local' });
  const text = fs.readFileSync(file, 'utf-8');
  assert.ok(text.includes('http://remote:8080/v1'), 'remote URL preserved');
  assert.ok(!text.includes('provider: endpoint'), 'embeddings untouched');
  writeProviders(file, { synthesis: 'remote' });
  assert.equal(scanProviders(file).synthesis, 'remote');
  assert.equal(scanProviders(file).jev, 'local');
});

test('write inserts missing provider lines without mangling', () => {
  const bare = TEMPLATE.replace('    provider: remote\n', '').replace('  provider: remote\n', '');
  const file = tmpConfig(bare);
  writeProviders(file, { synthesis: 'local', jev: 'local' });
  assert.deepEqual(scanProviders(file),
    { synthesis: 'local', jev: 'local', stt: 'local' });
});

test('write rejects invalid modes and unknown keys', () => {
  const file = tmpConfig();
  assert.throws(() => writeProviders(file, { synthesis: 'cloud' }), /invalid/);
  assert.throws(() => writeProviders(file, { nope: 'local' }), /unknown/);
  assert.equal(scanProviders(file).synthesis, 'remote', 'file untouched after refusal');
});

test('scanSetup reads endpoints without leaking secrets', () => {
  const file = tmpConfig([
    'discord:',
    '  token: sekrit',
    '  guild_id: "111"',
    '  dm_user_id: "222"',
    'models:',
    '  synthesis:',
    '    base_url: http://llm:8080/v1',
    '    model_id: m1',
    '    api_key: ${LLM_API_KEY}',
    '  stt:',
    '    stream_host: 192.168.0.9',
    '    stream_port: 43111',
    'openjev:',
    '  base_url: http://jev:8199/',
  ].join('\n'));
  const out = scanSetup(file);
  assert.equal(out.synthesis.base_url, 'http://llm:8080/v1');
  assert.equal(out.synthesis.has_api_key, false, '${VAR} placeholder is not configured');
  assert.equal(out.jev.base_url, 'http://jev:8199/');
  assert.deepEqual(out.stt, { mode: 'remote', stream_host: '192.168.0.9',
    stream_port: '43111' });
  assert.deepEqual(out.discord, { has_token: true, guild_id: '111', dm_user_id: '222' });
  assert.ok(!JSON.stringify(out).includes('sekrit'), 'token never echoed');
});

test('writeSetupConfig writes endpoints, discord, and STT mode', () => {
  const repoRoot = path.resolve(__dirname, '..', '..', '..');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fam-cfg-'));
  const file = path.join(dir, 'familiar-config.yaml');
  fs.copyFileSync(path.join(repoRoot, 'config.example.yaml'), file);
  writeSetupConfig(file, {
    discord: { token: 'tok-1', guild_id: '111', dm_user_id: '222' },
    synthesisRemote: { base_url: 'http://llm:8080/v1/', model_id: 'mx', api_key: 'sk-1' },
    jevRemote: { base_url: 'http://jev:8199' },
    stt: { mode: 'remote', stream_host: '192.168.0.9', stream_port: '43111' },
  });
  const text = fs.readFileSync(file, 'utf-8');
  assert.ok(text.includes('token: tok-1'));
  assert.ok(text.includes('base_url: http://llm:8080/v1\n') || text.includes('base_url: http://llm:8080/v1 '),
    'trailing slash stripped');
  assert.ok(text.includes('model_id: mx'));
  assert.ok(text.includes('stream_host: 192.168.0.9'));
  assert.ok(text.includes('stream_port: 43111'));
  const back = scanSetup(file);
  assert.equal(back.synthesis.base_url, 'http://llm:8080/v1');
  assert.equal(back.synthesis.has_api_key, true);
  assert.equal(back.stt.mode, 'remote');
  assert.equal(back.stt.stream_host, '192.168.0.9');
  assert.equal(back.stt.stream_port, '43111');
  assert.equal(back.discord.has_token, true);
  // Empty values leave existing config untouched (never clear secrets).
  writeSetupConfig(file, { synthesisRemote: { base_url: '', model_id: '', api_key: '' } });
  assert.equal(scanSetup(file).synthesis.base_url, 'http://llm:8080/v1');
  // Owner-only permissions.
  assert.equal(fs.statSync(file).mode & 0o777, 0o600);
});

test('writeSetupConfig flips STT to local streaming and rejects bad input', () => {
  const file = tmpConfig();
  writeSetupConfig(file, { stt: { mode: 'local' } });
  assert.equal(scanProviders(file).stt, 'local');
  assert.equal(scanSetup(file).stt.stream_host, '127.0.0.1');
  assert.equal(scanSetup(file).stt.stream_port, '43007');
  assert.throws(() => writeSetupConfig(file, { stt: { mode: 'cloud' } }), /stt.mode/);
  assert.throws(() => writeSetupConfig(file, { synthesisRemote: { base_url: 'nope' } }), /http/);
  assert.throws(() => writeSetupConfig(file, { jevRemote: { base_url: 'ftp://x' } }), /http/);
  assert.throws(() => writeSetupConfig(file,
    { stt: { mode: 'remote', stream_host: '', stream_port: '43007' } }), /host/);
  assert.throws(() => writeSetupConfig(file,
    { stt: { mode: 'remote', stream_host: 'h', stream_port: '99999' } }), /port/);
});

test('round-trip works on the real config.example.yaml (trailing comments)', () => {
  // Guards the section-header bug where `openjev:  # comment` lines were
  // not recognized as sections (setup showed remote for a local seed).
  const repoRoot = path.resolve(__dirname, '..', '..', '..');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fam-cfg-'));
  const file = path.join(dir, 'familiar-config.yaml');
  fs.copyFileSync(path.join(repoRoot, 'config.example.yaml'), file);
  assert.deepEqual(scanProviders(file),
    { synthesis: 'remote', jev: 'remote', stt: 'local' });
  writeProviders(file, { synthesis: 'local', jev: 'local' });
  assert.deepEqual(scanProviders(file),
    { synthesis: 'local', jev: 'local', stt: 'local' });
});
