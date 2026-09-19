/** Sidecar launcher tests: argv shapes and health-wait behavior. */
'use strict';

const assert = require('node:assert/strict');
const http = require('node:http');
const net = require('node:net');
const test = require('node:test');

const { basePython, jevArgv, sttArgv, waitForHttp, waitForTcp } = require('../sidecars.js');

test('sttArgv pins the verified server flags', () => {
  const out = sttArgv('/venv/bin/python', '/srv', '/model', 43007);
  assert.equal(out.command, '/venv/bin/python');
  assert.deepEqual(out.args, [
    '/srv/whisper_online_server.py',
    '--host', '127.0.0.1', '--port', '43007',
    '--model', 'large-v3-turbo', '--model_dir', '/model',
    '--backend', 'faster-whisper', '--vac',
  ]);
});

test('jevArgv serves the sidecar on the local port with the pinned rev', () => {
  const out = jevArgv('/venv/bin/python', '/sidecar', '/weights', 8299, 'abc123');
  assert.equal(out.command, '/venv/bin/python');
  assert.deepEqual(out.args, [
    '/sidecar/serve.py',
    '--host', '127.0.0.1', '--port', '8299',
    '--model', '/weights',
    '--revision', 'abc123',
  ]);
});

test('basePython prefers dev python, else bundled, else PATH', () => {
  process.env.FAMILIAR_DEV = '1';
  try {
    assert.equal(basePython('/res'), 'python3');
  } finally {
    delete process.env.FAMILIAR_DEV;
  }
  assert.equal(basePython('/nonexistent-resources'), 'python3');
});

test('waitForHttp resolves true on 200, false on refusal', async () => {
  const server = http.createServer((_req, res) => {
    res.writeHead(200).end('{}');
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  try {
    assert.equal(await waitForHttp(`http://127.0.0.1:${port}/health`, 3000), true);
  } finally {
    server.close();
  }
  assert.equal(await waitForHttp('http://127.0.0.1:9/health', 1200), false);
});

test('waitForTcp resolves true on accept, false on refusal', async () => {
  const server = net.createServer((sock) => sock.destroy());
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  try {
    assert.equal(await waitForTcp('127.0.0.1', port, 3000), true);
  } finally {
    server.close();
  }
  assert.equal(await waitForTcp('127.0.0.1', 9, 1200), false);
});
