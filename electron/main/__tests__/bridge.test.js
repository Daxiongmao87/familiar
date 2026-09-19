/** Bridge tests: OpenAI-dialect surface over a stub inference function. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { MODEL_ID, startBridge } = require('../bridge.js');

async function start(t, { infer, status, files }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fam-br-'));
  for (const [rel, body] of Object.entries(files || { 'mlc-chat-config.json': '{}' })) {
    fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true });
    fs.writeFileSync(path.join(dir, rel), body);
  }
  const server = await startBridge({
    host: '127.0.0.1', port: 0, modelDir: dir,
    infer: infer || (async () => ({ content: 'hi', usage: null })),
    status: status || (() => ({ modelLoaded: true, webgpu: { supported: true } })),
  });
  t.after(() => server.close());
  return `http://127.0.0.1:${server.address().port}`;
}

test('GET /models advertises the resident model id', async (t) => {
  const base = await start(t, {});
  const res = await fetch(`${base}/models`);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.deepEqual(body, { data: [{ id: MODEL_ID }] });
});

test('GET /health reflects load state and hardware', async (t) => {
  const base = await start(t, {
    status: () => ({ modelLoaded: false, webgpu: { supported: false, reason: 'no-webgpu' } }),
  });
  const body = await (await fetch(`${base}/health`)).json();
  assert.equal(body.status, 'loading');
  assert.equal(body.model_loaded, false);
  assert.equal(body.webgpu.reason, 'no-webgpu');
});

test('GET /local-model serves files but blocks path escape', async (t) => {
  const base = await start(t, { files: { 'a/b.bin': '0123456789' } });
  const ok = await fetch(`${base}/local-model/a/b.bin`);
  assert.equal(ok.status, 200);
  assert.equal(await ok.text(), '0123456789');
  const viaResolve = await fetch(`${base}/local-model/resolve/local/a/b.bin`);
  assert.equal(viaResolve.status, 200);
  assert.equal(await viaResolve.text(), '0123456789');
  const missing = await fetch(`${base}/local-model/nope.bin`);
  assert.equal(missing.status, 404);
  const escape = await fetch(`${base}/local-model/..%2F..%2Fetc%2Fhostname`);
  assert.ok([403, 404].includes(escape.status));
});

test('POST /chat/completions returns OpenAI shape', async (t) => {
  let seen = null;
  const base = await start(t, {
    infer: async (payload) => {
      seen = payload;
      return { content: 'hello', usage: { prompt_tokens: 3 } };
    },
  });
  const res = await fetch(`${base}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'x', messages: [{ role: 'user', content: 'hi' }],
      max_tokens: 64, temperature: 0.5, enable_thinking: false }),
  });
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.choices[0].message.content, 'hello');
  assert.equal(body.model, MODEL_ID);
  assert.deepEqual(seen.messages, [{ role: 'user', content: 'hi' }]);
  assert.equal(seen.maxTokens, 64);
  assert.equal(seen.temperature, 0.5);
  assert.equal(seen.json, false);
});

test('POST maps json_schema response_format to json+schema', async (t) => {
  let seen = null;
  const base = await start(t, {
    infer: async (payload) => {
      seen = payload;
      return { content: '{"a":1}', usage: null };
    },
  });
  const schema = { type: 'object', properties: { a: { type: 'number' } } };
  const res = await fetch(`${base}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'x', messages: [{ role: 'user', content: 'hi' }],
      response_format: { type: 'json_schema', json_schema: { name: 'o', schema } } }),
  });
  assert.equal(res.status, 200);
  assert.equal((await res.json()).choices[0].message.content, '{"a":1}');
  assert.equal(seen.json, true);
  assert.deepEqual(seen.schema, schema);
});

test('POST validates messages strictly', async (t) => {
  const base = await start(t, {});
  for (const body of [{}, { messages: [] }, { messages: [{ role: 'user' }] },
    { messages: [{ role: 'user', content: 42 }] }]) {
    const res = await fetch(`${base}/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    assert.equal(res.status, 400, JSON.stringify(body));
  }
  const res = await fetch(`${base}/chat/completions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{bad json',
  });
  assert.equal(res.status, 400);
});

test('POST surfaces inference failure as 500/503, never leaks internals', async (t) => {
  const base = await start(t, { infer: async () => { throw new Error('model not loaded'); } });
  const res = await fetch(`${base}/chat/completions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
  });
  assert.equal(res.status, 503);
  const bad = await start(t, { infer: async () => { throw new Error('boom'); } });
  const res2 = await fetch(`${bad}/chat/completions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
  });
  assert.equal(res2.status, 500);
});

test('bridge refuses non-loopback bind', () => {
  assert.throws(() => startBridge({
    host: '0.0.0.0', port: 0, modelDir: os.tmpdir(),
    infer: async () => ({ content: '' }), status: () => ({}),
  }), /loopback/);
});
