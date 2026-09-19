/** Localhost OpenAI-dialect bridge: Python Gateway -> WebLLM worker.
 *
 * Endpoints (all 127.0.0.1 only, never exposed off-host):
 * - GET  /models               -> {data:[{id:"minicpm5-2b-mlc"}]}
 * - GET  /health               -> {status, model_loaded, webgpu}
 * - GET  /local-model/*        -> static model assets for WebLLM fetch
 * - POST /chat/completions     -> OpenAI chat shape in/out
 *
 * The bridge validates and normalizes the Gateway body, forwards one
 * request at a time to the hidden inference window (MLC serializes
 * internally anyway), and maps `response_format.json_schema` onto the
 * JSON-object mode WebLLM 0.2.85 supports, keeping the schema text in
 * the prompt so the model still sees the exact contract.
 */
'use strict';

const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');

const MODEL_ID = 'minicpm5-2b-mlc';
const MAX_BODY_BYTES = 256 * 1024;

/**
 * Start the bridge server.
 * @param {object} opts
 * @param {string} opts.host bind host (must be a loopback address).
 * @param {number} opts.port bind port.
 * @param {string} opts.modelDir dir serving /local-model/* (pinned, no escape).
 * @param {(req:{messages:object[],maxTokens:number,temperature:number|null,json:boolean,schema:object|null},timeoutMs:number)=>Promise<{content:string,usage:object|null}>} opts.infer
 *   forward call into the inference window; rejects on timeout/unready.
 * @param {()=>{modelLoaded:boolean,webgpu:object|null}} opts.status
 * @returns {Promise<import('node:http').Server>}
 */
function startBridge(opts) {
  if (!['127.0.0.1', 'localhost', '::1'].includes(opts.host)) {
    throw new Error(`bridge must bind loopback, got ${opts.host}`);
  }
  const server = http.createServer((req, res) => {
    handle(req, res, opts).catch((err) => {
      if (!res.headersSent) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: String(err && err.message || err) } }));
      }
    });
  });
  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(opts.port, opts.host, () => resolve(server));
  });
}

async function handle(req, res, opts) {
  const url = new URL(req.url || '/', `http://${opts.host}`);
  if (req.method === 'GET' && url.pathname === '/models') {
    return json(res, 200, { data: [{ id: MODEL_ID }] });
  }
  if (req.method === 'GET' && url.pathname === '/health') {
    const st = opts.status();
    return json(res, 200, {
      status: st.modelLoaded ? 'ok' : 'loading',
      model_loaded: st.modelLoaded,
      webgpu: st.webgpu,
    });
  }
  if (req.method === 'GET' && url.pathname.startsWith('/local-model/')) {
    // WebLLM's cleanModelUrl requires a /resolve/<branch>/ segment; accept
    // it and strip it so the same files serve under both URL shapes.
    let rel = url.pathname.slice('/local-model/'.length);
    const m = /^resolve\/[^/]+\/(.*)$/.exec(rel);
    if (m) rel = m[1];
    return serveFile(res, opts.modelDir, rel);
  }
  if (req.method === 'POST' && url.pathname === '/chat/completions') {
    return chat(req, res, opts);
  }
  return json(res, 404, { error: { message: 'not found' } });
}

function serveFile(res, root, rel) {
  const dest = path.normalize(path.join(root, rel));
  if (!dest.startsWith(path.normalize(root + path.sep))) {
    return json(res, 403, { error: { message: 'path escape' } });
  }
  fs.stat(dest, (err, st) => {
    if (err || !st.isFile()) return json(res, 404, { error: { message: 'not found' } });
    const type = dest.endsWith('.wasm') ? 'application/wasm'
      : dest.endsWith('.json') ? 'application/json'
        : 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': type, 'Content-Length': st.size });
    fs.createReadStream(dest).pipe(res);
  });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => {
      size += c.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
    req.on('error', reject);
  });
}

async function chat(req, res, opts) {
  let body;
  try {
    body = JSON.parse(await readBody(req));
  } catch {
    return json(res, 400, { error: { message: 'invalid json' } });
  }
  if (!body || !Array.isArray(body.messages) || body.messages.length === 0) {
    return json(res, 400, { error: { message: 'messages must be a non-empty array' } });
  }
  for (const m of body.messages) {
    if (!m || typeof m.role !== 'string' || typeof m.content !== 'string') {
      return json(res, 400, { error: { message: 'each message needs string role/content' } });
    }
    if (m.content.length > MAX_BODY_BYTES) {
      return json(res, 400, { error: { message: 'message too large' } });
    }
  }
  const maxTokens = clampInt(body.max_tokens, 1, 8192, 1024);
  const temperature = body.temperature === undefined || body.temperature === null
    ? null : clampFloat(body.temperature, 0, 2, 0);
  const rf = body.response_format || null;
  const wantJson = !!rf && (rf.type === 'json_object'
    || (rf.type === 'json_schema' && !!rf.json_schema));
  const schema = rf && rf.type === 'json_schema' ? (rf.json_schema.schema || null) : null;
  try {
    const out = await opts.infer({
      messages: body.messages.map((m) => ({ role: m.role, content: m.content })),
      maxTokens, temperature, json: wantJson, schema,
    }, 300_000);
    return json(res, 200, {
      id: `mlc-${Date.now()}`,
      object: 'chat.completion',
      model: MODEL_ID,
      choices: [{ index: 0, message: { role: 'assistant', content: out.content },
        finish_reason: 'stop' }],
      usage: out.usage || undefined,
    });
  } catch (err) {
    const msg = String((err && err.message) || err);
    const code = msg.includes('not loaded') || msg.includes('not ready') ? 503 : 500;
    return json(res, code, { error: { message: msg.slice(0, 500) } });
  }
}

function clampInt(v, lo, hi, fallback) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, Math.floor(n)));
}

function clampFloat(v, lo, hi, fallback) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, n));
}

function json(res, code, obj) {
  const data = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json', 'Content-Length': data.length });
  res.end(data);
}

module.exports = { startBridge, MODEL_ID };
