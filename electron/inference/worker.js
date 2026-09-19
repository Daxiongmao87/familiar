/** WebLLM worker: resident MiniCPM5-2B engine, one generation at a time.
 *
 * Performance posture (only mechanisms WebLLM/MLC 0.2.85 actually has):
 * - the engine is created once and kept resident (KV/prompt state reuse
 *   is internal to the resident engine; there is no manual cache API);
 * - q4f16_1 quantized build (the pinned MLC artifact);
 * - prefill chunk / context window from the model's own mlc-chat-config
 *   (not overridden — the build was tuned for them);
 * - thinking disabled via extra_body.enable_thinking=false (Qwen path;
 *   MiniCPM ignores it harmlessly);
 * - no invented attention flags: MLC selects its own GPU kernels.
 *
 * Structured output: when the bridge asks for JSON, request WebLLM's
 * JSON-object mode and keep the schema text in the prompt; on any
 * response_format rejection, retry once unconstrained (Python's JSON
 * extraction tolerates prose around the object).
 */
'use strict';

importScripts('./vendor/webllm.bundle.js');

const MODEL_ID = 'minicpm5-2b-mlc';
let engine = null;

function progress(frac, text) {
  postMessage({ type: 'load-progress', frac, text });
}

onmessage = async (ev) => {
  const msg = ev.data || {};
  if (msg.type === 'load') {
    await doLoad(msg);
  } else if (msg.type === 'chat') {
    await doChat(msg);
  }
};

async function doLoad(msg) {
  try {
    if (typeof webllm === 'undefined' || !webllm.CreateMLCEngine) {
      throw new Error('web-llm bundle missing (run npm run bundle:webllm at build time)');
    }
    // Fail fast when the bridge isn't serving the model dir.
    // modelUrl already carries the /resolve/<branch>/ segment WebLLM wants.
    const cfgResp = await fetch(`${msg.modelUrl}/mlc-chat-config.json`);
    if (!cfgResp.ok) throw new Error(`model assets unreachable: ${cfgResp.status}`);
    progress(0.02, 'creating WebLLM engine…');
    engine = await webllm.CreateMLCEngine(MODEL_ID, {
      logLevel: 'WARN',
      appConfig: {
        model_list: [{
          model: msg.modelUrl.endsWith('/') ? msg.modelUrl : `${msg.modelUrl}/`,
          model_id: MODEL_ID,
          model_lib: msg.wasmUrl,
        }],
      },
      initProgressCallback: (p) => progress(p.progress || 0, p.text || 'loading…'),
    });
    // Warmup: compile a real pass before timing-sensitive requests.
    progress(0.99, 'warming up…');
    await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'ok' }],
      max_tokens: 1, temperature: 0, extra_body: { enable_thinking: false },
    });
    postMessage({ type: 'load-done', ok: true });
  } catch (err) {
    postMessage({ type: 'load-done', ok: false, error: String((err && err.message) || err) });
  }
};

async function doChat(msg) {
  if (!engine) {
    postMessage({ type: 'chat-done', id: msg.id, ok: false, error: 'model not loaded' });
    return;
  }
  try {
    const messages = msg.json ? withJsonHint(msg.messages, msg.schema) : msg.messages;
    const base = {
      messages,
      max_tokens: msg.maxTokens || 1024,
      extra_body: { enable_thinking: false },
    };
    if (typeof msg.temperature === 'number') base.temperature = msg.temperature;
    let resp;
    if (msg.json) {
      // response_format per the 0.2.85 .d.ts: type json_object (+ schema
      // string when Familiar supplied a json_schema). The prompt hint is
      // still required — without it the model can spin on whitespace.
      const responseFormat = { type: 'json_object' };
      if (msg.schema) {
        try {
          responseFormat.schema = JSON.stringify(msg.schema);
        } catch { /* fall through to un-constrained JSON mode */ }
      }
      try {
        resp = await engine.chat.completions.create(
          Object.assign({ response_format: responseFormat }, base));
      } catch (err) {
        // Older/odd builds may reject response_format: retry plain.
        if (!/response_format|response format/i.test(String((err && err.message) || ''))) throw err;
        resp = await engine.chat.completions.create(base);
      }
    } else {
      resp = await engine.chat.completions.create(base);
    }
    const choice = (resp.choices && resp.choices[0]) || {};
    const content = (choice.message && choice.message.content) || '';
    postMessage({ type: 'chat-done', id: msg.id, ok: true,
      content: String(content), usage: resp.usage || null });
  } catch (err) {
    postMessage({ type: 'chat-done', id: msg.id, ok: false,
      error: String((err && err.message) || err).slice(0, 800) });
  }
}

function withJsonHint(messages, schema) {
  const hint = schema
    ? `Respond with a single JSON object matching this schema (no prose outside it):\n${JSON.stringify(schema).slice(0, 4000)}`
    : 'Respond with a single JSON object (no prose outside it).';
  const out = messages.map((m) => ({ role: m.role, content: m.content }));
  const last = out[out.length - 1];
  if (last && last.role === 'user') {
    last.content = `${last.content}\n\n${hint}`;
  } else {
    out.push({ role: 'user', content: hint });
  }
  return out;
}
