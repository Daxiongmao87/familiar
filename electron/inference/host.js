/** Hidden-window router: main process <-> WebLLM worker.
 *
 * This page's own thread never runs inference; it only routes. Chat
 * requests serialize through a promise chain (single-flight) because the
 * MLC engine executes one generation at a time — queueing here keeps
 * ordering explicit and backpressure visible instead of piling up GPU jobs.
 */
'use strict';

(function () {
  const bridge = window.familiarInfer;
  const worker = new Worker('worker.js');
  const pending = new Map();
  let seq = 0;
  let chain = Promise.resolve();
  let modelLoaded = false;

  function reportStatus() {
    bridge.status({ modelLoaded });
  }

  worker.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.type === 'load-progress') {
      bridge.loadProgress({ frac: msg.frac || 0, text: msg.text || '' });
      return;
    }
    if (msg.type === 'load-done') {
      modelLoaded = !!msg.ok;
      reportStatus();
      bridge.loadDone({ ok: msg.ok, error: msg.error || null });
      return;
    }
    if (msg.type === 'chat-done') {
      const entry = pending.get(msg.id);
      if (entry) {
        pending.delete(msg.id);
        entry(msg);
      }
    }
  };

  worker.onerror = (ev) => {
    bridge.loadDone({ ok: false, error: `worker error: ${ev.message || 'unknown'}` });
  };

  bridge.onProbeHardware(async () => {
    bridge.hardware(await probeHardware());
  });

  async function probeHardware() {
    try {
      if (!navigator.gpu) return { supported: false, reason: 'no-webgpu' };
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { supported: false, reason: 'no-adapter' };
      const features = Array.from(adapter.features || []);
      if (!features.includes('shader-f16')) {
        return { supported: false, reason: 'missing-shader-f16', features };
      }
      let name = '';
      try {
        const info = await adapter.requestAdapterInfo();
        name = (info && (info.device || info.description)) || '';
      } catch { /* optional */ }
      return { supported: true, adapter: name, features };
    } catch (err) {
      return { supported: false, reason: String((err && err.message) || err) };
    }
  }

  bridge.onLoad((req) => {
    if (!req || typeof req.modelUrl !== 'string' || typeof req.wasmUrl !== 'string') {
      bridge.loadDone({ ok: false, error: 'modelUrl/wasmUrl required' });
      return;
    }
    worker.postMessage({ type: 'load', modelUrl: req.modelUrl, wasmUrl: req.wasmUrl });
  });

  bridge.onChat((req) => {
    if (!req || typeof req.id !== 'string' || !Array.isArray(req.messages)) {
      bridge.chatDone({ id: (req && req.id) || '', ok: false, error: 'bad chat payload' });
      return;
    }
    chain = chain.then(() => runOne(req));
  });

  function runOne(req) {
    return new Promise((resolve) => {
      pending.set(req.id, (msg) => {
        bridge.chatDone({
          id: req.id, ok: !!msg.ok,
          content: msg.content || '', usage: msg.usage || null,
          error: msg.error || null,
        });
        resolve();
      });
      worker.postMessage({
        type: 'chat', id: req.id, messages: req.messages,
        maxTokens: req.maxTokens, temperature: req.temperature,
        json: !!req.json, schema: req.schema || null,
      });
    });
  }

  reportStatus();
})();
