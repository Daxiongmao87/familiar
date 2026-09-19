/** Hardware gating tests: CUDA detection and per-provider verdicts. */
'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { detectCuda, verdict } = require('../hardware.js');

test('detectCuda parses GPU list', async () => {
  const out = await detectCuda(async () => ({
    stdout: 'GPU 0: Tesla T4 (UUID: GPU-aaa)\nGPU 1: NVIDIA RTX A2000 12GB (UUID: GPU-bbb)\n',
  }));
  assert.equal(out.available, true);
  assert.deepEqual(out.gpus, ['Tesla T4', 'NVIDIA RTX A2000 12GB']);
});

test('detectCuda reports missing driver and empty lists as unavailable', async () => {
  const missing = await detectCuda(async () => { throw new Error('spawn nvidia-smi ENOENT'); });
  assert.equal(missing.available, false);
  assert.match(missing.detail, /no NVIDIA driver/);
  const empty = await detectCuda(async () => ({ stdout: '\n' }));
  assert.equal(empty.available, false);
  const timeout = await detectCuda(async () => { throw new Error('timed out'); });
  assert.equal(timeout.available, false);
});

test('verdict grants each local provider independently', () => {
  const wgpu = { supported: true, adapter: 'Mesa' };
  const cuda = { available: true, detail: 'CUDA via 1 GPU(s)' };
  const full = verdict(wgpu, cuda);
  assert.equal(full.allLocal, true);
  assert.equal(full.allRemote, false);

  // WebGPU-only box (e.g. AMD/Intel): synthesis local, JEV/STT forced remote.
  const noCuda = verdict(wgpu, { available: false, detail: 'nope' });
  assert.equal(noCuda.synthesisLocal.ok, true);
  assert.equal(noCuda.jevLocal.ok, false);
  assert.equal(noCuda.sttLocal.ok, false);
  assert.equal(noCuda.allLocal, false);

  // CUDA box without WebGPU: JEV/STT local, synthesis forced remote.
  const noWgpu = verdict({ supported: false, reason: 'no-adapter' }, cuda);
  assert.equal(noWgpu.synthesisLocal.ok, false);
  assert.equal(noWgpu.jevLocal.ok, true);

  // Neither: everything forced remote.
  const none = verdict(null, null);
  assert.equal(none.allRemote, true);
  assert.match(none.synthesisLocal.reason, /unavailable/);
});
