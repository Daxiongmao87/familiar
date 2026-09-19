/** Hardware capability detection for local-inference gating.
 *
 * Local providers have hard requirements: synthesis needs WebGPU with
 * shader-f16 (probed in the renderer, see inference/host.js), while JEV
 * and the streaming STT server need NVIDIA CUDA (AWQ kernels;
 * faster-whisper's pinned server hardcodes device="cuda").
 *
 * This module detects the main-process side (CUDA) and combines it with
 * the renderer's WebGPU probe into a per-provider verdict. Anything not
 * capable is forced to its endpoint mode — the user is never offered a
 * local option their machine cannot run.
 *
 * Node stdlib only (child_process), exec-injectable for tests.
 */
'use strict';

const { execFile } = require('node:child_process');

const NVIDIA_SMI_TIMEOUT_MS = 8000;

/**
 * Detect NVIDIA CUDA via nvidia-smi.
 * @param {(file:string,args:string[],opts:object)=>Promise<{stdout:string}>} [run]
 *   exec shim (default: real execFile). Rejects on missing binary/timeout.
 * @returns {Promise<{available:boolean,gpus:string[],detail:string}>}
 */
async function detectCuda(run) {
  const exec = run || defaultExec;
  try {
    const { stdout } = await exec('nvidia-smi', ['-L'], { timeout: NVIDIA_SMI_TIMEOUT_MS });
    const gpus = String(stdout || '').split('\n')
      .map((l) => l.trim()).filter(Boolean)
      .map((l) => l.replace(/^GPU \d+:\s*/, '').split(' (UUID')[0].trim())
      .filter(Boolean);
    if (gpus.length === 0) {
      return { available: false, gpus: [], detail: 'nvidia-smi reports no GPUs' };
    }
    return { available: true, gpus, detail: `CUDA via ${gpus.length} GPU(s)` };
  } catch (err) {
    const msg = String((err && err.message) || err);
    if (/ENOENT|not found|not recognized/i.test(msg)) {
      return { available: false, gpus: [], detail: 'nvidia-smi not found (no NVIDIA driver)' };
    }
    return { available: false, gpus: [], detail: `nvidia-smi failed: ${msg.slice(0, 120)}` };
  }
}

function defaultExec(file, args, opts) {
  return new Promise((resolve, reject) => {
    execFile(file, args, opts, (err, stdout) => {
      if (err) reject(err);
      else resolve({ stdout });
    });
  });
}

/**
 * Combine WebGPU + CUDA probes into a per-provider local verdict.
 * @param {{supported:boolean,reason?:string,adapter?:string}|null} webgpu
 * @param {{available:boolean,detail:string}|null} cuda
 * @returns {{synthesisLocal:{ok:boolean,reason:string},
 *            jevLocal:{ok:boolean,reason:string},
 *            sttLocal:{ok:boolean,reason:string},
 *            allLocal:boolean, allRemote:boolean}}
 */
function verdict(webgpu, cuda) {
  const wgpuOk = !!(webgpu && webgpu.supported);
  const cudaOk = !!(cuda && cuda.available);
  const wgpuWhy = !webgpu ? 'WebGPU probe unavailable'
    : wgpuOk ? `WebGPU ready (${(webgpu.adapter || 'adapter ok')})`
      : `WebGPU unavailable (${webgpu.reason || 'unknown'})`;
  const cudaWhy = !cuda ? 'CUDA probe unavailable'
    : cudaOk ? cuda.detail : `no CUDA (${cuda.detail})`;
  const synth = { ok: wgpuOk, reason: wgpuWhy };
  const jev = { ok: cudaOk, reason: cudaWhy };
  const stt = { ok: cudaOk, reason: cudaWhy };
  return {
    synthesisLocal: synth, jevLocal: jev, sttLocal: stt,
    allLocal: synth.ok && jev.ok,
    allRemote: !synth.ok && !jev.ok,
  };
}

module.exports = { detectCuda, verdict };
