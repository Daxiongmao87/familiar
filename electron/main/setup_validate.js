/** Pure validation for setup provider/credential choices (testable).
 *
 * The IPC handlers in ipc.js are thin wrappers over these functions so
 * the forcing and requirement rules are unit-covered without Electron.
 */
'use strict';

const MODES = ['local', 'remote'];

/**
 * Validate a provider-mode patch shape.
 * @param {object} modes {synthesis?, jev?}.
 * @throws {Error} on shape violations.
 */
function validateProviderModes(modes) {
  if (!modes || typeof modes !== 'object') throw new Error('invalid provider modes');
  for (const [key, mode] of Object.entries(modes)) {
    if (!['synthesis', 'jev'].includes(key)) throw new Error('invalid provider modes');
    if (!MODES.includes(mode)) throw new Error('invalid provider modes');
  }
}

/**
 * Refuse local modes the hardware verdict forbids.
 * @param {{synthesis?:string, jev?:string}} modes requested modes.
 * @param {object|null} hwVerdict combined verdict (see hardware.js).
 * @throws {Error} naming the incapable provider and reason.
 */
function checkVerdict(modes, hwVerdict) {
  const v = hwVerdict;
  if (!v) return;
  if (modes.synthesis === 'local' && !v.synthesisLocal.ok) {
    throw new Error(`local synthesis unavailable: ${v.synthesisLocal.reason}`);
  }
  if (modes.jev === 'local' && !v.jevLocal.ok) {
    throw new Error(`local JEV unavailable: ${v.jevLocal.reason}`);
  }
}

/**
 * Validate a step-2 save: required remote URLs for remote providers,
 * forced-remote STT must flip, local STT needs CUDA. Discord is soft
 * (absent/empty skips). Secrets already saved satisfy requirements.
 * @param {object} setup {discord?, synthesisRemote?, jevRemote?, stt?}.
 * @param {{synthesis:string, jev:string}} providers current provider modes.
 * @param {object} saved scanSetup() of the current config file.
 * @param {object|null} hwVerdict combined verdict.
 * @param {boolean} sttMustBeRemote STT set to local on CUDA-less hardware.
 * @throws {Error} on the first unmet requirement.
 */
function validateSaveSetup(setup, providers, saved, hwVerdict, sttMustBeRemote) {
  if (!setup || typeof setup !== 'object') throw new Error('invalid setup');
  const need = (cond, value, have, label) => {
    if (cond && (!value || value === '') && !have) {
      throw new Error(`${label} is required for the chosen providers`);
    }
  };
  const synRemote = providers.synthesis === 'remote';
  const jevRemote = providers.jev === 'remote';
  const stt = setup.stt || {};
  need(synRemote, (setup.synthesisRemote || {}).base_url,
    saved.synthesis.base_url, 'synthesis base URL');
  need(synRemote, (setup.synthesisRemote || {}).model_id,
    saved.synthesis.model_id, 'synthesis model ID');
  need(jevRemote, (setup.jevRemote || {}).base_url,
    saved.jev.base_url, 'JEV base URL');
  if (sttMustBeRemote && stt.mode !== 'remote') {
    throw new Error('this machine cannot run the local STT server: choose a remote STT server');
  }
  if (stt.mode === 'remote') {
    // A saved loopback host is the local default — it cannot satisfy remote.
    const savedHost = saved.stt.stream_host || '';
    const savedRemote = ['127.0.0.1', 'localhost', '::1'].includes(savedHost) ? '' : savedHost;
    need(true, stt.stream_host, savedRemote, 'STT stream host');
    need(true, stt.stream_port, saved.stt.stream_port, 'STT stream port');
  }
  if (stt.mode === 'local' && hwVerdict && !hwVerdict.sttLocal.ok) {
    throw new Error(`local STT unavailable: ${hwVerdict.sttLocal.reason}`);
  }
}

module.exports = { validateProviderModes, checkVerdict, validateSaveSetup };
