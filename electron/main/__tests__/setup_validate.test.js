/** Setup validation tests: forcing rules and step-2 requirements. */
'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { checkVerdict, validateProviderModes, validateSaveSetup } = require('../setup_validate.js');

const FULL = {
  synthesisLocal: { ok: true, reason: 'WebGPU ready' },
  jevLocal: { ok: true, reason: 'CUDA ok' },
  sttLocal: { ok: true, reason: 'CUDA ok' },
};
const NOCUDA = {
  synthesisLocal: { ok: true, reason: 'WebGPU ready' },
  jevLocal: { ok: false, reason: 'no CUDA (none)' },
  sttLocal: { ok: false, reason: 'no CUDA (none)' },
};
function saved(over) {
  return Object.assign({
    synthesis: { base_url: '', model_id: '', has_api_key: false },
    jev: { base_url: '' },
    stt: { mode: 'remote', dialect: 'openai', base_url: '', has_api_key: false },
    discord: { has_token: false, guild_id: '', dm_user_id: '' },
  }, over || {});
}

test('validateProviderModes accepts known keys and modes only', () => {
  validateProviderModes({ synthesis: 'local' });
  validateProviderModes({ synthesis: 'remote', jev: 'local' });
  assert.throws(() => validateProviderModes({ stt: 'local' }), /invalid/);
  assert.throws(() => validateProviderModes({ synthesis: 'cloud' }), /invalid/);
  assert.throws(() => validateProviderModes(null), /invalid/);
});

test('checkVerdict refuses incapable local modes', () => {
  checkVerdict({ synthesis: 'local', jev: 'local' }, FULL);
  assert.throws(() => checkVerdict({ jev: 'local' }, NOCUDA), /local JEV unavailable/);
  checkVerdict({ synthesis: 'local', jev: 'remote' }, NOCUDA);
  checkVerdict({ jev: 'local' }, null);
});

test('validateSaveSetup requires remote URLs for remote providers', () => {
  const providers = { synthesis: 'remote', jev: 'remote' };
  assert.throws(() => validateSaveSetup({ stt: { mode: 'local' } }, providers, saved(), FULL, false),
    /synthesis base URL/);
  const s1 = saved({ synthesis: { base_url: 'http://x', model_id: '', has_api_key: false } });
  assert.throws(() => validateSaveSetup({ stt: { mode: 'local' } }, providers, s1, FULL, false),
    /synthesis model ID/);
  // Already-saved values satisfy requirements without re-entry.
  const s2 = saved({
    synthesis: { base_url: 'http://x', model_id: 'm', has_api_key: true },
    jev: { base_url: 'http://j' },
  });
  validateSaveSetup({ stt: { mode: 'local' } }, providers, s2, FULL, false);
  // Local providers need no URLs; discord always optional.
  validateSaveSetup({ stt: { mode: 'local' } },
    { synthesis: 'local', jev: 'local' }, saved(), FULL, false);
  validateSaveSetup({ discord: { token: 't' }, stt: { mode: 'local' } },
    { synthesis: 'local', jev: 'local' }, saved(), FULL, false);
});

test('validateSaveSetup forces STT remote on CUDA-less hardware', () => {
  const providers = { synthesis: 'remote', jev: 'remote' };
  const s = saved({
    synthesis: { base_url: 'http://x', model_id: 'm', has_api_key: false },
    jev: { base_url: 'http://j' },
  });
  assert.throws(
    () => validateSaveSetup({ stt: { mode: 'local' } }, providers, s, NOCUDA, true),
    /cannot run the local STT server/);
  assert.throws(
    () => validateSaveSetup({ stt: { mode: 'remote', base_url: '' } }, providers, s, NOCUDA, true),
    /STT base URL/);
  validateSaveSetup({ stt: { mode: 'remote', dialect: 'openai', base_url: 'http://s' } },
    providers, s, NOCUDA, true);
});
