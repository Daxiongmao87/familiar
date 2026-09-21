/** Provider flags in familiar-config.yaml without a YAML dependency.
 *
 * The seeded config keeps the committed template's line-oriented shape,
 * so provider reads/writes are small section scans. Unknown layouts fall
 * back to safe defaults (remote) rather than corrupting the file: writes
 * only touch lines they positively identify, and refuse when a section
 * is missing.
 */
'use strict';

const fs = require('node:fs');

const MODES = ['local', 'remote'];

function sectionOf(text, name) {
  const parts = text.split(new RegExp(`^${name}:\\s*(?:#.*)?$`, 'm'));
  if (parts.length < 2) return null;
  return parts[1].split(/^\S/m)[0];
}

function synthesisBlock(modelsSection) {
  return roleBlock(modelsSection, 'synthesis');
}

/** @private Loopback hosts mean the provisioned local STT server. */
function isLoopbackStt(host) {
  return host === '' || host === '127.0.0.1' || host === 'localhost' || host === '::1';
}

/**
 * Read provider modes + STT mode from a config file.
 * @param {string} configPath path to familiar-config.yaml.
 * @returns {{synthesis:string, jev:string, stt:string}} stt is local|remote
 *   (local ⇔ stream_host is loopback, served by the provisioned server).
 */
function scanProviders(configPath) {
  const out = { synthesis: 'remote', jev: 'remote', stt: 'remote' };
  const text = readText(configPath);
  if (text === null) return out;
  const models = sectionOf(text, 'models');
  const mProv = /provider:\s*(local|remote)/.exec(synthesisBlock(models));
  if (mProv) out.synthesis = mProv[1];
  const openjev = sectionOf(text, 'openjev');
  const mJev = openjev && /provider:\s*(local|remote)/.exec(openjev);
  if (mJev) out.jev = mJev[1];
  // Missing models section: corrupt layout, stay on safe remote. A missing
  // stt block or host key means the backend default (loopback → local).
  if (models) {
    const host = scalarOf(roleBlock(models, 'stt'), 'stream_host');
    out.stt = isLoopbackStt(host) ? 'local' : 'remote';
  }
  return out;
}

function readText(configPath) {
  try {
    return fs.readFileSync(configPath, 'utf-8');
  } catch {
    return null;
  }
}

/**
 * Read the setup-step-2 state: remote endpoint values and Discord ids.
 * Secrets are never returned — keys/token surface as booleans only.
 * @param {string} configPath path to familiar-config.yaml.
 * @returns {{synthesis:{base_url:string,model_id:string,has_api_key:boolean},
 *            jev:{base_url:string}, stt:{mode:string,stream_host:string,
 *            stream_port:string},
 *            discord:{has_token:boolean,guild_id:string,dm_user_id:string}}}
 */
function scanSetup(configPath) {
  const out = {
    synthesis: { base_url: '', model_id: '', has_api_key: false },
    jev: { base_url: '' },
    stt: { mode: 'local', stream_host: '127.0.0.1', stream_port: '43007' },
    discord: { has_token: false, guild_id: '', dm_user_id: '' },
  };
  const text = readText(configPath);
  if (text === null) return out;
  const models = sectionOf(text, 'models') || '';
  const synth = synthesisBlock(models);
  out.synthesis.base_url = scalarOf(synth, 'base_url');
  out.synthesis.model_id = scalarOf(synth, 'model_id');
  out.synthesis.has_api_key = secretSet(synth, 'api_key');
  const openjev = sectionOf(text, 'openjev') || '';
  out.jev.base_url = scalarOf(openjev, 'base_url');
  const sttBlock = roleBlock(models, 'stt');
  const streamHost = scalarOf(sttBlock, 'stream_host') || '127.0.0.1';
  const streamPort = scalarOf(sttBlock, 'stream_port') || '43007';
  out.stt.stream_host = streamHost;
  out.stt.stream_port = streamPort;
  out.stt.mode = isLoopbackStt(streamHost) ? 'local' : 'remote';
  const discord = sectionOf(text, 'discord') || '';
  out.discord.has_token = secretSet(discord, 'token');
  out.discord.guild_id = scalarOf(discord, 'guild_id');
  out.discord.dm_user_id = scalarOf(discord, 'dm_user_id');
  return out;
}

/** @private First `key: value` scalar in a block (quotes/comments stripped). */
function scalarOf(block, key) {
  const m = new RegExp(`^\\s*${key}:\\s*(.*?)\\s*(?:\\s+#.*)?$`, 'm').exec(block || '');
  if (!m) return '';
  let v = m[1].trim();
  if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
    v = v.slice(1, -1);
  }
  if (v === '' || v === 'null' || v === '~') return '';
  return v;
}

/** @private True when a secret key holds a real (non-placeholder) value. */
function secretSet(block, key) {
  const v = scalarOf(block, key);
  if (!v) return false;
  // ${VAR} placeholders and empty markers mean "not configured here".
  if (/^\$\{.+\}$/.test(v)) return false;
  return true;
}

function roleBlock(modelsSection, role) {
  if (!modelsSection) return '';
  const chunks = modelsSection.split(/^\s{2}(?=\S)/m);
  return chunks.find((b) => b.startsWith(`${role}:`)) || '';
}

/**
 * Rewrite provider flags in place (line surgery on the template shape).
 * @param {string} configPath path to familiar-config.yaml.
 * @param {{synthesis?:string, jev?:string}} modes validated modes.
 * @throws {Error} on invalid modes or unrecognized file layout.
 */
function writeProviders(configPath, modes) {
  for (const [key, val] of Object.entries(modes)) {
    if (!['synthesis', 'jev'].includes(key)) throw new Error(`unknown provider key: ${key}`);
    if (!MODES.includes(val)) throw new Error(`invalid ${key} mode: ${val}`);
  }
  let text = fs.readFileSync(configPath, 'utf-8');
  if (modes.synthesis !== undefined) {
    text = replaceInBlock(text, 'models', 'synthesis', '    ', modes.synthesis);
  }
  if (modes.jev !== undefined) {
    text = replaceInSection(text, 'openjev', '  ', modes.jev);
  }
  fs.writeFileSync(configPath, text, 'utf-8');
}

/** @private Replace or insert `provider:` inside models.<role>. */
function replaceInBlock(text, section, role, indent, mode) {
  return setKey(text, { section, role, indent, key: 'provider', value: mode });
}

/** @private Replace or insert `provider:` directly under a section. */
function replaceInSection(text, section, indent, mode) {
  return setKey(text, { section, role: null, indent, key: 'provider', value: mode });
}

/**
 * Set one scalar key inside a section (optionally inside a 2-space role
 * block), replacing the line in place (trailing comment preserved) or
 * inserting after the header when absent.
 * @private
 * @param {string} text full file text.
 * @param {{section:string, role:string|null, indent:string, key:string,
 *   value:string}} opts what to set.
 * @returns {string} updated text.
 * @throws {Error} when the section (or role) is missing.
 */
function setKey(text, opts) {
  const { section, role, indent, key, value } = opts;
  const secRe = new RegExp(`^${section}:\\s*(?:#.*)?$`, 'm');
  const secMatch = secRe.exec(text);
  if (!secMatch) throw new Error(`config missing ${section} section`);
  let spanStart = secMatch.index;
  let spanEnd = text.length;
  let insertAt = secMatch.index + secMatch[0].length;
  if (role !== null) {
    const roleRe = new RegExp(`^\\s{2}${role}:\\s*(?:#.*)?$`, 'm');
    const tail = text.slice(secMatch.index);
    const roleMatch = roleRe.exec(tail);
    if (!roleMatch) throw new Error(`config missing ${section}.${role}`);
    spanStart = secMatch.index + roleMatch.index;
    const afterRole = text.slice(spanStart + roleMatch[0].length);
    const nextRole = /^\s{2}(?=\S)/m.exec(afterRole);
    const nextSection = /^\S/m.exec(afterRole);
    for (const m of [nextRole, nextSection]) {
      if (m) spanEnd = Math.min(spanEnd, spanStart + roleMatch[0].length + m.index);
    }
    insertAt = spanStart + roleMatch[0].length;
  } else {
    const afterSec = text.slice(insertAt);
    const next = /^\S/m.exec(afterSec);
    if (next) spanEnd = insertAt + next.index;
  }
  const block = text.slice(spanStart, spanEnd);
  const lineRe = new RegExp(`^${indent}${key}:.*$`, 'm');
  const lineMatch = lineRe.exec(block);
  const rendered = `${indent}${key}: ${yamlScalar(value)}`;
  let nextBlock;
  if (lineMatch) {
    const comment = /(\s+#.*)$/.exec(lineMatch[0]);
    nextBlock = block.slice(0, lineMatch.index) + rendered + (comment ? comment[1] : '')
      + block.slice(lineMatch.index + lineMatch[0].length);
  } else {
    const at = insertAt - spanStart;
    nextBlock = `${block.slice(0, at)}\n${rendered}${block.slice(at)}`;
  }
  return text.slice(0, spanStart) + nextBlock + text.slice(spanEnd);
}

/** @private Render a string as a YAML scalar (quote only when needed). */
function yamlScalar(value) {
  const v = String(value);
  if (v !== '' && !/[\s#'"\\]|:\s/.test(v)) return v;
  return JSON.stringify(v);
}

/**
 * Validate + write the setup-step-2 credentials and endpoint choices.
 * Empty-string values mean "leave unchanged" (keys/token are never
 * cleared or echoed by setup). Restricts the config file to owner-only.
 * @param {string} configPath path to familiar-config.yaml.
 * @param {object} setup {discord?, synthesisRemote?, jevRemote?, stt?}.
 * @throws {Error} on invalid values or unrecognized layout.
 */
function writeSetupConfig(configPath, setup) {
  const s = setup || {};
  let text = fs.readFileSync(configPath, 'utf-8');
  if (s.discord !== undefined && s.discord !== null) {
    const d = s.discord;
    if (typeof d !== 'object') throw new Error('discord must be an object');
    for (const [key, label] of [['token', 'token'], ['guild_id', 'guild ID'], ['dm_user_id', 'DM user ID']]) {
      const v = d[key];
      if (v === undefined || v === null || v === '') continue;
      if (typeof v !== 'string' || v.trim() === '') throw new Error(`discord ${label} invalid`);
      text = setKey(text, { section: 'discord', role: null, indent: '  ', key, value: v.trim() });
    }
  }
  if (s.synthesisRemote !== undefined && s.synthesisRemote !== null) {
    const r = s.synthesisRemote;
    if (typeof r !== 'object') throw new Error('synthesisRemote must be an object');
    // Setup exposes one generation endpoint. Keep the optional fast role used
    // by background monitoring on that endpoint too.
    const hasFast = roleBlock(sectionOf(text, 'models'), 'fast') !== '';
    const generationRoles = hasFast ? ['synthesis', 'fast'] : ['synthesis'];
    if (r.base_url !== undefined && r.base_url !== '') {
      assertUrl(r.base_url, 'synthesis base URL');
      for (const role of generationRoles) {
        text = setKey(text, { section: 'models', role, indent: '    ',
          key: 'base_url', value: r.base_url.replace(/\/+$/, '') });
      }
    }
    if (r.model_id !== undefined && r.model_id !== '') {
      assertNonEmpty(r.model_id, 'synthesis model ID');
      for (const role of generationRoles) {
        text = setKey(text, { section: 'models', role, indent: '    ',
          key: 'model_id', value: r.model_id.trim() });
      }
    }
    if (r.api_key !== undefined && r.api_key !== null && r.api_key !== '') {
      assertNonEmpty(r.api_key, 'synthesis API key');
      for (const role of generationRoles) {
        text = setKey(text, { section: 'models', role, indent: '    ',
          key: 'api_key', value: r.api_key });
      }
    }
  }
  if (s.jevRemote !== undefined && s.jevRemote !== null) {
    const r = s.jevRemote;
    if (typeof r !== 'object') throw new Error('jevRemote must be an object');
    if (r.base_url !== undefined && r.base_url !== '') {
      assertUrl(r.base_url, 'JEV base URL');
      text = setKey(text, { section: 'openjev', role: null, indent: '  ',
        key: 'base_url', value: r.base_url.replace(/\/+$/, '') });
    }
  }
  if (s.stt !== undefined && s.stt !== null) {
    const r = s.stt;
    if (typeof r !== 'object') throw new Error('stt must be an object');
    if (r.mode !== 'local' && r.mode !== 'remote') throw new Error('stt.mode invalid');
    // Streaming-only: local points at the provisioned sidecar, remote at
    // a user-supplied whisper_online_server host:port. No dialects, no key.
    const host = r.mode === 'local' ? '127.0.0.1' : r.stream_host;
    const port = r.mode === 'local' ? '43007' : r.stream_port;
    assertNonEmpty(host, 'STT stream host');
    assertPort(port, 'STT stream port');
    text = setKey(text, { section: 'models', role: 'stt', indent: '    ',
      key: 'stream_host', value: String(host).trim() });
    text = setKey(text, { section: 'models', role: 'stt', indent: '    ',
      key: 'stream_port', value: String(port).trim() });
  }
  fs.writeFileSync(configPath, text, 'utf-8');
  try {
    fs.chmodSync(configPath, 0o600); // config holds secrets: owner-only
  } catch { /* best effort on odd filesystems */ }
}

function assertUrl(value, label) {
  if (typeof value !== 'string' || !value.startsWith('http://') && !value.startsWith('https://')) {
    throw new Error(`${label} must be an http(s) URL`);
  }
}

function assertNonEmpty(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function assertPort(value, label) {
  const n = Number(typeof value === 'string' ? value.trim() : value);
  if (!Number.isInteger(n) || n < 1 || n > 65535) {
    throw new Error(`${label} must be a port 1-65535`);
  }
}

module.exports = { scanProviders, scanSetup, writeProviders, writeSetupConfig };
