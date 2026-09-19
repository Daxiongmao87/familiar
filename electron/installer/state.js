/** Install-state tracking and migration for downloaded components.
 *
 * `install-state.json` records, per component, the installed version and
 * per-file sizes, so subsequent launches re-verify in milliseconds and
 * only fetch what changed. Writes are atomic (temp file + rename).
 */
'use strict';

const fs = require('node:fs');
const path = require('node:path');

/**
 * Load state; a missing or corrupt file yields a fresh empty state
 * (recover cleanly, never crash first-run on a torn write — the writer
 * is atomic, but disks and kills happen).
 * @param {string} stateFile path to install-state.json.
 * @returns {{manifest_version:number|null,components:Object}}
 */
function loadState(stateFile) {
  try {
    const raw = fs.readFileSync(stateFile, 'utf-8');
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && parsed.components) return parsed;
  } catch { /* fall through to fresh state */ }
  return { manifest_version: null, components: {} };
}

/**
 * Atomically persist state.
 * @param {string} stateFile path to install-state.json.
 * @param {object} state state object.
 */
function saveState(stateFile, state) {
  fs.mkdirSync(path.dirname(stateFile), { recursive: true });
  const tmp = `${stateFile}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(state, null, 2), 'utf-8');
  fs.renameSync(tmp, stateFile);
}

/**
 * Migrate old state onto a new manifest: keep per-file records whose
 * component version AND file size still match; drop everything else so
 * changed files re-download while untouched ones are kept (no blind
 * redownload of gigabytes on a version bump).
 * @param {object} manifest parsed manifest.json.
 * @param {object} state loaded state (mutated in place, also returned).
 * @returns {object} the migrated state.
 */
function migrateState(manifest, state) {
  if (state.manifest_version === manifest.manifest_version) return state;
  const wanted = new Map((manifest.components || []).map((c) => [c.id, c]));
  const kept = {};
  for (const [id, rec] of Object.entries(state.components || {})) {
    const comp = wanted.get(id);
    if (!comp || rec.version !== comp.version) continue;
    const sizes = new Map((comp.files || []).map((f) => [f.path, f.size]));
    const files = {};
    for (const [rel, meta] of Object.entries(rec.files || {})) {
      if (sizes.get(rel) === (meta && meta.size)) files[rel] = meta;
    }
    kept[id] = { version: rec.version, status: 'partial', files };
  }
  state.manifest_version = manifest.manifest_version;
  state.components = kept;
  return state;
}

/**
 * Plan the remaining work: files missing on disk or failing the
 * size check, for the wanted components only (provider selection
 * decides which components are wanted).
 * @param {object} manifest parsed manifest.json.
 * @param {object} state loaded (and migrated) state.
 * @param {(id:string, dir:string)=>string} dirFor maps install_dir to path.
 * @param {string[]} wantedIds component ids to install (default: all files).
 * @returns {{files:Array, totalBytes:number, doneBytes:number}}
 */
function planInstall(manifest, state, dirFor, wantedIds) {
  const wanted = wantedIds || (manifest.components || [])
    .filter((c) => c.kind === 'files').map((c) => c.id);
  const files = [];
  let totalBytes = 0;
  let doneBytes = 0;
  for (const comp of manifest.components || []) {
    if (!wanted.includes(comp.id) || comp.kind !== 'files') continue;
    const dir = dirFor(comp.install_dir);
    for (const f of comp.files || []) {
      totalBytes += f.size || 0;
      const dest = path.join(dir, f.path);
      let ok = false;
      try {
        ok = fs.statSync(dest).size === f.size;
      } catch { ok = false; }
      if (ok) {
        doneBytes += f.size || 0;
      } else {
        files.push({ component: comp, file: f, dest });
      }
    }
  }
  return { files, totalBytes, doneBytes };
}

/**
 * Record one verified file (call after its atomic promote).
 * @param {object} state loaded state (mutated in place).
 * @param {object} component manifest component entry.
 * @param {string} relPath file path relative to the component dir.
 * @param {number} size verified byte count.
 */
function markFileDone(state, component, relPath, size) {
  const comps = state.components || (state.components = {});
  const rec = comps[component.id] || (comps[component.id] = {
    version: component.version, status: 'partial', files: {},
  });
  rec.version = component.version;
  rec.files[relPath] = { size, finished_at: new Date().toISOString() };
  const want = (component.files || []).length;
  if (Object.keys(rec.files).length >= want) rec.status = 'done';
}

/**
 * True when every wanted `files`-kind component is fully on disk.
 * @param {object} manifest parsed manifest.json.
 * @param {(id:string, dir:string)=>string} dirFor maps install_dir to path.
 * @param {string[]} wantedIds component ids required.
 * @returns {boolean}
 */
function allInstalled(manifest, dirFor, wantedIds) {
  const { files } = planInstall(manifest, { components: {} }, dirFor, wantedIds);
  return files.length === 0;
}

/**
 * Copy one manifest-vendored file (shipped in app resources, e.g. the
 * patched STT server) into its component dir, size-verified.
 * @param {string} rel resources-relative path from the manifest entry.
 * @param {string} dest absolute destination path.
 * @param {number} expectedSize manifest byte count (enforced).
 * @param {{dev:boolean, resourcesPath:string, repoRoot:string}} roots
 *   dev reads from the repo services/ tree, prod from app resources.
 * @throws {Error} when the resource is missing or the size mismatches.
 */
function installVendoredFile(rel, dest, expectedSize, roots) {
  const src = roots.dev
    ? path.join(roots.repoRoot, 'services', rel)
    : path.join(roots.resourcesPath, rel);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
  const actual = fs.statSync(dest).size;
  if (actual !== expectedSize) {
    throw new Error(`vendored ${rel}: size ${actual} != manifest ${expectedSize}`);
  }
}

module.exports = { loadState, saveState, migrateState, planInstall, markFileDone, allInstalled, installVendoredFile };
