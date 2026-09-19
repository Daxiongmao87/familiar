/** Per-user data/cache locations for downloaded runtimes and models.
 *
 * Everything mutable lives outside the read-only AppImage filesystem.
 * `baseDir` is the Electron `userData` dir in production
 * (`~/.config/Familiar` via app.getPath) and a temp dir in tests.
 */
'use strict';

const path = require('node:path');

/**
 * Resolve the on-disk layout for installer state and components.
 * @param {string} baseDir per-user app-data dir (Electron userData).
 * @returns {{root:string, stateFile:string, componentDir:(id:string, sub:string)=>string}}
 */
function layout(baseDir) {
  const root = path.join(baseDir, 'local-ai');
  return {
    root,
    stateFile: path.join(root, 'install-state.json'),
    componentDir: (installDir) => path.join(root, installDir),
  };
}

module.exports = { layout };
