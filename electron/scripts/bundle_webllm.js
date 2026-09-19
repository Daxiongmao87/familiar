/** Build step: bundle @mlc-ai/web-llm into one IIFE for the worker.
 *
 * The worker runs under file:// with importScripts (no bare specifiers),
 * so the npm package is bundled with esbuild at build time. Output is a
 * build artifact (gitignored): electron/inference/vendor/webllm.bundle.js.
 *
 * Usage: node scripts/bundle_webllm.js  (or npm run bundle:webllm)
 */
'use strict';

const fs = require('node:fs');
const path = require('node:path');

async function main() {
  const esbuild = require('esbuild');
  const root = path.resolve(__dirname, '..');
  const candidates = [
    path.join(root, 'node_modules', '@mlc-ai', 'web-llm', 'lib', 'index.js'),
  ];
  try {
    candidates.unshift(require.resolve('@mlc-ai/web-llm'));
  } catch { /* fall back to the lib path */ }
  const entry = candidates.find((p) => fs.existsSync(p));
  if (!entry) {
    throw new Error('web-llm not installed (npm install) and no bundle entry found');
  }
  const outdir = path.join(root, 'inference', 'vendor');
  fs.mkdirSync(outdir, { recursive: true });
  await esbuild.build({
    entryPoints: [entry],
    bundle: true,
    format: 'iife',
    globalName: 'webllm',
    platform: 'browser',
    target: 'es2022',
    minify: true,
    outfile: path.join(outdir, 'webllm.bundle.js'),
    logLevel: 'warning',
  });
  // eslint-disable-next-line no-console
  console.log(`bundled ${entry} -> inference/vendor/webllm.bundle.js`);
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err.message || err);
  process.exit(1);
});
