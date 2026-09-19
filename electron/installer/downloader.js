/** Resumable, verified, atomic file downloads for the first-run installer.
 *
 * Guarantees per file:
 * - stream to disk (never buffer whole files in RAM);
 * - resume with Range requests when the server allows (else restart);
 * - retry transient failures with backoff;
 * - verify expected size and optional sha256 before promoting;
 * - download to `<final>.part` and atomically rename after verification;
 * - recover cleanly from termination: a leftover `.part` resumes, a
 *   complete-but-unverified file is never treated as installed.
 *
 * Node stdlib only (http/https/fs/crypto) so tests run with plain node.
 */
'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const https = require('node:https');
const path = require('node:path');

const MAX_REDIRECTS = 5;
const BACKOFF_MS = [1000, 2000, 4000];

class DownloadError extends Error {
  /**
   * @param {string} message human-readable failure.
   * @param {string} code 'checksum' | 'size' | 'network' | 'http'.
   * @param {object} [extra] e.g. {status}.
   */
  constructor(message, code, extra) {
    super(message);
    this.name = 'DownloadError';
    this.code = code;
    Object.assign(this, extra || {});
  }
}

/**
 * One HTTP(S) request following redirects, resolving with {res, url}.
 * @private
 */
function requestOnce(url, headers, timeoutMs, redirectsLeft) {
  return new Promise((resolve, reject) => {
    const lib = url.startsWith('https:') ? https : http;
    const req = lib.get(url, { headers, timeout: timeoutMs }, (res) => {
      const status = res.statusCode || 0;
      if ([301, 302, 303, 307, 308].includes(status) && res.headers.location) {
        res.resume();
        if (redirectsLeft <= 0) {
          reject(new DownloadError(`too many redirects for ${url}`, 'network'));
          return;
        }
        const next = new URL(res.headers.location, url).toString();
        requestOnce(next, headers, timeoutMs, redirectsLeft - 1)
          .then(resolve, reject);
        return;
      }
      resolve({ res, url });
    });
    req.on('timeout', () => req.destroy(new Error('request timeout')));
    req.on('error', reject);
  });
}

/**
 * Download one file with resume, retry, verification, and atomic promote.
 *
 * @param {string} url source URL (redirects followed).
 * @param {string} finalPath destination path (parent created as needed).
 * @param {object} [opts]
 * @param {number|null} [opts.expectedSize] byte count enforced after download.
 * @param {string|null} [opts.sha256] hex digest enforced after download.
 * @param {number} [opts.maxRetries=3]
 * @param {number} [opts.timeoutMs=30000] per-request socket timeout.
 * @param {(downloaded:number,total:number|null)=>void} [opts.onProgress]
 * @param {AbortSignal} [opts.signal]
 * @returns {Promise<{bytes:number,resumed:boolean,sha256:string|null}>}
 */
async function downloadFile(url, finalPath, opts) {
  const o = Object.assign(
    { expectedSize: null, sha256: null, maxRetries: 3, timeoutMs: 30000 },
    opts || {},
  );
  const partPath = `${finalPath}.part`;
  await fs.promises.mkdir(path.dirname(finalPath), { recursive: true });

  let attempt = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    if (o.signal && o.signal.aborted) {
      throw new DownloadError('download aborted', 'network');
    }
    try {
      return await attemptOnce(url, finalPath, partPath, o);
    } catch (err) {
      if (err instanceof DownloadError && (err.code === 'checksum' || err.code === 'size')) {
        throw err; // verified bytes are wrong: retrying the same URL is pointless
      }
      attempt += 1;
      if (attempt > o.maxRetries) {
        if (err instanceof DownloadError) throw err;
        throw new DownloadError(`download failed: ${err.message}`, 'network');
      }
      await new Promise((r) => setTimeout(r, BACKOFF_MS[Math.min(attempt - 1, 2)]));
    }
  }
}

/** @private Single attempt; may resume an existing `.part`. */
async function attemptOnce(url, finalPath, partPath, o) {
  let start = 0;
  try {
    const st = await fs.promises.stat(partPath);
    start = st.size;
  } catch { start = 0; }
  if (o.expectedSize !== null && start > o.expectedSize) start = 0;
  const headers = {};
  if (start > 0) headers.Range = `bytes=${start}-`;
  const { res } = await requestOnce(url, headers, o.timeoutMs, MAX_REDIRECTS);
  const status = res.statusCode || 0;
  try {
    if (status !== 200 && status !== 206) {
      throw new DownloadError(`HTTP ${status} for ${url}`, 'http', { status });
    }
    let resumed = status === 206 && start > 0;
    if (status === 200 && start > 0) {
      start = 0; // server ignored Range: restart from byte zero
      resumed = false;
    }
    const total = o.expectedSize !== null ? o.expectedSize : null;
    const hash = o.sha256 ? crypto.createHash('sha256') : null;
    if (hash && resumed) {
      // Hash must cover the resumed prefix too; stream it back through.
      await hashFile(partPath, hash);
    }
    const out = fs.createWriteStream(partPath, { flags: resumed ? 'a' : 'w' });
    let written = start;
    if (o.onProgress) o.onProgress(written, total);
    await new Promise((resolve, reject) => {
      res.on('data', (chunk) => {
        written += chunk.length;
        if (hash) hash.update(chunk);
        if (!out.write(chunk)) res.pause();
        if (o.onProgress) o.onProgress(written, total);
      });
      out.on('drain', () => res.resume());
      res.on('end', () => out.end(resolve));
      res.on('error', reject);
      out.on('error', reject);
      if (o.signal) {
        o.signal.addEventListener('abort', () => {
          res.destroy();
          reject(new DownloadError('download aborted', 'network'));
        }, { once: true });
      }
    });
    if (o.expectedSize !== null && written !== o.expectedSize) {
      throw new DownloadError(
        `size mismatch for ${finalPath}: got ${written}, want ${o.expectedSize}`,
        'size',
      );
    }
    let digest = null;
    if (hash) {
      digest = hash.digest('hex');
      if (digest !== o.sha256.toLowerCase()) {
        await fs.promises.unlink(partPath).catch(() => {});
        throw new DownloadError(
          `checksum mismatch for ${finalPath}: got ${digest.slice(0, 16)}…`,
          'checksum',
        );
      }
    }
    await fs.promises.rename(partPath, finalPath);
    return { bytes: written, resumed, sha256: digest };
  } finally {
    res.resume(); // drain on error paths so sockets close promptly
  }
}

/** @private Stream an existing file through a hash (resume prefix). */
function hashFile(filePath, hash) {
  return new Promise((resolve, reject) => {
    const stream = fs.createReadStream(filePath);
    stream.on('data', (c) => hash.update(c));
    stream.on('end', resolve);
    stream.on('error', reject);
  });
}

module.exports = { downloadFile, DownloadError };
