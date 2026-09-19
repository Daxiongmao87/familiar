/** Downloader tests: resume, retry, checksum, atomicity (real HTTP server). */
'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { downloadFile, DownloadError } = require('../downloader.js');

function sha256(buf) {
  return crypto.createHash('sha256').update(buf).digest('hex');
}

/** Test server with /file (range-aware), /flaky (dies once), /wrong (bad bytes). */
function startServer(payload) {
  let flakyHits = 0;
  const server = http.createServer((req, res) => {
    if (req.url === '/flaky') {
      flakyHits += 1;
      if (flakyHits === 1) {
        res.writeHead(200, { 'Content-Length': payload.length });
        res.write(payload.subarray(0, 8));
        setTimeout(() => res.destroy(), 20); // truncated: forces retry+resume
        return;
      }
    }
    if (req.url !== '/file' && req.url !== '/flaky' && req.url !== '/wrong') {
      res.writeHead(404).end();
      return;
    }
    const body = req.url === '/wrong' ? Buffer.from('corrupted-bytes!!') : payload;
    const range = req.headers.range;
    if (range) {
      const m = /^bytes=(\d+)-$/.exec(range);
      const start = m ? parseInt(m[1], 10) : 0;
      if (start >= body.length) {
        res.writeHead(416).end();
        return;
      }
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${body.length - 1}/${body.length}`,
        'Content-Length': body.length - start,
        'Accept-Ranges': 'bytes',
      });
      res.end(body.subarray(start));
      return;
    }
    res.writeHead(200, { 'Content-Length': body.length, 'Accept-Ranges': 'bytes' });
    res.end(body);
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

function tmpdir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'fam-dl-'));
}

test('happy path streams to disk and verifies size+checksum', async () => {
  const payload = crypto.randomBytes(200_000);
  const server = await startServer(payload);
  try {
    const dir = tmpdir();
    const dest = path.join(dir, 'model.bin');
    const progress = [];
    const out = await downloadFile(
      `http://127.0.0.1:${server.address().port}/file`, dest,
      { expectedSize: payload.length, sha256: sha256(payload),
        onProgress: (d, t) => progress.push([d, t]) },
    );
    assert.equal(out.bytes, payload.length);
    assert.equal(out.resumed, false);
    assert.deepEqual(fs.readFileSync(dest), payload);
    assert.ok(!fs.existsSync(`${dest}.part`), 'no .part left behind');
    assert.ok(progress.length > 1, 'progress reported incrementally');
    assert.equal(progress[progress.length - 1][0], payload.length);
  } finally {
    server.close();
  }
});

test('interrupted download resumes with Range and completes', async () => {
  const payload = crypto.randomBytes(100_000);
  const server = await startServer(payload);
  try {
    const dir = tmpdir();
    const dest = path.join(dir, 'model.bin');
    // Simulate a killed first run: a partial .part file on disk.
    fs.writeFileSync(`${dest}.part`, payload.subarray(0, 30_000));
    const out = await downloadFile(
      `http://127.0.0.1:${server.address().port}/file`, dest,
      { expectedSize: payload.length, sha256: sha256(payload) },
    );
    assert.equal(out.resumed, true);
    assert.deepEqual(fs.readFileSync(dest), payload);
  } finally {
    server.close();
  }
});

test('truncated response retries and still verifies', async () => {
  const payload = crypto.randomBytes(60_000);
  const server = await startServer(payload);
  try {
    const dir = tmpdir();
    const dest = path.join(dir, 'model.bin');
    const out = await downloadFile(
      `http://127.0.0.1:${server.address().port}/flaky`, dest,
      { expectedSize: payload.length, sha256: sha256(payload), maxRetries: 3 },
    );
    assert.equal(out.bytes, payload.length);
    assert.deepEqual(fs.readFileSync(dest), payload);
  } finally {
    server.close();
  }
});

test('checksum failure deletes the part file and reports checksum', async () => {
  const payload = crypto.randomBytes(1000);
  const server = await startServer(payload);
  try {
    const dir = tmpdir();
    const dest = path.join(dir, 'model.bin');
    await assert.rejects(
      downloadFile(`http://127.0.0.1:${server.address().port}/wrong`, dest,
        { expectedSize: 17, sha256: sha256(payload), maxRetries: 1 }),
      (err) => {
        assert.ok(err instanceof DownloadError);
        assert.equal(err.code, 'checksum');
        return true;
      },
    );
    assert.ok(!fs.existsSync(`${dest}.part`), 'bad bytes removed for clean retry');
    assert.ok(!fs.existsSync(dest), 'never promoted without verification');
  } finally {
    server.close();
  }
});

test('size mismatch is reported and never promoted', async () => {
  const payload = crypto.randomBytes(1000);
  const server = await startServer(payload);
  try {
    const dir = tmpdir();
    const dest = path.join(dir, 'model.bin');
    await assert.rejects(
      downloadFile(`http://127.0.0.1:${server.address().port}/file`, dest,
        { expectedSize: payload.length + 5, maxRetries: 1 }),
      (err) => {
        assert.ok(err instanceof DownloadError);
        assert.equal(err.code, 'size');
        return true;
      },
    );
    assert.ok(!fs.existsSync(dest));
  } finally {
    server.close();
  }
});

test('HTTP errors surface with status after retries', async () => {
  const server = await startServer(Buffer.from('x'));
  try {
    const dir = tmpdir();
    await assert.rejects(
      downloadFile(`http://127.0.0.1:${server.address().port}/nope`,
        path.join(dir, 'f'), { maxRetries: 1 }),
      (err) => {
        assert.ok(err instanceof DownloadError);
        assert.equal(err.code, 'http');
        assert.equal(err.status, 404);
        return true;
      },
    );
  } finally {
    server.close();
  }
});
