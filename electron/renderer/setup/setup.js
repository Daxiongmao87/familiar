/** Setup window: providers (1) -> credentials (2) -> install (3). */
'use strict';

(function () {
  const $ = (id) => document.getElementById(id);
  const stepSub = $('step-sub');
  const forcedBanner = $('forced-banner');
  const step1 = $('step1');
  const step2 = $('step2');
  const overallSec = $('overall-sec');
  const compList = $('comp-list');
  const overallFill = $('overall-fill');
  const overallPct = $('overall-pct');
  const statusLine = $('status-line');
  const errorLine = $('error-line');
  const btnNext = $('btn-next');
  const btnBack = $('btn-back');
  const btnInstall = $('btn-install');
  const btnAnyway = $('btn-anyway');
  const hwLine = $('hw-line');
  const cudaLine = $('cuda-line');

  let plan = null;
  let step = 1;
  let installing = false;

  function gb(bytes) {
    if (!bytes) return '0 MB';
    if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
    if (bytes < 1e9) return `${Math.round(bytes / 1e6)} MB`;
    return `${(bytes / 1e9).toFixed(1)} GB`;
  }

  function hwText(el, label, info, badReasons) {
    if (!info) {
      el.textContent = `${label}: capability unknown — remote remains available.`;
      el.className = 'hw-line warn';
      return;
    }
    const ok = info.supported !== undefined ? info.supported : info.available;
    if (ok) {
      el.textContent = `${label}: ready${info.adapter ? ` (${info.adapter})` : ''}${info.detail ? ` — ${info.detail}` : ''}.`;
      el.className = 'hw-line ok';
    } else if ((badReasons || []).includes(info.reason)) {
      el.textContent = `${label}: unavailable (${info.reason || info.detail || 'unknown'}) — endpoint required.`;
      el.className = 'hw-line bad';
    } else {
      el.textContent = `${label}: unclear (${info.reason || info.detail || 'unknown'}) — local may fail; remote remains available.`;
      el.className = 'hw-line warn';
    }
  }

  function renderHardware(p) {
    hwText(hwLine, 'WebGPU', p.hardware, ['no-webgpu', 'no-adapter', 'missing-shader-f16']);
    hwText(cudaLine, 'NVIDIA CUDA', p.cuda ? {
      available: p.cuda.available, detail: p.cuda.detail,
      reason: p.cuda.available ? '' : 'no-cuda',
    } : null, ['no-cuda']);
  }

  function disableLocal(name, verdictEntry, reqId) {
    const radio = document.querySelector(`input[name="${name}"][value="local"]`);
    const req = $(reqId);
    if (!radio) return;
    if (verdictEntry && !verdictEntry.ok) {
      radio.disabled = true;
      document.querySelector(`input[name="${name}"][value="remote"]`).checked = true;
      if (req) {
        req.hidden = false;
        req.textContent = `Local unavailable: ${verdictEntry.reason} — endpoint forced.`;
      }
    } else {
      radio.disabled = false;
      if (req) req.hidden = true;
    }
  }

  function renderForced(p) {
    const forced = p.forcedRemote || [];
    if (forced.length === 0) {
      forcedBanner.hidden = true;
      return;
    }
    forcedBanner.hidden = false;
    forcedBanner.textContent = 'Hardware change: forced to endpoints — '
      + forced.map((f) => `${f.provider} (${f.reason})`).join('; ');
  }

  function renderPlan(p) {
    plan = p;
    document.querySelector(`input[name="synth"][value="${p.providers.synthesis}"]`).checked = true;
    document.querySelector(`input[name="jev"][value="${p.providers.jev}"]`).checked = true;
    const v = p.hwVerdict || {};
    disableLocal('synth', v.synthesisLocal, 'synth-req');
    disableLocal('jev', v.jevLocal, 'jev-req');
    renderHardware(p);
    renderForced(p);
    compList.innerHTML = '';
    const rows = [{ label: 'Application runtime', state: 'done', id: '' }];
    for (const c of (p.components || [])) {
      rows.push({ label: `${c.label} — ${gb(c.bytes)}`, state: 'pending', id: c.id });
    }
    if (p.wanted && p.wanted.includes('stt-model')) {
      rows.push({ label: 'Speech recognition runtime (pip)', state: 'pending', id: '' });
    }
    if (p.wanted && p.wanted.includes('jev-model')) {
      rows.push({ label: 'OpenJEV runtime (pip)', state: 'pending', id: '' });
    }
    for (const r of rows) {
      const li = document.createElement('li');
      li.className = `comp ${r.state}`;
      li.dataset.comp = r.id || '';
      li.innerHTML = '<span class="mark"></span><span class="label"></span>';
      li.querySelector('.mark').textContent = r.state === 'done' ? '✓' : '○';
      li.querySelector('.label').textContent = r.label;
      compList.appendChild(li);
    }
    const frac = p.totalBytes ? p.doneBytes / p.totalBytes : 1;
    setOverall(frac);
    statusLine.textContent = p.files === 0
      ? 'Everything is already installed.'
      : `${p.files} files · ${gb(p.totalBytes)} total.`;
    btnInstall.textContent = p.files === 0 ? 'Launch' : 'Install & Launch';
  }

  function setOverall(frac) {
    const pct = Math.round(Math.min(1, Math.max(0, frac)) * 100);
    overallFill.style.width = `${pct}%`;
    overallPct.textContent = `${pct}%`;
  }

  function showError(msg) {
    errorLine.textContent = msg || '';
  }

  function showStep(n) {
    step = n;
    showError('');
    step1.hidden = n !== 1;
    step2.hidden = n !== 2;
    overallSec.hidden = n !== 3;
    btnNext.hidden = n !== 1 && n !== 2;
    btnNext.textContent = n === 1 ? 'Continue' : 'Save & Continue';
    btnBack.hidden = n === 1 || installing;
    btnInstall.hidden = n !== 3;
    stepSub.textContent = n === 1 ? 'Step 1 of 3 — providers and downloads.'
      : n === 2 ? 'Step 2 of 3 — credentials and endpoints.'
        : 'Step 3 of 3 — install and launch.';
    if (n === 2) renderStep2();
    if (n === 3) refreshPlan();
  }

  function renderStep2() {
    const saved = (plan && plan.saved) || {};
    const syn = saved.synthesis || {};
    const jev = saved.jev || {};
    const stt = saved.stt || {};
    const disc = saved.discord || {};
    $('sec-syn-remote').hidden = plan.providers.synthesis !== 'remote';
    $('sec-jev-remote').hidden = plan.providers.jev !== 'remote';
    if (plan.providers.synthesis === 'remote') {
      $('in-syn-base').value = syn.base_url || '';
      $('in-syn-model').value = syn.model_id || '';
      $('in-syn-key').placeholder = syn.has_api_key ? 'configured — blank = keep' : 'blank = none';
    }
    if (plan.providers.jev === 'remote') {
      $('in-jev-base').value = jev.base_url || '';
    }
    $('in-disc-guild').value = disc.guild_id || '';
    $('in-disc-dm').value = disc.dm_user_id || '';
    $('in-disc-token').placeholder = disc.has_token ? 'configured — blank = keep' : 'blank = skip';
    const v = (plan.hwVerdict || {}).sttLocal;
    const sttLocalRadio = document.querySelector('input[name="sttmode"][value="local"]');
    const sttReq = $('stt-req');
    if (v && !v.ok) {
      sttLocalRadio.disabled = true;
      document.querySelector('input[name="sttmode"][value="remote"]').checked = true;
      sttReq.hidden = false;
      sttReq.textContent = `Local server unavailable: ${v.reason} — endpoint forced.`;
    } else {
      sttLocalRadio.disabled = false;
      sttReq.hidden = true;
      document.querySelector(`input[name="sttmode"][value="${stt.mode || 'local'}"]`).checked = true;
    }
    if (stt.mode === 'remote' || (v && !v.ok)) {
      $('in-stt-host').value = stt.stream_host || '';
      $('in-stt-port').value = stt.stream_port || '43007';
    }
    syncSttFields();
  }

  function syncSttFields() {
    const mode = document.querySelector('input[name="sttmode"]:checked').value;
    $('stt-remote-fields').hidden = mode !== 'remote';
  }

  function collectStep2() {
    const val = (id) => $(id).value.trim();
    const out = {};
    const discToken = $('in-disc-token').value;
    const discGuild = val('in-disc-guild');
    const discDm = val('in-disc-dm');
    if (discToken !== '' || discGuild !== '' || discDm !== '') {
      out.discord = {};
      if (discToken !== '') out.discord.token = discToken;
      if (discGuild !== '') out.discord.guild_id = discGuild;
      if (discDm !== '') out.discord.dm_user_id = discDm;
    }
    if (plan.providers.synthesis === 'remote') {
      out.synthesisRemote = { base_url: val('in-syn-base'), model_id: val('in-syn-model') };
      if ($('in-syn-key').value !== '') out.synthesisRemote.api_key = $('in-syn-key').value;
    }
    if (plan.providers.jev === 'remote') {
      out.jevRemote = { base_url: val('in-jev-base') };
    }
    const sttMode = document.querySelector('input[name="sttmode"]:checked').value;
    out.stt = { mode: sttMode };
    if (sttMode === 'remote') {
      out.stt.stream_host = val('in-stt-host');
      out.stt.stream_port = val('in-stt-port');
    }
    return out;
  }

  async function refreshPlan() {
    try {
      const p = await window.familiarSetup.getPlan();
      renderPlan(Object.assign({}, plan || {}, p));
    } catch (e) {
      showError(`plan refresh failed: ${e.message}`);
    }
  }

  async function onProvidersChanged() {
    if (installing || step !== 1) return;
    const modes = {
      synthesis: document.querySelector('input[name="synth"]:checked').value,
      jev: document.querySelector('input[name="jev"]:checked').value,
    };
    try {
      const p = await window.familiarSetup.setProviders(modes);
      renderPlan(Object.assign({}, plan || {}, p));
    } catch (e) {
      showError(`provider switch failed: ${e.message}`);
      refreshPlan();
    }
  }

  async function onNext() {
    if (step === 1) {
      showStep(2);
      return;
    }
    if (step === 2) {
      btnNext.disabled = true;
      try {
        const p = await window.familiarSetup.saveSetup(collectStep2());
        renderPlan(Object.assign({}, plan || {}, p));
        showStep(3);
      } catch (e) {
        showError(`save failed: ${e.message}`);
      } finally {
        btnNext.disabled = false;
      }
    }
  }

  function markDownloading(fileRel) {
    const compId = (fileRel || '').split('/')[0];
    for (const li of compList.children) {
      if (li.dataset.comp === compId) {
        li.className = 'comp active';
        li.querySelector('.mark').textContent = '↓';
      }
    }
  }

  function markAllDone() {
    for (const li of compList.children) {
      li.className = 'comp done';
      li.querySelector('.mark').textContent = '✓';
    }
  }

  async function onInstall() {
    if (installing) return;
    installing = true;
    btnInstall.disabled = true;
    btnBack.hidden = true;
    showError('');
    btnAnyway.hidden = true;
    try {
      const r = await window.familiarSetup.startInstall();
      if (!r.ok) {
        showError(`Install failed: ${r.error} — retry to resume.`);
        installing = false;
        btnInstall.disabled = false;
        btnInstall.textContent = 'Retry';
        btnBack.hidden = false;
        return;
      }
      markAllDone();
      statusLine.textContent = 'Starting services…';
      await window.familiarSetup.boot();
    } catch (e) {
      showError(`Install failed: ${e.message} — retry to resume.`);
      installing = false;
      btnInstall.disabled = false;
      btnInstall.textContent = 'Retry';
      btnBack.hidden = false;
    }
  }

  async function onOpenAnyway() {
    try {
      await window.familiarSetup.openAnyway();
    } catch (e) {
      showError(`Cannot open yet: ${e.message}`);
    }
  }

  window.familiarSetup.onState((p) => { renderPlan(p); showStep(1); });
  window.familiarSetup.onProgress((p) => {
    if (typeof p.overall === 'number') setOverall(p.overall);
    if (p.detail) statusLine.textContent = p.detail;
    if (p.file) markDownloading(p.file);
  });
  window.familiarSetup.onBoot((p) => {
    if (p.detail) statusLine.textContent = p.detail;
  });
  window.familiarSetup.onError((e) => {
    installing = false;
    btnInstall.disabled = false;
    btnInstall.textContent = 'Retry';
    btnBack.hidden = false;
    showError(`Startup failed: ${e.message}`);
    btnAnyway.hidden = !e.canOpenAnyway;
  });

  document.querySelectorAll('input[name="synth"], input[name="jev"]')
    .forEach((el) => el.addEventListener('change', onProvidersChanged));
  document.querySelectorAll('input[name="sttmode"]')
    .forEach((el) => el.addEventListener('change', syncSttFields));
  btnNext.addEventListener('click', onNext);
  btnBack.addEventListener('click', () => { if (!installing) showStep(step - 1); });
  btnInstall.addEventListener('click', onInstall);
  btnAnyway.addEventListener('click', onOpenAnyway);
  refreshPlan();
})();
