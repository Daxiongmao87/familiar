# Familiar Desktop (AppImage) — developer guide

Self-contained Linux desktop application: an Electron shell around the
existing Python backend, with optional local inference (WebLLM synthesis,
bundled OpenJEV scorer, provisioned streaming STT) and preserved remote
endpoint support.

## 1. Project layout

```text
familiar/
├── dmd/                        # Python backend (preserved, minimal deltas)
│   ├── providers.py            # NEW: local/remote routing beneath chat()/score()
│   ├── desktop_api.py          # NEW: /api/desktop/status + /api/desktop/providers
│   ├── config.py               # + provider flags, + DesktopConfig
│   ├── gateway.py              # + optional router (None = legacy behavior)
│   ├── openjev.py              # + set_base_url (provider switch)
│   ├── pipeline.py             # + provider-aware gate URL, apply_providers()
│   └── server.py               # + router wiring, frozen web path, desktop mount
├── electron/                   # desktop shell (Electron 44)
│   ├── main/                   # lifecycle, supervisors, bridge, IPC
│   ├── preload/                # context-bridge scripts (setup/infer/app)
│   ├── renderer/setup/         # first-run setup window
│   ├── inference/              # hidden WebLLM host page + worker
│   ├── installer/              # downloader, install-state (stdlib-only)
│   └── shared/                 # manifest.json + generator + paths
├── services/jev_sidecar/       # local JEV: serve.py + vendor.py (_vendored/ is
│                               #   build output, gitignored, MIT from ../openjev)
├── packaging/                  # backend.spec, electron-builder.yml, build script
├── tests/unit/test_providers.py, test_desktop_api.py, test_sidecar_contract.py
├── tests/regression/test_jev_local_parity.py
└── docs/DESKTOP.md             # this file
```

## 2. Live startup paths (traced from executable code)

Web/server mode (unchanged):

```text
python dmd/server.py config.yaml
  → main(): load_config → IndexStore → Gateway+router → Embedder
  → lexicon → JobPool → SpeakingTracker → SessionEngine
  → BrowserAudioSource + consumer task → uvicorn serve
  → GET / (web/index.html) · /api/* · /ws · /ws/audio
```

`SessionEngine.consume_source` is the single audio entry point (browser,
Discord, and replay sources all implement `AudioSource`). STT is
streaming-only: PCM goes to `StreamingSttAdapter` (TCP stream_host /
stream_port, SimulStreaming protocol: raw s16le PCM in, newline-JSON
partials/finals out). There is no batch path — batch HTTP transcription
was removed for its multi-second delay. Every in-character final routes to
`OpenjevGate.decide`; deploys route through the deterministic `JevWorker`.

Desktop mode (new): Electron starts first, seeds
`<userData>/familiar-config.yaml` from the packaged example once, verifies
installer state, supervises the backend binary with `DMD_HOST/PORT` +
`cwd=userData`, supervises sidecars, hosts the synthesis bridge, then
loads the backend URL in the main window. The backend perceives local
inference as plain localhost URLs — no WebGPU detail crosses into Python.

### Diagnostic mode

Launch the AppImage with `--familiar-debug` to persist a support log at
`~/.config/familiar-desktop/logs/familiar-debug.log` and finalized utterances
as JSON Lines at `~/.config/familiar-desktop/logs/familiar-transcript.jsonl`.
The debug log includes desktop startup, supervised-service output, STT health,
trigger decisions, job drops, and card outcomes. The transcript contains the
timestamp, speaker identity/name, and corrected final text; partial hypotheses
and raw audio are never saved. Both files use mode `0600`, rotate to one bounded
5 MiB backup, and are opt-in because transcript text is private. The debug log
also redacts common credential forms and omits card bodies. Launch without
`--familiar-debug` for normal non-persistent operation.

## 3. OpenJEV scoring (inspected in ../openjev @ b4782a6c953f)

Do not approximate this as text generation. The actual path
(`src/openjev_phase1/{core,direct,server}.py`):

- Request: `{id, state, question, options[{id, description}]}` (2–16
  options). `state` may be a string or JSON value.
- Prompt: `DIRECT_SYSTEM` + a JSON payload
  `{evidence, criterion, options[{letter, description}]}` rendered through
  the model chat template with `enable_thinking=False`.
- Scoring: ONE forward pass, no tokens generated. Read last-position
  logits at the single-token letter slots (`A`…`P`; each must round-trip
  as exactly one token or scoring refuses), softmax over the slots only.
- Response: `{id, option_ids, probabilities, option_logits, input_tokens,
  forward_seconds, prompt_sha256, ...}`.
- `/health` returns `{status: "ok", model: {source, revision, ...}}`.
- Loader requires exactly one visible CUDA GPU; AWQ checkpoints load via
  their `quantization_config` (CUDA kernels).

Production pin (also in `tests/regression/golden/jev_gate_v39.json`):
`QuantTrio/Qwen3.5-4B-AWQ @ 32c292e3a73afe1138518180b1b6d2868c980ee2`
(3 safetensors shards, 6.07 GB), gate wording `jev-gate-v39` LOCKED.

### Why local JEV is a sidecar, not WebLLM

WebLLM 0.2.85 *can* expose top-5 logprobs (proven by openjev's own
`webgpu-demo/`), so ≤5-option scoring is mechanically possible there —
but it cannot run the pinned AWQ checkpoint, and the golden fixture
states probabilities do not transfer between revisions/quantizations.
Gate behavior would drift by construction. Correct semantics win: local
JEV runs the byte-identical vendored algorithm over the byte-identical
pinned weights (CUDA required, like upstream). CPU-only machines use the
remote JEV provider. The vendoring pin is enforced by
`services/jev_sidecar/vendor.py` — a sibling move refuses to vendor until
a human re-inspects and re-runs the golden suite.

## 4. Provider abstraction

`dmd/providers.py` is the only routing knowledge:

| selection | synthesis `chat()` | JEV `score()` |
|---|---|---|
| remote (default) | configured `base_url` (unchanged) | configured `openjev.base_url` (unchanged) |
| local | `desktop.bridge_url` (Electron WebLLM bridge, OpenAI dialect) | `desktop.jev_local_url` (sidecar, identical `/score`) |

`Gateway` takes an optional router (`None` = legacy resolution, used by
all existing tests). The router resolves per call, so synthesis switches
apply live; the JEV gate holds its URL and is repointed by
`SessionEngine.apply_providers()`. Remote URLs/keys are never rewritten
by a switch, and cached models are never deleted by one.

`POST /api/desktop/providers` validates `{synthesis?, jev?}` strictly,
applies live, persists to the YAML file, and never echoes secrets.
`GET /api/desktop/status` reports routing + reachability, secret-free.
The in-app settings modal (topbar Settings button) exposes both provider
selects plus the remote endpoint fields and saves through the desktop
API (live apply) before the rest of the config (restart to apply).

Hardware gating (local is CUDA + WebGPU or nothing): at startup the
shell probes WebGPU (`navigator.gpu` + `shader-f16`, in the hidden
window) and CUDA (`nvidia-smi -L`, in main). Each local option the
verdict forbids is disabled in setup with its reason, previously-local
providers are force-flipped to endpoints (bannered, models kept), and
the main process refuses local selections for incapable providers even
if the renderer is tampered with. Machines without WebGPU, without
CUDA, or both get a working endpoint-only app — never an offered local
mode that cannot run.

## 5. Local inference bridge

`electron/main/bridge.js` (127.0.0.1 only): `GET /models`, `GET /health`,
`GET /local-model/*` (pinned static assets for WebLLM fetch, including
the `/resolve/<branch>/` shape WebLLM requires), `POST
/chat/completions` (OpenAI shape in/out, ≤256 KB bodies).

Familiar's `{type: "json_schema", ...}` has no WebLLM equivalent, so the
bridge maps it to WebLLM 0.2.85's `{type: "json_object", schema: "<json
string>"}` (verified in the shipped `.d.ts`) and keeps a JSON instruction
in the prompt (required by WebLLM — without it the model can spin on
whitespace). On any `response_format` rejection the worker retries once
unconstrained; Python's JSON extraction tolerates prose.

The MLC engine lives in a Web Worker inside a hidden window: created
once, kept resident (KV reuse is internal to the resident engine),
single-flight queue, q4f16_1 pinned build, `enable_thinking: false`,
1-token warmup after load. No invented attention flags — MLC owns its
kernels. Model: `ozhyhinas/MiniCPM5-2B-q4f16_1-MLC @ 602e6da8`
(model_type llama, own webgpu wasm, 1.43 GB). There is no official
`mlc-ai` MiniCPM5 build as of 2026-09-19; the community build is pinned
by revision and size-checked per file.

## 6. First-run installer

`electron/shared/manifest.json` (generated by `make_manifest.py` from
verified pins — see §8) lists every downloadable component: id, version,
source URL/revision, expected files + sizes, install dir, requirements.
`electron/installer/downloader.js` streams each file to `<final>.part`
with Range resume, 3x backoff retry, size (+sha256 when known)
verification, and atomic rename. `installer/state.js` keeps
`install-state.json` (atomic writes) so relaunches skip verified files
and version bumps migrate (only changed files re-download).

Setup window (3 steps):

1. Providers + downloads. Radios for synthesis/JEV; per-component rows;
   overall plan. Local options the hardware verdict forbids are disabled
   with their reason, and previously-local providers are force-flipped
   to endpoints with an explanatory banner (models stay cached).
2. Credentials + endpoints. Discord token/guild/DM id (soft —
   skippable; without it speaker names are guessed), remote endpoint
   URL/model/key fields for every provider set to remote, and the STT
   mode choice (local provisioned server vs remote host:port of a
   whisper_online_server on the network — same protocol, no key).
   Required fields are enforced against the provider choice; secrets
   are write-only and the config file is chmod 600.
3. Install + launch. Downloads with per-file/overall progress, then
   supervised boot with health gates.

`Open anyway` appears only when the backend itself is up (degraded
entry). Fast-path boot failures reopen setup for recovery instead of
quitting. Setup always shows on first launch; afterwards it is
skipped only when nothing needs downloading and no forcing occurred.

Mutable data lives under Electron `userData`
(`~/.config/Familiar/local-ai/`): installer state, venvs, models. The
AppImage carries code only (~9.2 GB of weights download on first run
when all-local is selected).

Pip envs (`stt-runtime`: faster-whisper 1.2.1 + librosa/soundfile +
torch/numpy; `jev-runtime`: the openjev torch/transformers/gptqmodel
pins) are created from the bundled standalone CPython 3.12
(`astral-sh/python-build-standalone@20260901`, stripped install_only,
venv+ssl verified).

## 7. STT provisioning

Live path is unchanged: Electron/browser capture → mixed 16 kHz PCM →
`/ws/audio` → `StreamingSttAdapter`. Stock upstream sends plain-text
lines with no utterance-final signal, so the installer provisions a
patched server: `services/stt_server/whisper_online_server.py`
(`ufal/whisper_streaming @ 6da90b44` + a JSON/finals patch — newline
`{"text","start","end","is_final"}` from the VAC endpoint flag) copied
from app resources next to the three stock downloads, launched on port
43007 with `--backend faster-whisper --model large-v3-turbo --model_dir
<downloaded> --vac` (`--vac` is what makes finals exist) with
weights `dropbox-dash/faster-whisper-large-v3-turbo @ 0a363e91`
(canonical id after the mobiuslabsgmbh move; 1.62 GB). Model choice
`large-v3-turbo` is a provisioning default (best live speed/quality in
the server's `--model` choices), not a code-derived pin — the adapter
protocol is model-independent. Note: the pinned server hardcodes
`device="cuda"` for the faster-whisper backend, so local STT also
requires NVIDIA CUDA (CPU-only machines are forced to a remote
streaming server host:port). There is no batch fallback.

## 8. Manifest pins (verified live 2026-09-19)

| component | source | revision |
|---|---|---|
| stt-server | github ufal/whisper_streaming | 6da90b44b7e50d79695e68166d2a2c7609c75abb |
| stt-model | hf dropbox-dash/faster-whisper-large-v3-turbo | 0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf |
| synth-model | hf ozhyhinas/MiniCPM5-2B-q4f16_1-MLC | 602e6da83b3c3db304d34439fe968fd65f223e07 |
| jev-model | hf QuantTrio/Qwen3.5-4B-AWQ | 32c292e3a73afe1138518180b1b6d2868c980ee2 |
| jev-runtime | pypi (openjev pyproject pins) | torch 2.10.0 / transformers 5.17.0 / gptqmodel 7.5.0 … |
| stt-runtime | pypi | faster-whisper 1.2.1, librosa 1.0.0, soundfile 0.14.0, numpy 2.2.6, torch 2.10.0 |
| web-llm | npm @mlc-ai/web-llm | 0.2.85 (bundled at build) |
| electron | npm electron | 44.4.3 |
| python (sidecars) | astral-sh/python-build-standalone | 20260901, cpython 3.12.14 stripped |
| openjev code | ../openjev (sibling, MIT) | b4782a6c953f05c6255706d7a219f4e032af5b58 |

### How to add/update a component

1. Verify the new source + revision live (HF model API / GitHub API —
   never from memory). Record per-file sizes.
2. Edit the pins in `electron/shared/make_manifest.py`, bump
   `MANIFEST_VERSION`, run it, commit the regenerated `manifest.json`.
3. If the shape changed (new kind/field), update `installer/state.js`
   planning + `main/index.js` `wantedFromProviders`/`runInstall`.
4. Re-run `npm test` and `scripts/desktop_e2e.py`.

## 9. Electron/Python lifecycle

```text
Electron ready → seed config → verify state → setup window (iff needed)
  → install (downloads + venvs) → start backend → wait /api/status
  → start STT (iff streaming) → wait TCP 43007
  → start JEV (iff local) → wait /health
  → load WebLLM model (iff local synth) → main window
```

Supervision: 3 restarts per 60 s, then report + manual restart
(`app:restart-service`). `before-quit` stops sidecars then backend
(SIGTERM process-group, SIGKILL after 5 s) and closes the bridge — no
orphans. Service events surface on `service:event`.

Security: `contextIsolation: true`, `nodeIntegration: false`, no
renderer Node access; three minimal preload bridges (pinned by
`preload_surface` tests); IPC payloads validated; main-window
navigation locked to the backend origin; bridge binds loopback only.

## 10. Capture

Unchanged by design: the existing `getDisplayMedia` (system audio) +
`getUserMedia` (mic) → AudioContext mix → 16 kHz Int16 → `/ws/audio`
flow in `web/app.js` runs inside Electron as-is (secure context,
same-origin WS). The preload exposes `window.familiarDesktop`
(`isElectron`, services, screen-source enumeration) for future picker
UI; no `web/` rewrite was needed. Discord gateway audio stays dead;
speaking-state metadata still feeds attribution where the code uses it.

## 11. Build

```bash
packaging/build_appimage.sh
# → electron/dist/Familiar-0.1.0.AppImage
```

Steps: vendor JEV sidecar (pinned rev enforced) → `npm ci` + `npm test`
+ WebLLM bundle → Python unit tests → PyInstaller backend bundle →
stage standalone CPython → `electron-builder --linux AppImage`.
Flags: `--skip-backend`, `--skip-python` (iterate on the shell).

Prereqs: Node 20+, Python 3.12 + project `.venv`, sibling `../openjev`
at the pinned rev, FUSE for running the AppImage (`libfuse2`).

## 12. Tests

```bash
.venv/bin/python -m pytest tests/unit -q          # python unit (incl. providers)
.venv/bin/python -m pytest tests/regression -q    # golden + local parity (live parts skip)
.venv/bin/python -m pytest tests/e2e -q           # UI pass, setup wizard, settings save
cd electron && npm test                            # installer/main unit tests
python3 scripts/desktop_e2e.py                     # cross-language provider-matrix smoke
```

Coverage of the task matrix: first launch (plan-all), interrupted
download + restart (resume), checksum failure, already-installed launch,
4 provider combos, STT/backend startup failure (supervisor), settings
persistence, clean shutdown (no orphan), JEV golden parity, structured
output through the bridge, preload/capture surface, AppImage execution
(see §13).

## 13. Verification status and limitations

Measured in-session:

- Python unit suite: 298 passed, 1 skipped (baseline 274+1; +24 new).
- Electron suite (`npm test`): 60/60 (installer, bridge, supervisors,
  preload surface, provider file handling, sidecar launchers, hardware,
  setup validation).
- E2E suite green, including the setup wizard (3-step click-through),
  the settings providers save, and the previously-failing UI pass
  (fixed by adding the missing settings button).
- `scripts/desktop_e2e.py`: all checks pass (real node bridge + real
  Gateway over HTTP, 4 provider combos, desktop API, dev-backend boot).
- JEV golden (`test_jev_gate_golden.py`) vs the live pinned scorer:
  PASS with exact baseline reproduction (min-deploy 0.8808, max-wait
  0.0953, gap 0.7854). Local-parity test written; its live run needs a
  sidecar on 8299 (see limitations).
- PyInstaller backend (194 MB, hermetic build venv): boots, serves
  `/api/status`, `/api/desktop/status`, and the UI; degraded boot fixed
  (`stt_monitor` UnboundLocalError, pre-existing) with a red/green
  regression test.
- AppImage `Familiar-0.1.0.AppImage` (246 MB): launches headless under
  Xvfb, seeds config, supervises the backend (all endpoints live),
  serves the bridge with an honest WebGPU probe (`no-adapter` on
  SwiftShader), renders main UI + setup window (screenshots audited),
  quits with zero orphaned processes.
- Manifest: all JEV weights downloaded live (6.09 GB) and byte-size
  matched on all 14 files; STT server files size-matched; STT
  pip-env pins install cleanly and `import whisper_online` succeeds.
- WebLLM bundle builds (6.0 MB IIFE); `response_format` mapping and
  `/resolve/<branch>/` serving verified against the shipped 0.2.85
  sources; standalone CPython venv+ssl verified.
- Pre-existing failure, unchanged by this work (verified identical on
  the clean tree): `test_replay_golden` (2 tests, trigger/STT-mock
  behavior). The former `test_ui_pass` failure is fixed (settings
  button added).

Measured performance (this machine):

- Backend bundle cold boot to `/api/status`: ~12 s (degraded, dead
  endpoints); AppImage launch to main window: ~30 s (Electron init +
  backend boot, all-remote fast path).
- JEV gate+tier `decide()` vs the pinned AWQ scorer: 567 ms wall
  (deploy p=0.9325 on golden eval-1).
- Artifact sizes: AppImage 246 MB (backend bundle 194 MB, standalone
  CPython 104 MB unpacked, WebLLM IIFE 6.0 MB, Electron runtime rest).
- First-run downloads (all-local): 9.15 GB weights + pip envs (~7 GB
  STT runtime, ~8 GB JEV runtime with CUDA wheels).
- WebLLM model-load/inference latency: not measurable here (no WebGPU);
  measure on target hardware via bridge `/health` + card-loop timings.

Honest limitations (v1):

- Local JEV requires one NVIDIA CUDA GPU (AWQ kernels, like upstream
  openjev). CPU-only machines must use remote JEV; the setup hardware
  line and provider choice make this explicit. A live sidecar golden
  run was blocked in-session (all GPUs saturated by other workloads);
  run `test_jev_local_parity.py` against a sidecar on 8299 before
  release — by construction (identical code + weights) zero drift is
  expected.
- Local STT likewise requires CUDA (the pinned server hardcodes
  `device="cuda"`); CPU-only machines are forced to a remote
  streaming server. Live transcription through a provisioned server
  was not run in-session (same GPU saturation); argv, sizes, and the
  import closure were verified live.
- Local synthesis requires WebGPU + `shader-f16`. The MLC build is a
  pinned community artifact (no official MiniCPM5 MLC build exists);
  JSON-schema conformance rides WebLLM's `json_object`+schema mode plus
  prompt hints, validated by bridge tests but not against a live
  GPU run of the full card loop in this session (no WebGPU here).
- First-run downloads are large (~9.2 GB all-local + pip envs); resume
  and per-provider selection mitigate, but there is no torrent/P2P.
- The AppImage was built and launched on this machine (not a clean
  VM) — run `./Familiar-*.AppImage` on a fresh Ubuntu 22.04+ with FUSE
  before release.
- No application icon yet (electron-builder default); `files` allowlist
  in `electron-builder.yml` must gain any new runtime asset.

## 14. Architectural changes (summary)

- Added provider routing under the existing `chat()`/`score()`
  interfaces (new `dmd/providers.py`; 4 combos; defaults = legacy).
- Added secret-free desktop endpoints (`dmd/desktop_api.py`).
- Added the Electron shell (lifecycle, supervision, setup UI, localhost
  bridges, preloads) reusing the Familiar web UI unchanged.
- Added the first-run manifest/download/state system (verified pins).
- Added the local JEV sidecar (vendored algorithm + pinned weights,
  enforced provenance) and STT provisioning (pinned server + model).
- Added AppImage packaging (PyInstaller backend + electron-builder).
- No changes to SessionEngine/RAG/orchestration/transcript/monitor
  semantics, gate wording/ordering, or the capture pipeline.
