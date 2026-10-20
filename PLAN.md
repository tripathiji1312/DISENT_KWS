# PLAN — DISENT_KWS Multi-Person Voice Assistant (Demo)

**Status:** Ready to build
**Target:** A working, demo-ready, multi-person voice assistant on this Linux host,
with a browser UI, by 2026-10-21 (demo day).
**Research log:** `research/findings-2026-10-20.md` + `research/artifacts/`
(evaluation scripts and raw JSON). Read it before changing thresholds.

---

## 1. Objective & Success Criteria

Build a local service that:

1. Enrolls multiple people, each with a display name, color, one or more
   keywords, and a per-person action.
2. Lets the **same keyword** be assigned to multiple people.
3. Detects a spoken keyword from a short isolated utterance, identifies **which
   enrolled person** said it (including when they share a keyword), and triggers
   that person’s action and/or announces their name.
4. Identifies the speaker for **arbitrary speech** (not just keywords) among all
   enrolled people, and rejects unknown speakers.
5. Shows everything live in a polished web UI: waveform/orb, VAD state, detected
   keyword, person reveal, confidence bars, event timeline, enrollment wizard,
   people/keyword management, and settings.
6. Works without a microphone on the host (browser mic via SSH-forwarded
   localhost, or WAV upload/replay).

**Done means all of these are true:**

- `GET /api/health` returns `status=ok` with both models loaded and device `cpu`.
- Two people can be enrolled from the UI (5 takes each) in under ~60 s per person.
- Alice and Bob both use keyword `stop`; saying `stop` as Alice announces Alice,
  as Bob announces Bob, and as an unenrolled voice produces
  `keyword_unknown_speaker` (no action).
- Arbitrary 2–6 s speech from an enrolled person returns their name; an
  unenrolled speaker is rejected.
- A WAV upload to `/api/identify` returns the same decision as live audio.
- End-to-end latency for a 1–2 s utterance is **< 700 ms** (compute < 250 ms;
  the rest is the 300 ms VAD silence wait + buffering), displayed in the UI.
- Existing suite still passes: **60 baseline tests + new app tests**, all green.
- The exact demo script in §5.7 runs twice in a row without a server restart.

**Quantitative evidence targets** (from `research/`):

- ECAPA top-1 on held-out LibriSpeech utterances: **12/12** for 6 enrolled
  speakers; genuine min cosine **0.500** vs impostor max **0.278** at ≥2 s.
- Keyword `stop`: same-word min **0.780** vs different-word max **0.177**;
  `yes`: min **0.530** vs `no` max **0.361**.
- VAD rejects 10 s silence/noise with **0** transitions.
- No use of `torchaudio.load/save`, `model_final.onnx`, or ALSA.

---

## 2. Locked Scope & Constraints

### In scope

- Local FastAPI + uvicorn service, one process, one worker, CPU only.
- Browser UI served by the same app (vanilla HTML/CSS/ES modules; no build step).
- Browser microphone capture via Web Audio + AudioWorklet, binary WebSocket
  streaming; WAV upload/replay fallback.
- Multi-person registry with shared keywords, per-person actions, JSON + `.npy`
  persistence, atomic writes.
- Actions: `announce` (browser TTS + UI), `webhook`, `log`, `shell`
  (allowlisted executables only, disabled by default).
- Silero VAD segmentation; ECAPA speaker embeddings; DISENT phonetic embeddings
  for keywords.
- Fast enrollment (raw takes + fast gain/time-stretch variants), no pitch shift.
- Tests, docs, demo seed script/assets, run script.

### Out of scope / explicitly not promised

- **Continuous wake-word detection inside arbitrary sentences.** Measured
  negative result: non-`yes` sentences reached max-window score **0.735**, above
  the `yes` minimum **0.548**. The demo requires the keyword to be spoken as a
  short isolated utterance (≤2.5 s). The UI says “Say just your keyword.”
- **Reliable custom/out-of-vocabulary keywords.** OOV words (`country`,
  `promise`, `morning`, `letter`) overlapped heavily with negatives. Recommend
  curated in-vocabulary keywords; custom keywords are marked “experimental” and
  offer a calibration step.
- Simultaneous/overlapping speakers, diarization, far-field/noisy-room accuracy
  guarantees, cross-language, children’s voices.
- Authentication/multi-tenant security. Bind to `127.0.0.1` by default; LAN mode
  is a trusted demo network only.
- Training/fine-tuning the model, ONNX export, GPU.
- Cloud ASR/LLM/TTS. No network dependency at inference time.

### Constraints discovered (do not fight them)

- `torchaudio.load/save` fail (`libtorchcodec`/FFmpeg mismatch) → use
  `soundfile`.
- `model_final.onnx` fails (missing `.onnx.data`) → use `model_final.pt`.
- No sound card, no browser on the host → browser mic from another device or
  upload/replay; SSH local port forwarding is the recommended demo path.
- Importing `silero_vad` sets `torch.set_num_threads(1)` → call
  `torch.set_num_threads(4)` after imports (measured 26.2 ms → 17.9 ms).
- `augment_enrollment(5→30)` takes **142 s** → do not use pitch-shift
  augmentation in the web enrollment path.
- `trim_silence` crashes for <1600 samples → guard it.

---

## 3. Research Findings (evidence-backed)

Full detail and commands: `research/findings-2026-10-20.md`. Key facts the build
depends on:

| # | Finding | Evidence |
|---|---|---|
| 1 | `model_final.pt` loads with `strict=False`, 0 missing/unexpected, 1,806,068 params; softplus weights 0.5160/0.4840. | checkpoint inspection; `research/findings` §3 |
| 2 | `model_final.onnx` fails: missing `./model_final.onnx.data`. | onnxruntime error, §3 |
| 3 | DISENT unconditioned forward on `(1,80,200)` = **23.2 ms** batch-1; batch 8 = 91.8 ms. | timing script, §3 |
| 4 | ECAPA `spkrec-ecapa-voxceleb` loads offline (`HF_HUB_OFFLINE=1` works); `(B,T)` → `(B,1,192)`. | load test, §2/§4 |
| 5 | ECAPA held-out: 12/12; genuine min 0.500/mean 0.673; impostor max 0.278. | dev-clean eval, §4 |
| 6 | ECAPA short clips: 0.5 s genuine min 0.250, 1.0 s min 0.418, 2 s min 0.592; impostor max ~0.10–0.14. | §4 table |
| 7 | 5 × 1 s enrollment takes: 12/12, genuine min 0.506, impostor max 0.154. | §4 |
| 8 | Keyword `yes` vs `no` isolated: min yes 0.530, max no 0.361 (gap 0.168). | §5 |
| 9 | Keyword in sentences overlaps (yes min 0.548 vs non-yes max 0.735) → isolated only. | §5 |
| 10 | OOV words overlap; `stop` is clean (0.780 vs 0.177); `happy`/`house` marginal. | §5 table |
| 11 | Silero VAD: 512 samples @16 kHz; start/end dicts; 25–30 ms per 5 s; 0 false positives on silence/noise. | §6 |
| 12 | Enrollment pitch-shift augmentation: 142 s for 30 variants; fast gain/interpolate: 70 ms for 15. | §6 |
| 13 | `trim_silence` crashes below 1600 samples. | §6 |
| 14 | Web stack resolves: FastAPI 0.156.0, Starlette 1.7.0, uvicorn 0.54.0, websockets 16.1.1, python-multipart 0.0.32, silero-vad 6.2.3. | dry-run, §2 |
| 15 | FastAPI TestClient binary WS echo + StaticFiles work with these versions. | §2 |
| 16 | getUserMedia requires localhost/HTTPS; AudioContext sampleRate may throw; AudioWorklet is 128-frame. | MDN, §7 |
| 17 | No browser/mic on host; SSH `-L 8000:127.0.0.1:8000` gives the laptop a localhost secure context. | §7 |
| 18 | Baseline: 60 tests pass in 3.21 s; `pytest.ini` wins over `pyproject.toml`. | §1 |

---

## 4. Chosen Approach & Why

### Architecture

```
Browser (mic / upload)
   │  WebSocket binary Float32 + JSON control        REST (enroll, CRUD, replay)
   ▼                                                   ▼
FastAPI app  ──► per-session VAD (Silero)  ──►  single-worker inference executor
   │                                               ├─ DISENT z_phn → keyword cosine
   │                                               ├─ ECAPA embedding → speaker cosine
   │                                               └─ decision + action
   ▼
JSON registry + .npy profiles (atomic)  +  event timeline (JSONL)
```

- **Keyword = DISENT unconditioned `z_phn` cosine** to keyword centroids. This is
  the only path supported by the checkpoint and `extract_prototypes`; the model
  has no classifier logits.
- **Speaker = SpeechBrain ECAPA** (192-d) cosine to per-person centroids. It is
  substantially stronger than DISENT’s 17.86% EER speaker head and measured
  12/12 on held-out data.
- **Do not use the joint `DualGateScorer` for ranking people.** It is a
  single-profile gate (`score = w_kw·sim_kw + w_spk·sim_spk`), not a multi-class
  ranker. Use separate scores and explicit rules.
- **VAD-delimited utterances**, not fixed windows. One decision per utterance
  gives ECAPA enough audio and avoids max-over-windows false positives.
- **Shared keywords** are a first-class entity: a keyword stores a global
  centroid built from all owners’ per-person centroids; speaker ID is restricted
  to the keyword’s owners to disambiguate.
- **FastAPI + vanilla JS** because it is one process, no build step, and gives
  full control over the visual design. React/Next adds build/tooling risk for a
  one-day demo; Gradio/Streamlit cannot deliver the requested bespoke UI.
- **JSON + `.npy`** because the registry is tiny (tens of people), atomic writes
  are trivial, and there is no migration/tooling overhead. SQLite is more robust
  but unnecessary for one demo day.

### Rejected alternatives

| Option | Why rejected |
|---|---|
| DISENT `z_spk` for speaker ID | 17.86% EER vs ECAPA 0.80% / 12/12 held-out; much weaker. |
| Joint scorer threshold per person | Requires one model pass per person and conflates keyword/speaker; no multi-class ranking. |
| Continuous wake-word detection | Measured overlap (0.735 negative vs 0.548 positive); would false-trigger. |
| Custom names as keywords | Measured OOV overlap; only `stop`/`yes`/`no` are validated. |
| ONNX runtime | Broken artifact; PyTorch CPU is fast enough. |
| ALSA/`arecord` | No sound card on host; original demo hardcodes `hw:2,0`. |
| `torchaudio.load/save` | TorchCodec/FFmpeg failure; `soundfile` works. |
| MediaRecorder + server FFmpeg | Codec variance + broken FFmpeg libs; Web Audio WAV is deterministic. |
| SQLite | Extra schema/migration work; JSON atomic writes suffice for a demo. |
| Cloud APIs | Network dependency, latency, privacy, no offline demo. |

---

## 5. Implementation Plan

All paths are relative to `/home/tripathiji/projects/DISENT_KWS`.

### 5.0 Phase 0 — Preflight, dependencies, fixtures (30 min)

**Tasks**

1. Add dependencies (exact, verified resolution):
   ```bash
   cd /home/tripathiji/projects/DISENT_KWS
   uv add --no-sync \
     "fastapi==0.156.0" \
     "starlette==1.7.0" \
     "uvicorn[standard]==0.54.0" \
     "websockets==16.1.1" \
     "python-multipart==0.0.32" \
     "silero-vad==6.2.3"
   uv sync
   ```
   If `uv add` refuses a pin, edit `pyproject.toml` `dependencies` manually with
   those six lines, then `uv lock && uv sync`.
2. Verify imports/versions:
   ```bash
   .venv/bin/python -c "import fastapi, starlette, uvicorn, websockets, multipart, silero_vad; print(fastapi.__version__, starlette.__version__, uvicorn.__version__, websockets.__version__)"
   ```
   Expected: `0.156.0 1.7.0 0.54.0 16.1.1`.
3. Copy small test fixtures into the repo (force-add past `.gitignore`):
   ```bash
   mkdir -p tests/fixtures
   cp /tmp/opencode/libri/yes_1353.wav tests/fixtures/keyword_yes_1353.wav
   cp /tmp/opencode/libri/yes_1593.wav tests/fixtures/keyword_yes_1593.wav
   cp /tmp/opencode/libri/no_1665.wav tests/fixtures/keyword_no_1665.wav
   cp /tmp/opencode/samples/spk1_snt1.wav tests/fixtures/speaker_a_1.wav
   cp /tmp/opencode/samples/spk1_snt2.wav tests/fixtures/speaker_a_2.wav
   cp /tmp/opencode/samples/spk2_snt1.wav tests/fixtures/speaker_b_1.wav
   ```
   If `/tmp/opencode` was cleared, re-download the SpeechBrain samples or use
   any 3 short 16 kHz WAVs; the tests only need same/different speakers.
4. Update `.gitignore` (append after the existing `*.wav`/`*.flac` rules):
   ```gitignore
   # App runtime state
   data/registry.json
   data/profiles/
   data/events.jsonl
   data/*.pem
   # Keep demo/test audio
   !tests/fixtures/**/*.wav
   !demo_assets/**/*.wav
   ```
5. Run the baseline suite:
   ```bash
   .venv/bin/python -m pytest -q
   ```
   Must still be **60 passed**.

**Done when:** imports/versions match, fixtures exist, baseline green.

---

### 5.1 Phase 1 — Core inference, decision logic, audio utilities (3 h)

Create these files:

#### `app/__init__.py`
- Add repo `src/` to `sys.path` so `import models...` works without env vars:
  ```python
  import sys
  from pathlib import Path
  ROOT = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(ROOT / "src"))
  ```

#### `app/settings.py`
- `APP_ROOT`, `DATA_DIR = APP_ROOT / "data"`, `PROFILE_DIR = DATA_DIR / "profiles"`,
  `MODEL_PATH = APP_ROOT / "model_final.pt"`, `STATIC_DIR = APP_ROOT / "app/static"`.
- `Settings` Pydantic model with defaults:
  ```python
  keyword_threshold: float = 0.45      # 0..1
  keyword_margin: float = 0.06
  speaker_margin: float = 0.08
  speaker_threshold_scale: float = 1.0 # 0.5..2.0
  vad_threshold: float = 0.5
  min_speech_ms: int = 350
  max_utterance_sec: float = 6.0
  max_keyword_utterance_sec: float = 2.5
  cooldown_sec: float = 1.5
  announce: bool = True
  actions_enabled: bool = True
  ```
- `adaptive_speaker_threshold(duration_sec: float) -> float` implementing the
  measured table:
  ```python
  if duration < 0.75: return 0.20
  if duration < 1.5:  return 0.25
  if duration < 2.5:  return 0.32
  return 0.36
  ```
  Multiply by `settings.speaker_threshold_scale` at call sites.

#### `app/audio.py`
- `read_wav_bytes(data: bytes) -> tuple[torch.Tensor, int]` using
  `soundfile.SoundFile(io.BytesIO(...))`, `dtype="float32"`, `always_2d=True`,
  mean across channels. Never `torchaudio.load`.
- `to_16k_mono(wav, sr) -> torch.Tensor` using
  `torchaudio.functional.resample` (verified working). Return 1-D.
- `safe_trim(wav)` — if `wav.numel() < 1600`, return as-is; else call the repo’s
  `trim_silence` (see the one-line fix below).
- `select_window(wav, seconds) -> torch.Tensor` — pick the highest-energy
  contiguous window (100 ms frame RMS), pad if shorter. Used to choose the 2 s
  keyword window and the ≤3 s speaker window.
- `fast_augment(clips, n_total=15)` — gain ±1/2 dB + `F.interpolate` time-stretch
  factors `[0.95, 1.05, 0.92, 1.08]`, pad/truncate to the first clip’s length.
  Measured 70 ms for 5→15. **Do not** call `augment_enrollment` in the web path.
- `encode_wav_bytes(wav, sr) -> bytes` — PCM16 WAV via `soundfile.write` to
  `io.BytesIO`; used only in tests/demo assets.

#### One-line bug fix in `src/enrollment/enroll.py`
Inside `trim_silence`, after `wav = wav.unsqueeze(0)` (line 18):
```python
if wav.shape[-1] < window:
    return wav
```
Add a regression test (400/1000 samples do not raise).

#### `app/inference.py`
- `class Engine`:
  - `__init__(model_path=MODEL_PATH, device="cpu", ecapa_savedir=...)`:
    - Load `DISENT_KWS_v2()`; `torch.load(..., map_location=device)`;
      `sd = ckpt.get("model", ckpt)`; `load_state_dict(strict=False)`; `.eval()`.
    - Load `EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
      savedir=..., run_opts={"device": device})`.
    - Create one `torchaudio.transforms.MelSpectrogram` with the **same params as
      `LFBETransform`** (`config.SAMPLE_RATE`, `config.WIN_LENGTH`,
      `config.HOP_LENGTH`, `config.N_MELS`, `f_min=20`, `f_max=7600`).
    - Set `torch.set_num_threads(4)` **after** importing `silero_vad` and all
      model modules (measured 26.2 → 17.9 ms).
  - `logmel(wav_1d, pad_to_200=True) -> Tensor` — `(80,T)` or `(80,200)`.
  - `embed_phonetic_batch(feats: Tensor) -> Tensor` — `(B,80,200) → (B,192)`,
    `F.normalize`.
  - `embed_speaker_batch(wavs: list[Tensor]) -> Tensor` — pad batch to max
    length (cap 3 s), `ecapa.encode_batch`, reshape `(B,192)`, normalize.
    Verified `(B,T) → (B,1,192)`.
  - `score_keywords(wav, keyword_matrix) -> list[(keyword_id, score)]`.
  - `score_speakers(wav, person_matrix) -> list[(person_id, score)]`.
  - `analyze(wav, runtime, settings) -> Decision` — compute both score lists and
    call the pure `decide()` below.
- `@dataclass Decision` with: `event`, `person_id`, `person_name`, `keyword_id`,
  `keyword_label`, `keyword_score`, `keyword_runner_up`, `speaker_score`,
  `speaker_runner_up`, `speaker_threshold`, `margin`, `duration_sec`,
  `latency_ms: dict`, `confidence`.
- `decide(keyword_scores, speaker_scores, keyword_owners, duration, settings) ->
  Decision` — **pure function**, unit-testable without models. Exact rules:
  ```python
  kw_sorted = sorted(keyword_scores, key=score, reverse=True)
  kw_match = (
      kw_sorted and kw_sorted[0].score >= settings.keyword_threshold
      and (len(kw_sorted) == 1 or kw_sorted[0].score - kw_sorted[1].score >= settings.keyword_margin)
      and duration <= settings.max_keyword_utterance_sec
  )
  if kw_match:
      owners = keyword_owners[kw_id]
      owner_scores = [(p, s) for p, s in speaker_scores if p in owners]
      thresh = adaptive_speaker_threshold(duration) * settings.speaker_threshold_scale
      spk_match = owner_scores and owner_scores[0].score >= thresh and (
          len(owner_scores) == 1 or owner_scores[0].score - owner_scores[1].score >= settings.speaker_margin)
      event = "keyword" if spk_match else "keyword_unknown_speaker"
  else:
      thresh = adaptive_speaker_threshold(duration) * settings.speaker_threshold_scale
      spk_match = speaker_scores and speaker_scores[0].score >= thresh and (
          len(speaker_scores) == 1 or speaker_scores[0].score - speaker_scores[1].score >= settings.speaker_margin)
      event = "speech" if spk_match else "unrecognized"
  ```
  - One-person edge case: skip the margin check (no runner-up).
  - `confidence` = clamp01((top_score - threshold) / max(1e-6, 1 - threshold));
    label it “similarity” in the UI, not probability.
  - If `duration < settings.min_speech_ms/1000` → `event="too_short"`, no action.
- `class RuntimeProfiles` (frozen): keyword IDs/labels/matrix, person IDs/names/
  matrix, `keyword_owners: dict[str, tuple[str,...]]`. Built by the registry and
  swapped atomically.
- `@dataclass KeywordHit`/`SpeakerHit` or simple tuples; keep the pure function
  inputs as `list[tuple[str,float]]`.

**Tests — `tests/test_app_inference.py`**
- `test_trim_silence_short_input` (400/1000 samples) — no raise.
- `test_adaptive_thresholds` — exact values at 0.5/1.0/2.0/3.0 s.
- `test_decide_shared_keyword` — Alice/Bob both own `stop`; high speaker score
  picks the right one.
- `test_decide_unknown_speaker` — keyword matches, owner scores below threshold →
  `keyword_unknown_speaker`.
- `test_decide_arbitrary_speech` — no keyword, speaker above threshold → `speech`.
- `test_decide_margin` — runner-up too close → `unrecognized`.
- `test_decide_long_keyword_ignored` — >2.5 s keyword match → not a keyword event.
- `@pytest.mark.integration` real-model tests (skip if `model_final.pt` missing):
  - keyword: prototype `yes_1353`, assert `sim(yes_1593) - sim(no_1665) > 0.10`
    (evidence: 0.6063 vs 0.3399).
  - speaker: profile from `speaker_a_1`, assert
    `cos(a_2) > cos(b_1) + 0.20` (evidence: 0.643 vs 0.092).
  - shapes `(192,)` and L2 norm ≈ 1.

**Done when:** `pytest -q tests/test_app_inference.py` passes; real-model tests
show the expected gaps.

---

### 5.2 Phase 2 — Registry, profiles, actions (2 h)

#### `app/registry.py`
- Pydantic models: `ActionConfig`, `PersonKeyword`, `Person`, `Keyword`,
  `RegistryState` (`version=1`, `settings`, `keywords`, `people`).
- IDs: `kw_<slug>_<6hex>`, `p_<slug>_<6hex>`; validate against
  `^[a-z0-9_]{1,40}$`.
- Storage:
  - `data/registry.json` — atomic write: `tempfile.mkstemp(dir=DATA_DIR)` →
    `json.dump` → `flush` + `fsync` → `os.replace`.
  - `data/profiles/<id>.npy` — `np.save` to `.tmp.npy` → `os.replace`;
    `np.load(..., allow_pickle=False)`.
  - `data/events.jsonl` — append one JSON line per event; keep an in-memory
    `deque(maxlen=500)`.
- `Registry` methods:
  - `load()`, `save()`, `snapshot() -> RegistryState` (deep copy under
    `threading.RLock`).
  - `runtime_profiles() -> RuntimeProfiles` (rebuilt on mutation; loads `.npy`
    once).
  - `create_keyword(label, threshold=None)`, `get_keyword`, `list_keywords`,
    `delete_keyword(force=False)`.
  - `create_person(name, color, action, trigger_on_any_speech)`.
  - `attach_keyword(person_id, keyword_id, person_centroid, n_samples)` —
    writes `profiles/pk_<person>_<keyword>.npy`, recomputes the global keyword
    centroid as the **normalized weighted mean** of all owners’ per-person
    centroids (weights = `n_samples`), writes `profiles/kw_<id>.npy`.
  - `detach_keyword`, `delete_person` — remove profiles; recompute keyword
    centroids; delete orphan keywords only when `force=True`.
  - `set_person_profile(person_id, centroid)`, `set_action`,
    `update_settings`.
  - `append_event(event)`, `recent_events(limit)`.
- Path safety: every stored profile path is relative; resolve under
  `DATA_DIR` and reject escapes (`Path.is_relative_to`).
- Startup recovery: if `registry.json` is corrupt, load `registry.json.bak`
  (write a `.bak` copy before each replace) or start empty with a clear log.
- **Do not** share `RuntimeProfiles` mutably; replace the reference atomically.

#### `app/enrollment.py`
- `enroll_person(engine, registry, name, keyword_label_or_id, clips, action,
  trigger_on_any_speech) -> dict`:
  1. Validate 3–8 clips, each 0.25–5 s after trim; reject silence
     (peak < 0.01 or RMS < 0.003) with a clear 422.
  2. Speaker: `safe_trim` each; `select_window(..., 3.0)`; batch ECAPA;
     centroid = normalize(mean); leave-one-out quality (`min`, `mean`); warn if
     `min < 0.35`.
  3. Keyword: raw takes + `fast_augment` to 15; batch DISENT; per-person
     centroid = normalize(mean); batch time ~160 ms.
  4. Create or reuse keyword (case-insensitive label match); attach both
     profiles; save registry atomically.
  5. Return person, keyword, quality, timings.
- `leave_one_out(vectors) -> {"min":..., "mean":...}`.

#### `app/actions.py`
- `async def execute(action, person, keyword, decision) -> dict`:
  - `disabled` → `{"status":"disabled"}`.
  - `announce` → `{"status":"ok","text": rendered}`; the server includes
    `"speak": text` in the event; the browser does TTS.
  - `log` → log at INFO and return `{"status":"ok"}`.
  - `webhook` → `httpx.AsyncClient(timeout=3.0).post(url, json=payload)`;
    return status/body; never raise into the WS loop.
  - `shell` → `shlex.split(cmd)`, executable must be in
    `ALLOWED_SHELL = {"notify-send","paplay","aplay"}`; `shell=False`,
    `timeout=3`; return exit code/stderr. Disabled by default in the UI.
  - Render `message_template` with `.format(name=..., keyword=...)`; sanitize
    unknown fields (catch `KeyError` → default text).
- Cooldown: per person, `settings.cooldown_sec`; suppress duplicate actions but
  still emit the event with `"cooldown": true`.

**Tests — `tests/test_app_registry.py`**
- create/attach/reload round-trip; profiles are normalized and loadable with
  `allow_pickle=False`.
- shared keyword: Alice + Bob attach `stop`; global centroid is the weighted
  mean; `runtime_profiles().keyword_owners["stop"] == {alice,bob}`.
- delete Bob → keyword centroid recomputed, Alice remains.
- path traversal attempt (`../../etc/passwd`) rejected.
- atomic write under 20 concurrent `update_settings` calls → valid JSON.
- events append/recent limit.

**Tests — `tests/test_app_actions.py`**
- announce renders `"{name} said {keyword}"`.
- webhook uses `httpx.MockTransport`; timeout path returns error, doesn't raise.
- shell rejects `rm -rf /`; accepts `notify-send` (mock subprocess).

**Done when:** registry/actions tests pass; a full enroll→reload→runtime
round-trip preserves cosine scores within 1e-6.

---

### 5.3 Phase 3 — FastAPI server: REST, WebSocket, streaming (3 h)

#### `app/server.py` (app factory) and `app/main.py`
- `create_app(data_dir=None, model_path=None, load_models=True) -> FastAPI`.
- `lifespan`:
  - `Registry(data_dir)`, `Engine(model_path)` (unless `load_models=False`),
    `ThreadPoolExecutor(max_workers=1)`, `torch.set_num_threads(4)`.
  - Fail fast with a clear message if `model_final.pt` or the ECAPA cache is
    missing (print the exact download command).
- Mount `/static` and `GET /` → `FileResponse(STATIC_DIR/"index.html")`.
- `app/main.py`: `app = create_app()` for `uvicorn app.main:app`.

**REST endpoints (exact):**

| Method | Path | Body / params | Returns |
|---|---|---|---|
| GET | `/api/health` | — | status, model names, device, torch threads, uptime, loaded people/keywords |
| GET | `/api/state` | — | people, keywords, settings, recent events, version |
| GET | `/api/people` | — | list |
| POST | `/api/people` | JSON `{name,color,action,trigger_on_any_speech}` | person (no voice yet) |
| PATCH | `/api/people/{id}` | JSON partial | person |
| DELETE | `/api/people/{id}` | `?force=false` | 204 / 409 if keyword shared |
| GET | `/api/keywords` | — | list |
| POST | `/api/keywords` | JSON `{label,threshold?}` | keyword |
| PATCH | `/api/keywords/{id}` | JSON `{label?,threshold?}` | keyword |
| DELETE | `/api/keywords/{id}` | `?force=false` | 204 / 409 if owners |
| POST | `/api/enroll` | multipart: `name`, `keyword_label` or `keyword_id`, `files[]`, `action_json?`, `trigger_on_any_speech?` | person, keyword, quality, timings |
| POST | `/api/people/{id}/keywords` | multipart: `keyword_label`/`keyword_id`, `files[]` | person, keyword, quality |
| POST | `/api/identify` | multipart `file`, `?execute_actions=false` | decision + segments (no broadcast) |
| POST | `/api/replay` | multipart `file`, `?announce=true` | decision + broadcast + action |
| POST | `/api/settings` | JSON partial settings | settings |
| GET | `/api/events` | `?limit=100` | events |
| POST | `/api/demo/seed` | `?force=false` | seeded people (409 if assets missing) |

- Upload limits: max 8 files, 15 MB each, 40 MB total; reject non-WAV with 415;
  reject >30 s audio with 413.
- `POST /api/identify` reads with `soundfile`, resamples, runs `engine.analyze`
  in the executor, returns JSON (no actions unless `execute_actions=true`).
- `POST /api/replay` runs the same analysis, executes actions, broadcasts the
  event to all live WS clients, returns the event.

**WebSocket `/ws/live` protocol (exact):**

Client → server:
```json
{"type":"hello","sample_rate":16000,"client":"browser"}
```
then binary frames: little-endian Float32 mono samples at the declared rate.
Also text:
```json
{"type":"reset"}
{"type":"ping"}
```
Server → client:
```json
{"type":"ready","settings":{...},"models":{"disent":"model_final.pt","ecapa":"spkrec-ecapa-voxceleb"}}
{"type":"level","rms":0.031,"speech":true,"vad":0.82}
{"type":"event","event":{...}}
{"type":"error","message":"..."}
```
- `level` at most 10 Hz; `event` once per utterance.
- Event object: `id`, `ts`, `type` (`keyword`/`speech`/`keyword_unknown_speaker`/
  `unrecognized`/`too_short`), `person_id`, `person_name`, `person_color`,
  `keyword`, `keyword_score`, `speaker_score`, `speaker_threshold`, `margin`,
  `duration_sec`, `latency_ms`, `speak`, `action`.
- Session implementation:
  - Per-connection `StreamSession`: `VADIterator(model, threshold=0.5,
    sampling_rate=16000, min_silence_duration_ms=300, speech_pad_ms=100)`;
    deque of 512-sample chunks (keep last 30 s); `fed_samples`.
  - Feed exactly 512 samples per VAD call; keep the remainder in a small
    pending buffer if the incoming resampled chunk is not a multiple of 512.
  - On `{'start': s}`: record `seg_start = s` (absolute), `speech_active=True`.
  - On `{'end': e}`: slice `[seg_start:e]` from the stored chunks, enqueue
    analysis on the executor, clear `seg_start`, keep VAD state.
  - If `seg_start` and `(fed_samples - seg_start) > max_utterance_sec*16000`:
    force-finalize at `max_utterance_sec`, call `vad.reset_states()`, clear
    `seg_start`. This is required because a clip that ends mid-speech emits no
    `end` (verified).
  - On `reset`/disconnect: clear buffers; do not execute actions on disconnect.
  - Run VAD in the WS coroutine (fast, ~0.1 ms/chunk); run DISENT/ECAPA via
    `loop.run_in_executor(app.state.executor, ...)` so reading continues.
  - Serialize events with a per-session `asyncio.Lock` and include `seq`.
  - Resampling: if `sample_rate != 16000`, resample each chunk with
    `torchaudio.functional.resample`; log a warning that chunk-boundary
    resampling is a fallback.
  - Disable compression (`ws.send_json`/`send_bytes` default); binary payloads
    are small.
- Broadcast manager: `set[WebSocket]` with an `asyncio.Lock`; remove dead
  sockets; `await asyncio.gather(*[ws.send_json(...)], return_exceptions=True)`.

**Tests — `tests/test_app_api.py`**
- `create_app(data_dir=tmp, load_models=False)`; inject `FakeEngine` with a
  deterministic `analyze` (e.g., always Alice/stop) and a real `Registry`.
- CRUD: create keyword/person, patch, delete with shared keyword → 409.
- Enroll: generate 3 WAV bytes with `soundfile` (1 s tone/speech fixture), post
  multipart; assert 200, profile files exist, registry reloads.
- Identify: post a WAV; assert decision shape and no action executed.
- Replay: assert event appended and broadcast (use a TestClient WS connection).
- Settings validation: `keyword_threshold=2` → 422; `vad_threshold=0` → 422.
- Upload guards: non-WAV → 415; >8 files → 422; oversized → 413.
- WS: hello → ready; send 2 s of silence → no event; send fixture audio in
  512-sample chunks → at least one event; binary echo path.
- Static `/` returns HTML containing `id="app"`.

**Done when:** API tests pass; `curl` health/state/enroll/identify/replay work
against a live uvicorn.

---

### 5.4 Phase 4 — Frontend: live experience + enrollment wizard (5 h)

Create `app/static/`:

#### `index.html`
Semantic layout, no CDN:
```
<div id="app">
  <aside class="sidebar">
    brand (orb mark + "DISENT")
    nav: Live / People / Keywords / Settings
    connection pill (dot + "Live"/"Offline")
    footer: model + latency
  </aside>
  <main>
    <section id="view-live" class="view is-active">
      <header> title + mic button + device picker </header>
      <div class="live-grid">
        <div class="stage glass"> orb canvas + waveform canvas + status text </div>
        <div class="result glass"> last detection: name, keyword, bars, latency </div>
      </div>
      <div class="people-rail"> person chips with active glow </div>
      <div class="timeline glass"> event feed </div>
    </section>
    <section id="view-people" class="view"> ... </section>
    <section id="view-keywords" class="view"> ... </section>
    <section id="view-settings" class="view"> ... </section>
  </main>
</div>
<div id="enroll-modal" class="modal" hidden> 4-step wizard </div>
<div id="toast" role="status" aria-live="polite"></div>
```

#### `styles.css` — visual system
- Dark aurora background: `#07070c` base, two blurred gradient blobs
  (`#7c5cff`, `#22d3ee`) at low opacity, subtle grain via CSS gradient.
- Glass panels: `background: rgba(255,255,255,.045)`, `backdrop-filter:
  blur(24px)`, `1px solid rgba(255,255,255,.09)`, `border-radius: 24px`,
  layered shadows.
- Typography: system UI stack; scores in `ui-monospace`.
- Orb: 220 px circle, radial gradient, conic ring, `breathe` animation; states
  `idle` (dim), `listening` (cyan pulse driven by `--level`), `thinking`
  (rotating conic), `hit` (person color burst), `unknown` (amber).
- Result card: large name reveal (48–64 px), keyword chip, two confidence bars
  with threshold ticks, latency badge.
- People cards: avatar initials in the person color, name, keyword chips,
  quality ring, action badge, hover lift.
- Timeline: rows with color dot, name, keyword, relative time, score pills.
- Responsive: `grid-template-columns: 1fr 380px`; collapse to one column under
  900 px; sidebar becomes a bottom/top bar under 720 px.
- Accessibility: `:focus-visible` rings, `prefers-reduced-motion` disables
  aurora/orb animations, `aria-live` on the result/timeline, labels on inputs.
- No external fonts/images; inline SVG icons only.

#### `audio-worklet.js`
- `CaptureProcessor` accumulates 128-frame input into a 2048-sample buffer and
  posts a copied `Float32Array` when full; returns `true`.
- `registerProcessor("capture-processor", CaptureProcessor)`.

#### `app.js`
State + modules:
- `connectWS()`: `ws://` or `wss://` based on `location.protocol`; on `ready`
  load `/api/state`; reconnect with 1→5 s backoff; show pill.
- `MicCapture`:
  - `start(deviceId)` → `getUserMedia({audio:{channelCount:1,
    echoCancellation:false, noiseSuppression:false, autoGainControl:false,
    deviceId}})`.
  - `new AudioContext({sampleRate:16000, latencyHint:"interactive"})`; catch
    `NotSupportedError` and use `new AudioContext()`.
  - `audioWorklet.addModule("/static/audio-worklet.js")`; connect
    source → worklet → zero-gain → destination; source → analyser.
  - On worklet messages, batch and `ws.send(float32.buffer)`; send hello with
    `sample_rate: audioContext.sampleRate`.
  - `stop()` stops tracks/closes context; sends `{type:"reset"}`.
  - `startRecording(ms)` collects samples and resolves a WAV `Blob` via
    `encodeWav` (PCM16 mono); `stopRecording()` early.
- `encodeWav(samples, sampleRate)`: standard 44-byte RIFF header + PCM16.
- `speak(text)`: `speechSynthesis.cancel(); speechSynthesis.speak(new
  SpeechSynthesisUtterance(text))`; respect `settings.announce`.
- `renderState()`, `renderPeople()`, `renderKeywords()`, `renderSettings()`,
  `renderEvent(event)`, `renderHealth()`.
- Enrollment wizard:
  1. Identity: name, color swatches, optional action (announce/webhook/log/
     shell), `trigger_on_any_speech` toggle.
  2. Keyword: select existing keyword cards or type a new one; show
     “Recommended: stop, yes, no” and an “experimental” warning for custom
     labels.
  3. Voice: 5 takes; each take has Record/Stop/Play/Retake, a 1.5 s countdown,
     live level meter, and a thumbnail. Require ≥3 takes (UI asks for 5).
  4. Review: quality ring, keyword/keyword owner list, Save.
  - `fetch("/api/enroll", {method:"POST", body: formData})`; show progress and
    errors; on success refresh state, close modal, toast “Alice enrolled”.
- People view: cards + Add person; edit name/color/action; delete with confirm;
  “Add keyword” opens the same keyword/voice steps.
- Keywords view: cards with label, owners’ avatars, sample count, threshold
  (global or override), “Test” button (record/upload → `/api/identify`, show
  keyword/speaker scores without saving).
- Settings view: sliders for keyword threshold/margin, speaker margin/scale,
  VAD threshold; number inputs for max utterance/cooldown; toggles for announce/
  actions; Save; “Reset to defaults”; server health.
- Live mic errors:
  - `navigator.mediaDevices === undefined` → banner: “Microphone needs
    localhost or HTTPS. Use SSH port forwarding (`ssh -L 8000:127.0.0.1:8000
    user@host`) or upload a WAV.”
  - `NotAllowedError` → permission instructions.
  - `NotFoundError` → device picker + upload fallback.
- Upload/replay: a drop zone on Live (“Drop a WAV to replay”); posts to
  `/api/replay`; the server broadcasts; the UI renders the event and animates.
- Demo help panel: 5-step script (see §5.7).
- Empty states: “No people yet — add your first person”, “No keywords yet”.

**Frontend checks:**
- `node --check app/static/app.js` and `node --check app/static/audio-worklet.js`
  (Node v26 available).
- `curl -s localhost:8000/ | grep 'id="app"'`.
- Manual visual pass on the demo device: all views, enrollment wizard, mic,
  upload, responsive layout.

**Done when:** live mic/upload produces a name reveal; enrollment from the UI
creates a person visible in People and usable in Live.

---

### 5.5 Phase 5 — Demo assets, seed path, docs (2 h)

1. Create `demo_assets/` with 6–10 short WAVs and `manifest.json`:
   - Alice: 2 `stop` clips; Bob: 2 `stop` clips; Carol: 2 `yes` clips.
   - Source: `/tmp/opencode/words/stop_*.wav` (Whisper-extracted LibriSpeech)
     and `/tmp/opencode/libri/yes_1353.wav` / `no_1665.wav`.
   - If absent, use `tests/fixtures/keyword_yes_1353.wav` for Alice and
     `keyword_no_1665.wav` for Bob, plus a third fixture or record live.
   - `demo_assets/README.md`: LibriSpeech CC BY 4.0 attribution
     (<https://www.openslr.org/12>), note that clips are excerpted words.
2. `scripts/seed_demo.py`:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/seed_demo.py --data-dir data --assets demo_assets
   ```
   - Calls `app.enrollment.enroll_person` for each manifest entry (fast augment
     to 15 variants).
   - Idempotent: skip existing names unless `--force`.
3. `POST /api/demo/seed` calls the same service; UI button “Load demo people”
   on the People empty state. 409 with a clear message if assets are missing.
4. `scripts/run_demo.sh`:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   cd "$(dirname "$0")/.."
   export PYTHONPATH="$PWD/src:$PWD"
   if [ -d "$HOME/.cache/huggingface/speechbrain/spkrec-ecapa-voxceleb" ]; then
     export HF_HUB_OFFLINE=1
   else
     echo "WARNING: ECAPA cache missing; first run will download the model."
   fi
   exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
   ```
   `chmod +x scripts/run_demo.sh`.
5. `docs/multi_user_demo.md`:
   - Install/run, SSH port-forward path, LAN/HTTPS path, upload fallback.
   - Demo script, keyword recommendations/limitations, troubleshooting
     (mic, permission, model cache, port, torch threads).
   - Explain why continuous wake-word and custom names are experimental.
6. Update `README.md` with a short “Multi-person web demo” section linking the
   doc.
7. Add `data/.gitkeep`; ensure `data/` runtime files are ignored.

**Done when:** `scripts/seed_demo.py` creates 3 people; `/api/state` shows them;
the UI can replay a demo clip and announce the correct person.

---

### 5.6 Phase 6 — Hardening, E2E, rehearsal (2 h)

1. **Latency instrumentation:** record `vad_ms`, `keyword_ms`, `speaker_ms`,
   `total_ms` per decision; surface in `/api/health` (rolling p50/p95) and the UI
   badge. Target p95 < 250 ms compute for ≤2 s audio.
2. **Cooldown/dedupe:** ensure one action per utterance; suppress repeats within
   `cooldown_sec`; test double-send of the same audio.
3. **Error paths:** model missing, ECAPA cache missing, corrupt registry,
   full disk (atomic write failure), WS disconnect mid-utterance, invalid WAV.
4. **Security pass:** bind 127.0.0.1 by default; path traversal tests; shell
   allowlist; upload size limits; no stack traces to clients; `--workers 1`
   documented.
5. **Full E2E with real models:** enroll Alice/Bob from fixtures (both `stop`),
   identify Alice vs Bob, unknown speaker, arbitrary speech, replay; assert
   decisions and timings.
6. **Demo rehearsal:** run the exact script in §5.7 twice; restart the server and
   confirm registry persistence.
7. **Final test run:** `.venv/bin/python -m pytest -q` → all green; `node --check`
   both JS files; `curl /api/health`.

**Done when:** the §7 checklist passes top-to-bottom and the demo script is
repeatable.

---

### 5.7 Exact demo script (rehearse this)

**Setup (once):**
```bash
cd /home/tripathiji/projects/DISENT_KWS
./scripts/run_demo.sh          # terminal 1, host
# laptop terminal:
ssh -N -L 8000:127.0.0.1:8000 tripathiji@<host>
# laptop browser: http://localhost:8000
```

**On stage:**
1. People → Add person → **Alice**, keyword **stop** (recommended chip), 5 takes,
   action Announce. Save.
2. Add person → **Bob**, keyword **stop** (same keyword), 5 takes, action
   Announce. Save. Point out both cards show the shared `stop` chip.
3. Live → Start listening. Say **“stop”** as Alice → orb hits, Alice’s name
   reveals, browser says “Alice said stop”, timeline logs scores/latency.
4. Say **“stop”** as Bob → Bob reveals. Emphasize same keyword, different person.
5. Have an unenrolled person say **“stop”** → “Keyword heard — unknown speaker”
   (amber), no action.
6. Alice says a normal sentence (“This is a test of the speaker system”) →
   “Alice” appears as arbitrary-speech identification, no keyword action.
7. Drop a WAV onto the Live view → replay the same decision (fallback if mic
   fails).
8. Settings → lower speaker threshold slightly → repeat step 6 → show the
   sensitivity trade-off.

**Do not demo:** keyword inside a long sentence; a custom name as keyword
(unless pre-calibrated); two people talking at once.

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No browser/mic on host | Certain | High | SSH local port forwarding to laptop (`http://localhost:8000` = secure context); WAV upload/replay fallback; demo assets. |
| Phone mic over HTTPS blocked by self-signed cert | Medium | High | Prefer SSH tunnel; if phone, use a trusted cert (mkcert) or upload/replay; test before demo. |
| Continuous-speech keyword false positives | High if attempted | High | Locked scope: isolated keywords only; `max_keyword_utterance_sec=2.5`; UI hint; script says keyword alone. |
| Custom/OOV keyword fails | Medium | Medium | Curated chips (`stop`, `yes`, `no` tested); custom marked experimental; per-keyword threshold + calibration. |
| Short keyword clip speaker ID below threshold | Medium | Medium | Duration-adaptive thresholds (0.20/0.25/0.32/0.36) + margin 0.08; show “uncertain” rather than guess; 5 enrollment takes. |
| Background noise / far-field | Medium | Medium | Silero VAD; enroll in the demo room; keep noise suppression off (consistent with training); test with the actual device; upload fallback. |
| Enrollment feels slow | Low after fix | Medium | Fast augment (70 ms vs 142 s), batch DISENT/ECAPA (~0.6 s total); progress UI. |
| Model cache/network stall at startup | Medium | High | `HF_HUB_OFFLINE=1` when cache exists; preflight health; pre-download ECAPA; fail fast with instructions. |
| Registry corruption / partial write | Low | High | Atomic temp+`os.replace`, `.bak` copy, validation on load, path safety. |
| Two people share a keyword and are confused | Medium | Medium | Restrict speaker candidates to keyword owners; require margin; show “uncertain” if margin < 0.08. |
| Uvicorn multiple workers duplicate models | Low | Medium | `--workers 1` in run script/docs; health shows PID/loaded models. |
| Silero import slows DISENT | Certain | Low | Set `torch.set_num_threads(4)` after imports; measure; health exposes threads. |
| Shell action dangerous | Low | High | Allowlist executables, `shell=False`, timeout, disabled by default, no auth assumed local-only. |
| Demo device browser lacks AudioWorklet | Low | Medium | Fallback to `ScriptProcessorNode` (deprecated but widely supported) behind a feature check; or upload/replay. |
| `AudioContext({sampleRate:16000})` unsupported | Medium | Low | Catch `NotSupportedError`; native-rate context; server-side resample per chunk (degraded but functional). |

---

## 7. Verification Checklist

### Environment / build
- [ ] `uv add ...` + `uv sync` succeeded; no dependency conflicts.
- [ ] `.venv/bin/python -c "import fastapi, starlette, uvicorn, websockets, multipart, silero_vad"` prints versions.
- [ ] `model_final.pt` loads with 0 missing/unexpected; `/api/health` says `disent=loaded`, `ecapa=loaded`.
- [ ] `HF_HUB_OFFLINE=1` ECAPA load succeeds (cache present).
- [ ] `torch.get_num_threads() == 4` after startup; DISENT batch-1 ≤ 25 ms.
- [ ] `model_final.onnx` is never referenced; `torchaudio.load/save` never called.

### Tests
- [ ] `.venv/bin/python -m pytest -q` → 60 baseline + new tests pass.
- [ ] `tests/test_app_inference.py` real-model tests: keyword gap > 0.10; speaker gap > 0.20.
- [ ] `tests/test_app_registry.py`: shared keyword + atomic write + path traversal.
- [ ] `tests/test_app_api.py`: CRUD, enroll, identify, replay, settings, WS.
- [ ] `node --check app/static/app.js`; `node --check app/static/audio-worklet.js`.

### API / server
- [ ] `curl -s localhost:8000/api/health | python -m json.tool` → `status=ok`.
- [ ] `curl -s localhost:8000/api/state` → people/keywords/settings.
- [ ] Enroll 3 people via `curl -F` (≥3 WAVs each); profiles exist in `data/profiles/`.
- [ ] Restart server; `/api/state` still shows all people; identify scores match pre-restart.
- [ ] `POST /api/identify` with Alice’s clip → Alice; Bob’s clip → Bob; unknown clip → rejected.
- [ ] `POST /api/replay` broadcasts exactly one event and executes one action.
- [ ] WS: hello → ready; 2 s silence → no event; fixture audio → event with `latency_ms`.

### UI (on the demo device)
- [ ] `http://localhost:8000` loads with no console errors; no external network requests.
- [ ] Start listening → orb pulses; speaking shows level; stopping shows “thinking” then result.
- [ ] Add person wizard: 5 takes, playback, retake, save; card appears.
- [ ] Shared keyword: both people show `stop`; speaking as each reveals the right name.
- [ ] Unknown speaker: amber “unknown speaker”, no action, event logged.
- [ ] Arbitrary speech: name reveal, no keyword action.
- [ ] Drop WAV → replay works.
- [ ] People/Keywords/Settings CRUD works; settings persist after reload.
- [ ] Mic permission denied → actionable banner; upload fallback still works.
- [ ] Responsive at 1440/1024/390 px; reduced-motion respected.

### Demo readiness
- [ ] SSH tunnel path tested from the actual laptop/phone.
- [ ] Demo script §5.7 runs twice without restart.
- [ ] Demo assets seed works from a clean `data/`.
- [ ] Backup plan: upload/replay WAVs on the desktop; server restart command handy.
- [ ] Latency badge < 700 ms end-to-end for 1–2 s utterances.

---

## 8. Open Questions (decisions for the user)

1. **Demo device/browser:** Which laptop/phone will open the UI, and can it SSH
   port-forward to this host? This determines whether live mic (SSH tunnel) or
   upload/replay is the primary path. The build supports both.
2. **Demo names/keywords:** Default seed is Alice/Bob/Carol with `stop`/`stop`/
   `yes`. If you want real names as keywords, they are “experimental” and need a
   calibration pass; confirm whether to use curated words instead.
3. **Actions for the live demo:** Is announce/display enough, or must a webhook
   or shell action visibly fire? Shell is allowlisted and off by default for
   safety.
4. **Committing demo audio:** OK to commit ~1 MB of LibriSpeech-derived WAVs
   under CC BY 4.0 with attribution in `demo_assets/README.md`? If not, the seed
   button will be omitted and only live enrollment/upload will be available.

---

## Appendix A — Data model (exact JSON shape)

```json
{
  "version": 1,
  "settings": {
    "keyword_threshold": 0.45,
    "keyword_margin": 0.06,
    "speaker_margin": 0.08,
    "speaker_threshold_scale": 1.0,
    "vad_threshold": 0.5,
    "min_speech_ms": 350,
    "max_utterance_sec": 6.0,
    "max_keyword_utterance_sec": 2.5,
    "cooldown_sec": 1.5,
    "announce": true,
    "actions_enabled": true
  },
  "keywords": [
    {
      "id": "kw_stop_ab12cd",
      "label": "stop",
      "profile": "profiles/kw_stop_ab12cd.npy",
      "n_samples": 10,
      "threshold": null,
      "created_at": "2026-10-21T00:00:00Z"
    }
  ],
  "people": [
    {
      "id": "p_alice_7f3a1c",
      "name": "Alice",
      "color": "#8b5cf6",
      "avatar": "AL",
      "profile": "profiles/p_alice_7f3a1c.npy",
      "keywords": [
        {
          "keyword_id": "kw_stop_ab12cd",
          "profile": "profiles/pk_p_alice_7f3a1c_kw_stop_ab12cd.npy",
          "n_samples": 5
        }
      ],
      "action": {
        "type": "announce",
        "enabled": true,
        "webhook_url": null,
        "shell_command": null,
        "message_template": "{name} said {keyword}"
      },
      "trigger_on_any_speech": false,
      "created_at": "2026-10-21T00:00:00Z"
    }
  ]
}
```

- `avatar` = first two letters of the name (server-generated; UI may override).
- All profile paths are relative to `data/`; resolve and validate on load.
- `threshold: null` means use the global keyword threshold.

## Appendix B — Decision event shape

```json
{
  "id": "evt_01J...",
  "seq": 42,
  "ts": "2026-10-21T00:12:34.567Z",
  "type": "keyword",
  "person_id": "p_alice_7f3a1c",
  "person_name": "Alice",
  "person_color": "#8b5cf6",
  "keyword_id": "kw_stop_ab12cd",
  "keyword": "stop",
  "keyword_score": 0.842,
  "keyword_runner_up": 0.311,
  "speaker_score": 0.714,
  "speaker_runner_up": 0.052,
  "speaker_threshold": 0.36,
  "margin": 0.662,
  "duration_sec": 1.18,
  "latency_ms": {"vad": 0.1, "keyword": 22.8, "speaker": 101.4, "total": 124.3},
  "confidence": 0.859,
  "action": {"type": "announce", "status": "ok", "text": "Alice said stop"},
  "speak": "Alice said stop"
}
```

## Appendix C — Threshold quick reference

| Signal | Default | Evidence |
|---|---|---|
| Keyword cosine threshold | 0.45 | yes min 0.530 vs no max 0.361; stop 0.780 vs 0.177 |
| Keyword top1−top2 margin | 0.06 | separates yes/no; tune in Settings |
| Max keyword utterance | 2.5 s | isolated-only; continuous speech overlaps |
| Speaker threshold <0.75 s | 0.20 | 0.5 s: genuine min 0.250 vs impostor 0.103 |
| Speaker threshold 0.75–1.5 s | 0.25 | 0.75 s: 0.324 vs 0.138; 1.0 s: 0.418 vs 0.130 |
| Speaker threshold 1.5–2.5 s | 0.32 | 1.5 s: 0.467 vs 0.114; 2 s: 0.592 vs 0.138 |
| Speaker threshold ≥2.5 s | 0.36 | 3 s: 0.577 vs 0.096; full: 0.618 vs 0.126 |
| Speaker top1−top2 margin | 0.08 | held-out margin min 0.342 (same-speaker) |
| VAD threshold | 0.5 | Silero default; 0 FP on silence/noise |
| VAD min silence | 300 ms | natural end-of-utterance latency |
| VAD speech pad | 100 ms | keep onset/offset |
| Max utterance | 6.0 s | force-finalize if no VAD end |
| Cooldown | 1.5 s | prevent duplicate actions |
