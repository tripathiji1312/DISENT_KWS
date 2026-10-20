# Multi-Person Voice Assistant — Demo Guide

A local, Google Home/Alexa-style demo built on DISENT-KWS:

- Several people can be enrolled at once.
- The same keyword can be assigned to more than one person (`stop` for both
  Alice and Bob in the demo).
- The system detects an isolated keyword, identifies the enrolled speaker, and
  runs that person's action (announce, webhook, allowlisted shell command, or
  log).
- Arbitrary speech is attributed to an enrolled person when the voice matches.
  Unknown speakers are rejected instead of being forced onto the nearest person.
- Everything runs on the local machine. No cloud service and no API keys are
  required.

The browser dashboard is served by the same FastAPI process as the models. The
microphone path streams Float32 audio over a WebSocket; WAV upload and replay
are available when microphone access is not possible.

---

## 1. What you need

- The repo's Python 3.10 environment and `uv`.
- `model_final.pt` at the repository root (the DISENT checkpoint).
- A cached SpeechBrain ECAPA model, or internet access for the first run:

  ```text
  ~/.cache/huggingface/speechbrain/spkrec-ecapa-voxceleb
  ```

- For browser microphone input: a browser on the **same machine** or an SSH
  tunnel to `localhost` (see §4). Browsers only allow microphone capture on a
  secure context; `http://localhost` counts as secure, `http://<server-ip>` does
  not.

The host running this repo does not need a browser or sound card. You can do
the whole demo with WAV upload/replay from another machine.

---

## 2. Install and seed

From the repository root:

```bash
uv sync --all-extras

# Create data/registry.json and data/profiles/*.npy from demo_assets/.
PYTHONPATH=src .venv/bin/python scripts/seed_demo.py --data-dir data --force
```

`uv sync --all-extras` is required for the test/dev extras. Plain `uv sync`
removes `pytest`.

The seed creates two people who share the keyword **stop**:

| Person | Keyword | Enrollment | Action |
|:---|:---|:---|:---|
| Alice | `stop` | `demo_assets/alice_sentence_{1,2,3}.wav` | announce `"Alice said stop"` |
| Bob | `stop` | `demo_assets/bob_sentence_{1,2,3}.wav` | announce `"Bob said stop"` |

The seed also carries held-out clips used to demonstrate rejection:

- `unknown_stop.wav` — says "stop", but the speaker is not enrolled.
- `unknown_sentence.wav` — arbitrary speech from an unknown speaker.
- `alice_speech.wav` — arbitrary speech from Alice.

Re-run the seed with `--force` any time the speaker-embedding code changes.
Changing thresholds or actions does **not** require re-seeding.

---

## 3. Start the server

```bash
./scripts/run_demo.sh
```

The default address is `http://127.0.0.1:8000`. To use a different port:

```bash
PORT=8010 ./scripts/run_demo.sh
```

Run **one worker**. The dashboard, registry, in-memory connections, and model
engine are per-process state; multiple workers would serve inconsistent state.
`scripts/run_demo.sh` already passes `--workers 1`.

Check it directly on the host:

```bash
curl -s http://127.0.0.1:8000/api/health | python -m json.tool
```

`status` should be `ok`, both models should say `loaded`, and the people count
should be `2` after seeding.

---

## 4. Open the dashboard from your laptop

On the machine running the server, keep it bound to `127.0.0.1`. From the
laptop, forward the port:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<server>
```

Then open **`http://localhost:8000`** in the laptop browser. The browser sees
`localhost`, so microphone capture is allowed.

If the server runs on port 8010:

```bash
ssh -N -L 8000:127.0.0.1:8010 <user>@<server>
# open http://localhost:8000
```

Do not open `http://<server-ip>:8000` and expect the mic to work. Use the
tunnel, or use the upload/replay buttons.

---

## 5. Demo script (about 3 minutes)

1. **Open the dashboard.** The header pill should show **Live** and the footer
   should show the loaded models.
2. **Confirm the cast.** In **People**, Alice and Bob should both show the
   `stop` keyword. If the registry is empty, click **Load demo people** (or
   re-run the seed command).
3. **Alice says stop.** Click **Start listening**, say **"stop"** once, then
   pause. The result card should show **Alice**, keyword `stop`, and a keyword
   similarity around `0.85`.
4. **Bob says stop.** Repeat with Bob. The same keyword must resolve to **Bob**,
   not Alice. On the host's measured run: Bob keyword `0.891`, speaker `0.415`.
5. **Held-out person says stop.** Use **Upload WAV** and choose
   `demo_assets/unknown_stop.wav`, or drag it onto the page. Expected event:
   **`keyword_unknown_speaker`** — the keyword matched (`~0.92`) but the best
   speaker score (`~0.145`) stayed below the `0.20` short-clip threshold.
6. **Arbitrary enrolled speech.** Upload `demo_assets/alice_speech.wav`.
   Expected: **`speech` / Alice**. This is not a keyword match; it is speaker
   identification on a normal sentence.
7. **Arbitrary unknown speech.** Upload `demo_assets/unknown_sentence.wav`.
   Expected: **`unrecognized`** — the best speaker score stays below the
   `0.36` long-clip threshold.
8. **Actions.** For a keyword event, the event timeline shows the action
   result. Announce actions also return the text the dashboard can speak through
   the browser's speech synthesis.

Measured on the host (CPU, one worker):

| Clip | Event | Person | Keyword score | Speaker score | Latency |
|:---|:---|:---|:---:|:---:|:---:|
| `alice_stop.wav` | `keyword` | Alice | 0.851 | 0.305 | ~119 ms |
| `bob_stop.wav` | `keyword` | Bob | 0.891 | 0.415 | ~89 ms |
| `unknown_stop.wav` | `keyword_unknown_speaker` | — | 0.921 | 0.145 | ~103 ms |
| `alice_speech.wav` | `speech` | Alice | — | 0.858 | ~338 ms |
| `bob_sentence_1.wav` | `speech` | Bob | — | 0.863 | ~341 ms |
| `unknown_sentence.wav` | `unrecognized` | — | — | 0.248 max | ~411 ms |

Latency depends on clip length and machine load; short keywords are faster
because there are fewer frames to encode.

### Enrollment from the UI

1. Open **People → Add person**.
2. Enter a name and choose an existing keyword or create a new one.
3. If the microphone is available, record 3–5 takes of about 2 seconds each.
   Otherwise upload 3–5 WAV files.
4. Record the keyword takes separately when prompted. Isolated words work best.
5. Save. The person appears immediately and the live session picks up the new
   profile on the next event.

A person may share a keyword with others. The keyword detector produces a
keyword score; the speaker matcher then ranks the enrolled people. The UI does
not need one keyword per person.

---

## 6. How it works

```
browser mic (AudioWorklet, 16 kHz)
        │  binary Float32LE chunks over /ws/live
        ▼
FastAPI WebSocket ──► Silero VAD (512-sample frames, 300 ms silence)
        │
        ├─ segment < 2.5 s ──► DISENT keyword score (cosine to keyword centroid)
        ├─ segment >= 2.5 s ─► speaker-only path
        ▼
SpeechBrain ECAPA speaker embedding (natural clip length)
        │
        ├─ keyword + known speaker ──► keyword event + action
        ├─ keyword + unknown speaker ► keyword_unknown_speaker
        ├─ speech + known speaker ───► speech event (action only if enabled)
        └─ speech + unknown speaker ─► unrecognized
```

Key implementation choices:

- **Keyword score:** DISENT's unconditioned phonetic embedding (`z_phn`) from
  `model_final.pt`, compared by cosine to the enrolled keyword centroid. The
  joint `DualGateScorer` and the broken `model_final.onnx` are not used for
  multi-person ranking.
- **Speaker score:** SpeechBrain ECAPA-TDNN. The DISENT `z_spk` output is only a
  fallback; ECAPA is much better calibrated for this demo.
- **Short clips are not padded.** A 0.5 s keyword padded to 6 s changes ECAPA's
  embedding enough to swap speakers. Speaker embeddings keep the natural clip
  length; only DISENT's fixed-length mel input is padded.
- **Duration-aware speaker thresholds:** `<0.75 s: 0.20`, `0.75–1.5 s: 0.25`,
  `1.5–2.5 s: 0.32`, `>=2.5 s: 0.36`, plus a `0.08` margin over the runner-up.
- **VAD:** Silero `VADIterator`, 16 kHz, 512-sample frames, threshold `0.5`,
  minimum silence `300 ms`, speech pad `100 ms`. Utterances are force-finalized
  at 6 s because a clip that ends while speech is still active may never emit an
  `end` event otherwise.
- **Persistence:** `data/registry.json` plus `data/profiles/*.npy`, written
  atomically. The runtime directory is gitignored and local to the machine.

Only isolated keyword utterances up to about 2.5 s are treated as keyword
candidates. Continuous keyword spotting inside a sentence was measured
unreliable and is not claimed here. `stop`, `yes`, and `no` are the validated
keywords; arbitrary/OOV keywords are marked experimental in the UI.

---

## 7. HTTP and WebSocket reference

| Method | Path | Purpose |
|:---|:---|:---|
| `GET` | `/api/health` | Model state, PID, people/keyword counts, latency p50/p95 |
| `GET` | `/api/state` | People, keywords, settings, recent events |
| `GET` | `/api/people` | People and their enrolled keywords |
| `GET` | `/api/keywords` | Keyword thresholds and centroids |
| `POST` | `/api/people` | Create an empty person |
| `PATCH` | `/api/people/{id}` | Update name/color/action/trigger |
| `DELETE` | `/api/people/{id}?force=` | Delete a person |
| `POST` | `/api/keywords` | Create a keyword |
| `PATCH` | `/api/keywords/{id}` | Update threshold/label |
| `DELETE` | `/api/keywords/{id}?force=` | Delete a keyword |
| `POST` | `/api/enroll` | Multipart: name, files, keyword, action |
| `POST` | `/api/people/{id}/keywords` | Add a keyword take to a person |
| `POST` | `/api/identify` | Single WAV → decision, action optional |
| `POST` | `/api/replay` | Single WAV → decision, record + broadcast |
| `POST` | `/api/settings` | Update runtime thresholds/options |
| `GET` | `/api/events` | Recent persisted events |
| `POST` | `/api/demo/seed?force=true` | Seed Alice/Bob from `demo_assets` |
| `WS` | `/ws/live` | Live audio and events |

### WebSocket protocol

The server sends:

```json
{"type": "ready", "settings": {...}, "models": {...}}
{"type": "level", "rms": 0.0123, "speech": false, "vad": false}
{"type": "event", "event": {"type": "keyword", "person_name": "Alice", ...}}
{"type": "state", "state": {...}}
{"type": "settings", "settings": {...}}
{"type": "error", "message": "..."}
```

The client sends text control messages and binary audio:

```json
{"type": "hello", "sample_rate": 16000}
{"type": "reset"}
{"type": "ping"}
```

Binary frames are little-endian Float32 mono. Any chunk size is accepted as
long as the byte length is a multiple of 4; the server buffers to 512-sample
VAD frames. The browser AudioWorklet requests 16 kHz mono Float32; if the
browser chooses a different native rate, `hello` tells the server that rate and
the server resamples.

Event types:

- `keyword` — keyword matched and an enrolled speaker was identified.
- `keyword_unknown_speaker` — keyword matched, but no enrolled speaker passed
  the threshold.
- `speech` — arbitrary speech attributed to an enrolled person.
- `unrecognized` — no keyword and no enrolled speaker above threshold.

Each event includes keyword/speaker scores, the threshold used, the runner-up
margin, duration, latency breakdown, and the action result.

---

## 8. Settings and tuning

The **Settings** view edits the runtime values stored in `registry.json`:

| Setting | Default | Effect |
|:---|:---:|:---|
| `keyword_threshold` | `0.45` | Minimum cosine for a keyword match |
| `keyword_margin` | `0.06` | Required lead of top keyword over second |
| `speaker_margin` | `0.08` | Required lead of top person over runner-up |
| `speaker_threshold_scale` | `1.0` | Multiplies the duration-aware threshold |
| `vad_threshold` | `0.5` | Silero speech probability threshold |
| `min_speech_ms` | `350` | Minimum speech duration |
| `max_utterance_sec` | `6.0` | Force-finalize an ongoing segment |
| `max_keyword_utterance_sec` | `2.5` | Keyword path vs speaker-only path |
| `cooldown_sec` | `1.5` | Per-person action cooldown |
| `announce` | `true` | Return announce text to the browser |
| `actions_enabled` | `true` | Master switch for actions |

Raise `speaker_threshold_scale` if a known person is being rejected. Lower it
if an unknown person is being accepted. Do not lower the short-clip threshold
below `0.18`; held-out impostors reached `0.145–0.16`.

Shell actions are restricted to `notify-send`, `paplay`, and `aplay`; the
master `actions_enabled` switch is on by default. Webhook actions should only
target hosts you control.

---

## 9. Troubleshooting

| Symptom | Fix |
|:---|:---|
| Dashboard says **Offline** | Server not running, wrong port, or SSH tunnel not active |
| Mic button does nothing | Open `http://localhost:8000` via the tunnel; check browser site permissions; try upload |
| `status: degraded`, HTTP 503 | `model_final.pt` missing or engine failed to load; check server log |
| ECAPA download/offline error | Cache `spkrec-ecapa-voxceleb` or run once with internet and `HF_HUB_OFFLINE` unset |
| Upload returns 415 | Not a RIFF/WAVE file; export WAV, mono or stereo is fine |
| No events while speaking | Speak an isolated word after the pill shows Live; watch `level`/VAD badge; check `vad_threshold` |
| Known person rejected | Re-seed after embedding changes; raise `speaker_threshold_scale` slightly |
| Unknown person accepted | Lower `speaker_threshold_scale`; use a longer enrollment sample |
| Speaker swapped after code change | Re-seed. Old profiles were built with the zero-padding bug and are invalid |
| Port already in use | `PORT=8010 ./scripts/run_demo.sh` and forward the matching port |

Runtime logs are written to the terminal running `run_demo.sh`. The registry is
`data/registry.json`; delete `data/` and re-seed for a clean demo.

---

## 10. Tests

Fast checks (no model weights):

```bash
.venv/bin/python -m pytest -q -m "not integration"
node --check app/static/app.js
node --check app/static/audio-worklet.js
```

Real-model integration tests (needs `model_final.pt`, the ECAPA cache, and
`demo_assets/`):

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q -m integration
```

The integration suite includes a regression test for the short-keyword
zero-padding bug: Bob's 0.56 s `stop` clip must match Bob's voiceprint, not
Alice's.

---

## 11. Privacy and licenses

- Audio is processed locally; uploads are not sent to a cloud service.
- `data/` contains voiceprints and event history. It is gitignored. Do not
  commit or share it.
- Demo clips are LibriSpeech `dev-clean` excerpts, CC BY 4.0. See
  `demo_assets/README.md` for attribution.
- The DISENT model and the SpeechBrain ECAPA model keep their original
  licenses; see the repository README.
