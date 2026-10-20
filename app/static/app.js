/**
 * DISENT Voice dashboard.
 * Vanilla ES module: WebSocket live audio, enrollment wizard, registry CRUD.
 */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const PALETTE = ["#8b5cf6", "#06b6d4", "#f59e0b", "#ec4899", "#22c55e", "#ef4444", "#3b82f6", "#a3e635"];
const RECOMMENDED = [
  "stop",
  "yes",
  "no",
  "hey google",
  "alexa",
  "lights on",
  "lights off",
  "play music",
  "weather",
  "timer",
];

const smartHome = {
  light: { on: false, color: "#f59e0b", person: null },
  music: { playing: false, track: "Personalized Playlist", person: null },
  timer: { running: false, seconds: 10, total: 10, timerId: null, person: null },
  weather: { temp: "24°C", condition: "Sunny", person: null },
};

function playChime(kind = "google") {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") ctx.resume();
    const now = ctx.currentTime;
    if (kind === "google") {
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();
      osc1.type = "sine";
      osc2.type = "sine";
      osc1.frequency.setValueAtTime(783.99, now); // G5
      osc2.frequency.setValueAtTime(1046.50, now + 0.11); // C6
      gain.gain.setValueAtTime(0.14, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.38);
      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.11);
      osc2.start(now + 0.11);
      osc2.stop(now + 0.38);
    } else if (kind === "timer") {
      [523.25, 659.25, 783.99].forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.frequency.setValueAtTime(freq, now + idx * 0.12);
        gain.gain.setValueAtTime(0.18, now + idx * 0.12);
        gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.12 + 0.28);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now + idx * 0.12);
        osc.stop(now + idx * 0.12 + 0.28);
      });
    } else if (kind === "click") {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.frequency.setValueAtTime(587.33, now);
      gain.gain.setValueAtTime(0.08, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.08);
    }
  } catch (_) {
    // audio context blocked or unavailable
  }
}

function renderSmartHub() {
  const lightCard = $("#device-light");
  const lightPill = $("#light-status-pill");
  const lightDesc = $("#light-desc");
  if (lightCard && lightPill) {
    if (smartHome.light.on) {
      lightCard.classList.add("is-light-on");
      lightPill.className = "pill pill-on";
      lightPill.textContent = "ON";
      lightCard.style.setProperty("--person-color", smartHome.light.color || "#f59e0b");
      lightCard.style.setProperty("--person-glow", (smartHome.light.color || "#f59e0b") + "44");
      lightDesc.textContent = smartHome.light.person ? `${smartHome.light.person}’s color glow` : "Living room lit";
    } else {
      lightCard.classList.remove("is-light-on");
      lightPill.className = "pill pill-off";
      lightPill.textContent = "OFF";
      lightDesc.textContent = "Voice: “lights on” / “lights off”";
    }
  }

  const musicCard = $("#device-music");
  const musicPill = $("#music-status-pill");
  const musicTrack = $("#music-track");
  if (musicCard && musicPill) {
    if (smartHome.music.playing) {
      musicCard.classList.add("is-music-playing");
      musicPill.className = "pill pill-playing";
      musicPill.textContent = "PLAYING";
      musicTrack.textContent = smartHome.music.track;
    } else {
      musicCard.classList.remove("is-music-playing");
      musicPill.className = "pill pill-off";
      musicPill.textContent = "PAUSED";
      musicTrack.textContent = "Voice: “play music” / “stop”";
    }
  }

  const timerCard = $("#device-timer");
  const timerPill = $("#timer-status-pill");
  const timerDisplay = $("#timer-display");
  const timerFill = $("#timer-fill");
  if (timerCard && timerPill) {
    if (smartHome.timer.running) {
      timerCard.classList.add("is-timer-running");
      timerPill.className = "pill pill-running";
      timerPill.textContent = `${smartHome.timer.seconds}s`;
      timerDisplay.textContent = smartHome.timer.person ? `${smartHome.timer.person}’s timer` : "Counting down...";
      if (timerFill) {
        const pct = ((smartHome.timer.total - smartHome.timer.seconds) / smartHome.timer.total) * 100;
        timerFill.style.width = `${pct}%`;
      }
    } else {
      timerCard.classList.remove("is-timer-running");
      timerPill.className = "pill pill-off";
      timerPill.textContent = smartHome.timer.seconds === 0 ? "DONE" : "READY";
      timerDisplay.textContent = smartHome.timer.seconds === 0 ? "Timer finished!" : "Voice: “timer” (10s countdown)";
      if (timerFill) timerFill.style.width = smartHome.timer.seconds === 0 ? "100%" : "0%";
    }
  }
}

function startTimer(seconds = 10, person = null) {
  clearInterval(smartHome.timer.timerId);
  smartHome.timer.running = true;
  smartHome.timer.seconds = seconds;
  smartHome.timer.total = seconds;
  smartHome.timer.person = person;
  renderSmartHub();
  playChime("click");
  smartHome.timer.timerId = setInterval(() => {
    smartHome.timer.seconds -= 1;
    if (smartHome.timer.seconds <= 0) {
      clearInterval(smartHome.timer.timerId);
      smartHome.timer.running = false;
      smartHome.timer.seconds = 0;
      renderSmartHub();
      playChime("timer");
      if (state.data.settings.announce !== false) {
        speak(person ? `${person}, your timer is done!` : "Timer complete!");
      }
    } else {
      renderSmartHub();
    }
  }, 1000);
}

function handleSmartAssistantAction(event) {
  const kw = (event.keyword || "").toLowerCase().trim();
  const person = event.person_name;
  const color = event.person_color || "#8b5cf6";
  const hubBadge = $("#hub-speaker-badge");
  const hubName = $("#hub-speaker-name");

  if (event.type === "keyword_unknown_speaker") {
    if (hubBadge && hubName) {
      hubBadge.className = "hub-speaker-badge is-rejected";
      hubName.textContent = "Voice Match Failed · Impostor Rejected";
    }
    const leds = $("#google-leds");
    if (leds) {
      leds.className = "google-leds is-listening";
      setTimeout(() => { if (leds) leds.className = "google-leds"; }, 2000);
    }
    return;
  }

  if (event.type === "keyword" || event.type === "speech") {
    if (hubBadge && hubName && person) {
      hubBadge.className = "hub-speaker-badge is-verified";
      const conf = Math.round(Number(event.speaker_score || 0.85) * 100);
      hubName.textContent = `Verified: ${person} (${conf}% Voice Match)`;
    }

    const leds = $("#google-leds");
    if (leds) {
      leds.className = "google-leds is-hit";
      setTimeout(() => { if (leds) leds.className = "google-leds"; }, 2600);
    }

    if (kw === "hey google" || kw === "alexa") {
      playChime("google");
      if (state.data.settings.announce !== false && !event.speak) {
        speak(`Hi ${person}, how can I help you?`);
      }
    } else if (kw.includes("light")) {
      playChime("click");
      if (kw.includes("off")) {
        smartHome.light.on = false;
        smartHome.light.person = person;
        renderSmartHub();
        if (state.data.settings.announce !== false && !event.speak) {
          speak(`Turned off lights for ${person}`);
        }
      } else {
        smartHome.light.on = true;
        smartHome.light.color = color;
        smartHome.light.person = person;
        renderSmartHub();
        if (state.data.settings.announce !== false && !event.speak) {
          speak(`Lights turned on in ${person}’s color`);
        }
      }
    } else if (kw.includes("music") || kw === "play") {
      playChime("click");
      smartHome.music.playing = true;
      smartHome.music.track = `${person}’s Daily Mix · Chill Beats`;
      smartHome.music.person = person;
      renderSmartHub();
      if (state.data.settings.announce !== false && !event.speak) {
        speak(`Playing ${person}’s favorite mix`);
      }
    } else if (kw === "stop") {
      if (smartHome.music.playing) {
        smartHome.music.playing = false;
        renderSmartHub();
        playChime("click");
      }
      if (smartHome.timer.running) {
        clearInterval(smartHome.timer.timerId);
        smartHome.timer.running = false;
        renderSmartHub();
        playChime("click");
      }
    } else if (kw.includes("timer")) {
      startTimer(10, person);
    } else if (kw.includes("weather")) {
      playChime("google");
      if (state.data.settings.announce !== false && !event.speak) {
        speak(`Good day ${person}, the forecast is 24 degrees and sunny`);
      }
    }
  }
}

const DEFAULT_SETTINGS = {
  keyword_threshold: 0.45,
  keyword_margin: 0.06,
  speaker_margin: 0.08,
  speaker_threshold_scale: 1.0,
  vad_threshold: 0.5,
  min_speech_ms: 350,
  max_utterance_sec: 6.0,
  max_keyword_utterance_sec: 2.5,
  cooldown_sec: 1.5,
  announce: true,
  actions_enabled: true,
};

const state = {
  data: {
    people: [],
    keywords: [],
    settings: { ...DEFAULT_SETTINGS },
    events: [],
    recommended_keywords: RECOMMENDED,
  },
  events: [],
  health: null,
  ws: null,
  wsReady: false,
  reconnectAttempt: 0,
  reconnectTimer: null,
  listening: false,
  capture: null,
  level: 0,
  waveLevels: new Array(96).fill(0),
  orbLockUntil: 0,
  lastEvent: null,
  activePersonTimer: null,
  wizard: null,
  recording: false,
  activeRecorder: null,
  saving: false,
  settingsDirty: false,
  audioUnlocked: false,
};

const TYPE_LABEL = {
  keyword: "Keyword matched",
  speech: "Speaker identified",
  keyword_unknown_speaker: "Keyword / unknown speaker",
  unrecognized: "No confident match",
  too_short: "Too short",
};

const TYPE_COLOR = {
  keyword: "#8b5cf6",
  speech: "#22d3ee",
  keyword_unknown_speaker: "#f59e0b",
  unrecognized: "#6f6f84",
  too_short: "#6f6f84",
};

const OUTPUT_IDS = {
  keyword_threshold: "keyword-threshold-out",
  keyword_margin: "keyword-margin-out",
  speaker_margin: "speaker-margin-out",
  speaker_threshold_scale: "speaker-scale-out",
  vad_threshold: "vad-threshold-out",
  min_speech_ms: "min-speech-out",
  max_utterance_sec: "max-utterance-out",
  max_keyword_utterance_sec: "max-keyword-out",
  cooldown_sec: "cooldown-out",
};
const outputFor = (key) => (OUTPUT_IDS[key] ? document.getElementById(OUTPUT_IDS[key]) : null);

// --------------------------------------------------------------------- utils
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const fmtScore = (value) => Number(value || 0).toFixed(3);
const fmtMs = (value) => `${Math.round(Number(value || 0))} ms`;
const fmtSeconds = (value) => `${Number(value || 0).toFixed(2)} s`;
const initials = (name) => {
  const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  return (parts.length > 1 ? parts[0][0] + parts[1][0] : parts[0].slice(0, 2)).toUpperCase();
};
const relativeTime = (timestamp) => {
  if (!timestamp) return "—";
  const then = new Date(timestamp).getTime();
  if (!Number.isFinite(then)) return "—";
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};

function toast(message, kind = "") {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast${kind ? ` is-${kind}` : ""}`;
  node.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { node.hidden = true; }, kind === "error" ? 7000 : 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload && payload.detail) detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    } catch (_) { /* keep status text */ }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

// ----------------------------------------------------------------- websocket
function setConnection(mode) {
  const pill = $("#connection-pill");
  const text = $("#connection-text");
  pill.classList.remove("is-online", "is-offline");
  if (mode === "online") {
    pill.classList.add("is-online");
    text.textContent = "Live";
  } else if (mode === "connecting") {
    text.textContent = "Connecting";
  } else {
    pill.classList.add("is-offline");
    text.textContent = "Offline";
  }
}

function connectWS() {
  clearTimeout(state.reconnectTimer);
  setConnection("connecting");
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  let socket;
  try {
    socket = new WebSocket(`${protocol}//${location.host}/ws/live`);
  } catch (error) {
    setConnection("offline");
    scheduleReconnect();
    return;
  }
  socket.binaryType = "arraybuffer";
  state.ws = socket;

  socket.addEventListener("open", () => {
    state.wsReady = true;
    state.reconnectAttempt = 0;
    setConnection("online");
    if (state.listening) sendHello();
  });

  socket.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (_) {
      return;
    }
    handleWSMessage(message);
  });

  socket.addEventListener("close", () => {
    state.wsReady = false;
    setConnection("offline");
    state.ws = null;
    if (state.listening) stopListening(false);
    scheduleReconnect();
  });

  socket.addEventListener("error", () => {
    setConnection("offline");
  });
}

function scheduleReconnect() {
  clearTimeout(state.reconnectTimer);
  const delay = Math.min(5000, 800 * 2 ** state.reconnectAttempt);
  state.reconnectAttempt += 1;
  state.reconnectTimer = setTimeout(connectWS, delay);
}

function handleWSMessage(message) {
  if (!message || !message.type) return;
  if (message.type === "ready") {
    state.wsReady = true;
    setConnection("online");
    if (message.settings) {
      state.data.settings = { ...state.data.settings, ...message.settings };
    }
    if (state.listening) sendHello();
  } else if (message.type === "level") {
    updateLevel(message.rms || 0, Boolean(message.speech));
  } else if (message.type === "event") {
    handleEvent(message.event);
  } else if (message.type === "state") {
    if (message.state) {
      state.data = message.state;
      state.events = mergeEvents(message.state.events || [], state.events);
      renderAll();
    }
  } else if (message.type === "settings") {
    state.data.settings = message.settings;
    renderSettings();
  } else if (message.type === "error") {
    toast(message.message || "Server error", "error");
  }
}

function sendHello() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.ws.send(JSON.stringify({
    type: "hello",
    sample_rate: state.capture ? state.capture.sampleRate : 16000,
    client: "browser",
  }));
}

function sendControl(payload) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(payload));
  }
}

function mergeEvents(primary, secondary) {
  const seen = new Set();
  const merged = [];
  for (const event of [...primary, ...secondary]) {
    if (!event || !event.id || seen.has(event.id)) continue;
    seen.add(event.id);
    merged.push(event);
  }
  merged.sort((a, b) => String(b.ts || "").localeCompare(String(a.ts || "")));
  return merged.slice(0, 100);
}

// ------------------------------------------------------------------ level/ui
function updateLevel(rms, speech) {
  const level = clamp(rms * 7.5, 0, 0.9);
  state.level = level;
  const orb = $("#orb");
  orb.style.setProperty("--level", level.toFixed(3));
  state.waveLevels.shift();
  state.waveLevels.push(level);
  const vad = $("#vad-badge");
  vad.textContent = speech ? "VAD speech" : "VAD idle";
  vad.classList.toggle("good", speech);
  if (state.listening && Date.now() > state.orbLockUntil) {
    setOrb("listening");
    setStageStatus("Listening — say your keyword");
  }
}

function updateLocalLevel(rms) {
  const level = clamp(rms * 3.2, 0, 0.75);
  $("#orb").style.setProperty("--level", level.toFixed(3));
  state.waveLevels.shift();
  state.waveLevels.push(level);
}

function setOrb(kind, color = null) {
  const orb = $("#orb");
  orb.classList.remove("is-idle", "is-listening", "is-thinking", "is-hit", "is-unknown");
  orb.classList.add(`is-${kind}`);
  if (color) orb.style.setProperty("--person-color", color);
}

function setStageStatus(main, sub = null) {
  $("#stage-status").textContent = main;
  if (sub !== null) $("#stage-substatus").innerHTML = sub;
}

// ------------------------------------------------------------------- capture
class MicCapture {
  constructor({ onSamples, onLevel, onError }) {
    this.onSamples = onSamples || (() => {});
    this.onLevel = onLevel || (() => {});
    this.onError = onError || (() => {});
    this.stream = null;
    this.ctx = null;
    this.source = null;
    this.node = null;
    this.sink = null;
    this.analyser = null;
    this.buffer = null;
    this.raf = null;
    this.stopped = false;
  }

  get sampleRate() {
    return this.ctx ? this.ctx.sampleRate : 16000;
  }

  async start(deviceId) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "Microphone needs a secure context. Open http://localhost:8000 (SSH port forwarding) or use Upload WAV."
      );
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
    });

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    try {
      this.ctx = new AudioCtx({ sampleRate: 16000, latencyHint: "interactive" });
    } catch (_) {
      this.ctx = new AudioCtx();
    }
    await this.ctx.resume().catch(() => {});
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.source.connect(this.analyser);

    let usedWorklet = false;
    if (this.ctx.audioWorklet && typeof AudioWorkletNode !== "undefined") {
      try {
        await this.ctx.audioWorklet.addModule("/static/audio-worklet.js");
        this.node = new AudioWorkletNode(this.ctx, "capture-processor");
        this.node.port.onmessage = (event) => {
          if (!this.stopped) this.onSamples(event.data);
        };
        this.source.connect(this.node);
        usedWorklet = true;
      } catch (_) {
        usedWorklet = false;
      }
    }

    if (!usedWorklet) {
      const processor = this.ctx.createScriptProcessor(2048, 1, 1);
      processor.onaudioprocess = (event) => {
        if (this.stopped) return;
        this.onSamples(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      this.source.connect(processor);
      this.node = processor;
    }

    this.sink = this.ctx.createGain();
    this.sink.gain.value = 0;
    this.node.connect(this.sink);
    this.sink.connect(this.ctx.destination);
    this.stopped = false;
    this._levelLoop();
    return this;
  }

  _levelLoop() {
    if (!this.analyser || this.stopped) return;
    const data = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i += 1) sum += data[i] * data[i];
    this.onLevel(Math.sqrt(sum / data.length));
    this.raf = requestAnimationFrame(() => this._levelLoop());
  }

  stop() {
    this.stopped = true;
    cancelAnimationFrame(this.raf);
    this.raf = null;
    try { this.node?.disconnect(); } catch (_) { /* noop */ }
    try { this.source?.disconnect(); } catch (_) { /* noop */ }
    try { this.sink?.disconnect(); } catch (_) { /* noop */ }
    try { this.stream?.getTracks().forEach((track) => track.stop()); } catch (_) { /* noop */ }
    try { this.ctx?.close(); } catch (_) { /* noop */ }
    this.ctx = null;
    this.stream = null;
  }
}

async function startListening() {
  if (state.listening) return;
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    toast("Server is not connected yet. Check the Live pill, then try again.", "error");
    return;
  }
  const deviceId = $("#device-select").value;
  const capture = new MicCapture({
    onSamples: (samples) => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(samples.buffer);
      }
    },
    onLevel: updateLocalLevel,
    onError: (error) => toast(error.message, "error"),
  });
  try {
    await capture.start(deviceId);
  } catch (error) {
    const name = error.name || "";
    if (name === "NotAllowedError") {
      toast("Microphone permission was denied. Allow it in the browser, or use Upload WAV.", "error");
    } else if (name === "NotFoundError") {
      toast("No microphone found. Connect one, or use Upload WAV.", "error");
    } else {
      toast(error.message || "Could not start the microphone.", "error");
    }
    return;
  }
  state.capture = capture;
  state.listening = true;
  $("#mic-btn").classList.add("is-recording");
  $("#mic-btn-label").textContent = "Stop listening";
  $("#stream-badge").textContent = `${capture.sampleRate} Hz stream`;
  sendHello();
  setOrb("listening");
  setStageStatus("Listening — say your keyword", "Say just <strong>stop</strong>, <strong>yes</strong>, or <strong>no</strong>, then pause");
  await populateDevices();
}

function stopListening(sendReset = true) {
  if (state.capture) {
    state.capture.stop();
    state.capture = null;
  }
  state.listening = false;
  $("#mic-btn").classList.remove("is-recording");
  $("#mic-btn-label").textContent = "Start listening";
  $("#vad-badge").textContent = "VAD idle";
  $("#vad-badge").classList.remove("good");
  if (sendReset) sendControl({ type: "reset" });
  if (!state.wizard || state.wizard.mode !== "voice") {
    setOrb("idle");
    setStageStatus("Ready when you are", "Isolated words work best · try <strong>stop</strong>, <strong>yes</strong>, or <strong>no</strong>");
  }
}

async function populateDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((device) => device.kind === "audioinput");
    const select = $("#device-select");
    const current = select.value;
    select.innerHTML = '<option value="">Default microphone</option>';
    inputs.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || `Microphone ${index + 1}`;
      select.appendChild(option);
    });
    if (current && inputs.some((device) => device.deviceId === current)) select.value = current;
  } catch (_) { /* devices are optional */ }
}

// ------------------------------------------------------------------ recording
function mergeFloat32(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = clamp(samples[i], -1, 1);
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

function setTakeStatus(index, text) {
  const node = $(`#take-status-${index}`);
  if (node) node.textContent = text;
}

function setTakeLevel(index, level) {
  const node = $(`#take-level-${index}`);
  if (node) node.style.width = `${clamp(level * 220, 0, 100)}%`;
}

async function recordTake(index) {
  if (!state.wizard || state.recording) return;
  state.recording = true;
  renderWizard();
  const chunks = [];
  let sampleRate = 16000;
  const capture = new MicCapture({
    onSamples: (samples) => chunks.push(samples),
    onLevel: (level) => setTakeLevel(index, level),
    onError: (error) => toast(error.message, "error"),
  });
  state.activeRecorder = capture;
  try {
    await capture.start($("#device-select").value);
    sampleRate = capture.sampleRate;
    for (let countdown = 3; countdown >= 1; countdown -= 1) {
      if (!state.wizard) throw new Error("cancelled");
      setTakeStatus(index, `Recording in ${countdown}…`);
      await sleep(650);
    }
    setTakeStatus(index, "Recording — say your keyword");
    await sleep(2200);
    capture.stop();
    const samples = mergeFloat32(chunks);
    if (samples.length < sampleRate * 0.25) throw new Error("Recording was too short");
    const blob = encodeWav(samples, sampleRate);
    state.wizard.takes[index] = {
      blob,
      url: URL.createObjectURL(blob),
      duration: samples.length / sampleRate,
      sampleRate,
    };
    toast(`Take ${index + 1} ready`, "success");
  } catch (error) {
    if (error.message !== "cancelled") toast(error.message || "Recording failed", "error");
  } finally {
    capture.stop();
    state.activeRecorder = null;
    state.recording = false;
    if (state.wizard) renderWizard();
  }
}

function addUploadedTakes(files) {
  if (!state.wizard) return;
  const list = Array.from(files || []);
  let slot = 0;
  for (const file of list) {
    while (slot < 5 && state.wizard.takes[slot]) slot += 1;
    if (slot >= 5) break;
    state.wizard.takes[slot] = {
      blob: file,
      url: URL.createObjectURL(file),
      duration: 0,
      sampleRate: 0,
      uploaded: true,
    };
    slot += 1;
  }
  renderWizard();
}

// -------------------------------------------------------------------- wizard
const WIZARD_STEPS = {
  new: ["Identity", "Keyword", "Voice", "Review"],
  keyword: ["Keyword", "Voice", "Review"],
  edit: ["Identity"],
  "keyword-create": ["Keyword"],
  "keyword-edit": ["Keyword"],
};

function openWizard({ mode = "new", person = null } = {}) {
  const actions = person?.action || { type: "announce", enabled: true, message_template: "{name} said {keyword}" };
  state.wizard = {
    mode,
    person,
    step: 0,
    identity: {
      name: person?.name || "",
      color: person?.color || PALETTE[(state.data.people || []).length % PALETTE.length],
      action: { ...actions },
      trigger_on_any_speech: Boolean(person?.trigger_on_any_speech),
    },
    keyword: { id: null, label: "" },
    takes: new Array(5).fill(null),
  };
  state.saving = false;
  state.settingsDirty = false;
  $("#modal").hidden = false;
  renderWizard();
  setTimeout(() => $("#wiz-name")?.focus(), 50);
}

function closeModal() {
  if (state.activeRecorder) {
    try { state.activeRecorder.stop(); } catch (_) { /* noop */ }
    state.activeRecorder = null;
  }
  state.recording = false;
  state.wizard = null;
  state.saving = false;
  $("#modal").hidden = true;
  $("#modal-body").innerHTML = "";
}

function openKeywordEditor(keyword = null) {
  state.wizard = {
    mode: keyword ? "keyword-edit" : "keyword-create",
    keyword: {
      id: keyword?.id || null,
      label: keyword?.label || "",
      threshold: keyword?.threshold ?? null,
    },
    step: 0,
  };
  state.saving = false;
  $("#modal").hidden = false;
  renderWizard();
  setTimeout(() => $("#wiz-keyword")?.focus(), 50);
}

function renderWizard() {
  const wizard = state.wizard;
  if (!wizard) return;
  const steps = WIZARD_STEPS[wizard.mode] || WIZARD_STEPS.new;
  const titles = {
    new: ["Enroll a person", "Create a voice profile"],
    keyword: ["Add a keyword", wizard.person?.name || "Existing person"],
    edit: ["Edit person", wizard.person?.name || ""],
    "keyword-create": ["New keyword", "No owner yet"],
    "keyword-edit": ["Edit keyword", wizard.keyword?.label || ""],
  };
  $("#modal-eyebrow").textContent = titles[wizard.mode]?.[1] || "Enrollment";
  $("#modal-title").textContent = titles[wizard.mode]?.[0] || "Enrollment";
  $("#modal-steps").innerHTML = steps
    .map((_, index) => `<span class="step-dot ${index < wizard.step ? "is-done" : ""} ${index === wizard.step ? "is-current" : ""}"></span>`)
    .join("");

  if (wizard.mode === "keyword-create" || wizard.mode === "keyword-edit") {
    $("#modal-body").innerHTML = keywordEditorTemplate(wizard);
  } else if (wizard.step === 0) {
    $("#modal-body").innerHTML = identityTemplate(wizard);
  } else if (wizard.step === 1) {
    $("#modal-body").innerHTML = keywordStepTemplate(wizard);
  } else if (wizard.step === 2) {
    $("#modal-body").innerHTML = voiceStepTemplate(wizard);
  } else {
    $("#modal-body").innerHTML = reviewTemplate(wizard);
  }
  renderWizardFooter(wizard, steps);
  wireWizard(wizard);
}

function renderWizardFooter(wizard, steps) {
  const isLast = wizard.step >= steps.length - 1;
  const saveLabel = wizard.mode === "edit" ? "Save changes"
    : wizard.mode === "keyword-create" ? "Create keyword"
    : wizard.mode === "keyword-edit" ? "Save keyword"
    : isLast ? (wizard.mode === "keyword" ? "Add keyword" : "Enroll person")
    : "Continue";
  const nextDisabled = state.saving ? "disabled" : "";
  $("#modal-footer").innerHTML = `
    <button type="button" class="btn ghost" data-modal-cancel>Cancel</button>
    <div class="spacer"></div>
    ${wizard.step > 0 ? '<button type="button" class="btn ghost" data-modal-back>Back</button>' : ""}
    <button type="button" class="btn primary" data-modal-next ${nextDisabled}>${state.saving ? "Saving…" : saveLabel}</button>
  `;
}

function identityTemplate(wizard) {
  const identity = wizard.identity;
  const action = identity.action;
  return `
    <div class="form-grid">
      <div class="field">
        <label class="input-label" for="wiz-name">Display name</label>
        <input id="wiz-name" class="input" maxlength="40" value="${esc(identity.name)}" placeholder="Alice" autocomplete="off" />
      </div>
      <div class="field">
        <span class="input-label">Color</span>
        <div class="swatches">
          ${PALETTE.map((color) => `<button type="button" class="swatch ${identity.color === color ? "is-selected" : ""}" data-color="${color}" style="--swatch:${color}" aria-label="Choose color ${color}"></button>`).join("")}
        </div>
      </div>
      <div class="field">
        <label class="input-label" for="wiz-action">Action</label>
        <select id="wiz-action" class="select">
          <option value="announce" ${action.type === "announce" ? "selected" : ""}>Announce (browser TTS + name)</option>
          <option value="log" ${action.type === "log" ? "selected" : ""}>Log only</option>
          <option value="webhook" ${action.type === "webhook" ? "selected" : ""}>Webhook POST</option>
          <option value="shell" ${action.type === "shell" ? "selected" : ""}>Shell (allowlisted)</option>
        </select>
      </div>
      <div class="field">
        <label class="input-label" for="wiz-template">Message template</label>
        <input id="wiz-template" class="input" maxlength="250" value="${esc(action.message_template || "{name} said {keyword}")}" />
      </div>
      ${action.type === "webhook" ? `
        <div class="field full">
          <label class="input-label" for="wiz-webhook">Webhook URL</label>
          <input id="wiz-webhook" class="input" value="${esc(action.webhook_url || "")}" placeholder="https://example.test/hook" />
        </div>` : ""}
      ${action.type === "shell" ? `
        <div class="field full">
          <label class="input-label" for="wiz-shell">Command</label>
          <input id="wiz-shell" class="input" value="${esc(action.shell_command || "")}" placeholder="notify-send '{name} said {keyword}'" />
          <p class="hint">Only notify-send, paplay, and aplay are allowed. Actions can be disabled in Settings.</p>
        </div>` : ""}
      <label class="switch-row full">
        <span><strong>Trigger on any speech</strong><small>Also run this action when arbitrary speech is identified, not only for a keyword.</small></span>
        <input id="wiz-any-speech" type="checkbox" class="switch" ${identity.trigger_on_any_speech ? "checked" : ""} />
      </label>
    </div>
    <div class="notice info">The keyword must be spoken alone. Continuous phrases are treated as arbitrary speech by design.</div>
  `;
}

function keywordEditorTemplate(wizard) {
  const keyword = wizard.keyword || {};
  return `
    <div class="form-grid">
      <div class="field full">
        <label class="input-label" for="wiz-keyword">Keyword label</label>
        <input id="wiz-keyword" class="input" maxlength="40" value="${esc(keyword.label || "")}" placeholder="stop" autocomplete="off" />
        <p class="hint">Lowercase is enforced. Existing labels are reused case-insensitively.</p>
      </div>
      <div class="field full">
        <label class="input-label" for="wiz-threshold">Per-keyword threshold (optional)</label>
        <input id="wiz-threshold" class="input" type="number" min="0" max="1" step="0.01" value="${keyword.threshold ?? ""}" placeholder="Use global (0.45)" />
        <p class="hint">Leave blank to use the global keyword threshold from Settings.</p>
      </div>
    </div>
  `;
}

function keywordStepTemplate(wizard) {
  const keywords = state.data.keywords || [];
  const selected = wizard.keyword.id;
  const custom = selected ? "" : (wizard.keyword.label || "");
  const warning = custom && !RECOMMENDED.includes(custom.trim().toLowerCase());
  return `
    <div class="field">
      <span class="input-label">Choose an existing keyword</span>
      ${keywords.length ? `<div class="keyword-options">
        ${keywords.map((keyword) => `
          <button type="button" class="keyword-option ${selected === keyword.id ? "is-selected" : ""}" data-keyword-id="${esc(keyword.id)}" data-keyword-label="${esc(keyword.label)}">
            <strong>${esc(keyword.label)}</strong>
            <span>${keyword.n_people || 0} owner${keyword.n_people === 1 ? "" : "s"}${keyword.n_samples ? ` · ${keyword.n_samples} samples` : ""}</span>
          </button>`).join("")}
      </div>` : '<p class="empty-inline">No keywords yet. Create one below.</p>'}
    </div>
    <div class="field">
      <label class="input-label" for="wiz-keyword">Or use a new keyword</label>
      <input id="wiz-keyword" class="input" maxlength="40" value="${esc(custom)}" placeholder="stop" autocomplete="off" />
      <div class="chip-row">
        ${RECOMMENDED.map((word) => `<button type="button" class="chip-btn ${custom.toLowerCase() === word ? "is-selected" : ""}" data-recommended="${word}">${word}</button>`).join("")}
      </div>
    </div>
    <div id="keyword-warning" class="notice info">
      ${warning
        ? `Custom keyword “<strong>${esc(custom.trim())}</strong>” ready. DISENT-KWS phonetic encoder handles any word or phrase.`
        : "Assistant keyword selected. Say it as a short, isolated utterance."}
    </div>
  `;
}

function voiceStepTemplate(wizard) {
  const ready = wizard.takes.filter(Boolean).length;
  const micAvailable = Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  return `
    <div class="notice ${micAvailable ? "info" : ""}">
      ${micAvailable
        ? "Record 5 short takes (2 seconds each). 3 are required; 5 gives the best profile."
        : "No microphone API here. Use the instant AI generator below, or upload WAV files."}
    </div>
    <div style="margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
      <button type="button" class="auto-gen-btn" id="btn-auto-gen-takes">
        <svg class="icon" viewBox="0 0 24 24"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3L12 3Z"/></svg>
        <span>⚡ Auto-Generate Takes (AI Voice)</span>
      </button>
      <span class="muted small">Zero mic effort · creates 5 diverse takes</span>
    </div>
    <div class="takes">
      ${wizard.takes.map((take, index) => `
        <div class="take ${take ? "is-ready" : ""}">
          <div class="take-index">${index + 1}</div>
          <div class="take-info">
            <strong>${take ? "Ready" : "Not recorded"}</strong>
            <span id="take-status-${index}">${take ? (take.uploaded ? "Ready (WAV)" : `${take.duration.toFixed(2)}s · ${take.sampleRate} Hz`) : "Record a 2 s take"}</span>
            <div class="take-level"><span id="take-level-${index}"></span></div>
          </div>
          <div class="take-actions">
            ${take ? `<button type="button" class="btn tiny ghost" data-take-play="${index}">Play</button>` : ""}
            <button type="button" class="btn tiny ${take ? "ghost" : "primary"}" data-take-record="${index}" ${state.recording ? "disabled" : ""}>${take ? "Retake" : "Record"}</button>
          </div>
        </div>`).join("")}
    </div>
    <div class="field">
      <label class="input-label" for="take-upload">Upload WAV takes (fallback)</label>
      <input id="take-upload" class="input" type="file" accept=".wav,audio/wav,audio/x-wav" multiple />
    </div>
    <p class="hint">${ready} of 5 takes ready · ${Math.min(ready, 3)} of 3 required</p>
  `;
}

function reviewTemplate(wizard) {
  const identity = wizard.identity || {};
  const keyword = wizard.keyword || {};
  const ready = wizard.takes.filter(Boolean).length;
  const personLabel = wizard.mode === "keyword" ? wizard.person?.name : identity.name;
  return `
    <div class="review">
      <div class="review-card">
        <div class="review-row"><span>Person</span><span>${esc(personLabel || "—")}</span></div>
        <div class="review-row"><span>Keyword</span><span>${esc(keyword.label || keyword.id || "—")}</span></div>
        <div class="review-row"><span>Takes</span><span>${ready}</span></div>
        <div class="review-row"><span>Action</span><span>${esc(identity.action?.type || "—")}</span></div>
        <div class="review-row"><span>Message</span><span>${esc(identity.action?.message_template || "{name} said {keyword}")}</span></div>
      </div>
      <div class="notice info">The speaker profile and keyword centroid are built now. Quality is reported after enrollment; re-record if the takes were noisy.</div>
    </div>
  `;
}

function wireWizard(wizard) {
  const body = $("#modal-body");
  body.querySelectorAll("[data-color]").forEach((button) => {
    button.addEventListener("click", () => {
      wizard.identity.color = button.dataset.color;
      renderWizard();
    });
  });
  const bindInput = (selector, callback) => {
    const node = $(selector);
    if (!node) return;
    node.addEventListener("input", () => callback(node.value));
    node.addEventListener("change", () => callback(node.value));
  };
  bindInput("#wiz-name", (value) => { wizard.identity.name = value; });
  bindInput("#wiz-template", (value) => { wizard.identity.action.message_template = value; });
  bindInput("#wiz-webhook", (value) => { wizard.identity.action.webhook_url = value; });
  bindInput("#wiz-shell", (value) => { wizard.identity.action.shell_command = value; });
  bindInput("#wiz-threshold", (value) => {
    wizard.keyword.threshold = value === "" ? null : Number(value);
  });

  const actionSelect = $("#wiz-action");
  if (actionSelect) {
    actionSelect.addEventListener("change", () => {
      wizard.identity.action.type = actionSelect.value;
      renderWizard();
    });
  }
  const anySpeech = $("#wiz-any-speech");
  if (anySpeech) {
    anySpeech.addEventListener("change", () => {
      wizard.identity.trigger_on_any_speech = anySpeech.checked;
    });
  }

  const keywordInput = $("#wiz-keyword");
  if (keywordInput && wizard.mode !== "keyword-create" && wizard.mode !== "keyword-edit") {
    keywordInput.addEventListener("input", () => {
      wizard.keyword = { id: null, label: keywordInput.value };
      body.querySelectorAll(".keyword-option").forEach((node) => node.classList.remove("is-selected"));
      updateKeywordWarning(keywordInput.value);
      updateRecommendedChips(keywordInput.value);
    });
  }
  body.querySelectorAll("[data-keyword-id]").forEach((button) => {
    button.addEventListener("click", () => {
      wizard.keyword = { id: button.dataset.keywordId, label: button.dataset.keywordLabel };
      body.querySelectorAll(".keyword-option").forEach((node) => node.classList.remove("is-selected"));
      button.classList.add("is-selected");
      if (keywordInput) keywordInput.value = "";
    });
  });
  body.querySelectorAll("[data-recommended]").forEach((button) => {
    button.addEventListener("click", () => {
      const word = button.dataset.recommended;
      wizard.keyword = { id: null, label: word };
      if (keywordInput) keywordInput.value = word;
      body.querySelectorAll(".keyword-option").forEach((node) => node.classList.remove("is-selected"));
      body.querySelectorAll("[data-recommended]").forEach((node) => node.classList.remove("is-selected"));
      button.classList.add("is-selected");
      updateKeywordWarning(word);
    });
  });

  body.querySelectorAll("[data-take-record]").forEach((button) => {
    button.addEventListener("click", () => recordTake(Number(button.dataset.takeRecord)));
  });
  body.querySelectorAll("[data-take-play]").forEach((button) => {
    button.addEventListener("click", () => {
      const take = wizard.takes[Number(button.dataset.takePlay)];
      if (take) {
        const url = take.url || (take.blob ? URL.createObjectURL(take.blob) : null);
        if (url) new Audio(url).play().catch(() => toast("Could not play this take", "error"));
      }
    });
  });
  const upload = $("#take-upload");
  if (upload) {
    upload.addEventListener("change", () => addUploadedTakes(upload.files));
  }

  const autoGenBtn = $("#btn-auto-gen-takes");
  if (autoGenBtn) {
    autoGenBtn.addEventListener("click", async () => {
      const kw = (wizard.keyword.label || wizard.keyword.id || "Hey Google").trim();
      autoGenBtn.disabled = true;
      autoGenBtn.innerHTML = `<span>⏳ Synthesizing 5 takes for “${esc(kw)}”...</span>`;
      try {
        const res = await api("/api/keywords/synthesize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: kw, count: 5 }),
        });
        if (res && res.takes && res.takes.length) {
          for (let i = 0; i < res.takes.length && i < 5; i++) {
            const take = res.takes[i];
            const blobRes = await fetch(take.data_url);
            const blob = await blobRes.blob();
            wizard.takes[i] = {
              blob,
              url: URL.createObjectURL(blob),
              duration: take.duration,
              sampleRate: take.sample_rate || 16000,
              uploaded: true,
            };
          }
          toast(`Generated 5 takes for “${kw}”!`, "success");
          renderWizard();
        }
      } catch (err) {
        toast(`Synthesis failed: ${err.message}`, "error");
        autoGenBtn.disabled = false;
        autoGenBtn.innerHTML = `<span>⚡ Auto-Generate Takes (AI Voice)</span>`;
      }
    });
  }
}

function updateKeywordWarning(value) {
  const warning = $("#keyword-warning");
  if (!warning) return;
  const kw = value.trim();
  const experimental = kw && !RECOMMENDED.includes(kw.toLowerCase());
  warning.className = "notice info";
  warning.innerHTML = experimental
    ? `Custom keyword “<strong>${esc(kw)}</strong>” ready. DISENT-KWS phonetic encoder handles any phrase.`
    : "Assistant keyword selected. Say it as a short, isolated utterance.";
}

function updateRecommendedChips(value) {
  const normalized = value.trim().toLowerCase();
  $$("[data-recommended]").forEach((node) => node.classList.toggle("is-selected", node.dataset.recommended === normalized));
}

function wizardNext() {
  const wizard = state.wizard;
  if (!wizard || state.saving) return;
  if (wizard.mode === "keyword-create" || wizard.mode === "keyword-edit") {
    const label = ($("#wiz-keyword")?.value || "").trim();
    if (!label) return toast("Enter a keyword label", "error");
    wizard.keyword.label = label;
    const thresholdRaw = $("#wiz-threshold")?.value ?? "";
    wizard.keyword.threshold = thresholdRaw === "" ? null : Number(thresholdRaw);
    saveWizard();
    return;
  }
  if (wizard.step === 0) {
    const name = ($("#wiz-name")?.value || "").trim();
    if (!name) return toast("Enter a display name", "error");
    wizard.identity.name = name;
    const action = wizard.identity.action;
    if (action.type === "webhook" && !/^https?:\/\//.test(action.webhook_url || "")) {
      return toast("Webhook actions need an http:// or https:// URL", "error");
    }
    if (action.type === "shell" && !(action.shell_command || "").trim()) {
      return toast("Shell actions need a command", "error");
    }
  } else if (wizard.step === 1) {
    const label = (wizard.keyword.label || "").trim();
    if (!wizard.keyword.id && !label) return toast("Choose or type a keyword", "error");
    wizard.keyword.label = label;
  } else if (wizard.step === 2) {
    const ready = wizard.takes.filter(Boolean).length;
    if (ready < 3) return toast("Record or upload at least 3 takes", "error");
  }
  const steps = WIZARD_STEPS[wizard.mode] || WIZARD_STEPS.new;
  if (wizard.step >= steps.length - 1) {
    saveWizard();
  } else {
    wizard.step += 1;
    renderWizard();
    $("#modal-body").scrollTop = 0;
  }
}

async function saveWizard() {
  const wizard = state.wizard;
  if (!wizard || state.saving) return;
  state.saving = true;
  renderWizardFooter(wizard, WIZARD_STEPS[wizard.mode] || WIZARD_STEPS.new);
  try {
    if (wizard.mode === "edit") {
      await api(`/api/people/${wizard.person.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: wizard.identity.name,
          color: wizard.identity.color,
          action: { ...wizard.identity.action, enabled: true },
          trigger_on_any_speech: wizard.identity.trigger_on_any_speech,
        }),
      });
      toast("Person updated", "success");
    } else if (wizard.mode === "keyword-create" || wizard.mode === "keyword-edit") {
      const payload = {
        label: wizard.keyword.label,
        threshold: wizard.keyword.threshold,
      };
      if (wizard.mode === "keyword-create") {
        await api("/api/keywords", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        toast(`Keyword “${wizard.keyword.label}” created`, "success");
      } else {
        await api(`/api/keywords/${wizard.keyword.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        toast("Keyword updated", "success");
      }
    } else {
      const form = new FormData();
      form.append("action_json", JSON.stringify({ ...wizard.identity.action, enabled: true }));
      form.append("trigger_on_any_speech", String(wizard.identity.trigger_on_any_speech));
      if (wizard.keyword.id) form.append("keyword_id", wizard.keyword.id);
      else form.append("keyword_label", wizard.keyword.label);
      wizard.takes.filter(Boolean).forEach((take, index) => {
        form.append("files", take.blob, `take_${index + 1}.wav`);
      });
      if (wizard.mode === "new") {
        form.append("name", wizard.identity.name);
        form.append("color", wizard.identity.color);
        const result = await api("/api/enroll", { method: "POST", body: form });
        const quality = result.quality || {};
        const qualityText = quality.n >= 2 ? ` · voice consistency ${(quality.min ?? 0).toFixed(2)}` : "";
        toast(`${result.person.name} enrolled with “${result.keyword.label}”${qualityText}`, "success");
        if (result.warnings?.length) toast(result.warnings[0], "error");
      } else {
        form.append("keyword_label", wizard.keyword.label);
        const result = await api(`/api/people/${wizard.person.id}/keywords`, { method: "POST", body: form });
        toast(`Keyword “${result.keyword.label}” added to ${result.person.name}`, "success");
      }
    }
    closeModal();
    await loadState();
  } catch (error) {
    toast(error.message, "error");
    state.saving = false;
    if (state.wizard) renderWizardFooter(state.wizard, WIZARD_STEPS[state.wizard.mode] || WIZARD_STEPS.new);
  }
}

// -------------------------------------------------------------------- events
function handleEvent(event) {
  if (!event) return;
  if (event.id && state.events.some((item) => item.id === event.id)) return;
  state.lastEvent = event;
  state.events = mergeEvents([event], state.events);
  state.orbLockUntil = Date.now() + 2600;
  renderResult(event);
  renderTimeline();
  renderLivePeople();
  handleSmartAssistantAction(event);
  if (event.latency_ms?.total != null) {
    $("#latency-mini").textContent = `${Math.round(event.latency_ms.total)} ms`;
  }
  if (event.speak && state.data.settings.announce !== false) speak(event.speak);

  const orbKind = event.type === "keyword" || event.type === "speech" ? "hit"
    : event.type === "keyword_unknown_speaker" ? "unknown" : "idle";
  setOrb(orbKind, event.person_color || null);

  const status = event.type === "keyword"
    ? `${esc(event.person_name || "Someone")} said “${esc(event.keyword || "")}”`
    : event.type === "speech"
      ? `That was ${esc(event.person_name || "someone enrolled")}`
      : event.type === "keyword_unknown_speaker"
        ? "Keyword heard — unknown speaker"
        : event.type === "too_short"
          ? "Utterance too short"
          : "No confident match";
  const sub = event.type === "keyword"
    ? `Keyword similarity ${fmtScore(event.keyword_score)} · speaker ${fmtScore(event.speaker_score)}`
    : event.type === "speech"
      ? `Arbitrary speech · speaker similarity ${fmtScore(event.speaker_score)}`
      : event.type === "keyword_unknown_speaker"
        ? "No action triggered. Enroll this voice to use the keyword."
        : "Try again a little closer to the microphone.";
  setStageStatus(status, sub);
  $("#stage-status").setAttribute("aria-live", "polite");
}

function renderResult(event) {
  $("#result-empty").hidden = true;
  $("#result-content").hidden = false;
  const person = event.person_name || (event.type === "keyword_unknown_speaker" ? "Unknown speaker" : "Unrecognized voice");
  const color = event.person_color || TYPE_COLOR[event.type] || "#8b5cf6";
  const avatar = $("#result-avatar");
  avatar.textContent = event.person_name ? initials(event.person_name) : "?";
  avatar.style.setProperty("--avatar-color", color);
  $("#result-name").textContent = person;
  $("#result-kicker").textContent = TYPE_LABEL[event.type] || "Detection";
  $("#result-time").textContent = relativeTime(event.ts);

  const hasKeyword = Boolean(event.keyword);
  $("#result-keyword-row").style.display = hasKeyword ? "flex" : "none";
  if (hasKeyword) $("#result-keyword").textContent = event.keyword;

  const keywordScore = Number(event.keyword_score || 0);
  const speakerScore = Number(event.speaker_score || 0);
  const speakerThreshold = Number(event.speaker_threshold || 0);
  const keywordThreshold = Number(state.data.settings.keyword_threshold || 0.45);
  $("#keyword-score-label").textContent = fmtScore(keywordScore);
  $("#speaker-score-label").textContent = fmtScore(speakerScore);
  $("#keyword-fill").style.width = `${clamp(keywordScore * 100, 0, 100)}%`;
  $("#speaker-fill").style.width = `${clamp(speakerScore * 100, 0, 100)}%`;
  $("#keyword-tick").style.left = `${clamp(keywordThreshold * 100, 0, 100)}%`;
  $("#speaker-tick").style.left = `${clamp(speakerThreshold * 100, 0, 100)}%`;
  $("#result-latency").textContent = fmtMs(event.latency_ms?.total || 0);
  $("#result-duration").textContent = fmtSeconds(event.duration_sec || 0);
  $("#result-margin").textContent = `margin ${fmtScore(event.margin)}`;

  const action = event.action;
  let actionText = "No action";
  if (action) {
    if (action.status === "ok") actionText = action.type === "announce" ? `Spoke: “${action.text || ""}”` : `Action: ${action.type}`;
    else if (action.status === "cooldown") actionText = "Action suppressed by cooldown";
    else if (action.status === "disabled") actionText = "Actions disabled";
    else actionText = `Action ${action.type || ""} ${action.status || "error"}`;
  }
  $("#result-action").textContent = actionText;

  let note = "";
  if (event.type === "keyword_unknown_speaker") {
    note = "The keyword matched, but the voice was not confidently matched to one of its owners.";
  } else if (event.type === "unrecognized") {
    note = "Speaker similarity stayed below the duration-adaptive threshold, or the runner-up was too close.";
  } else if (event.type === "too_short") {
    note = "The VAD segment was shorter than the minimum speech duration.";
  } else if (event.type === "speech") {
    note = "Arbitrary speech identified. This person's action only runs if “Trigger on any speech” is enabled.";
  } else if (event.confidence != null) {
    note = `Similarity confidence ${(Number(event.confidence) * 100).toFixed(0)}% — this is similarity, not a probability.`;
  }
  $("#result-note").textContent = note;

  $$(".person-chip").forEach((chip) => chip.classList.remove("is-active"));
  const chip = $(`.person-chip[data-person-id="${CSS.escape(event.person_id || "")}"]`);
  if (chip) {
    chip.classList.add("is-active");
    clearTimeout(state.activePersonTimer);
    state.activePersonTimer = setTimeout(() => chip.classList.remove("is-active"), 8000);
  }
}

function renderTimeline() {
  const list = $("#timeline");
  $("#event-count").textContent = `${state.events.length} event${state.events.length === 1 ? "" : "s"}`;
  if (!state.events.length) {
    list.innerHTML = '<p class="timeline-empty">No utterances yet. Start listening or drop a WAV on the stage.</p>';
    return;
  }
  list.innerHTML = state.events.slice(0, 40).map((event) => {
    const color = event.person_color || TYPE_COLOR[event.type] || "#8b5cf6";
    const name = event.person_name || (event.type === "keyword_unknown_speaker" ? "Unknown speaker" : "No match");
    const keyword = event.keyword ? ` · “${esc(event.keyword)}”` : "";
    const action = event.action?.status === "ok" ? " · action ok" : event.action?.status === "cooldown" ? " · cooldown" : "";
    return `
      <div class="timeline-row">
        <span class="timeline-dot" style="--dot:${color}"></span>
        <div class="timeline-main">
          <strong>${esc(name)}</strong>
          <p>${esc(TYPE_LABEL[event.type] || event.type)}${keyword}${action} · ${relativeTime(event.ts)}</p>
        </div>
        <div class="timeline-scores">
          <span>kw ${fmtScore(event.keyword_score)}</span>
          <span>spk ${fmtScore(event.speaker_score)}</span>
        </div>
      </div>`;
  }).join("");
}

function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  if (state.data.settings.announce === false) return;
  try {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.02;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  } catch (_) { /* TTS is best-effort */ }
}

// ------------------------------------------------------------------ rendering
function renderAll() {
  renderPeople();
  renderKeywords();
  renderLivePeople();
  renderTimeline();
  renderSettings();
  renderHealth();
}

function renderLivePeople() {
  const container = $("#live-people");
  const people = state.data.people || [];
  if (!people.length) {
    container.innerHTML = '<p class="empty-inline">No people enrolled yet — add your first person to begin.</p>';
    return;
  }
  container.innerHTML = people.map((person) => {
    const labels = (person.keyword_labels || []).join(" · ") || "no keyword";
    return `
      <div class="person-chip" data-person-id="${esc(person.id)}" style="--avatar-color:${esc(person.color)}">
        <span class="avatar" style="--avatar-color:${esc(person.color)}">${esc(person.avatar || initials(person.name))}</span>
        <span class="chip-copy"><strong>${esc(person.name)}</strong><span>${esc(labels)}</span></span>
      </div>`;
  }).join("");
}

function renderPeople() {
  const grid = $("#people-grid");
  const people = state.data.people || [];
  if (!people.length) {
    grid.innerHTML = `
      <div class="empty-state">
        <h3>No people yet</h3>
        <p>Enroll a voice profile with 3–5 short takes, or load the demo people.</p>
        <button class="btn primary" type="button" data-empty-add>Add your first person</button>
      </div>`;
    grid.querySelector("[data-empty-add]")?.addEventListener("click", () => openWizard({ mode: "new" }));
    return;
  }
  grid.innerHTML = people.map((person) => {
    const labels = person.keyword_labels || [];
    const action = person.action || {};
    return `
      <article class="person-card glass">
        <div class="person-card-head">
          <span class="avatar" style="--avatar-color:${esc(person.color)}">${esc(person.avatar || initials(person.name))}</span>
          <div class="person-card-title">
            <h3>${esc(person.name)}</h3>
            <p>${labels.length ? "Voice profile ready" : "No keyword yet"}</p>
          </div>
        </div>
        <div class="chips">
          ${labels.length ? labels.map((label) => `<span class="keyword-chip">${esc(label)}</span>`).join("") : '<span class="tag warn">Needs keyword</span>'}
          <span class="tag ${person.trigger_on_any_speech ? "good" : ""}">${person.trigger_on_any_speech ? "any speech" : "keyword only"}</span>
        </div>
        <p class="card-note">Action: <strong>${esc(action.type || "announce")}</strong>${action.message_template ? ` — ${esc(action.message_template)}` : ""}</p>
        <div class="card-actions">
          <button class="btn tiny ghost" type="button" data-person-add-keyword="${esc(person.id)}">Add keyword</button>
          <button class="btn tiny ghost" type="button" data-person-edit="${esc(person.id)}">Edit</button>
          <button class="btn tiny danger" type="button" data-person-delete="${esc(person.id)}">Delete</button>
        </div>
      </article>`;
  }).join("");
}

function renderKeywords() {
  const grid = $("#keywords-grid");
  const keywords = state.data.keywords || [];
  if (!keywords.length) {
    grid.innerHTML = `
      <div class="empty-state">
        <h3>No keywords yet</h3>
        <p>Create a keyword, then attach owners by enrolling them. Shared keywords are allowed.</p>
        <button class="btn primary" type="button" data-empty-keyword>Create a keyword</button>
      </div>`;
    grid.querySelector("[data-empty-keyword]")?.addEventListener("click", () => openKeywordEditor(null));
    return;
  }
  grid.innerHTML = keywords.map((keyword) => {
    const owners = keyword.owners || [];
    const recommended = keyword.recommended || RECOMMENDED.includes(keyword.label);
    return `
      <article class="keyword-card glass">
        <div class="section-head">
          <div>
            <span class="eyebrow">Keyword</span>
            <h3 style="font-family:var(--mono);font-size:24px;margin-top:4px">${esc(keyword.label)}</h3>
          </div>
          <span class="tag ${recommended ? "good" : "warn"}">${recommended ? "validated" : "experimental"}</span>
        </div>
        <div class="card-stat">
          <div>
            <span>Owners</span>
            <div class="chip-row">
              ${owners.length ? owners.map((owner) => `<span class="keyword-chip">${esc(owner.name)}</span>`).join("") : '<span class="tag warn">unassigned</span>'}
            </div>
          </div>
          <div class="owner-stack">
            ${owners.slice(0, 5).map((owner) => `<span class="avatar" style="--avatar-color:${esc(owner.color)}" title="${esc(owner.name)}">${esc(owner.avatar || initials(owner.name))}</span>`).join("")}
          </div>
        </div>
        <div class="card-stat">
          <span>Threshold</span>
          <span class="pill">${keyword.threshold == null ? "global" : Number(keyword.threshold).toFixed(2)}</span>
          <span>${keyword.n_samples || 0} samples</span>
        </div>
        <div class="card-actions">
          <button class="btn tiny ghost" type="button" data-keyword-test="${esc(keyword.id)}">Test with WAV</button>
          <button class="btn tiny ghost" type="button" data-keyword-edit="${esc(keyword.id)}">Edit</button>
          <button class="btn tiny danger" type="button" data-keyword-delete="${esc(keyword.id)}">Delete</button>
        </div>
      </article>`;
  }).join("");
}

function renderSettings() {
  const settings = state.data.settings || DEFAULT_SETTINGS;
  const outputs = {
    keyword_threshold: (value) => Number(value).toFixed(2),
    keyword_margin: (value) => Number(value).toFixed(2),
    speaker_margin: (value) => Number(value).toFixed(2),
    speaker_threshold_scale: (value) => Number(value).toFixed(2),
    vad_threshold: (value) => Number(value).toFixed(2),
    min_speech_ms: (value) => `${Math.round(value)} ms`,
    max_utterance_sec: (value) => `${Number(value).toFixed(1)} s`,
    max_keyword_utterance_sec: (value) => `${Number(value).toFixed(1)} s`,
    cooldown_sec: (value) => `${Number(value).toFixed(1)} s`,
  };
  $$("[data-setting]").forEach((input) => {
    const key = input.dataset.setting;
    if (input.type === "checkbox") {
      input.checked = Boolean(settings[key]);
    } else if (!state.settingsDirty) {
      input.value = settings[key];
    }
    const output = outputFor(key);
    if (output && outputs[key]) output.textContent = outputs[key](settings[key]);
  });
  $$("[data-setting]").forEach((input) => {
    const key = input.dataset.setting;
    const output = outputFor(key);
    if (output && outputs[key]) output.textContent = outputs[key](input.type === "checkbox" ? input.checked : input.value);
  });
  const health = $("#health-panel");
  if (state.health) {
    const models = state.health.models || {};
    health.innerHTML = `
      <div class="health-row"><span>Server</span><span>${esc(state.health.status)}</span></div>
      <div class="health-row"><span>DISENT</span><span>${esc(models.disent?.state || "—")}</span></div>
      <div class="health-row"><span>ECAPA</span><span>${esc(models.ecapa?.state || "—")}</span></div>
      <div class="health-row"><span>Device / threads</span><span>${esc(state.health.device || "cpu")} · ${esc(state.health.torch_threads ?? "—")}</span></div>
      <div class="health-row"><span>Compute p50 / p95</span><span>${state.health.latency_ms?.p50 ?? 0} / ${state.health.latency_ms?.p95 ?? 0} ms</span></div>
      <div class="health-row"><span>Uptime</span><span>${Math.round(state.health.uptime_sec || 0)} s</span></div>`;
  } else {
    health.innerHTML = '<div class="health-row"><span>Server</span><span>checking…</span></div>';
  }
}

function renderHealth() {
  if (!state.health) return;
  $("#latency-mini").textContent = `${state.health.latency_ms?.p50 ?? 0} ms p50`;
}

// ------------------------------------------------------------------ load/API
async function loadState() {
  const data = await api("/api/state");
  state.data = data;
  state.events = mergeEvents(data.events || [], state.events);
  renderAll();
}

async function loadHealth() {
  try {
    state.health = await api("/api/health");
    renderHealth();
    renderSettings();
  } catch (_) {
    $("#latency-mini").textContent = "offline";
  }
}

async function uploadWav(file) {
  if (!file) return;
  if (!/\.wav$/i.test(file.name || "") && file.type !== "audio/wav" && file.type !== "audio/x-wav") {
    toast("Please choose a .wav file (the server decodes WAV only).", "error");
    return;
  }
  setOrb("thinking");
  setStageStatus("Replaying WAV…", esc(file.name));
  const form = new FormData();
  form.append("file", file, file.name || "replay.wav");
  try {
    const event = await api("/api/replay", { method: "POST", body: form });
    handleEvent(event);
    toast("Replay complete", "success");
  } catch (error) {
    toast(error.message, "error");
    setOrb("idle");
    setStageStatus("Ready when you are", "Isolated words work best · try <strong>stop</strong>, <strong>yes</strong>, or <strong>no</strong>");
  }
}

async function testKeyword(keyword) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".wav,audio/wav,audio/x-wav";
  input.addEventListener("change", async () => {
    const file = input.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file, file.name || "test.wav");
    try {
      const result = await api("/api/identify", { method: "POST", body: form });
      const decision = result.decision || {};
      const who = decision.person_name || "no confident speaker";
      const heard = decision.keyword || "no keyword";
      toast(`“${keyword.label}” test: ${heard} · ${who} · kw ${fmtScore(decision.keyword_score)} / spk ${fmtScore(decision.speaker_score)}`, "success");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      input.remove();
    }
  });
  input.style.display = "none";
  document.body.appendChild(input);
  input.click();
}

async function deletePerson(person) {
  if (!confirm(`Delete ${person.name}? Their voice profile and keyword links will be removed.`)) return;
  try {
    await api(`/api/people/${person.id}`, { method: "DELETE" });
    toast(`${person.name} deleted`, "success");
    await loadState();
  } catch (error) {
    if (/no owners|force/i.test(error.message)) {
      if (confirm(`${error.message}\n\nDelete the orphaned keyword(s) too?`)) {
        try {
          await api(`/api/people/${person.id}?force=true`, { method: "DELETE" });
          toast(`${person.name} and orphaned keywords deleted`, "success");
          await loadState();
        } catch (secondError) {
          toast(secondError.message, "error");
        }
      }
    } else {
      toast(error.message, "error");
    }
  }
}

async function deleteKeyword(keyword) {
  if (!confirm(`Delete keyword “${keyword.label}”?`)) return;
  try {
    await api(`/api/keywords/${keyword.id}`, { method: "DELETE" });
    toast(`Keyword “${keyword.label}” deleted`, "success");
    await loadState();
  } catch (error) {
    if (/assigned|force/i.test(error.message)) {
      if (confirm(`${error.message}\n\nRemove it from all owners?`)) {
        try {
          await api(`/api/keywords/${keyword.id}?force=true`, { method: "DELETE" });
          toast(`Keyword “${keyword.label}” deleted`, "success");
          await loadState();
        } catch (secondError) {
          toast(secondError.message, "error");
        }
      }
    } else {
      toast(error.message, "error");
    }
  }
}

async function seedDemo() {
  const button = $("#seed-btn");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Seeding voices…";
  try {
    const result = await api("/api/demo/seed", { method: "POST" });
    const names = (result.seeded || []).map((entry) => entry.name).join(", ");
    toast(names ? `Demo people loaded: ${names}` : "Demo people were already loaded", "success");
    await loadState();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

// ---------------------------------------------------------------------- init
function bindUI() {
  $$(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".nav-item").forEach((node) => node.classList.toggle("is-active", node === button));
      $$(".view").forEach((view) => view.classList.toggle("is-active", view.id === `view-${button.dataset.view}`));
      if (button.dataset.view === "settings") loadHealth();
    });
  });

  $("#mic-btn").addEventListener("click", () => {
    if (state.listening) stopListening();
    else startListening();
  });
  $("#device-select").addEventListener("change", async () => {
    if (!state.listening) return;
    stopListening(false);
    await startListening();
  });
  $("#upload-btn").addEventListener("click", () => $("#wav-input").click());
  $("#wav-input").addEventListener("change", () => {
    const file = $("#wav-input").files?.[0];
    uploadWav(file);
    $("#wav-input").value = "";
  });

  $("#add-person-btn").addEventListener("click", () => openWizard({ mode: "new" }));
  $("#rail-add").addEventListener("click", () => openWizard({ mode: "new" }));
  $("#add-keyword-btn").addEventListener("click", () => openKeywordEditor(null));
  $("#seed-btn").addEventListener("click", seedDemo);

  // Smart Hub Device Controls
  $("#toggle-light-btn")?.addEventListener("click", () => {
    smartHome.light.on = !smartHome.light.on;
    playChime("click");
    renderSmartHub();
  });
  $("#toggle-music-btn")?.addEventListener("click", () => {
    smartHome.music.playing = !smartHome.music.playing;
    playChime("click");
    renderSmartHub();
  });
  $("#start-timer-btn")?.addEventListener("click", () => {
    startTimer(10, null);
  });
  $("#trigger-weather-btn")?.addEventListener("click", () => {
    playChime("google");
    if (state.data.settings.announce !== false) {
      speak("Good day, the current weather is 24 degrees and sunny.");
    }
  });

  $("#modal-close").addEventListener("click", closeModal);
  $("#modal").addEventListener("click", (event) => {
    if (event.target.dataset.close) closeModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("#modal").hidden) closeModal();
  });
  $("#modal-footer").addEventListener("click", (event) => {
    const wizard = state.wizard;
    if (!wizard) return;
    if (event.target.closest("[data-modal-cancel]")) closeModal();
    else if (event.target.closest("[data-modal-back]")) {
      wizard.step = Math.max(0, wizard.step - 1);
      renderWizard();
    } else if (event.target.closest("[data-modal-next]")) {
      wizardNext();
    }
  });

  $("#people-grid").addEventListener("click", (event) => {
    const addKeyword = event.target.closest("[data-person-add-keyword]");
    const edit = event.target.closest("[data-person-edit]");
    const remove = event.target.closest("[data-person-delete]");
    if (addKeyword) {
      const person = state.data.people.find((item) => item.id === addKeyword.dataset.personAddKeyword);
      if (person) openWizard({ mode: "keyword", person });
    } else if (edit) {
      const person = state.data.people.find((item) => item.id === edit.dataset.personEdit);
      if (person) openWizard({ mode: "edit", person });
    } else if (remove) {
      const person = state.data.people.find((item) => item.id === remove.dataset.personDelete);
      if (person) deletePerson(person);
    }
  });

  $("#keywords-grid").addEventListener("click", (event) => {
    const edit = event.target.closest("[data-keyword-edit]");
    const remove = event.target.closest("[data-keyword-delete]");
    const test = event.target.closest("[data-keyword-test]");
    if (edit) {
      const keyword = state.data.keywords.find((item) => item.id === edit.dataset.keywordEdit);
      if (keyword) openKeywordEditor(keyword);
    } else if (remove) {
      const keyword = state.data.keywords.find((item) => item.id === remove.dataset.keywordDelete);
      if (keyword) deleteKeyword(keyword);
    } else if (test) {
      const keyword = state.data.keywords.find((item) => item.id === test.dataset.keywordTest);
      if (keyword) testKeyword(keyword);
    }
  });

  $$("[data-setting]").forEach((input) => {
    input.addEventListener("input", () => {
      state.settingsDirty = true;
      const key = input.dataset.setting;
      const output = outputFor(key);
      if (output) {
        const value = input.type === "checkbox" ? input.checked : input.value;
        if (key.endsWith("_sec")) output.textContent = `${Number(value).toFixed(1)} s`;
        else if (key === "min_speech_ms") output.textContent = `${Math.round(value)} ms`;
        else if (key.endsWith("_threshold") || key.endsWith("_margin") || key.endsWith("_scale")) output.textContent = Number(value).toFixed(2);
      }
    });
  });
  $("#save-settings").addEventListener("click", saveSettings);
  $("#reset-settings").addEventListener("click", resetSettings);

  // Drag and drop anywhere on the page.
  let dragDepth = 0;
  window.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    $("#drop-overlay").hidden = false;
    $("#stage").classList.add("is-dragging");
  });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) {
      $("#drop-overlay").hidden = true;
      $("#stage").classList.remove("is-dragging");
    }
  });
  window.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    $("#drop-overlay").hidden = true;
    $("#stage").classList.remove("is-dragging");
    const file = Array.from(event.dataTransfer?.files || []).find((item) => /\.wav$/i.test(item.name));
    if (file) uploadWav(file);
    else toast("Drop a .wav file to replay it", "error");
  });
  window.addEventListener("beforeunload", () => {
    try { state.capture?.stop(); } catch (_) { /* noop */ }
    try { state.ws?.close(); } catch (_) { /* noop */ }
  });
}

function collectSettings() {
  const payload = {};
  $$("[data-setting]").forEach((input) => {
    const key = input.dataset.setting;
    if (input.type === "checkbox") payload[key] = input.checked;
    else if (key === "min_speech_ms") payload[key] = Math.round(Number(input.value));
    else payload[key] = Number(input.value);
  });
  return payload;
}

async function saveSettings() {
  const button = $("#save-settings");
  button.disabled = true;
  try {
    const settings = await api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectSettings()),
    });
    state.data.settings = settings;
    state.settingsDirty = false;
    renderSettings();
    toast("Settings saved", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function resetSettings() {
  state.data.settings = { ...DEFAULT_SETTINGS };
  state.settingsDirty = true;
  renderSettings();
  await saveSettings();
}

function drawWaveform() {
  const canvas = $("#waveform");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== Math.floor(rect.width * dpr) || canvas.height !== Math.floor(rect.height * dpr)) {
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const centerX = rect.width / 2;
  const centerY = rect.height / 2;
  const radius = Math.min(rect.width, rect.height) * 0.29;
  const levels = state.waveLevels;
  const count = levels.length;
  const gradient = ctx.createLinearGradient(0, 0, rect.width, 0);
  gradient.addColorStop(0, "rgba(34,211,238,0.05)");
  gradient.addColorStop(0.5, "rgba(167,139,250,0.45)");
  gradient.addColorStop(1, "rgba(34,211,238,0.05)");
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < count; i += 1) {
    const angle = (i / count) * Math.PI * 2 - Math.PI / 2;
    const amplitude = levels[i] * (state.listening ? 1 : 0.25);
    const idle = state.listening ? 0 : Math.sin(Date.now() / 900 + i * 0.35) * 0.008;
    const r = radius + amplitude * 58 + idle * 20;
    const x = centerX + Math.cos(angle) * r;
    const y = centerY + Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
  // Decay bars so the ring settles after speech.
  for (let i = 0; i < levels.length; i += 1) levels[i] *= 0.94;
  requestAnimationFrame(drawWaveform);
}

async function init() {
  bindUI();
  connectWS();
  drawWaveform();
  try {
    await loadState();
  } catch (error) {
    toast(`Could not load state: ${error.message}`, "error");
  }
  await loadHealth();
  populateDevices();
  renderSmartHub();
  setInterval(loadHealth, 15000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
