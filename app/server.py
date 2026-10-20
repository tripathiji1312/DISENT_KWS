"""FastAPI application: REST registry/enrollment/identify plus live WS audio."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.functional as AF
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.actions import ActionRunner
from app.audio import SAMPLE_RATE, read_wav_bytes, to_16k_mono
from app.demo import seed_demo
from app.enrollment import enroll_keyword_for_person, enroll_person
from app.inference import Decision, Engine, make_vad
from app.registry import (
    ActionConfig,
    Conflict,
    Invalid,
    Keyword,
    NotFound,
    Person,
    Registry,
    RegistryError,
    utc_now,
)
from app.settings import (
    DEMO_ASSETS_DIR,
    MODEL_PATH,
    STATIC_DIR,
    Settings,
    ensure_data_dirs,
)

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_BYTES = 40 * 1024 * 1024
MAX_AUDIO_SEC = 30.0
MAX_FILES = 8
MIN_KEYWORD_CLIPS = 3


# --------------------------------------------------------------------- schemas
class PersonCreate(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    color: str | None = None
    action: ActionConfig | None = None
    trigger_on_any_speech: bool = False


class PersonPatch(BaseModel):
    name: str | None = None
    color: str | None = None
    action: ActionConfig | None = None
    trigger_on_any_speech: bool | None = None
    avatar: str | None = None


class KeywordCreate(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class KeywordPatch(BaseModel):
    label: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SettingsPatch(BaseModel):
    keyword_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    keyword_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker_threshold_scale: float | None = Field(default=None, ge=0.5, le=2.0)
    vad_threshold: float | None = Field(default=None, ge=0.05, le=0.95)
    min_speech_ms: int | None = Field(default=None, ge=100, le=2000)
    max_utterance_sec: float | None = Field(default=None, ge=1.0, le=20.0)
    max_keyword_utterance_sec: float | None = Field(default=None, ge=0.5, le=10.0)
    cooldown_sec: float | None = Field(default=None, ge=0.0, le=30.0)
    announce: bool | None = None
    actions_enabled: bool | None = None


@dataclass(eq=False)
class LiveConnection:
    websocket: WebSocket
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    seq: int = 0

    async def send_json(self, payload: dict) -> None:
        async with self.lock:
            await self.websocket.send_json(payload)

    async def send_bytes(self, payload: bytes) -> None:
        async with self.lock:
            await self.websocket.send_bytes(payload)


# ------------------------------------------------------------------ app factory
def create_app(
    data_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    load_models: bool = True,
    assets_dir: Path | str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        data_root, _ = ensure_data_dirs(Path(data_dir) if data_dir is not None else None)
        app.state.data_dir = data_root
        app.state.registry = Registry(data_root)
        app.state.assets_dir = Path(assets_dir) if assets_dir is not None else DEMO_ASSETS_DIR
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        app.state.runner = ActionRunner()
        app.state.connections: set[LiveConnection] = set()
        app.state.latencies: deque[float] = deque(maxlen=200)
        app.state.started_at = time.time()
        app.state.seq = 0
        app.state.engine = None

        if load_models:
            checkpoint = Path(model_path) if model_path is not None else MODEL_PATH
            loop = asyncio.get_running_loop()
            try:
                app.state.engine = await loop.run_in_executor(
                    None, functools.partial(Engine, model_path=checkpoint)
                )
            except Exception:
                log.exception("model load failed")
                raise
            log.info(
                "models loaded: disent missing=%d unexpected=%d, torch threads=%d",
                len(app.state.engine.missing_keys),
                len(app.state.engine.unexpected_keys),
                torch.get_num_threads(),
            )
        else:
            log.info("create_app(load_models=False): inference endpoints return 503 until an engine is injected")

        try:
            yield
        finally:
            app.state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="DISENT KWS Multi-Person Voice Assistant",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def upload_size_guard(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and request.url.path.startswith("/api/") and int(length) > MAX_TOTAL_BYTES + 5 * 1024 * 1024:
            return JSONResponse({"detail": "request body is too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(NotFound)
    async def not_found_handler(_request: Request, exc: NotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(Conflict)
    async def conflict_handler(_request: Request, exc: Conflict):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(Invalid)
    async def invalid_handler(_request: Request, exc: Invalid):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(RegistryError)
    async def registry_handler(_request: Request, exc: RegistryError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # ------------------------------------------------------------------ static
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        path = STATIC_DIR / "index.html"
        if not path.exists():
            return JSONResponse(
                {"detail": "UI not built yet; static/index.html is missing"}, status_code=404
            )
        return FileResponse(path)

    # ------------------------------------------------------------------ health
    @app.get("/api/health")
    async def health():
        registry: Registry = app.state.registry
        latencies = list(app.state.latencies)
        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
        engine: Engine | None = app.state.engine
        return {
            "status": "ok" if engine is not None else "degraded",
            "models": {
                "disent": {
                    "name": "model_final.pt",
                    "state": "loaded" if engine is not None else "unavailable",
                    "missing_keys": len(engine.missing_keys) if engine else None,
                    "unexpected_keys": len(engine.unexpected_keys) if engine else None,
                },
                "ecapa": {
                    "name": "speechrec-ecapa-voxceleb",
                    "state": "loaded" if engine is not None else "unavailable",
                },
            },
            "device": engine.device if engine else "cpu",
            "torch_threads": torch.get_num_threads(),
            "uptime_sec": round(time.time() - app.state.started_at, 1),
            "people": len(registry.state.people),
            "keywords": len(registry.state.keywords),
            "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1), "count": len(latencies)},
            "pid": __import__("os").getpid(),
        }

    @app.get("/api/state")
    async def state(limit: int = Query(default=50, ge=0, le=500)):
        return app.state.registry.state_payload(limit=limit)

    @app.get("/api/people")
    async def list_people():
        registry: Registry = app.state.registry
        return {"people": [registry.person_payload(p) for p in registry.list_people()]}

    @app.get("/api/keywords")
    async def list_keywords():
        registry: Registry = app.state.registry
        return {"keywords": [registry.keyword_payload(k) for k in registry.list_keywords()]}

    # -------------------------------------------------------------------- CRUD
    @app.post("/api/people", status_code=201)
    async def create_person(payload: PersonCreate):
        person = app.state.registry.create_person(
            payload.name,
            color=payload.color,
            action=payload.action,
            trigger_on_any_speech=payload.trigger_on_any_speech,
        )
        await broadcast_state(app)
        return app.state.registry.person_payload(person)

    @app.patch("/api/people/{person_id}")
    async def patch_person(person_id: str, payload: PersonPatch):
        patch = payload.model_dump(exclude_unset=True)
        person = app.state.registry.update_person(person_id, patch)
        await broadcast_state(app)
        return app.state.registry.person_payload(person)

    @app.delete("/api/people/{person_id}", status_code=204)
    async def delete_person(person_id: str, force: bool = False):
        app.state.registry.delete_person(person_id, force=force)
        await broadcast_state(app)
        return Response(status_code=204)

    @app.post("/api/keywords", status_code=201)
    async def create_keyword(payload: KeywordCreate):
        keyword = app.state.registry.create_keyword(payload.label, payload.threshold)
        await broadcast_state(app)
        return app.state.registry.keyword_payload(keyword)

    @app.patch("/api/keywords/{keyword_id}")
    async def patch_keyword(keyword_id: str, payload: KeywordPatch):
        patch = payload.model_dump(exclude_unset=True)
        keyword = app.state.registry.update_keyword(keyword_id, patch)
        await broadcast_state(app)
        return app.state.registry.keyword_payload(keyword)

    @app.delete("/api/keywords/{keyword_id}", status_code=204)
    async def delete_keyword(keyword_id: str, force: bool = False):
        app.state.registry.delete_keyword(keyword_id, force=force)
        await broadcast_state(app)
        return Response(status_code=204)

    @app.delete("/api/people/{person_id}/keywords/{keyword_id}")
    async def detach_keyword(person_id: str, keyword_id: str):
        keyword = app.state.registry.detach_keyword(person_id, keyword_id)
        person = app.state.registry.get_person(person_id)
        await broadcast_state(app)
        return {
            "person": app.state.registry.person_payload(person),
            "keyword": app.state.registry.keyword_payload(keyword),
        }

    # --------------------------------------------------------------- enrollment
    async def _read_files(files: list[UploadFile]) -> list[torch.Tensor]:
        if not files:
            raise HTTPException(status_code=422, detail="no audio files supplied")
        if len(files) > MAX_FILES:
            raise HTTPException(status_code=422, detail=f"at most {MAX_FILES} takes are allowed")
        total = 0
        clips: list[torch.Tensor] = []
        for upload in files:
            data = await upload.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds 15 MB")
            total += len(data)
            if total > MAX_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="uploads exceed 40 MB total")
            clips.append(_decode_wav(data, filename=upload.filename or "upload.wav"))
        return clips

    @app.post("/api/enroll")
    async def enroll(
        name: str = Form(...),
        files: list[UploadFile] = File(...),
        keyword_label: str | None = Form(None),
        keyword_id: str | None = Form(None),
        action_json: str | None = Form(None),
        trigger_on_any_speech: bool = Form(False),
        color: str | None = Form(None),
        keyword_threshold: float | None = Form(None),
    ):
        require_engine(app)
        if len(files) < MIN_KEYWORD_CLIPS:
            raise HTTPException(status_code=422, detail=f"need at least {MIN_KEYWORD_CLIPS} takes")
        clips = await _read_files(files)
        action = None
        if action_json:
            try:
                action = ActionConfig.model_validate(json.loads(action_json))
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"invalid action_json: {exc}") from exc
        loop = asyncio.get_running_loop()
        func = functools.partial(
            enroll_person,
            app.state.engine,
            app.state.registry,
            name=name,
            keyword_label=keyword_label,
            keyword_id=keyword_id,
            clips=clips,
            action=action,
            trigger_on_any_speech=trigger_on_any_speech,
            color=color,
            keyword_threshold=keyword_threshold,
        )
        result = await loop.run_in_executor(app.state.executor, func)
        await broadcast_state(app)
        return result

    @app.post("/api/people/{person_id}/keywords")
    async def add_person_keyword(
        person_id: str,
        files: list[UploadFile] = File(...),
        keyword_label: str | None = Form(None),
        keyword_id: str | None = Form(None),
        keyword_threshold: float | None = Form(None),
    ):
        require_engine(app)
        if len(files) < MIN_KEYWORD_CLIPS:
            raise HTTPException(status_code=422, detail=f"need at least {MIN_KEYWORD_CLIPS} takes")
        clips = await _read_files(files)
        loop = asyncio.get_running_loop()
        func = functools.partial(
            enroll_keyword_for_person,
            app.state.engine,
            app.state.registry,
            person_id=person_id,
            clips=clips,
            keyword_label=keyword_label,
            keyword_id=keyword_id,
            keyword_threshold=keyword_threshold,
        )
        result = await loop.run_in_executor(app.state.executor, func)
        await broadcast_state(app)
        return result

    # -------------------------------------------------------------- identify
    @app.post("/api/identify")
    async def identify(
        file: UploadFile = File(...),
        execute_actions: bool = False,
    ):
        wav = _decode_wav(await file.read(MAX_FILE_BYTES + 1), filename=file.filename or "upload.wav")
        decision = await analyze_wav(app, wav)
        event = await finalize_decision(
            app,
            decision,
            execute_actions=execute_actions,
            broadcast_event=False,
            record=False,
        )
        return {
            "decision": event,
            "segments": [{"start": 0.0, "end": round(decision.duration_sec, 3)}],
        }

    @app.post("/api/replay")
    async def replay(
        file: UploadFile = File(...),
        announce: bool = True,
    ):
        wav = _decode_wav(await file.read(MAX_FILE_BYTES + 1), filename=file.filename or "upload.wav")
        decision = await analyze_wav(app, wav)
        event = await finalize_decision(
            app,
            decision,
            execute_actions=announce,
            broadcast_event=True,
            record=True,
        )
        return event

    # -------------------------------------------------------------- settings
    @app.post("/api/settings")
    async def update_settings(payload: SettingsPatch):
        patch = payload.model_dump(exclude_unset=True)
        settings = app.state.registry.update_settings(patch)
        await broadcast(app, {"type": "settings", "settings": settings.model_dump(mode="json")})
        return settings

    @app.get("/api/events")
    async def events(limit: int = Query(default=100, ge=0, le=500)):
        return {"events": app.state.registry.recent_events(limit)}

    # ---------------------------------------------------------------- demo seed
    @app.post("/api/demo/seed")
    async def demo_seed(force: bool = False):
        require_engine(app)
        loop = asyncio.get_running_loop()
        func = functools.partial(
            seed_demo,
            app.state.engine,
            app.state.registry,
            assets_dir=app.state.assets_dir,
            force=force,
        )
        result = await loop.run_in_executor(app.state.executor, func)
        await broadcast_state(app)
        return result

    # --------------------------------------------------------------- websocket
    @app.websocket("/ws/live")
    async def live_socket(websocket: WebSocket):
        await websocket.accept()
        conn = LiveConnection(websocket)
        app.state.connections.add(conn)
        session = StreamSession(app, conn)
        try:
            await conn.send_json(
                {
                    "type": "ready",
                    "settings": app.state.registry.settings.model_dump(mode="json"),
                    "models": {
                        "disent": "model_final.pt",
                        "ecapa": "speechrec-ecapa-voxceleb",
                        "engine": "loaded" if app.state.engine is not None else "unavailable",
                    },
                }
            )
            await session.run()
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("websocket session failed")
        finally:
            session.closed = True
            for task in list(session.tasks):
                task.cancel()
            app.state.connections.discard(conn)

    return app


# ------------------------------------------------------------------- helpers
def require_engine(app: FastAPI) -> Engine:
    engine = app.state.engine
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="inference engine is not loaded (started with load_models=False)",
        )
    return engine


def _decode_wav(data: bytes, *, filename: str = "upload.wav") -> torch.Tensor:
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"{filename} exceeds 15 MB")
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(status_code=415, detail=f"{filename} is not a WAV file")
    try:
        wav, sample_rate = read_wav_bytes(data)
        wav = to_16k_mono(wav, sample_rate)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=f"could not decode {filename}: {exc}") from exc
    if wav.numel() == 0:
        raise HTTPException(status_code=422, detail=f"{filename} contains no samples")
    if wav.numel() > MAX_AUDIO_SEC * SAMPLE_RATE:
        raise HTTPException(status_code=413, detail=f"{filename} exceeds {MAX_AUDIO_SEC:.0f}s")
    return wav


async def analyze_wav(app: FastAPI, wav: torch.Tensor) -> Decision:
    engine = require_engine(app)
    registry: Registry = app.state.registry
    profiles = registry.runtime_profiles()
    settings = registry.settings
    loop = asyncio.get_running_loop()
    decision = await loop.run_in_executor(
        app.state.executor, engine.analyze, wav, profiles, settings
    )
    app.state.latencies.append(float(decision.latency_ms.get("total", 0.0)))
    return decision


def next_seq(app: FastAPI) -> int:
    app.state.seq += 1
    return int(app.state.seq)


async def finalize_decision(
    app: FastAPI,
    decision: Decision,
    *,
    execute_actions: bool,
    broadcast_event: bool,
    record: bool,
) -> dict:
    registry: Registry = app.state.registry
    settings: Settings = registry.settings
    person = None
    if decision.person_id:
        try:
            person = registry.get_person(decision.person_id)
        except NotFound:
            person = None

    action_result = None
    speak = None
    should_act = bool(person) and decision.event in ("keyword", "speech")
    if decision.event == "speech" and person is not None:
        should_act = should_act and person.trigger_on_any_speech
    if should_act and execute_actions:
        keyword_label = decision.keyword_label or ""
        action_result = await app.state.runner.run(
            person.action,
            person_id=person.id,
            name=person.name,
            keyword=keyword_label,
            decision=decision.as_dict(),
            cooldown_sec=settings.cooldown_sec,
            enabled=settings.actions_enabled,
        )
        if (
            settings.announce
            and action_result.get("status") == "ok"
            and action_result.get("type") == "announce"
        ):
            speak = action_result.get("text")

    event = decision.as_dict()
    event.update(
        {
            "id": f"evt_{int(time.time() * 1000)}_{next_seq(app)}",
            "ts": utc_now(),
            "person_color": person.color if person else None,
            "action": action_result,
            "speak": speak,
        }
    )
    if record:
        event = registry.append_event(event)
    if broadcast_event:
        await broadcast(app, {"type": "event", "event": event})
    return event


async def broadcast(app: FastAPI, payload: dict) -> None:
    connections: set[LiveConnection] = set(app.state.connections)
    if not connections:
        return
    results = await asyncio.gather(
        *[conn.send_json(payload) for conn in connections], return_exceptions=True
    )
    for conn, result in zip(connections, results):
        if isinstance(result, Exception):
            app.state.connections.discard(conn)


async def broadcast_state(app: FastAPI) -> None:
    await broadcast(
        app,
        {"type": "state", "state": app.state.registry.state_payload(limit=50)},
    )


# ------------------------------------------------------------- stream session
class StreamSession:
    """One browser connection: VAD segmentation + ordered analysis tasks."""

    def __init__(self, app: FastAPI, conn: LiveConnection) -> None:
        self.app = app
        self.conn = conn
        self.settings = app.state.registry.settings
        self.sample_rate = SAMPLE_RATE
        self.vad = make_vad(self.settings)
        self.pending = torch.zeros(0)
        self.buffer = torch.zeros(0)
        self.buffer_start = 0
        self.fed_samples = 0
        self.vad_base = 0
        self.seg_start: int | None = None
        self.last_level = 0.0
        self.tasks: set[asyncio.Task] = set()
        self.closed = False

    async def run(self) -> None:
        while True:
            message = await self.conn.websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                await self.on_audio(message["bytes"])
            elif message.get("text") is not None:
                await self.on_text(message["text"])

    async def on_text(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            await self.conn.send_json({"type": "error", "message": "invalid JSON control message"})
            return
        kind = payload.get("type")
        if kind == "hello":
            rate = int(payload.get("sample_rate") or SAMPLE_RATE)
            if rate < 8000 or rate > 192000:
                await self.conn.send_json({"type": "error", "message": f"unsupported sample rate {rate}"})
                return
            self.sample_rate = rate
        elif kind == "reset":
            self.pending = torch.zeros(0)
            self.buffer = torch.zeros(0)
            self.buffer_start = self.fed_samples
            self.seg_start = None
            self.vad.reset_states()
            self.vad_base = self.fed_samples
        elif kind == "ping":
            await self.conn.send_json({"type": "pong"})
        else:
            await self.conn.send_json({"type": "error", "message": f"unknown control type {kind!r}"})

    async def on_audio(self, data: bytes) -> None:
        if len(data) == 0:
            return
        if len(data) % 4 != 0:
            await self.conn.send_json(
                {"type": "error", "message": "binary audio must be little-endian Float32"}
            )
            return
        samples = np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True)
        if not np.all(np.isfinite(samples)):
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        wav = torch.from_numpy(samples)
        if self.sample_rate != SAMPLE_RATE and wav.numel() > 1:
            wav = AF.resample(wav.unsqueeze(0), self.sample_rate, SAMPLE_RATE).squeeze(0)
        self.pending = torch.cat([self.pending, wav])
        while self.pending.numel() >= 512:
            chunk = self.pending[:512]
            self.pending = self.pending[512:]
            await self.feed(chunk)
            if self.closed:
                return

    async def feed(self, chunk: torch.Tensor) -> None:
        self.buffer = torch.cat([self.buffer, chunk])
        self.fed_samples += chunk.numel()

        result = self.vad(chunk)
        if result:
            if "start" in result and self.seg_start is None:
                self.seg_start = self.vad_base + int(result["start"])
            if "end" in result and self.seg_start is not None:
                end = self.vad_base + int(result["end"])
                start = self.seg_start
                self.seg_start = None
                if end > start:
                    await self.schedule(start, end)

        max_samples = int(self.settings.max_utterance_sec * SAMPLE_RATE)
        if self.seg_start is not None and self.fed_samples - self.seg_start >= max_samples:
            start = self.seg_start
            self.seg_start = None
            await self.schedule(start, start + max_samples)
            self.vad.reset_states()
            self.vad_base = self.fed_samples

        self._trim_buffer()
        now = time.monotonic()
        if now - self.last_level >= 0.1:
            self.last_level = now
            await self.conn.send_json(
                {
                    "type": "level",
                    "rms": round(float(chunk.pow(2).mean().sqrt().item()), 5),
                    "speech": self.seg_start is not None,
                    "vad": self.seg_start is not None,
                }
            )

    def _trim_buffer(self) -> None:
        keep_from = self.fed_samples - 30 * SAMPLE_RATE
        if self.seg_start is not None:
            keep_from = min(keep_from, self.seg_start)
        drop = keep_from - self.buffer_start
        if drop > 0:
            drop = min(int(drop), self.buffer.numel())
            self.buffer = self.buffer[drop:]
            self.buffer_start += drop

    async def schedule(self, start: int, end: int) -> None:
        i0 = max(0, start - self.buffer_start)
        i1 = min(self.buffer.numel(), end - self.buffer_start)
        if i1 - i0 < 400:  # <25 ms: ignore blips
            return
        wav = self.buffer[i0:i1].clone()
        task = asyncio.create_task(self.process(wav))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def process(self, wav: torch.Tensor) -> None:
        try:
            decision = await analyze_wav(self.app, wav)
            if self.closed:
                return
            await finalize_decision(
                self.app,
                decision,
                execute_actions=True,
                broadcast_event=True,
                record=True,
            )
        except Exception:
            log.exception("stream analysis failed")
            if not self.closed:
                try:
                    await self.conn.send_json(
                        {"type": "error", "message": "analysis failed; check the server log"}
                    )
                except Exception:
                    pass
