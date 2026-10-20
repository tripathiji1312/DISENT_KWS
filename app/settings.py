"""Application settings and filesystem locations."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
PROFILE_DIR = DATA_DIR / "profiles"
MODEL_PATH = APP_ROOT / "model_final.pt"
STATIC_DIR = APP_ROOT / "app" / "static"
DEMO_ASSETS_DIR = APP_ROOT / "demo_assets"

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SAVEDIR = Path.home() / ".cache" / "huggingface" / "speechbrain" / "spkrec-ecapa-voxceleb"


class Settings(BaseModel):
    """Runtime-tunable settings shared by the API and the inference engine."""

    keyword_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    keyword_margin: float = Field(default=0.06, ge=0.0, le=1.0)
    speaker_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    speaker_threshold_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    vad_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    min_speech_ms: int = Field(default=350, ge=100, le=2000)
    max_utterance_sec: float = Field(default=6.0, ge=1.0, le=20.0)
    max_keyword_utterance_sec: float = Field(default=2.5, ge=0.5, le=10.0)
    cooldown_sec: float = Field(default=1.5, ge=0.0, le=30.0)
    announce: bool = True
    actions_enabled: bool = True


def adaptive_speaker_threshold(duration_sec: float) -> float:
    """Duration-aware ECAPA threshold calibrated on held-out LibriSpeech.

    Evidence (research/findings-2026-10-20.md, section 4):
      0.5 s  -> genuine min 0.250, impostor max 0.103
      1.0 s  -> genuine min 0.418, impostor max 0.130
      2.0 s  -> genuine min 0.592, impostor max 0.138
      3.0 s  -> genuine min 0.577, impostor max 0.096
    """
    if duration_sec < 0.75:
        return 0.20
    if duration_sec < 1.5:
        return 0.25
    if duration_sec < 2.5:
        return 0.32
    return 0.36


def ensure_data_dirs(data_dir: Path | None = None) -> tuple[Path, Path]:
    """Create the runtime data directories and return (data, profiles)."""
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    profiles = base / "profiles"
    base.mkdir(parents=True, exist_ok=True)
    profiles.mkdir(parents=True, exist_ok=True)
    return base, profiles
