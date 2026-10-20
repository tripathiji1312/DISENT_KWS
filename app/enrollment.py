"""Enrollment orchestration: audio validation, centroids, registry transaction."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch

from app.audio import SAMPLE_RATE, is_too_quiet, peak, rms, safe_trim
from app.inference import Engine
from app.registry import ActionConfig, Invalid, Keyword, Registry
from app.settings import Settings


def clip_duration(clip: torch.Tensor) -> float:
    return clip.numel() / SAMPLE_RATE


def validate_clips(
    clips: Sequence[torch.Tensor],
    *,
    min_clips: int = 3,
    max_clips: int = 8,
    min_sec: float = 0.25,
    max_sec: float = 8.0,
) -> list[torch.Tensor]:
    """Trim, validate, and return 1-D clips. Raises Invalid with a useful message."""
    if len(clips) < min_clips:
        raise Invalid(f"need at least {min_clips} audio take(s), got {len(clips)}")
    if len(clips) > max_clips:
        raise Invalid(f"at most {max_clips} audio takes are allowed, got {len(clips)}")

    prepared: list[torch.Tensor] = []
    for index, clip in enumerate(clips, start=1):
        wav = clip.reshape(-1).to(torch.float32)
        if wav.numel() == 0:
            raise Invalid(f"take {index} is empty")
        wav = safe_trim(wav)
        duration = clip_duration(wav)
        if duration < min_sec:
            raise Invalid(f"take {index} is too short ({duration:.2f}s; need ≥{min_sec:.2f}s)")
        if duration > max_sec:
            raise Invalid(f"take {index} is too long ({duration:.2f}s; need ≤{max_sec:.0f}s)")
        if is_too_quiet(wav):
            raise Invalid(
                f"take {index} is silent or too quiet (peak {peak(wav):.3f}, rms {rms(wav):.3f})"
            )
        prepared.append(wav.contiguous())
    return prepared


def leave_one_out(vectors: Sequence[np.ndarray] | torch.Tensor) -> dict:
    if isinstance(vectors, torch.Tensor):
        arr = vectors.detach().cpu().numpy()
    else:
        arr = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors])
    if arr.shape[0] < 2:
        return {"min": 0.0, "mean": 0.0, "n": int(arr.shape[0])}
    scores = []
    for i in range(arr.shape[0]):
        others = np.delete(arr, i, axis=0).mean(axis=0)
        denom = float(np.linalg.norm(arr[i]) * np.linalg.norm(others))
        scores.append(float(np.dot(arr[i], others) / max(denom, 1e-9)))
    return {
        "min": round(float(min(scores)), 4),
        "mean": round(float(np.mean(scores)), 4),
        "n": int(arr.shape[0]),
    }


def enroll_person(
    engine: Engine,
    registry: Registry,
    *,
    name: str,
    keyword_label: str | None = None,
    keyword_id: str | None = None,
    clips: Sequence[torch.Tensor],
    speaker_clips: Sequence[torch.Tensor] | None = None,
    action: ActionConfig | dict | None = None,
    trigger_on_any_speech: bool = False,
    color: str | None = None,
    keyword_threshold: float | None = None,
    min_keyword_clips: int = 3,
) -> dict:
    """Enroll a new person with a speaker profile and one keyword profile."""
    if not keyword_label and not keyword_id:
        raise Invalid("provide keyword_label or keyword_id")

    started = time.perf_counter()
    keyword_takes = validate_clips(clips, min_clips=min_keyword_clips)
    if speaker_clips is not None:
        speaker_takes = validate_clips(
            speaker_clips, min_clips=1, max_clips=8, min_sec=0.25, max_sec=8.0
        )
    else:
        speaker_takes = keyword_takes

    t0 = time.perf_counter()
    speaker_centroid, quality = engine.speaker_centroid(speaker_takes)
    t1 = time.perf_counter()
    keyword_centroid = engine.keyword_centroid(keyword_takes)
    t2 = time.perf_counter()

    warnings: list[str] = []
    if quality.get("n", 0) >= 2 and quality.get("min", 1.0) < 0.35:
        warnings.append("speaker takes are inconsistent; re-record in a quieter room")

    keyword_created = False
    person = None
    try:
        if keyword_id:
            keyword = registry.resolve_keyword(keyword_id=keyword_id)
            if keyword_threshold is not None:
                keyword = registry.update_keyword(keyword.id, {"threshold": keyword_threshold})
        else:
            keyword, keyword_created = registry.get_or_create_keyword(
                str(keyword_label), keyword_threshold
            )
        person = registry.create_person(
            name,
            color=color,
            action=action or ActionConfig(type="announce"),
            trigger_on_any_speech=trigger_on_any_speech,
        )
        registry.set_person_profile(person.id, speaker_centroid)
        keyword = registry.attach_keyword(
            person.id, keyword.id, keyword_centroid, n_samples=len(keyword_takes)
        )
        person = registry.get_person(person.id)
    except Exception:
        # Keep the registry consistent if any step after person creation fails.
        if person is not None:
            try:
                registry.delete_person(person.id, force=True)
            except Exception:
                pass
        elif keyword_created:
            try:
                existing = registry.find_keyword_by_label(str(keyword_label))
                if existing is not None:
                    registry.delete_keyword(existing.id, force=True)
            except Exception:
                pass
        raise

    total = time.perf_counter() - started
    return {
        "person": registry.person_payload(person),
        "keyword": registry.keyword_payload(keyword),
        "quality": quality,
        "warnings": warnings,
        "timings_ms": {
            "speaker": round((t1 - t0) * 1000.0, 1),
            "keyword": round((t2 - t1) * 1000.0, 1),
            "total": round(total * 1000.0, 1),
        },
        "keyword_created": keyword_created,
    }


def enroll_keyword_for_person(
    engine: Engine,
    registry: Registry,
    *,
    person_id: str,
    clips: Sequence[torch.Tensor],
    keyword_label: str | None = None,
    keyword_id: str | None = None,
    keyword_threshold: float | None = None,
    min_keyword_clips: int = 3,
) -> dict:
    """Add an extra keyword (and its voice centroid) to an existing person."""
    if not keyword_label and not keyword_id:
        raise Invalid("provide keyword_label or keyword_id")
    person = registry.get_person(person_id)
    takes = validate_clips(clips, min_clips=min_keyword_clips)

    started = time.perf_counter()
    centroid = engine.keyword_centroid(takes)
    keyword_took = (time.perf_counter() - started) * 1000.0

    if keyword_id:
        keyword = registry.resolve_keyword(keyword_id=keyword_id)
        if keyword_threshold is not None:
            keyword = registry.update_keyword(keyword.id, {"threshold": keyword_threshold})
    else:
        keyword, _ = registry.get_or_create_keyword(str(keyword_label), keyword_threshold)

    keyword = registry.attach_keyword(person.id, keyword.id, centroid, n_samples=len(takes))
    person = registry.get_person(person.id)
    return {
        "person": registry.person_payload(person),
        "keyword": registry.keyword_payload(keyword),
        "quality": {},
        "warnings": [],
        "timings_ms": {"keyword": round(keyword_took, 1)},
    }


def default_settings() -> Settings:
    return Settings()
