"""Demo seeding from a manifest of local WAV files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.audio import read_wav_bytes, to_16k_mono
from app.enrollment import enroll_person
from app.inference import Engine
from app.registry import Conflict, Invalid, Registry

log = logging.getLogger(__name__)


def load_manifest(assets_dir: Path | str) -> list[dict]:
    root = Path(assets_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise Conflict(
            f"demo assets not found at {root}. Add demo_assets/manifest.json and WAV files, "
            "or enroll people from the UI."
        )
    try:
        raw = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Invalid(f"could not read demo manifest: {exc}") from exc
    people = raw.get("people") if isinstance(raw, dict) else raw
    if not isinstance(people, list) or not people:
        raise Invalid("demo manifest must contain a non-empty 'people' list")
    return people


def _read_clip(path: Path):
    if not path.exists():
        raise Invalid(f"demo audio missing: {path}")
    wav, sample_rate = read_wav_bytes(path.read_bytes())
    return to_16k_mono(wav, sample_rate)


def seed_demo(
    engine: Engine,
    registry: Registry,
    *,
    assets_dir: Path | str,
    force: bool = False,
) -> dict:
    root = Path(assets_dir)
    entries = load_manifest(root)
    seeded: list[dict] = []
    skipped: list[str] = []

    for entry in entries:
        name = str(entry.get("name") or "").strip()
        if not name:
            raise Invalid("every manifest person needs a name")
        existing = registry.find_person_by_name(name)
        if existing is not None:
            if not force:
                skipped.append(name)
                continue
            registry.delete_person(existing.id, force=True)

        keyword = entry.get("keyword")
        keyword_files = entry.get("keyword_files") or []
        speaker_files = entry.get("speaker_files") or keyword_files
        if not keyword:
            raise Invalid(f"{name}: manifest entry needs a keyword")
        if not keyword_files:
            raise Invalid(f"{name}: manifest entry needs keyword_files")
        clips = [_read_clip(root / filename) for filename in keyword_files]
        speaker_clips = [_read_clip(root / filename) for filename in speaker_files]
        result = enroll_person(
            engine,
            registry,
            name=name,
            keyword_label=keyword,
            clips=clips,
            speaker_clips=speaker_clips,
            action=entry.get("action") or {"type": "announce"},
            trigger_on_any_speech=bool(entry.get("trigger_on_any_speech", False)),
            color=entry.get("color"),
            min_keyword_clips=1,
        )
        seeded.append(
            {
                "name": name,
                "person_id": result["person"]["id"],
                "keyword": result["keyword"]["label"],
                "quality": result["quality"],
            }
        )
        log.info("seeded demo person %s (%s)", name, result["keyword"]["label"])

    return {"seeded": seeded, "skipped": skipped, "assets_dir": str(root)}
