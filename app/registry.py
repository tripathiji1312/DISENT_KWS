"""Thread-safe JSON registry, voice profiles, and event log.

State is a single JSON document plus one `.npy` per centroid. Every write is
atomic (temp file + fsync + os.replace), so a crash cannot leave half a JSON
document. A `.bak` copy of the previous registry is kept for startup recovery.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.inference import RuntimeProfiles
from app.settings import Settings, ensure_data_dirs

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[a-z0-9_]{1,80}$")
COLOR_RE = re.compile(r"^#[0-9a-f]{6}$")
HEX_RE = re.compile(r"^[0-9a-f]{6}$")

PALETTE = (
    "#8b5cf6",
    "#06b6d4",
    "#f59e0b",
    "#ec4899",
    "#22c55e",
    "#ef4444",
    "#3b82f6",
    "#a3e635",
)

RECOMMENDED_KEYWORDS = ("stop", "yes", "no")


class RegistryError(Exception):
    """Base class for user-facing registry errors."""


class NotFound(RegistryError):
    pass


class Conflict(RegistryError):
    pass


class Invalid(RegistryError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (slug[:24] or "item").strip("_")


def _new_id(prefix: str, label: str) -> str:
    return f"{prefix}_{_slug(label)}_{secrets.token_hex(3)}"


def _validate(model_cls, data: dict):
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise Invalid(str(exc)) from exc


def _normalize_vector(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise Invalid("profile vector is empty")
    if not np.all(np.isfinite(arr)):
        raise Invalid("profile vector contains non-finite values")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-6:
        raise Invalid("profile vector has zero norm")
    return (arr / norm).astype(np.float32)


class ActionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["announce", "log", "webhook", "shell"] = "announce"
    enabled: bool = True
    webhook_url: str | None = None
    shell_command: str | None = None
    message_template: str = "{name} said {keyword}"

    @field_validator("message_template")
    @classmethod
    def _template(cls, value: str) -> str:
        value = value.strip() or "{name} said {keyword}"
        if len(value) > 250:
            raise ValueError("message_template must be at most 250 characters")
        return value

    @model_validator(mode="after")
    def _check_target(self) -> "ActionConfig":
        if self.type == "webhook":
            if not self.webhook_url:
                raise ValueError("webhook action requires webhook_url")
            if not self.webhook_url.startswith(("http://", "https://")):
                raise ValueError("webhook_url must start with http:// or https://")
        if self.type == "shell" and not self.shell_command:
            raise ValueError("shell action requires shell_command")
        return self


class PersonKeyword(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword_id: str
    profile: str
    n_samples: int = Field(default=1, ge=1, le=1000)


class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1, max_length=40)
    color: str
    avatar: str = ""
    profile: str | None = None
    keywords: list[PersonKeyword] = Field(default_factory=list)
    action: ActionConfig = Field(default_factory=ActionConfig)
    trigger_on_any_speech: bool = False
    created_at: str = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("name cannot be blank")
        return value

    @field_validator("color")
    @classmethod
    def _color(cls, value: str) -> str:
        value = value.strip().lower()
        if not COLOR_RE.match(value):
            raise ValueError("color must be a #rrggbb hex value")
        return value

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not ID_RE.match(value):
            raise ValueError("invalid person id")
        return value

    @model_validator(mode="after")
    def _avatar(self) -> "Person":
        if not self.avatar:
            words = self.name.split()
            letters = "".join(w[0] for w in words[:2]) if len(words) > 1 else self.name[:2]
            self.avatar = letters.upper()
        return self


class Keyword(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str = Field(min_length=1, max_length=40)
    profile: str | None = None
    n_samples: int = Field(default=0, ge=0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=utc_now)

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        value = " ".join(value.split()).lower()
        if not value:
            raise ValueError("keyword label cannot be blank")
        if not re.match(r"^[a-z0-9][a-z0-9 '\-]*$", value):
            raise ValueError("keyword label may contain letters, numbers, spaces, ' and -")
        return value

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not ID_RE.match(value):
            raise ValueError("invalid keyword id")
        return value


class RegistryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    settings: Settings = Field(default_factory=Settings)
    keywords: list[Keyword] = Field(default_factory=list)
    people: list[Person] = Field(default_factory=list)


class Registry:
    def __init__(self, data_dir: Path | str | None = None, *, max_events: int = 500) -> None:
        self.data_dir, self.profiles_dir = ensure_data_dirs(
            Path(data_dir) if data_dir is not None else None
        )
        self.registry_path = self.data_dir / "registry.json"
        self.backup_path = self.data_dir / "registry.json.bak"
        self.events_path = self.data_dir / "events.jsonl"
        self._lock = threading.RLock()
        self._events: deque[dict] = deque(maxlen=max_events)
        self.max_events = max_events
        self.state = RegistryState()
        self._runtime: RuntimeProfiles | None = None
        self.load()

    # --------------------------------------------------------------- storage
    def _safe_path(self, relative: str) -> Path:
        if not relative or Path(relative).is_absolute():
            raise Invalid("profile path must be relative")
        parts = Path(relative).parts
        if any(part in ("..", "") for part in parts):
            raise Invalid("profile path may not contain '..'")
        root = self.data_dir.resolve()
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise Invalid("profile path escapes the data directory")
        return resolved

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _write_json_atomic(self, path: Path, payload: dict, *, backup: bool = False) -> None:
        if backup and path.exists():
            try:
                shutil.copy2(path, self.backup_path)
            except OSError:
                log.warning("could not refresh registry backup", exc_info=True)
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self._atomic_write_bytes(path, data)

    def _write_npy(self, relative: str, vector: np.ndarray) -> str:
        path = self._safe_path(relative)
        buf = io.BytesIO()
        np.save(buf, _normalize_vector(vector), allow_pickle=False)
        self._atomic_write_bytes(path, buf.getvalue())
        return relative

    def _load_npy(self, relative: str | None) -> np.ndarray | None:
        if not relative:
            return None
        try:
            path = self._safe_path(relative)
        except Invalid:
            log.error("rejected unsafe profile path: %r", relative)
            return None
        if not path.exists():
            log.warning("profile file missing: %s", relative)
            return None
        try:
            arr = np.load(path, allow_pickle=False)
            return _normalize_vector(arr)
        except (ValueError, OSError) as exc:
            log.error("could not read profile %s: %s", relative, exc)
            return None

    def _delete_profile(self, relative: str | None) -> None:
        if not relative:
            return
        try:
            self._safe_path(relative).unlink(missing_ok=True)
        except (Invalid, OSError) as exc:
            log.warning("could not delete profile %s: %s", relative, exc)

    def save(self) -> None:
        with self._lock:
            self._write_json_atomic(
                self.registry_path, self.state.model_dump(mode="json"), backup=True
            )

    def load(self) -> None:
        with self._lock:
            self.state = self._read_state(self.registry_path)
            if self.state is None:
                self.state = self._read_state(self.backup_path)
                if self.state is not None:
                    log.warning("recovered registry from %s", self.backup_path)
            if self.state is None:
                self.state = RegistryState()
            self._runtime = None
            self._load_events()
            log.info(
                "registry loaded: %d people, %d keywords",
                len(self.state.people),
                len(self.state.keywords),
            )

    def _read_state(self, path: Path) -> RegistryState | None:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text("utf-8"))
            return RegistryState.model_validate(raw)
        except Exception as exc:
            log.error("could not load registry %s: %s", path, exc)
            if path == self.registry_path:
                corrupt = path.with_suffix(f".corrupt-{int(datetime.now().timestamp())}.json")
                try:
                    shutil.copy2(path, corrupt)
                    log.error("saved corrupt registry copy to %s", corrupt)
                except OSError:
                    pass
            return None

    def _load_events(self) -> None:
        self._events.clear()
        if not self.events_path.exists():
            return
        try:
            with self.events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            log.warning("could not read events log", exc_info=True)

    # -------------------------------------------------------------- snapshots
    def snapshot(self) -> RegistryState:
        with self._lock:
            return self.state.model_copy(deep=True)

    @property
    def settings(self) -> Settings:
        with self._lock:
            return self.state.settings.model_copy(deep=True)

    def runtime_profiles(self) -> RuntimeProfiles:
        with self._lock:
            if self._runtime is not None:
                return self._runtime
            keyword_ids: list[str] = []
            keyword_labels: list[str] = []
            keyword_thresholds: dict[str, float] = {}
            keyword_vectors: list[np.ndarray] = []
            for keyword in self.state.keywords:
                vector = self._load_npy(keyword.profile)
                if vector is None:
                    continue
                keyword_ids.append(keyword.id)
                keyword_labels.append(keyword.label)
                if keyword.threshold is not None:
                    keyword_thresholds[keyword.id] = keyword.threshold
                keyword_vectors.append(vector)

            person_ids: list[str] = []
            person_names: list[str] = []
            person_vectors: list[np.ndarray] = []
            for person in self.state.people:
                vector = self._load_npy(person.profile)
                if vector is None:
                    continue
                person_ids.append(person.id)
                person_names.append(person.name)
                person_vectors.append(vector)

            owners: dict[str, list[str]] = {}
            for person in self.state.people:
                for link in person.keywords:
                    owners.setdefault(link.keyword_id, []).append(person.id)

            import torch

            keyword_matrix = (
                torch.from_numpy(np.stack(keyword_vectors).astype(np.float32))
                if keyword_vectors
                else None
            )
            person_matrix = (
                torch.from_numpy(np.stack(person_vectors).astype(np.float32))
                if person_vectors
                else None
            )
            self._runtime = RuntimeProfiles(
                keyword_ids=tuple(keyword_ids),
                keyword_labels=tuple(keyword_labels),
                keyword_thresholds=MappingProxyType(keyword_thresholds),
                keyword_matrix=keyword_matrix,
                person_ids=tuple(person_ids),
                person_names=tuple(person_names),
                person_matrix=person_matrix,
                keyword_owners=MappingProxyType(
                    {key: tuple(sorted(value)) for key, value in owners.items()}
                ),
            )
            return self._runtime

    def _invalidate(self) -> None:
        self._runtime = None

    # --------------------------------------------------------------- lookups
    def get_person(self, person_id: str) -> Person:
        with self._lock:
            for person in self.state.people:
                if person.id == person_id:
                    return person
        raise NotFound(f"person not found: {person_id}")

    def get_keyword(self, keyword_id: str) -> Keyword:
        with self._lock:
            for keyword in self.state.keywords:
                if keyword.id == keyword_id:
                    return keyword
        raise NotFound(f"keyword not found: {keyword_id}")

    def list_people(self) -> list[Person]:
        with self._lock:
            return list(self.state.people)

    def list_keywords(self) -> list[Keyword]:
        with self._lock:
            return list(self.state.keywords)

    def find_person_by_name(self, name: str) -> Person | None:
        needle = " ".join(name.split()).casefold()
        with self._lock:
            for person in self.state.people:
                if person.name.casefold() == needle:
                    return person
        return None

    def find_keyword_by_label(self, label: str) -> Keyword | None:
        needle = " ".join(label.split()).casefold()
        with self._lock:
            for keyword in self.state.keywords:
                if keyword.label.casefold() == needle:
                    return keyword
        return None

    def resolve_keyword(self, *, keyword_id: str | None = None, label: str | None = None) -> Keyword:
        if keyword_id:
            return self.get_keyword(keyword_id)
        if label:
            keyword = self.find_keyword_by_label(label)
            if keyword is None:
                raise NotFound(f"keyword not found: {label}")
            return keyword
        raise Invalid("provide keyword_id or keyword_label")

    # --------------------------------------------------------------- mutation
    def create_keyword(self, label: str, threshold: float | None = None) -> Keyword:
        with self._lock:
            if self.find_keyword_by_label(label) is not None:
                raise Conflict(f"keyword already exists: {label.strip().lower()}")
            keyword = _validate(
                Keyword,
                {
                    "id": _new_id("kw", label),
                    "label": label,
                    "threshold": threshold,
                },
            )
            self.state.keywords.append(keyword)
            self._invalidate()
            self.save()
            return keyword.model_copy(deep=True)

    def get_or_create_keyword(self, label: str, threshold: float | None = None) -> tuple[Keyword, bool]:
        with self._lock:
            existing = self.find_keyword_by_label(label)
            if existing is not None:
                if threshold is not None and existing.threshold != threshold:
                    existing.threshold = threshold
                    self._invalidate()
                    self.save()
                return existing.model_copy(deep=True), False
        return self.create_keyword(label, threshold), True

    def update_keyword(self, keyword_id: str, patch: dict[str, Any]) -> Keyword:
        allowed = {"label", "threshold"}
        unknown = set(patch) - allowed
        if unknown:
            raise Invalid(f"unknown keyword fields: {', '.join(sorted(unknown))}")
        with self._lock:
            keyword = self.get_keyword(keyword_id)
            data = keyword.model_dump()
            data.update({k: v for k, v in patch.items() if v is not None or k == "threshold"})
            if "label" in patch and patch["label"] is not None:
                other = self.find_keyword_by_label(str(patch["label"]))
                if other is not None and other.id != keyword_id:
                    raise Conflict(f"keyword already exists: {patch['label']}")
            updated = _validate(Keyword, {**data, "id": keyword.id, "created_at": keyword.created_at})
            idx = self.state.keywords.index(keyword)
            self.state.keywords[idx] = updated
            self._invalidate()
            self.save()
            return updated.model_copy(deep=True)

    def delete_keyword(self, keyword_id: str, *, force: bool = False) -> None:
        with self._lock:
            keyword = self.get_keyword(keyword_id)
            owners = [p for p in self.state.people if any(k.keyword_id == keyword_id for k in p.keywords)]
            if owners and not force:
                raise Conflict(
                    f"keyword '{keyword.label}' is assigned to {len(owners)} person(s); pass force=true"
                )
            for person in owners:
                person.keywords = [k for k in person.keywords if k.keyword_id != keyword_id]
                self._delete_profile(f"profiles/pk_{person.id}_{keyword_id}.npy")
            self._delete_profile(keyword.profile)
            self.state.keywords.remove(keyword)
            self._invalidate()
            self.save()

    def create_person(
        self,
        name: str,
        *,
        color: str | None = None,
        action: ActionConfig | dict | None = None,
        trigger_on_any_speech: bool = False,
    ) -> Person:
        with self._lock:
            if self.find_person_by_name(name) is not None:
                raise Conflict(f"person already exists: {name}")
            chosen = color or PALETTE[len(self.state.people) % len(PALETTE)]
            action_config = _validate(ActionConfig, action or {}) if not isinstance(action, ActionConfig) else action
            person = _validate(
                Person,
                {
                    "id": _new_id("p", name),
                    "name": name,
                    "color": chosen,
                    "action": action_config,
                    "trigger_on_any_speech": trigger_on_any_speech,
                },
            )
            self.state.people.append(person)
            self._invalidate()
            self.save()
            return person.model_copy(deep=True)

    def update_person(self, person_id: str, patch: dict[str, Any]) -> Person:
        allowed = {"name", "color", "action", "trigger_on_any_speech", "avatar"}
        unknown = set(patch) - allowed
        if unknown:
            raise Invalid(f"unknown person fields: {', '.join(sorted(unknown))}")
        with self._lock:
            person = self.get_person(person_id)
            if "name" in patch and patch["name"] is not None:
                other = self.find_person_by_name(str(patch["name"]))
                if other is not None and other.id != person_id:
                    raise Conflict(f"person already exists: {patch['name']}")
            data = person.model_dump()
            data.update(patch)
            updated = _validate(Person, {**data, "id": person.id, "created_at": person.created_at})
            idx = self.state.people.index(person)
            self.state.people[idx] = updated
            self._invalidate()
            self.save()
            return updated.model_copy(deep=True)

    def set_person_profile(self, person_id: str, centroid: np.ndarray | Sequence[float]) -> str:
        with self._lock:
            person = self.get_person(person_id)
            relative = f"profiles/{person.id}.npy"
            self._write_npy(relative, np.asarray(centroid))
            person.profile = relative
            self._invalidate()
            self.save()
            return relative

    def attach_keyword(
        self,
        person_id: str,
        keyword_id: str,
        centroid: np.ndarray | Sequence[float],
        *,
        n_samples: int,
    ) -> Keyword:
        """Attach a person-specific keyword centroid and rebuild the global one."""
        with self._lock:
            person = self.get_person(person_id)
            keyword = self.get_keyword(keyword_id)
            n_samples = max(1, int(n_samples))
            relative = f"profiles/pk_{person.id}_{keyword.id}.npy"
            self._write_npy(relative, np.asarray(centroid))
            link = next((k for k in person.keywords if k.keyword_id == keyword_id), None)
            if link is None:
                person.keywords.append(
                    PersonKeyword(keyword_id=keyword_id, profile=relative, n_samples=n_samples)
                )
            else:
                link.profile = relative
                link.n_samples = n_samples
            self.recompute_keyword(keyword_id)
            self._invalidate()
            self.save()
            return self.get_keyword(keyword_id).model_copy(deep=True)

    def detach_keyword(self, person_id: str, keyword_id: str) -> Keyword:
        with self._lock:
            person = self.get_person(person_id)
            link = next((k for k in person.keywords if k.keyword_id == keyword_id), None)
            if link is None:
                raise NotFound("person does not have that keyword")
            person.keywords = [k for k in person.keywords if k.keyword_id != keyword_id]
            self._delete_profile(link.profile)
            self.recompute_keyword(keyword_id)
            self._invalidate()
            self.save()
            return self.get_keyword(keyword_id).model_copy(deep=True)

    def recompute_keyword(self, keyword_id: str) -> Keyword:
        """Global centroid = normalized weighted mean of owner-specific centroids."""
        with self._lock:
            keyword = self.get_keyword(keyword_id)
            vectors: list[np.ndarray] = []
            weights: list[float] = []
            for person in self.state.people:
                link = next((k for k in person.keywords if k.keyword_id == keyword_id), None)
                if link is None:
                    continue
                vector = self._load_npy(link.profile)
                if vector is None:
                    continue
                vectors.append(vector)
                weights.append(max(1, link.n_samples))
            if not vectors:
                self._delete_profile(keyword.profile)
                keyword.profile = None
                keyword.n_samples = 0
                return keyword.model_copy(deep=True)
            stacked = np.stack(vectors).astype(np.float32)
            weight_arr = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
            mean = (stacked * weight_arr).sum(axis=0) / float(weight_arr.sum())
            relative = f"profiles/{keyword.id}.npy"
            self._write_npy(relative, mean)
            keyword.profile = relative
            keyword.n_samples = int(sum(weights))
            return keyword.model_copy(deep=True)

    def delete_person(self, person_id: str, *, force: bool = False) -> list[str]:
        """Delete a person. Returns orphaned keyword IDs that were removed."""
        with self._lock:
            person = self.get_person(person_id)
            owned = {link.keyword_id for link in person.keywords}
            others = {
                link.keyword_id
                for other in self.state.people
                if other.id != person_id
                for link in other.keywords
            }
            orphans = sorted(owned - others)
            if orphans and not force:
                labels = [self.get_keyword(kid).label for kid in orphans]
                raise Conflict(
                    "deleting this person would leave keyword(s) with no owners: "
                    + ", ".join(labels)
                    + "; pass force=true to delete them too"
                )
            for link in list(person.keywords):
                self._delete_profile(link.profile)
            self._delete_profile(person.profile)
            self.state.people.remove(person)
            for keyword_id in orphans:
                keyword = self.get_keyword(keyword_id)
                self._delete_profile(keyword.profile)
                self.state.keywords.remove(keyword)
            for keyword_id in owned - set(orphans):
                self.recompute_keyword(keyword_id)
            self._invalidate()
            self.save()
            return orphans

    # ----------------------------------------------------------------- events
    def append_event(self, event: dict) -> dict:
        with self._lock:
            record = dict(event)
            record.setdefault("id", f"evt_{secrets.token_hex(6)}")
            record.setdefault("ts", utc_now())
            line = json.dumps(record, ensure_ascii=False)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._events.append(record)
            return record

    def recent_events(self, limit: int = 100) -> list[dict]:
        limit = max(0, min(int(limit), self.max_events))
        with self._lock:
            return list(reversed(self._events))[:limit]

    # --------------------------------------------------------------- settings
    def update_settings(self, patch: dict[str, Any]) -> Settings:
        with self._lock:
            current = self.state.settings.model_dump()
            current.update(patch)
            updated = _validate(Settings, current)
            self.state.settings = updated
            self._invalidate()
            self.save()
            return updated.model_copy(deep=True)

    # ----------------------------------------------------------------- payloads
    def person_payload(self, person: Person) -> dict:
        keyword_map = {k.id: k for k in self.state.keywords}
        labels = [
            keyword_map[link.keyword_id].label
            for link in person.keywords
            if link.keyword_id in keyword_map
        ]
        return {
            **person.model_dump(mode="json"),
            "keyword_labels": labels,
        }

    def keyword_payload(self, keyword: Keyword) -> dict:
        owners = [
            {"id": person.id, "name": person.name, "color": person.color, "avatar": person.avatar}
            for person in self.state.people
            if any(link.keyword_id == keyword.id for link in person.keywords)
        ]
        return {
            **keyword.model_dump(mode="json"),
            "owners": owners,
            "n_people": len(owners),
            "recommended": keyword.label in RECOMMENDED_KEYWORDS,
        }

    def state_payload(self, limit: int = 50) -> dict:
        with self._lock:
            return {
                "version": self.state.version,
                "settings": self.state.settings.model_dump(mode="json"),
                "people": [self.person_payload(p) for p in self.state.people],
                "keywords": [self.keyword_payload(k) for k in self.state.keywords],
                "events": self.recent_events(limit),
                "recommended_keywords": list(RECOMMENDED_KEYWORDS),
            }
