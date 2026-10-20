"""Registry persistence, shared-keyword centroid, path safety, and events."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from app.registry import Conflict, Invalid, NotFound, Registry


def vec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def test_attach_reload_roundtrip(tmp_path: Path):
    registry = Registry(tmp_path)
    keyword = registry.create_keyword("stop")
    person = registry.create_person("Alice", color="#8b5cf6")
    registry.set_person_profile(person.id, vec(1, 0, 0))
    registry.attach_keyword(person.id, keyword.id, vec(0, 1, 0), n_samples=3)

    person_file = tmp_path / "profiles" / f"{person.id}.npy"
    keyword_file = tmp_path / "profiles" / f"{keyword.id}.npy"
    link_file = tmp_path / "profiles" / f"pk_{person.id}_{keyword.id}.npy"
    assert person_file.exists() and keyword_file.exists() and link_file.exists()
    for path in (person_file, keyword_file, link_file):
        arr = np.load(path, allow_pickle=False)
        assert arr.shape == (3,)
        assert np.linalg.norm(arr) == pytest.approx(1.0, abs=1e-5)

    reloaded = Registry(tmp_path)
    runtime = reloaded.runtime_profiles()
    assert runtime.person_ids == (person.id,)
    assert runtime.keyword_ids == (keyword.id,)
    assert float(np.dot(runtime.person_matrix[0].numpy(), vec(1, 0, 0))) == pytest.approx(1.0)
    assert runtime.keyword_owners[keyword.id] == (person.id,)


def test_shared_keyword_weighted_centroid_and_delete(tmp_path: Path):
    registry = Registry(tmp_path)
    keyword = registry.create_keyword("stop")
    alice = registry.create_person("Alice")
    bob = registry.create_person("Bob")
    registry.set_person_profile(alice.id, vec(1, 0, 0))
    registry.set_person_profile(bob.id, vec(0, 1, 0))
    registry.attach_keyword(alice.id, keyword.id, vec(1, 0, 0), n_samples=1)
    registry.attach_keyword(bob.id, keyword.id, vec(0, 1, 0), n_samples=3)

    runtime = registry.runtime_profiles()
    assert runtime.keyword_owners[keyword.id] == tuple(sorted((alice.id, bob.id)))
    centroid = runtime.keyword_matrix[0].numpy()
    expected = vec(1, 3, 0) / np.linalg.norm(vec(1, 3, 0))
    assert np.allclose(centroid, expected, atol=1e-5)
    assert registry.get_keyword(keyword.id).n_samples == 4

    removed = registry.delete_person(bob.id)
    assert removed == []
    runtime = registry.runtime_profiles()
    assert runtime.keyword_owners[keyword.id] == (alice.id,)
    assert np.allclose(runtime.keyword_matrix[0].numpy(), vec(1, 0, 0), atol=1e-5)
    assert not (tmp_path / "profiles" / f"pk_{bob.id}_{keyword.id}.npy").exists()


def test_delete_person_force_removes_orphan_keyword(tmp_path: Path):
    registry = Registry(tmp_path)
    keyword = registry.create_keyword("yes")
    alice = registry.create_person("Alice")
    registry.set_person_profile(alice.id, vec(1, 0, 0))
    registry.attach_keyword(alice.id, keyword.id, vec(0, 1, 0), n_samples=2)
    with pytest.raises(Conflict):
        registry.delete_person(alice.id)
    removed = registry.delete_person(alice.id, force=True)
    assert removed == [keyword.id]
    assert registry.list_keywords() == []
    assert registry.runtime_profiles().person_ids == ()


def test_delete_keyword_shared_requires_force(tmp_path: Path):
    registry = Registry(tmp_path)
    keyword = registry.create_keyword("no")
    alice = registry.create_person("Alice")
    bob = registry.create_person("Bob")
    registry.set_person_profile(alice.id, vec(1, 0, 0))
    registry.set_person_profile(bob.id, vec(0, 1, 0))
    registry.attach_keyword(alice.id, keyword.id, vec(0, 1, 0), n_samples=1)
    registry.attach_keyword(bob.id, keyword.id, vec(0, 1, 0), n_samples=1)
    with pytest.raises(Conflict):
        registry.delete_keyword(keyword.id)
    registry.delete_keyword(keyword.id, force=True)
    assert registry.list_keywords() == []
    assert all(not person.keywords for person in registry.list_people())


def test_path_traversal_rejected(tmp_path: Path):
    registry = Registry(tmp_path)
    with pytest.raises(Invalid):
        registry._safe_path("../../etc/passwd")
    with pytest.raises(Invalid):
        registry._safe_path("/etc/passwd")
    assert registry._load_npy("../../etc/passwd") is None


def test_concurrent_settings_writes_stay_valid(tmp_path: Path):
    registry = Registry(tmp_path)
    errors: list[Exception] = []

    def worker(value: float) -> None:
        try:
            registry.update_settings({"keyword_threshold": value})
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(0.3 + i * 0.01,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    raw = json.loads((tmp_path / "registry.json").read_text("utf-8"))
    assert 0.3 <= raw["settings"]["keyword_threshold"] <= 0.49


def test_events_and_recovery(tmp_path: Path):
    registry = Registry(tmp_path)
    registry.create_keyword("stop")
    registry.create_person("Alice")
    registry.append_event({"type": "keyword", "person_name": "Alice"})
    registry.append_event({"type": "speech", "person_name": "Alice"})
    recent = registry.recent_events(1)
    assert len(recent) == 1
    assert recent[0]["type"] == "speech"
    assert recent[0]["id"].startswith("evt_")
    assert recent[0]["ts"].endswith("Z")

    # registry.json.bak holds the previous valid document (after keyword creation).
    (tmp_path / "registry.json").write_text("{not json", encoding="utf-8")
    recovered = Registry(tmp_path)
    assert len(recovered.list_keywords()) == 1
    assert recovered.recent_events(10)[0]["type"] == "speech"


def test_duplicates_and_validation(tmp_path: Path):
    registry = Registry(tmp_path)
    registry.create_keyword("Stop")
    with pytest.raises(Conflict):
        registry.create_keyword("stop")
    registry.create_person("Alice")
    with pytest.raises(Conflict):
        registry.create_person("alice")
    with pytest.raises(Invalid):
        registry.update_settings({"keyword_threshold": 2.0})
    with pytest.raises(NotFound):
        registry.get_person("p_missing_000000")
    person = registry.get_person(registry.list_people()[0].id)
    updated = registry.update_person(person.id, {"action": {"type": "log", "message_template": "hi {name}"}})
    assert updated.action.type == "log"
    with pytest.raises(Invalid):
        registry.update_person(person.id, {"nope": 1})
