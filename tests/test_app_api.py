"""API tests use a deterministic fake engine; real-model E2E is separate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.inference import Decision
from app.server import create_app

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
WAV = (FIXTURES / "keyword_yes_1353.wav").read_bytes()


class FakeEngine:
    device = "cpu"
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []

    def speaker_centroid(self, clips):
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32), {
            "min": 0.9,
            "mean": 0.95,
            "n": len(clips),
        }

    def keyword_centroid(self, clips):
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    def analyze(self, wav, profiles, settings):
        person_id = profiles.person_ids[0] if profiles.person_ids else None
        person_name = profiles.person_names[0] if profiles.person_names else None
        keyword_id = profiles.keyword_ids[0] if profiles.keyword_ids else None
        keyword_label = profiles.keyword_labels[0] if profiles.keyword_labels else None
        return Decision(
            event="keyword" if person_id else "unrecognized",
            person_id=person_id,
            person_name=person_name,
            keyword_id=keyword_id,
            keyword_label=keyword_label,
            keyword_score=0.91,
            keyword_runner_up=0.21,
            speaker_score=0.81,
            speaker_runner_up=0.11,
            speaker_threshold=0.25,
            margin=0.70,
            duration_sec=wav.numel() / 16000,
            confidence=0.84,
            latency_ms={"keyword": 1.0, "speaker": 2.0, "total": 3.0},
        )


@pytest.fixture()
def app(tmp_path: Path):
    return create_app(data_dir=tmp_path, load_models=False, assets_dir=tmp_path / "assets")


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        app.state.engine = FakeEngine()
        yield test_client


def _files(count: int = 3):
    return [("files", (f"take{i}.wav", WAV, "audio/wav")) for i in range(count)]


def _enroll(client: TestClient, name: str = "Alice", keyword: str = "stop"):
    response = client.post(
        "/api/enroll",
        data={"name": name, "keyword_label": keyword},
        files=_files(),
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_health_and_state(client: TestClient):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["models"]["disent"]["state"] == "loaded"
    state = client.get("/api/state").json()
    assert state["people"] == []
    assert state["recommended_keywords"] == ["stop", "yes", "no"]


def test_crud_shared_keyword_and_delete_rules(client: TestClient):
    alice = _enroll(client, "Alice", "stop")
    bob = _enroll(client, "Bob", "stop")
    state = client.get("/api/state").json()
    assert len(state["people"]) == 2
    assert len(state["keywords"]) == 1
    keyword = state["keywords"][0]
    assert keyword["label"] == "stop"
    assert keyword["n_people"] == 2
    assert {owner["name"] for owner in keyword["owners"]} == {"Alice", "Bob"}

    patched = client.patch(
        f"/api/people/{bob['person']['id']}",
        json={"name": "Robert", "action": {"type": "log", "message_template": "hi {name}"}},
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Robert"
    assert patched.json()["action"]["type"] == "log"

    # Shared keyword: deleting Bob is allowed and leaves Alice as owner.
    assert client.delete(f"/api/people/{bob['person']['id']}").status_code == 204
    state = client.get("/api/state").json()
    assert len(state["people"]) == 1
    assert state["keywords"][0]["n_people"] == 1

    # Alice is the last owner: deletion needs force because it orphans the keyword.
    assert client.delete(f"/api/people/{alice['person']['id']}").status_code == 409
    assert client.delete(f"/api/people/{alice['person']['id']}?force=true").status_code == 204
    assert client.get("/api/state").json()["keywords"] == []


def test_keyword_crud(client: TestClient):
    created = client.post("/api/keywords", json={"label": "Yes"})
    assert created.status_code == 201
    assert created.json()["label"] == "yes"
    keyword_id = created.json()["id"]
    assert client.post("/api/keywords", json={"label": "yes"}).status_code == 409
    patched = client.patch(f"/api/keywords/{keyword_id}", json={"threshold": 0.6})
    assert patched.status_code == 200
    assert patched.json()["threshold"] == 0.6
    assert client.delete(f"/api/keywords/{keyword_id}").status_code == 204


def test_enroll_requires_three_takes(client: TestClient):
    response = client.post(
        "/api/enroll",
        data={"name": "Alice", "keyword_label": "stop"},
        files=_files(2),
    )
    assert response.status_code == 422
    assert "at least 3" in response.json()["detail"]


def test_identify_returns_decision_without_event_log(client: TestClient):
    _enroll(client)
    response = client.post(
        "/api/identify",
        files={"file": ("sample.wav", WAV, "audio/wav")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["decision"]["type"] == "keyword"
    assert payload["decision"]["person_name"] == "Alice"
    assert payload["decision"]["keyword"] == "stop"
    assert payload["decision"]["latency_ms"]["total"] == 3.0
    assert client.get("/api/events").json()["events"] == []


def test_replay_broadcasts_and_logs(client: TestClient):
    _enroll(client)
    with client.websocket_connect("/ws/live") as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        response = client.post(
            "/api/replay",
            files={"file": ("sample.wav", WAV, "audio/wav")},
        )
        assert response.status_code == 200, response.text
        event = response.json()
        assert event["type"] == "keyword"
        assert event["speak"] == "Alice said stop"
        assert event["action"]["status"] == "ok"
        message = websocket.receive_json()
        assert message["type"] == "event"
        assert message["event"]["id"] == event["id"]

    events = client.get("/api/events").json()["events"]
    assert len(events) == 1
    assert events[0]["id"] == event["id"]


def test_websocket_ready_and_ping(client: TestClient):
    with client.websocket_connect("/ws/live") as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["models"]["engine"] == "loaded"
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json() == {"type": "pong"}


def test_settings_validation_and_persistence(client: TestClient):
    assert client.post("/api/settings", json={"keyword_threshold": 2}).status_code == 422
    assert client.post("/api/settings", json={"vad_threshold": 0}).status_code == 422
    response = client.post(
        "/api/settings", json={"keyword_threshold": 0.52, "speaker_margin": 0.1}
    )
    assert response.status_code == 200
    assert response.json()["keyword_threshold"] == 0.52
    state = client.get("/api/state").json()
    assert state["settings"]["keyword_threshold"] == 0.52
    assert state["settings"]["speaker_margin"] == 0.1


def test_upload_guards(client: TestClient):
    bad = client.post(
        "/api/identify",
        files={"file": ("x.txt", b"not a wav", "text/plain")},
    )
    assert bad.status_code == 415
    too_many = client.post(
        "/api/enroll",
        data={"name": "Alice", "keyword_label": "stop"},
        files=_files(9),
    )
    assert too_many.status_code == 422


def test_demo_seed_missing_assets_is_409(client: TestClient):
    response = client.post("/api/demo/seed")
    assert response.status_code == 409
    assert "demo assets not found" in response.json()["detail"]


def test_static_dashboard_is_served(client: TestClient):
    index = client.get("/")
    assert index.status_code == 200
    assert 'id="app"' in index.text
    assert "audio-worklet.js" not in index.text  # loaded by app.js
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/audio-worklet.js").status_code == 200
