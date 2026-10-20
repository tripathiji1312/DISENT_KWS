"""Unit tests for audio helpers, decision rules, and (optionally) real models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.audio import cap_window, encode_wav_bytes, fast_augment, read_wav_bytes, safe_trim, select_window
from app.inference import Decision, decide
from app.settings import Settings, adaptive_speaker_threshold
from enrollment.enroll import trim_silence

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_trim_silence_short_input():
    for n in (400, 1000, 1599):
        out = trim_silence(torch.randn(n))
        assert out.numel() == n
    assert safe_trim(torch.randn(400)).numel() == 400


def test_adaptive_thresholds():
    assert adaptive_speaker_threshold(0.5) == 0.20
    assert adaptive_speaker_threshold(0.74) == 0.20
    assert adaptive_speaker_threshold(0.75) == 0.25
    assert adaptive_speaker_threshold(1.0) == 0.25
    assert adaptive_speaker_threshold(1.5) == 0.32
    assert adaptive_speaker_threshold(2.0) == 0.32
    assert adaptive_speaker_threshold(2.5) == 0.36
    assert adaptive_speaker_threshold(6.0) == 0.36


def test_select_window_and_augment():
    wav = torch.randn(16000 * 4)
    window = select_window(wav, 2.0)
    assert window.numel() == 32000
    short = select_window(torch.randn(8000), 2.0)
    assert short.numel() == 32000
    variants = fast_augment([torch.randn(16000) for _ in range(5)], n_total=15)
    assert len(variants) == 15
    assert all(v.numel() == 16000 for v in variants)
    assert all(v.dtype == torch.float32 for v in variants)


def test_cap_window_keeps_short_clips_natural_length():
    short = torch.randn(8000)
    assert cap_window(short, 6.0).numel() == 8000
    assert cap_window(short, 3.0).numel() == 8000
    assert select_window(short, 3.0).numel() == 48000
    long = torch.randn(16000 * 8)
    assert cap_window(long, 6.0).numel() == 96000


def test_wav_roundtrip():
    wav = torch.sin(torch.linspace(0, 200, 16000)) * 0.1
    data = encode_wav_bytes(wav)
    decoded, rate = read_wav_bytes(data)
    assert rate == 16000
    assert decoded.numel() == wav.numel()
    assert torch.allclose(decoded, wav, atol=1e-3)


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def test_decide_shared_keyword():
    decision = decide(
        [("kw_stop", 0.82)],
        [("alice", 0.71), ("bob", 0.22)],
        keyword_labels={"kw_stop": "stop"},
        keyword_owners={"kw_stop": ("alice", "bob")},
        person_names={"alice": "Alice", "bob": "Bob"},
        duration_sec=1.2,
        settings=_settings(),
    )
    assert decision.event == "keyword"
    assert decision.person_name == "Alice"
    assert decision.keyword_label == "stop"
    assert decision.confidence > 0.5


def test_decide_unknown_speaker():
    decision = decide(
        [("kw_stop", 0.82)],
        [("alice", 0.12), ("bob", 0.10)],
        keyword_labels={"kw_stop": "stop"},
        keyword_owners={"kw_stop": ("alice", "bob")},
        person_names={"alice": "Alice", "bob": "Bob"},
        duration_sec=0.8,
        settings=_settings(),
    )
    assert decision.event == "keyword_unknown_speaker"
    assert decision.person_id is None


def test_decide_arbitrary_speech():
    decision = decide(
        [("kw_stop", 0.30)],
        [("alice", 0.72), ("bob", 0.10)],
        keyword_labels={"kw_stop": "stop"},
        keyword_owners={"kw_stop": ("alice", "bob")},
        person_names={"alice": "Alice", "bob": "Bob"},
        duration_sec=2.2,
        settings=_settings(),
    )
    assert decision.event == "speech"
    assert decision.person_name == "Alice"


def test_decide_margin():
    decision = decide(
        [("kw_stop", 0.30)],
        [("alice", 0.72), ("bob", 0.68)],
        keyword_labels={"kw_stop": "stop"},
        keyword_owners={"kw_stop": ("alice", "bob")},
        person_names={"alice": "Alice", "bob": "Bob"},
        duration_sec=2.2,
        settings=_settings(),
    )
    assert decision.event == "unrecognized"
    assert decision.speaker_score == 0.72


def test_decide_long_keyword_ignored():
    decision = decide(
        [("kw_stop", 0.95)],
        [("alice", 0.85), ("bob", 0.10)],
        keyword_labels={"kw_stop": "stop"},
        keyword_owners={"kw_stop": ("alice", "bob")},
        person_names={"alice": "Alice", "bob": "Bob"},
        duration_sec=3.2,
        settings=_settings(),
    )
    assert decision.event == "speech"


def test_decide_one_person_skips_margin():
    decision = decide(
        [("kw_yes", 0.70)],
        [("alice", 0.40)],
        keyword_labels={"kw_yes": "yes"},
        keyword_owners={"kw_yes": ("alice",)},
        person_names={"alice": "Alice"},
        duration_sec=0.8,
        settings=_settings(),
    )
    assert decision.event == "keyword"
    assert decision.person_name == "Alice"


def test_decide_threshold_override_and_too_short():
    decision = decide(
        [("kw_stop", 0.55)],
        [("alice", 0.6)],
        keyword_labels={"kw_stop": "stop"},
        keyword_owners={"kw_stop": ("alice",)},
        person_names={"alice": "Alice"},
        keyword_thresholds={"kw_stop": 0.60},
        duration_sec=1.0,
        settings=_settings(),
    )
    assert decision.event == "speech"

    short = decide([], [], keyword_labels={}, keyword_owners={}, person_names={}, duration_sec=0.1, settings=_settings())
    assert short.event == "too_short"


# ------------------------------------------------------------------ real models
def _engine_available() -> bool:
    from app.settings import ECAPA_SAVEDIR, MODEL_PATH

    return MODEL_PATH.exists() and ECAPA_SAVEDIR.exists()


@pytest.mark.integration
@pytest.mark.skipif(not _engine_available(), reason="model_final.pt or ECAPA cache missing")
def test_real_keyword_gap():
    from app.inference import Engine

    engine = Engine()
    yes = read_wav_bytes((FIXTURES / "keyword_yes_1353.wav").read_bytes())[0]
    no = read_wav_bytes((FIXTURES / "keyword_no_1665.wav").read_bytes())[0]
    yes_centroid = engine.keyword_centroid([yes])
    no_centroid = engine.keyword_centroid([no])
    sim = float(np.dot(yes_centroid, no_centroid))
    self_sim = float(np.dot(yes_centroid, yes_centroid))
    assert self_sim == pytest.approx(1.0, abs=1e-4)
    assert self_sim - sim > 0.10, f"yes-vs-no cosine gap too small: {self_sim - sim:.3f}"


@pytest.mark.integration
@pytest.mark.skipif(not _engine_available(), reason="model_final.pt or ECAPA cache missing")
def test_real_speaker_gap():
    from app.inference import Engine

    engine = Engine()
    a1 = read_wav_bytes((FIXTURES / "speaker_a_1.wav").read_bytes())[0]
    a2 = read_wav_bytes((FIXTURES / "speaker_a_2.wav").read_bytes())[0]
    b1 = read_wav_bytes((FIXTURES / "speaker_b_1.wav").read_bytes())[0]
    centroid, quality = engine.speaker_centroid([a1])
    emb = engine.embed_speaker_batch([a2, b1]).cpu().numpy()
    sim_a = float(np.dot(centroid, emb[0]))
    sim_b = float(np.dot(centroid, emb[1]))
    assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-4)
    assert sim_a > sim_b + 0.20, f"speaker gap too small: {sim_a:.3f} vs {sim_b:.3f}"
    assert quality["n"] == 1


DEMO = ROOT / "demo_assets"


@pytest.mark.integration
@pytest.mark.skipif(
    not _engine_available() or not (DEMO / "bob_sentence_1.wav").exists(),
    reason="model_final.pt, ECAPA cache, or demo assets missing",
)
def test_real_short_keyword_is_not_zero_padded():
    """Regression for a 0.5 s keyword being padded to 6 s before ECAPA."""
    from app.inference import Engine

    engine = Engine()

    def load(name: str) -> torch.Tensor:
        return read_wav_bytes((DEMO / name).read_bytes())[0]

    bob_centroid, _ = engine.speaker_centroid(
        [load(f"bob_sentence_{i}.wav") for i in (1, 2, 3)]
    )
    alice_centroid, _ = engine.speaker_centroid(
        [load(f"alice_sentence_{i}.wav") for i in (1, 2, 3)]
    )
    bob_stop = engine.embed_speaker_batch([load("bob_stop.wav")]).cpu().numpy()[0]
    alice_stop = engine.embed_speaker_batch([load("alice_stop.wav")]).cpu().numpy()[0]

    bob_to_bob = float(np.dot(bob_centroid, bob_stop))
    bob_to_alice = float(np.dot(alice_centroid, bob_stop))
    alice_to_alice = float(np.dot(alice_centroid, alice_stop))
    alice_to_bob = float(np.dot(bob_centroid, alice_stop))
    assert bob_to_bob > 0.30, f"Bob stop self-similarity too low: {bob_to_bob:.3f}"
    assert bob_to_bob > bob_to_alice + 0.15, (
        f"Bob stop matched Alice: {bob_to_bob:.3f} vs {bob_to_alice:.3f}"
    )
    assert alice_to_alice > 0.25, f"Alice stop self-similarity too low: {alice_to_alice:.3f}"
    assert alice_to_alice > alice_to_bob + 0.10, (
        f"Alice stop matched Bob: {alice_to_alice:.3f} vs {alice_to_bob:.3f}"
    )
