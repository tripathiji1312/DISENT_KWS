from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Sequence
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class LFBETransform:
    def __init__(self):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.WIN_LENGTH,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=20,
            f_max=7600,
        )
        self._target_len = int(config.SAMPLE_RATE * config.MAX_AUDIO_SEC)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        n = waveform.shape[-1]
        if n < self._target_len:
            waveform = F.pad(waveform, (0, self._target_len - n))
        else:
            waveform = waveform[..., : self._target_len]

        spec = self.mel(waveform)                          # (1, 80, T)
        return torch.log(spec + 1e-6).squeeze(0)          # (80, T)


def _pitch_shift(wav: torch.Tensor, semitones: int) -> torch.Tensor:
    try:
        return AF.pitch_shift(wav, config.SAMPLE_RATE, semitones)
    except Exception:
        return wav  # graceful fallback if sox not available


def _speed_perturb(wav: torch.Tensor, factor: float, target_len: int) -> torch.Tensor:
    try:
        out, _ = AF.speed(wav, config.SAMPLE_RATE, factor)
        if out.shape[-1] != target_len:
            out = F.interpolate(
                out.unsqueeze(0).float(), size=target_len, mode="linear",
                align_corners=False
            ).squeeze(0)
        return out
    except Exception:
        return wav


def augment_enrollment(
    waveforms: Sequence[torch.Tensor],
    n_total: int = 30,
) -> list[torch.Tensor]:
    if len(waveforms) < 5:
        raise ValueError(
            f"Need at least 5 enrollment samples, got {len(waveforms)}"
        )
    target_len = waveforms[0].shape[-1]
    variants: list[torch.Tensor] = list(waveforms)

    pitch_opts = [-2, -1, 1, 2]
    speed_opts = [0.90, 0.95, 1.05, 1.10]
    gain_opts  = [-3.0, -1.5, 1.5, 3.0]

    for wav in waveforms:
        for ps in pitch_opts:
            variants.append(_pitch_shift(wav, ps))
            if len(variants) >= n_total:
                return variants[:n_total]

    rng = random.Random(42)
    while len(variants) < n_total:
        base = rng.choice(waveforms)
        aug  = _pitch_shift(base, rng.choice(pitch_opts))
        aug  = _speed_perturb(aug, rng.choice(speed_opts), target_len)
        gain_db = rng.choice(gain_opts)
        aug  = aug * (10 ** (gain_db / 20.0))
        variants.append(aug)

    return variants[:n_total]


@torch.no_grad()
def extract_prototypes(
    model: torch.nn.Module,
    waveforms: list[torch.Tensor],
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    transform  = LFBETransform()
    model      = model.to(device).eval()

    z_phns, z_spks = [], []
    for wav in waveforms:
        feat = transform(wav).unsqueeze(0).to(device)   # (1, 80, T)
        z_phn, z_spk = model(feat)                       # (1, 192) each
        z_phns.append(z_phn.squeeze(0))
        z_spks.append(z_spk.squeeze(0))

    p_kw  = F.normalize(torch.stack(z_phns).mean(dim=0, keepdim=True), dim=-1)
    p_spk = F.normalize(torch.stack(z_spks).mean(dim=0, keepdim=True), dim=-1)
    return p_kw, p_spk

@torch.no_grad()
def calibrate_threshold(
    model:              torch.nn.Module,
    p_kw:              torch.Tensor,
    p_spk:             torch.Tensor,
    background_paths:  list[str],
    target_fa_per_hr:  float = 1.0,
    window_sec:        float = 2.0,
    device:            str   = "cpu",
) -> float:
    transform  = LFBETransform()
    model      = model.to(device).eval()
    hop_n      = int(config.SAMPLE_RATE * window_sec)
    scores_bg: list[float] = []

    for path in background_paths:
        wav, sr = torchaudio.load(path)
        if sr != config.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE)
        for start in range(0, wav.shape[-1] - hop_n + 1, hop_n):
            chunk = wav[:, start : start + hop_n]
            feat  = transform(chunk).unsqueeze(0).to(device)
            z_phn, z_spk = model(feat, p_spk=p_spk, p_kw=p_kw)
            score, _, _ = model.scorer(z_phn, z_spk, p_kw, p_spk)
            scores_bg.append(score.item())

    if not scores_bg:
        print("⚠️  No background audio scored — using default threshold 0.50")
        return 0.5

    total_hours = (len(scores_bg) * window_sec) / 3600.0

    scores_bg_sorted = sorted(scores_bg, reverse=True)
    max_fa           = int(target_fa_per_hr * total_hours) + 1
    threshold        = float(scores_bg_sorted[min(max_fa, len(scores_bg_sorted) - 1)])

    print(f"✅  Calibrated threshold = {threshold:.4f}  "
          f"(target FA ≤ {target_fa_per_hr}/hr, "
          f"{total_hours:.2f} hrs background audio)")
    return threshold

def enroll_user(
    model:                  torch.nn.Module,
    keyword_audio_paths:    list[str],
    background_audio_paths: list[str] | None = None,
    target_fa_per_hr:       float = 1.0,
    n_augmented:            int   = 30,
    device:                 str   = "cpu",
) -> dict:
    if len(keyword_audio_paths) < 5:
        raise ValueError(
            f"Need ≥ 5 enrollment recordings, got {len(keyword_audio_paths)}"
        )

    # 1. Load raw waveforms
    print(f"🔊  Loading {len(keyword_audio_paths)} enrollment recordings …")
    waveforms = []
    for path in keyword_audio_paths:
        wav, sr = torchaudio.load(path)
        if sr != config.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)   # stereo → mono
        waveforms.append(wav)

    # 2. DSP augmentation
    print(f"🔧  Augmenting to {n_augmented} variants …")
    augmented = augment_enrollment(waveforms, n_total=n_augmented)

    # 3. Extract prototypes
    print(f"🧠  Extracting embeddings ({n_augmented} passes) …")
    p_kw, p_spk = extract_prototypes(model, augmented, device=device)
    print(f"   p_kw  shape: {tuple(p_kw.shape)}  norm: {p_kw.norm():.4f}")
    print(f"   p_spk shape: {tuple(p_spk.shape)}  norm: {p_spk.norm():.4f}")

    # 4. Calibrate threshold
    threshold = config.DEFAULT_THRESHOLD
    if background_audio_paths:
        print(f"🎯  Calibrating threshold against {len(background_audio_paths)} background files …")
        threshold = calibrate_threshold(
            model, p_kw, p_spk,
            background_paths=background_audio_paths,
            target_fa_per_hr=target_fa_per_hr,
            device=device,
        )
    else:
        print(f"⚠️  No background audio provided — using default threshold {threshold:.2f}")

    enrollment = {"p_kw": p_kw.cpu(), "p_spk": p_spk.cpu(), "threshold": threshold}
    print(f"\n✅  Enrollment complete.  threshold = {threshold:.4f}\n")
    return enrollment

def save_enrollment(enrollment: dict, path: str | os.PathLike) -> None:
    torch.save(enrollment, path)
    print(f"💾  Enrollment saved → {path}")


def load_enrollment(path: str | os.PathLike) -> dict:
    enrollment = torch.load(path, map_location="cpu")
    print(f"📂  Enrollment loaded ← {path}  (threshold={enrollment['threshold']:.4f})")
    return enrollment

if __name__ == "__main__":
    import tempfile
    import numpy as np

    # Create 5 dummy 1-second 16 kHz WAV files in a temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(5):
            p = os.path.join(tmpdir, f"kw_{i}.wav")
            wav = torch.randn(1, config.SAMPLE_RATE) * 0.1
            torchaudio.save(p, wav, config.SAMPLE_RATE)
            paths.append(p)

        # Load a dummy model
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from models.disent_v2 import DISENT_KWS_v2
        model = DISENT_KWS_v2().eval()

        enrollment = enroll_user(model, paths)
        assert enrollment["p_kw"].shape  == (1, config.EMBED_DIM)
        assert enrollment["p_spk"].shape == (1, config.EMBED_DIM)
        print("✅  enroll_user smoke test passed")

        # Save / load round-trip
        save_path = os.path.join(tmpdir, "enrollment.pt")
        save_enrollment(enrollment, save_path)
        loaded = load_enrollment(save_path)
        assert torch.allclose(loaded["p_kw"], enrollment["p_kw"])
        print("✅  save/load round-trip passed")
