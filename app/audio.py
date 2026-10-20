"""Audio I/O and light-weight DSP helpers.

Important: torchaudio.load/save are broken on this host (torchcodec/FFmpeg
mismatch), so all file I/O uses soundfile. torchaudio is only used for
resampling and the existing MelSpectrogram transform.
"""

from __future__ import annotations

import io
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from enrollment.enroll import trim_silence

SAMPLE_RATE = 16000
_TRIM_MIN_SAMPLES = 1600  # trim_silence's 100 ms frame size


def read_wav_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    """Decode WAV bytes into a mono float32 tensor and its sample rate."""
    if not data:
        raise ValueError("empty audio file")
    try:
        with sf.SoundFile(io.BytesIO(data)) as fh:
            sample_rate = int(fh.samplerate)
            frames = fh.read(dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises LibsndfileError / RuntimeError
        raise ValueError(f"could not decode WAV: {exc}") from exc

    if frames.size == 0:
        raise ValueError("audio file contains no samples")
    mono = frames.mean(axis=1) if frames.ndim == 2 else frames
    return torch.from_numpy(np.ascontiguousarray(mono)), sample_rate


def to_16k_mono(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Return a 1-D mono tensor at 16 kHz."""
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    if wav.dim() != 1:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {tuple(wav.shape)}")
    wav = wav.to(torch.float32)
    if sample_rate != SAMPLE_RATE:
        wav = AF.resample(wav.unsqueeze(0), sample_rate, SAMPLE_RATE).squeeze(0)
    return wav.contiguous()


def safe_trim(wav: torch.Tensor) -> torch.Tensor:
    """trim_silence with a guard for clips shorter than one 100 ms frame."""
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    if wav.numel() < _TRIM_MIN_SAMPLES:
        return wav
    return trim_silence(wav).squeeze(0)


def peak(wav: torch.Tensor) -> float:
    return float(wav.abs().max().item()) if wav.numel() else 0.0


def rms(wav: torch.Tensor) -> float:
    return float(wav.pow(2).mean().sqrt().item()) if wav.numel() else 0.0


def is_too_quiet(wav: torch.Tensor) -> bool:
    return peak(wav) < 0.01 or rms(wav) < 0.003


def cap_window(wav: torch.Tensor, seconds: float) -> torch.Tensor:
    """Pick the highest-energy contiguous window, without padding short clips.

    Speaker embeddings must keep the natural duration: SpeechBrain's ECAPA
    changes substantially when a 0.5 s keyword is padded with seconds of zeros
    (measured Bob stop similarity +0.41 unpadded vs -0.07 at a fixed 6 s).
    """
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    wav = wav.to(torch.float32)
    need = max(1, int(round(seconds * SAMPLE_RATE)))
    if wav.numel() <= need:
        return wav
    if wav.numel() < _TRIM_MIN_SAMPLES:
        start = max(0, (wav.numel() - need) // 2)
        return wav[start:start + need]

    frame = 1600
    hop = 400
    frames = wav.unfold(0, frame, hop)
    energy = frames.pow(2).mean(dim=1)
    win_frames = max(1, int(round(need / hop)))
    if win_frames >= energy.numel():
        return wav[:need]

    csum = torch.cumsum(torch.cat([energy.new_zeros(1), energy]), dim=0)
    totals = csum[win_frames:] - csum[:-win_frames]
    start = int(totals.argmax().item()) * hop
    start = min(start, wav.numel() - need)
    return wav[start:start + need]


def select_window(wav: torch.Tensor, seconds: float) -> torch.Tensor:
    """Highest-energy window, zero-padded to exactly ``seconds`` when short.

    Used by DISENT keyword scoring, whose mel input is fixed-length. Speaker
    embeddings must use :func:`cap_window` instead.
    """
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    need = max(1, int(round(seconds * SAMPLE_RATE)))
    window = cap_window(wav, seconds)
    if window.numel() < need:
        window = F.pad(window, (0, need - window.numel()))
    return window


def fast_augment(clips: Sequence[torch.Tensor], n_total: int = 15) -> list[torch.Tensor]:
    """Fast enrollment variants: gain jitter + linear time-stretch.

    The repo's augment_enrollment uses pitch_shift and took 142 s for 5 -> 30
    clips on this CPU. This path measured ~70 ms for 5 -> 15 and is what the
    web enrollment flow uses.
    """
    clips = [c.squeeze(0) if c.dim() == 2 else c for c in clips]
    clips = [c.to(torch.float32).contiguous() for c in clips if c.numel() > 0]
    if not clips:
        return []
    n_total = max(len(clips), int(n_total))
    target = int(clips[0].numel())
    out: list[torch.Tensor] = []
    factors = (0.95, 1.05, 0.92, 1.08)
    gains_db = (-2.0, 2.0, -1.0, 1.0)

    for i in range(n_total):
        if i < len(clips):
            variant = clips[i].clone()
        else:
            base = clips[i % len(clips)]
            factor = factors[i % len(factors)]
            n = max(1, int(round(base.numel() * factor)))
            variant = F.interpolate(
                base.view(1, 1, -1), size=n, mode="linear", align_corners=False
            ).view(-1)
            gain = 10.0 ** (gains_db[i % len(gains_db)] / 20.0)
            variant = variant * gain

        if variant.numel() < target:
            variant = F.pad(variant, (0, target - variant.numel()))
        elif variant.numel() > target:
            variant = variant[:target]
        out.append(variant.clamp(-1.0, 1.0))
    return out


def encode_wav_bytes(wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode a mono float tensor as PCM16 WAV bytes (tests/demo assets)."""
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    arr = wav.detach().cpu().numpy().astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def concat_chunks(chunks: Iterable[torch.Tensor]) -> torch.Tensor:
    parts = [c for c in chunks if c.numel() > 0]
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts).contiguous()
