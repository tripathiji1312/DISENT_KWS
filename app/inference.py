"""Model loading, embedding extraction, scoring, and decision rules."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# Importing silero_vad calls torch.set_num_threads(1). Restore a small pool
# afterwards; measured DISENT batch-1 26.2 ms -> 17.9 ms with 4 threads.
from silero_vad import VADIterator, load_silero_vad

from app.audio import (
    SAMPLE_RATE,
    cap_window,
    fast_augment,
    safe_trim,
    select_window,
    to_16k_mono,
)
from app.settings import (
    ECAPA_SAVEDIR,
    ECAPA_SOURCE,
    MODEL_PATH,
    Settings,
    adaptive_speaker_threshold,
)
from models.disent_v2 import DISENT_KWS_v2

TORCH_THREADS = max(1, min(4, os.cpu_count() or 1))
torch.set_num_threads(TORCH_THREADS)


@dataclass(frozen=True)
class RuntimeProfiles:
    """Immutable snapshot used by inference; replaced atomically on mutation."""

    keyword_ids: tuple[str, ...] = ()
    keyword_labels: tuple[str, ...] = ()
    keyword_thresholds: Mapping[str, float] = field(default_factory=dict)
    keyword_matrix: torch.Tensor | None = None
    person_ids: tuple[str, ...] = ()
    person_names: tuple[str, ...] = ()
    person_matrix: torch.Tensor | None = None
    keyword_owners: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class Decision:
    event: str
    person_id: str | None = None
    person_name: str | None = None
    keyword_id: str | None = None
    keyword_label: str | None = None
    keyword_score: float = 0.0
    keyword_runner_up: float = 0.0
    speaker_score: float = 0.0
    speaker_runner_up: float = 0.0
    speaker_threshold: float = 0.0
    margin: float = 0.0
    duration_sec: float = 0.0
    confidence: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)
    keyword_scores: dict[str, float] = field(default_factory=dict)
    speaker_scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "type": self.event,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "keyword_id": self.keyword_id,
            "keyword": self.keyword_label,
            "keyword_score": round(float(self.keyword_score), 4),
            "keyword_runner_up": round(float(self.keyword_runner_up), 4),
            "speaker_score": round(float(self.speaker_score), 4),
            "speaker_runner_up": round(float(self.speaker_runner_up), 4),
            "speaker_threshold": round(float(self.speaker_threshold), 4),
            "margin": round(float(self.margin), 4),
            "duration_sec": round(float(self.duration_sec), 3),
            "confidence": round(float(self.confidence), 4),
            "latency_ms": {k: round(float(v), 2) for k, v in self.latency_ms.items()},
            "keyword_scores": {k: round(float(v), 4) for k, v in self.keyword_scores.items()},
            "speaker_scores": {k: round(float(v), 4) for k, v in self.speaker_scores.items()},
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _rank(scores: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted(scores, key=lambda item: item[1], reverse=True)


def decide(
    keyword_scores: Sequence[tuple[str, float]],
    speaker_scores: Sequence[tuple[str, float]],
    *,
    keyword_labels: Mapping[str, str],
    keyword_owners: Mapping[str, Sequence[str]],
    person_names: Mapping[str, str],
    keyword_thresholds: Mapping[str, float] | None = None,
    duration_sec: float,
    settings: Settings,
) -> Decision:
    """Pure decision function: no model calls, fully unit-testable.

    Rules:
      * keyword event requires a keyword above its threshold + margin, the
        utterance to be short (isolated keyword), and an owner above the
        duration-adaptive speaker threshold + margin;
      * a keyword heard from an unenrolled/uncertain speaker becomes
        ``keyword_unknown_speaker`` and triggers no action;
      * no keyword + confident speaker becomes ``speech`` (arbitrary-speech ID);
      * everything else is ``unrecognized``.
    """
    if duration_sec < settings.min_speech_ms / 1000.0:
        return Decision(event="too_short", duration_sec=duration_sec)

    kw_sorted = _rank(keyword_scores)
    kw_top = kw_sorted[0] if kw_sorted else ("", 0.0)
    kw_second = kw_sorted[1] if len(kw_sorted) > 1 else ("", 0.0)
    kw_threshold = (keyword_thresholds or {}).get(kw_top[0], settings.keyword_threshold)
    kw_match = bool(kw_sorted) and (
        kw_top[1] >= kw_threshold
        and (len(kw_sorted) == 1 or kw_top[1] - kw_second[1] >= settings.keyword_margin)
        and duration_sec <= settings.max_keyword_utterance_sec
    )

    spk_sorted = _rank(speaker_scores)
    base_threshold = adaptive_speaker_threshold(duration_sec) * settings.speaker_threshold_scale

    def above_threshold(candidate: Sequence[tuple[str, float]]) -> tuple[bool, float, float, float]:
        if not candidate:
            return False, 0.0, 0.0, base_threshold
        top = candidate[0]
        second = candidate[1][1] if len(candidate) > 1 else 0.0
        margin = top[1] - second
        ok = top[1] >= base_threshold and (
            len(candidate) == 1 or margin >= settings.speaker_margin
        )
        return ok, top[1], margin, base_threshold

    if kw_match:
        owners = tuple(keyword_owners.get(kw_top[0], ()))
        owner_scores = [(pid, score) for pid, score in spk_sorted if pid in owners]
        owner_scores = _rank(owner_scores)
        spk_match, spk_score, spk_margin, spk_threshold = above_threshold(owner_scores)
        if spk_match:
            person_id = owner_scores[0][0]
            kw_conf = _clamp01((kw_top[1] - kw_threshold) / max(1e-6, 1.0 - kw_threshold))
            spk_conf = _clamp01((spk_score - spk_threshold) / max(1e-6, 1.0 - spk_threshold))
            return Decision(
                event="keyword",
                person_id=person_id,
                person_name=person_names.get(person_id),
                keyword_id=kw_top[0],
                keyword_label=keyword_labels.get(kw_top[0], kw_top[0]),
                keyword_score=kw_top[1],
                keyword_runner_up=kw_second[1],
                speaker_score=spk_score,
                speaker_runner_up=owner_scores[1][1] if len(owner_scores) > 1 else 0.0,
                speaker_threshold=spk_threshold,
                margin=spk_margin,
                duration_sec=duration_sec,
                confidence=min(kw_conf, spk_conf),
                keyword_scores=dict(keyword_scores),
                speaker_scores=dict(speaker_scores),
            )
        return Decision(
            event="keyword_unknown_speaker",
            keyword_id=kw_top[0],
            keyword_label=keyword_labels.get(kw_top[0], kw_top[0]),
            keyword_score=kw_top[1],
            keyword_runner_up=kw_second[1],
            speaker_score=spk_score,
            speaker_runner_up=owner_scores[1][1] if len(owner_scores) > 1 else 0.0,
            speaker_threshold=spk_threshold,
            margin=spk_margin,
            duration_sec=duration_sec,
            confidence=0.0,
            keyword_scores=dict(keyword_scores),
            speaker_scores=dict(speaker_scores),
        )

    spk_match, spk_score, spk_margin, spk_threshold = above_threshold(spk_sorted)
    if spk_match:
        person_id = spk_sorted[0][0]
        spk_conf = _clamp01((spk_score - spk_threshold) / max(1e-6, 1.0 - spk_threshold))
        return Decision(
            event="speech",
            person_id=person_id,
            person_name=person_names.get(person_id),
            speaker_score=spk_score,
            speaker_runner_up=spk_sorted[1][1] if len(spk_sorted) > 1 else 0.0,
            speaker_threshold=spk_threshold,
            margin=spk_margin,
            duration_sec=duration_sec,
            confidence=spk_conf,
            keyword_scores=dict(keyword_scores),
            speaker_scores=dict(speaker_scores),
        )

    return Decision(
        event="unrecognized",
        keyword_score=kw_top[1],
        keyword_runner_up=kw_second[1],
        speaker_score=spk_score,
        speaker_runner_up=spk_sorted[1][1] if len(spk_sorted) > 1 else 0.0,
        speaker_threshold=spk_threshold,
        margin=spk_margin,
        duration_sec=duration_sec,
        keyword_scores=dict(keyword_scores),
        speaker_scores=dict(speaker_scores),
    )


class Engine:
    """Loads the two models once and exposes batched embedding/scoring calls."""

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        device: str = "cpu",
        ecapa_source: str = ECAPA_SOURCE,
        ecapa_savedir: Path | str = ECAPA_SAVEDIR,
    ) -> None:
        self.device = device
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model checkpoint not found: {self.model_path}. "
                "Place model_final.pt at the repo root."
            )

        self.disent = DISENT_KWS_v2().to(device)
        checkpoint = torch.load(self.model_path, map_location=device)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        result = self.disent.load_state_dict(state, strict=False)
        self.missing_keys = list(result.missing_keys)
        self.unexpected_keys = list(result.unexpected_keys)
        self.disent.eval()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            f_min=20,
            f_max=7600,
        )

        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self.ecapa = EncoderClassifier.from_hparams(
                source=ecapa_source,
                savedir=str(ecapa_savedir),
                run_opts={"device": device},
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load SpeechBrain ECAPA. Ensure the model is cached at "
                f"{ecapa_savedir} or run once with network access. Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------ audio
    def logmel(self, wav: torch.Tensor, pad_to_200: bool = True) -> torch.Tensor:
        """(80, T) log-mel; optionally pad/truncate to the model's 200 frames."""
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        wav = wav.to(self.device).to(torch.float32)
        if wav.numel() == 0:
            raise ValueError("empty waveform")
        spec = self.mel(wav.unsqueeze(0))  # (1, 80, T)
        feat = torch.log(spec + 1e-6).squeeze(0)
        if pad_to_200:
            target = 200
            if feat.shape[-1] < target:
                feat = F.pad(feat, (0, target - feat.shape[-1]))
            else:
                feat = feat[:, :target]
        return feat

    @torch.no_grad()
    def embed_phonetic_batch(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, 80, 200) -> (B, 192), L2-normalized."""
        z_phn, _ = self.disent(feats.to(self.device))
        return F.normalize(z_phn, dim=-1)

    @torch.no_grad()
    def embed_speaker_batch(self, wavs: Sequence[torch.Tensor]) -> torch.Tensor:
        """List of 1-D waveforms -> (B, 192), L2-normalized.

        Short clips are never padded to a fixed duration. Lengths are only
        padded within a batch when they are nearly equal; otherwise each clip is
        encoded separately so a long take cannot distort a short keyword.
        """
        if not wavs:
            return torch.empty(0, 192)
        prepared = []
        for wav in wavs:
            w = wav.squeeze(0) if wav.dim() == 2 else wav
            prepared.append(cap_window(w, 6.0))
        lengths = [w.numel() for w in prepared]
        with torch.no_grad():
            if max(lengths) / max(1, min(lengths)) <= 1.25:
                max_len = max(lengths)
                batch = torch.stack(
                    [F.pad(w, (0, max_len - w.numel())) for w in prepared]
                )
                emb = self.ecapa.encode_batch(batch.to(self.device))
                emb = emb.reshape(len(prepared), -1)
            else:
                parts = [
                    self.ecapa.encode_batch(w.reshape(1, -1).to(self.device))
                    for w in prepared
                ]
                emb = torch.cat(parts, dim=0).reshape(len(prepared), -1)
        return F.normalize(emb, dim=-1)

    # ---------------------------------------------------------------- scoring
    @torch.no_grad()
    def score_keywords(
        self, wav: torch.Tensor, profiles: RuntimeProfiles
    ) -> list[tuple[str, float]]:
        if profiles.keyword_matrix is None or not profiles.keyword_ids:
            return []
        window = select_window(safe_trim(wav), 2.0)
        feat = self.logmel(window).unsqueeze(0)
        z = self.embed_phonetic_batch(feat)  # (1, 192)
        matrix = profiles.keyword_matrix.to(z.device)
        sims = (matrix @ z.squeeze(0)).tolist()
        return [(kid, float(sim)) for kid, sim in zip(profiles.keyword_ids, sims)]

    @torch.no_grad()
    def score_speakers(
        self, wav: torch.Tensor, profiles: RuntimeProfiles
    ) -> list[tuple[str, float]]:
        if profiles.person_matrix is None or not profiles.person_ids:
            return []
        # Natural-length embedding: padding a short keyword to 6 s changes
        # ECAPA's embedding enough to swap the speaker.
        window = cap_window(safe_trim(wav), 6.0)
        z = self.embed_speaker_batch([window])  # (1, 192)
        matrix = profiles.person_matrix.to(z.device)
        sims = (matrix @ z.squeeze(0)).tolist()
        return [(pid, float(sim)) for pid, sim in zip(profiles.person_ids, sims)]

    def analyze(self, wav: torch.Tensor, profiles: RuntimeProfiles, settings: Settings) -> Decision:
        t0 = time.perf_counter()
        wav = wav.reshape(-1)
        duration = wav.numel() / SAMPLE_RATE
        if duration < settings.min_speech_ms / 1000.0:
            return Decision(event="too_short", duration_sec=duration)

        t_kw = time.perf_counter()
        kw_scores = self.score_keywords(wav, profiles)
        t_spk = time.perf_counter()
        spk_scores = self.score_speakers(wav, profiles)
        t_end = time.perf_counter()

        decision = decide(
            kw_scores,
            spk_scores,
            keyword_labels=dict(zip(profiles.keyword_ids, profiles.keyword_labels)),
            keyword_owners=profiles.keyword_owners,
            person_names=dict(zip(profiles.person_ids, profiles.person_names)),
            keyword_thresholds=profiles.keyword_thresholds,
            duration_sec=duration,
            settings=settings,
        )
        decision.latency_ms = {
            "keyword": (t_spk - t_kw) * 1000.0,
            "speaker": (t_end - t_spk) * 1000.0,
            "total": (t_end - t0) * 1000.0,
        }
        return decision

    # -------------------------------------------------------------- enrollment
    @torch.no_grad()
    def keyword_centroid(self, clips: Sequence[torch.Tensor], n_variants: int = 15) -> np.ndarray:
        variants = fast_augment([safe_trim(c) for c in clips], n_total=n_variants)
        feats = torch.stack([self.logmel(v) for v in variants])
        emb = self.embed_phonetic_batch(feats)
        centroid = F.normalize(emb.mean(dim=0), dim=-1)
        return centroid.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def speaker_centroid(self, clips: Sequence[torch.Tensor]) -> tuple[np.ndarray, dict]:
        prepared = [cap_window(safe_trim(c), 3.0) for c in clips]
        emb = self.embed_speaker_batch(prepared)
        centroid = F.normalize(emb.mean(dim=0), dim=-1)
        quality = {"min": 0.0, "mean": 0.0, "n": int(emb.shape[0])}
        if emb.shape[0] >= 2:
            loo = []
            for i in range(emb.shape[0]):
                others = torch.cat([emb[:i], emb[i + 1:]], dim=0)
                loo.append(float(torch.dot(emb[i], F.normalize(others.mean(dim=0), dim=-1))))
            quality = {
                "min": round(min(loo), 4),
                "mean": round(float(np.mean(loo)), 4),
                "n": int(emb.shape[0]),
            }
        return centroid.detach().cpu().numpy().astype(np.float32), quality


def make_vad(settings: Settings) -> VADIterator:
    """Fresh VAD model+iterator per stream session (VAD state is not shareable)."""
    model = load_silero_vad()
    return VADIterator(
        model,
        threshold=settings.vad_threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
    )
