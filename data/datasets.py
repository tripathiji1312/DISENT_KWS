import os
import glob
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class LFBETransform:
    """Shared feature extraction — used by ALL dataloaders."""
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 hop_length: int = 160,
                 n_mels: int = 80,
                 f_min: int = 20,
                 f_max: int = 7600,
                 max_audio_sec: float = 2.0,
                 max_frames: int = 200):
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * max_audio_sec)
        self.max_frames = max_frames
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Accept (T,) or (1, T) or (C, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Trim or pad to target length
        if waveform.size(-1) < self.target_len:
            pad = self.target_len - waveform.size(-1)
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[..., : self.target_len]

        mel = self.mel(waveform) + 1e-6
        log_mel = torch.log(mel)
        
        # Ensure we always get exactly max_frames frames
        # Trim or pad the time dimension to max_frames
        if log_mel.shape[-1] > self.max_frames:
            log_mel = log_mel[..., :self.max_frames]
        elif log_mel.shape[-1] < self.max_frames:
            pad = self.max_frames - log_mel.shape[-1]
            log_mel = F.pad(log_mel, (0, pad), value=-20.0)
        
        # Return (n_mels, T_frames)
        return log_mel.squeeze(0)


class GSCDataset(Dataset):
    """Google Speech Commands v2 — 35 words, ~105K utterances"""
    def __init__(self, root, subset='training', transform: Optional[LFBETransform] = None, augmentor=None):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset=subset)
        self.transform = transform or LFBETransform()
        self.augmentor = augmentor
        # Collect labels from dataset examples
        labels = set()
        for i in range(len(self.dataset)):
            try:
                _, _, label, *_ = self.dataset[i]
                labels.add(label)
            except Exception:
                continue
        self.labels = sorted(list(labels))

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.dataset[idx]
        if self.augmentor:
            waveform = self.augmentor(waveform)
        features = self.transform(waveform)
        label_idx = self.labels.index(label)
        return features, label_idx

    def __len__(self):
        return len(self.dataset)


class VoxCelebDataset(Dataset):
    """VoxCeleb — full 7205 speakers (Kaggle-hosted)"""
    def __init__(self, root, transform: Optional[LFBETransform] = None, augmentor=None, max_utts_per_spk=50):
        self.transform = transform or LFBETransform()
        self.augmentor = augmentor
        self.samples = self._scan_directory(root, max_utts_per_spk)

    def _scan_directory(self, root, max_per_spk):
        samples = []
        if not os.path.isdir(root):
            return samples
        speakers = sorted(os.listdir(root))
        for spk_idx, spk_id in enumerate(speakers):
            spk_dir = os.path.join(root, spk_id)
            if not os.path.isdir(spk_dir):
                continue
            utts = glob.glob(os.path.join(spk_dir, '**/*.wav'), recursive=True)
            for utt in utts[:max_per_spk]:
                samples.append((utt, spk_idx))
        return samples

    def __getitem__(self, idx):
        path, spk_label = self.samples[idx]
        waveform, sr = torchaudio.load(path)
        if self.augmentor:
            waveform = self.augmentor(waveform)
        features = self.transform(waveform)
        return features, spk_label

    def __len__(self):
        return len(self.samples)


class LibriPhraseDataset(Dataset):
    """LibriPhrase — keyword triplets (anchor, positive, confuser)"""
    def __init__(self, root, split='hard', transform: Optional[LFBETransform] = None, augmentor=None):
        self.transform = transform or LFBETransform()
        self.augmentor = augmentor
        self.triplets = self._load_triplets(root, split)

    def _load_triplets(self, root, split):
        triplets = []
        meta_file = os.path.join(root, f'{split}_triplets.csv')
        if not os.path.isfile(meta_file):
            return triplets
        with open(meta_file) as f:
            for line in f:
                anchor, pos, neg, label = line.strip().split(',')
                triplets.append((anchor, pos, neg, label))
        return triplets

    def __getitem__(self, idx):
        anchor_p, pos_p, neg_p, label = self.triplets[idx]
        anchor = self._load_audio(anchor_p)
        positive = self._load_audio(pos_p)
        confuser = self._load_audio(neg_p)
        return anchor, positive, confuser, label

    def _load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if self.augmentor:
            wav = self.augmentor(wav)
        return self.transform(wav)

    def __len__(self):
        return len(self.triplets)
