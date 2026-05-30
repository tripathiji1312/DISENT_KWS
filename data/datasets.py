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


import os
import torch
import torchaudio
from torch.utils.data import Dataset

# GSC v2: 35 classes (all words in the dataset)
GSC_CLASSES = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"
]
GSC_LABEL_MAP = {word: i for i, word in enumerate(GSC_CLASSES)}


class GSCDataset(Dataset):
    """
    Google Speech Commands v2 — reads directly from pre-extracted folder.
    Works regardless of whether Kaggle input has the torchaudio subfolder
    layout or a flat word-folder layout.
    """

    def __init__(self, root: str, subset: str = "training",
                 augmentor=None, transform=None):
        super().__init__()
        self.root      = self._resolve_root(root)
        self.augmentor = augmentor
        self.transform = transform or LFBETransform()
        self.samples   = []   # list of (wav_path, label_int)

        # Load the official split file if present, else do 80/10/10 split
        split_file = os.path.join(self.root, f"{subset}_list.txt")  # validation_list.txt etc.
        val_file   = os.path.join(self.root, "validation_list.txt")
        test_file  = os.path.join(self.root, "testing_list.txt")

        val_set  = self._read_list(val_file)
        test_set = self._read_list(test_file)

        for word in GSC_CLASSES:
            word_dir = os.path.join(self.root, word)
            if not os.path.isdir(word_dir):
                continue
            label = GSC_LABEL_MAP[word]
            for fname in os.listdir(word_dir):
                if not fname.endswith(".wav"):
                    continue
                rel = f"{word}/{fname}"
                if subset == "validation" and rel not in val_set:
                    continue
                if subset == "testing" and rel not in test_set:
                    continue
                if subset == "training" and (rel in val_set or rel in test_set):
                    continue
                self.samples.append((os.path.join(word_dir, fname), label))

        print(f"  GSCDataset [{subset}]: {len(self.samples):,} samples "
              f"from {self.root}")

    @staticmethod
    def _resolve_root(root: str) -> str:
        """Handle both flat and torchaudio-nested layouts."""
        for candidate in [
            root,
            os.path.join(root, "speech_commands_v0.02"),
            os.path.join(root, "SpeechCommands", "speech_commands_v0.02"),
        ]:
            if os.path.isdir(candidate) and os.path.isdir(
                    os.path.join(candidate, "yes")):   # sanity-check a word folder
                return candidate
        raise FileNotFoundError(
            f"Could not locate GSC word folders under {root}. "
            f"Run the diagnostic cell to inspect the structure."
        )

    @staticmethod
    def _read_list(path: str) -> set:
        if not os.path.exists(path):
            return set()
        with open(path) as f:
            return {line.strip() for line in f}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if self.augmentor is not None:
            waveform = self.augmentor(waveform)
        feat = self.transform(waveform)   # (80, T)
        return feat, label

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
