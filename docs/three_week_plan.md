# DISENT-KWS v2 — Final 3-Week Sprint Plan

## Team: Sohini (Track A: Architecture) + Swarnim (Track B: Data/Training)

---

## Upgrades vs 2-Week Plan

| Change | Rationale |
|:---|:---|
| **Full VoxCeleb (7205 speakers)** | Better AAM-Softmax angular margins, stronger speaker generalization |
| **Mamba SSM first, Conv1D fallback** | Try the better architecture; 1 day to fail is now acceptable |
| **Pre-trained ECAPA-TDNN transfer** | Freeze first 2 SE-Res2Net blocks from SpeechBrain — free quality boost |
| **Mid-training evaluation (Day 12)** | Catch problems with 9 days to fix, not 2 |
| **Full ablation study (6 variants)** | Proves each component's contribution — makes submission credible |
| **Proper QAT + ONNX benchmarking** | Real INT8 latency numbers, not estimates |

---

## Infrastructure

```
Local Machine:    Code development, git, unit tests with dummy tensors
Kaggle Notebooks: Training + evaluation (free T4 GPU, VoxCeleb hosted)
GitHub:           Sync between local ↔ Kaggle via git clone
W&B:              Experiment tracking (free tier)
```

---

## Repo Structure

```
DISENT_KWS/
├── config.py                # Shared constants & tensor shape contract
├── models/                  # ← SOHINI
│   ├── bc_resnet.py         #   Shared encoder (520K)
│   ├── temporal.py          #   Mamba SSM or Dilated Conv1D fallback (120-180K)
│   ├── film.py              #   FiLM conditioning (~30K)
│   ├── heads.py             #   Phonetic + Speaker heads (620K + 580K)
│   ├── scorer.py            #   Dual-gate + EMA (30K)
│   └── disent_v2.py         #   Unified model class
├── data/                    # ← SWARNIM
│   ├── datasets.py          #   GSC, VoxCeleb, LibriPhrase loaders
│   ├── augmentations.py     #   MUSAN, RIR, SpecAugment, codec
│   └── synthetic.py         #   DSP enrollment augmentation
├── training/                # ← SWARNIM
│   ├── losses.py            #   AAM-Softmax, Prototypical, Rejection, KD
│   ├── disentangle.py       #   GRL autograd + CLUB MI estimator
│   └── scheduler.py         #   λ ramp, cosine annealing
├── eval/                    # ← SWARNIM
│   ├── benchmark.py         #   TA, FA, DET curves, per-SNR breakdown
│   ├── ablation.py          #   Ablation study runner
│   └── export.py            #   ONNX/TFLite export + latency profiling
├── enrollment/              # ← SHARED
│   └── enroll.py            #   Prototype extraction + quality filter
├── demo.py                  # ← SHARED
└── train.py                 # ← SHARED
```

---

## Day 0 (Together, 2-3 hrs): The Contract

### 1. Write `config.py`

```python
# config.py — THE CONTRACT. Do not change without both agreeing.

# Audio
SAMPLE_RATE = 16000
N_MELS = 80
WIN_LENGTH = 400      # 25ms
HOP_LENGTH = 160      # 10ms
MAX_AUDIO_SEC = 2.0
MAX_FRAMES = 200      # ~2s at 10ms hop

# Architecture
EMBED_DIM = 192
BC_CHANNELS = [16, 16, 32, 48]
TEMPORAL_CHANNELS = 48
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
CONFORMER_HEADS = 4
CONFORMER_CONV_K = 15
ECAPA_SCALE = 4
ECAPA_SE_RATIO = 4

# Training
BATCH_SIZE = 128
NUM_SPEAKERS_VOXCELEB = 7205  # full VoxCeleb
AAM_SCALE = 30
AAM_MARGIN = 0.2
PROTO_SCALE = 32
PROTO_MARGIN = 0.25
REJECTION_MARGIN = 0.4
GRL_MAX_LAMBDA = 1.0
CLUB_WEIGHT = 0.1
KD_TEMPERATURE = 4
KD_ALPHA = 0.7

# Scoring
SCORE_W_KW = 0.55
SCORE_W_SPK = 0.45
EMA_ALPHA = 0.7

# Augmentation
SPEC_FREQ_MASK = 15
SPEC_TIME_MASK = 25
SPEC_NUM_MASKS = 2
SNR_RANGE = (-5, 30)
ROOM_DIM_RANGE = (3, 10)
RT60_RANGE = (0.1, 1.0)
DISTANCE_RANGE = (0.5, 5.0)
```

### 2. Set up GitHub repo with folder structure
### 3. Create Kaggle notebook, verify VoxCeleb dataset is accessible
### 4. Install all dependencies locally

```bash
pip install torch torchaudio speechbrain
pip install mamba-ssm  # try it; okay if it fails
pip install pyroomacoustics torch-audiomentations
pip install onnxruntime openai-whisper wandb
pip install librosa soundfile matplotlib sounddevice
```

### 5. Swarnim: Submit VoxCeleb access request NOW (if not using Kaggle-hosted)

---

# WEEK 1: Independent Build (Days 1-7)

---

## Day 1-2

### Sohini: BC-ResNet-2 Shared Encoder (`models/bc_resnet.py`)

```python
class BCResBlock(nn.Module):
    """Broadcasted Residual Block — the core BC-ResNet innovation."""
    def __init__(self, channels, stride_freq=1):
        super().__init__()
        # Frequency branch: 2D conv on freq axis
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 1), 
                      stride=(stride_freq, 1), padding=(1, 0)),
            nn.BatchNorm2d(channels), nn.ReLU()
        )
        # Temporal branch: 1D conv (mean over freq, then broadcast back)
        self.time_conv = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU()
        )
    
    def forward(self, x):
        # x: (B, C, F, T)
        freq_out = self.freq_conv(x)
        # Average over frequency → (B, C, T) → 1D conv → broadcast back
        time_in = x.mean(dim=2)                    # (B, C, T)
        time_out = self.time_conv(time_in)          # (B, C, T)
        time_out = time_out.unsqueeze(2)            # (B, C, 1, T)
        return freq_out + time_out                  # broadcast add


class BCResNet2(nn.Module):
    """BC-ResNet-2: 6 BC-ResBlocks, ~520K params"""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16), nn.ReLU()
        )
        self.blocks = nn.Sequential(
            BCResBlock(16), BCResBlock(16),
            BCResBlock(16, stride_freq=2),  # downsample freq
            # Channel expansion 16→32
            nn.Conv2d(16, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            BCResBlock(32), BCResBlock(32, stride_freq=2),
            # Channel expansion 32→48
            nn.Conv2d(32, 48, 1), nn.BatchNorm2d(48), nn.ReLU(),
            BCResBlock(48), BCResBlock(48),
        )
    
    def forward(self, x):
        # x: (B, 80, T) → (B, 1, 80, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)      # (B, 16, 80, T)
        x = self.blocks(x)    # (B, 48, F', T)
        x = x.mean(dim=2)     # (B, 48, T) — average over remaining freq bins
        return x
```

**Verification:**
```python
m = BCResNet2()
out = m(torch.randn(4, 80, 200))
assert out.shape[0] == 4 and out.shape[1] == 48
print(f"BC-ResNet-2 params: {sum(p.numel() for p in m.parameters()):,}")
```

### Swarnim: Dataloaders (`data/datasets.py`)

```python
import torchaudio
from torch.utils.data import Dataset, DataLoader

class LFBETransform:
    """Shared feature extraction — used by ALL dataloaders."""
    def __init__(self):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160,
            n_mels=80, f_min=20, f_max=7600
        )
    
    def __call__(self, waveform):
        # Pad/trim to MAX_AUDIO_SEC
        target_len = int(16000 * 2.0)
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        else:
            waveform = waveform[..., :target_len]
        return torch.log(self.mel(waveform) + 1e-6).squeeze(0)  # (80, T)


class GSCDataset(Dataset):
    """Google Speech Commands v2 — 35 words, ~105K utterances"""
    def __init__(self, root, subset='training', transform=None, augmentor=None):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset=subset)
        self.transform = transform or LFBETransform()
        self.augmentor = augmentor
        self.labels = sorted(list(set(self.dataset._walker)))  # word labels
    
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
    def __init__(self, root, transform=None, augmentor=None, max_utts_per_spk=50):
        self.transform = transform or LFBETransform()
        self.augmentor = augmentor
        self.samples = self._scan_directory(root, max_utts_per_spk)
    
    def _scan_directory(self, root, max_per_spk):
        # Walk VoxCeleb directory: root/id_XXXXX/video_id/XXXXX.wav
        samples = []
        speakers = sorted(os.listdir(root))
        for spk_idx, spk_id in enumerate(speakers):
            spk_dir = os.path.join(root, spk_id)
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
    def __init__(self, root, split='hard', transform=None, augmentor=None):
        self.transform = transform or LFBETransform()
        self.augmentor = augmentor
        self.triplets = self._load_triplets(root, split)
    
    def _load_triplets(self, root, split):
        # Load pre-built triplet lists from LibriPhrase metadata
        # Each triplet: (anchor_path, positive_path, confuser_path, word_label)
        triplets = []
        meta_file = os.path.join(root, f'{split}_triplets.csv')
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
```

---

## Day 3

### Sohini: Temporal Block — Mamba First, Conv1D Fallback (`models/temporal.py`)

```python
# Try Mamba first
USE_MAMBA = False
try:
    from mamba_ssm import Mamba
    USE_MAMBA = True
    print("✅ Mamba SSM available")
except ImportError:
    print("⚠️ Mamba unavailable, using Dilated Conv1D fallback")


class MambaTemporalBlock(nn.Module):
    """Mamba SSM — O(T) selective state space, ~180K params"""
    def __init__(self, d_model=48, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state,
                          d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, C, T) → (B, T, C) for Mamba
        residual = x
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.transpose(1, 2)
        return x + residual


class DilatedConvTemporalBlock(nn.Module):
    """Dilated DW-Conv1D fallback — O(T), ~120K params, causal"""
    def __init__(self, channels=48):
        super().__init__()
        self.layers = nn.ModuleList([
            self._make_layer(channels, kernel=3, dilation=1),
            self._make_layer(channels, kernel=5, dilation=2),
            self._make_layer(channels, kernel=7, dilation=4),
        ])
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
    
    def _make_layer(self, ch, kernel, dilation):
        # Causal padding: pad only on the left
        pad = (kernel - 1) * dilation
        return nn.Sequential(
            nn.ConstantPad1d((pad, 0), 0),  # causal left-pad
            nn.Conv1d(ch, ch, kernel, dilation=dilation, groups=ch),
            nn.BatchNorm1d(ch), nn.SiLU()
        )
    
    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        x = self.pointwise(x)
        return self.norm(x + residual)


def get_temporal_block(channels=48):
    """Factory: returns best available temporal block."""
    if USE_MAMBA:
        return MambaTemporalBlock(d_model=channels)
    return DilatedConvTemporalBlock(channels=channels)
```

**Sohini spends max 4 hours on Mamba install. If it fails → DilatedConv. Move on.**

### Swarnim: Augmentation Pipeline (`data/augmentations.py`)

```python
import pyroomacoustics as pra
import random
import torchaudio

class RIRSimulator:
    """Simulate room impulse responses at target distances."""
    def __init__(self, room_range=(3,10), rt60_range=(0.1,1.0), dist_range=(0.5,5.0)):
        self.room_range = room_range
        self.rt60_range = rt60_range
        self.dist_range = dist_range
    
    def generate(self):
        dim = random.uniform(*self.room_range)
        rt60 = random.uniform(*self.rt60_range)
        dist = random.uniform(*self.dist_range)
        
        room = pra.ShoeBox([dim, dim, 2.5], fs=16000,
                           materials=pra.Material(
                               pra.inverse_sabine(rt60, [dim, dim, 2.5])))
        room.add_source([dim/2 + dist/2, dim/2, 1.5])
        room.add_microphone([dim/2 - dist/2, dim/2, 1.5])
        room.compute_rir()
        return torch.tensor(room.rir[0][0], dtype=torch.float32)


class AudioAugmentor:
    def __init__(self, musan_path=None):
        self.rir_sim = RIRSimulator()
        self.musan_noise = self._load_musan(musan_path) if musan_path else []
    
    def _load_musan(self, path):
        """Pre-load MUSAN noise segments into memory."""
        noises = []
        for f in glob.glob(os.path.join(path, 'noise/**/*.wav'), recursive=True):
            wav, sr = torchaudio.load(f)
            noises.append(wav)
        return noises
    
    def __call__(self, waveform):
        # 1. RIR convolution (p=0.4)
        if random.random() < 0.4:
            rir = self.rir_sim.generate()
            waveform = torchaudio.functional.fftconvolve(waveform, rir.unsqueeze(0))
            waveform = waveform[..., :waveform.shape[-1]]  # trim to original length
        
        # 2. Additive noise from MUSAN (p=0.7)
        if random.random() < 0.7 and self.musan_noise:
            snr_db = random.uniform(-5, 30)
            noise = random.choice(self.musan_noise)
            noise = self._match_length(noise, waveform.shape[-1])
            waveform = self._add_noise(waveform, noise, snr_db)
        
        # 3. Speed perturbation (p=0.3)
        if random.random() < 0.3:
            factor = random.choice([0.9, 0.95, 1.05, 1.1])
            waveform, _ = torchaudio.functional.speed(waveform, 16000, factor)
        
        # 4. Gain jitter (p=0.5)
        if random.random() < 0.5:
            gain_db = random.uniform(-6, 6)
            waveform = waveform * (10 ** (gain_db / 20))
        
        return waveform
    
    def _match_length(self, noise, target_len):
        if noise.shape[-1] >= target_len:
            start = random.randint(0, noise.shape[-1] - target_len)
            return noise[..., start:start+target_len]
        else:
            repeats = target_len // noise.shape[-1] + 1
            return noise.repeat(1, repeats)[..., :target_len]
    
    def _add_noise(self, signal, noise, snr_db):
        s_power = signal.pow(2).mean()
        n_power = noise.pow(2).mean()
        scale = (s_power / (n_power + 1e-8) * 10 ** (-snr_db / 10)).sqrt()
        return signal + scale * noise


class SpecAugment(nn.Module):
    def __init__(self, freq_mask=15, time_mask=25, num_masks=2):
        super().__init__()
        self.aug = nn.Sequential(*[
            torchaudio.transforms.FrequencyMasking(freq_mask)
            for _ in range(num_masks)
        ] + [
            torchaudio.transforms.TimeMasking(time_mask)
            for _ in range(num_masks)
        ])
    
    def forward(self, x):
        return self.aug(x)
```

---

## Day 4-5

### Sohini: Phonetic + Speaker Heads (`models/heads.py`)

```python
class AttentiveStatsPool(nn.Module):
    """Preserves temporal dynamics — NOT global average pooling."""
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.Tanh(),
            nn.Linear(in_dim, 1)
        )
    
    def forward(self, x):
        # x: (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)
        alpha = F.softmax(self.attn(x), dim=1)         # (B, T, 1)
        mu = (alpha * x).sum(dim=1)                     # (B, C)
        var = (alpha * (x - mu.unsqueeze(1))**2).sum(dim=1)
        sigma = torch.sqrt(var + 1e-6)                  # (B, C)
        return torch.cat([mu, sigma], dim=1)             # (B, 2C)


class CausalConformerBlock(nn.Module):
    """Single causal Conformer block."""
    def __init__(self, d_model=192, n_heads=4, conv_k=15):
        super().__init__()
        self.ff1 = nn.Sequential(nn.Linear(d_model, d_model*2), nn.SiLU(),
                                  nn.Linear(d_model*2, d_model), nn.Dropout(0.1))
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, conv_k, groups=d_model,
                      padding=conv_k-1),  # causal: trim right
            nn.BatchNorm1d(d_model), nn.SiLU()
        )
        self.ff2 = nn.Sequential(nn.Linear(d_model, d_model*2), nn.SiLU(),
                                  nn.Linear(d_model*2, d_model), nn.Dropout(0.1))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, C, T) → (B, T, C) for attention
        x = x.transpose(1, 2)
        
        # FFN1
        x = x + 0.5 * self.ff1(self.norm1(x))
        
        # Causal self-attention
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + attn_out
        
        # Depthwise conv (causal)
        conv_in = self.norm2(x).transpose(1, 2)        # (B, C, T)
        conv_out = self.conv(conv_in)[..., :T]          # trim future padding
        x = x + conv_out.transpose(1, 2)
        
        # FFN2
        x = x + 0.5 * self.ff2(self.norm3(x))
        return x.transpose(1, 2)                        # back to (B, C, T)


class SEDWRes2NetBlock(nn.Module):
    """SE + Depthwise-Separable Res2Net block for speaker head."""
    def __init__(self, channels=48, scale=4, se_ratio=4):
        super().__init__()
        width = channels // scale
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, 3, padding=1, groups=width),
                nn.Conv1d(width, width, 1),
                nn.BatchNorm1d(width), nn.ReLU()
            ) for _ in range(scale - 1)
        ])
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // se_ratio), nn.ReLU(),
            nn.Linear(channels // se_ratio, channels), nn.Sigmoid()
        )
        self.scale = scale
        self.width = width
    
    def forward(self, x):
        residual = x
        splits = x.chunk(self.scale, dim=1)
        outputs = [splits[0]]
        for i, conv in enumerate(self.convs):
            inp = splits[i+1] + outputs[-1] if i > 0 else splits[i+1]
            outputs.append(conv(inp))
        x = torch.cat(outputs, dim=1)
        se_weight = self.se(x).unsqueeze(2)
        return residual + x * se_weight


class PhoneticHead(nn.Module):
    """Conformer-based phonetic embedding head, ~620K params"""
    def __init__(self, in_ch=48, embed_dim=192):
        super().__init__()
        self.film = FiLM(cond_dim=384, channels=in_ch)
        self.proj_up = nn.Conv1d(in_ch, embed_dim, 1)
        self.conformer1 = CausalConformerBlock(embed_dim)
        self.conformer2 = CausalConformerBlock(embed_dim)
        self.pool = AttentiveStatsPool(embed_dim)
        self.proj_out = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, x, cond=None):
        if cond is not None:
            x = self.film(x, cond)
        x = self.proj_up(x)
        x = self.conformer1(x)
        x = self.conformer2(x)
        x = self.pool(x)
        return self.proj_out(x)


class SpeakerHead(nn.Module):
    """ECAPA-TDNNLite speaker embedding head, ~580K params"""
    def __init__(self, in_ch=48, embed_dim=192):
        super().__init__()
        self.film = FiLM(cond_dim=384, channels=in_ch)
        self.blocks = nn.Sequential(
            SEDWRes2NetBlock(in_ch), SEDWRes2NetBlock(in_ch), SEDWRes2NetBlock(in_ch)
        )
        self.pool = AttentiveStatsPool(in_ch)
        self.proj = nn.Linear(in_ch * 2, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
    
    def forward(self, x, cond=None):
        if cond is not None:
            x = self.film(x, cond)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.proj(x)
        return self.bn(x)
```

### Swarnim: Loss Functions + Disentanglement (`training/losses.py`, `training/disentangle.py`)

```python
# --- training/disentangle.py ---

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class AdversarialHead(nn.Module):
    """Adversarial classifier attached via GRL."""
    def __init__(self, embed_dim=192, n_classes=35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 96), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(96, n_classes)
        )
    
    def forward(self, x, lambda_):
        x_rev = grad_reverse(x, lambda_)
        return self.net(x_rev)


class CLUB(nn.Module):
    """Contrastive Log-ratio Upper Bound on MI(z_spk; z_phn)."""
    def __init__(self, dim=192):
        super().__init__()
        self.mu_net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.logvar_net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    
    def forward(self, z_spk, z_phn):
        mu = self.mu_net(z_spk)
        logvar = self.logvar_net(z_spk)
        
        pos = -(mu - z_phn)**2 / (2 * logvar.exp() + 1e-6) - 0.5 * logvar
        
        z_phn_shuffle = z_phn[torch.randperm(z_phn.size(0))]
        neg = -(mu - z_phn_shuffle)**2 / (2 * logvar.exp() + 1e-6) - 0.5 * logvar
        
        mi_upper = (pos.sum(-1).mean() - neg.sum(-1).mean())
        return mi_upper


class DisentanglementLoss(nn.Module):
    """Combined GRL adversarial + CLUB MI loss."""
    def __init__(self, embed_dim=192, n_spk=7205, n_phn=35):
        super().__init__()
        self.adv_spk = AdversarialHead(embed_dim, n_spk)  # on z_phn
        self.adv_phn = AdversarialHead(embed_dim, n_phn)   # on z_spk
        self.club = CLUB(embed_dim)
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, z_phn, z_spk, spk_labels, phn_labels, lambda_):
        # Adversarial: force z_phn to NOT encode speaker
        spk_pred = self.adv_spk(z_phn, lambda_)
        loss_adv_spk = self.ce(spk_pred, spk_labels)
        
        # Adversarial: force z_spk to NOT encode phonemes
        phn_pred = self.adv_phn(z_spk, lambda_)
        loss_adv_phn = self.ce(phn_pred, phn_labels)
        
        # MI minimization
        loss_mi = self.club(z_spk.detach(), z_phn)
        
        return loss_adv_spk + loss_adv_phn + 0.1 * loss_mi


# --- training/losses.py ---

class AAMSoftmax(nn.Module):
    def __init__(self, in_dim=192, n_classes=7205, scale=30, margin=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = scale, margin
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, x, labels):
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(x, dim=1)
        cosine = F.linear(x, w)
        theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        logits = self.s * (torch.cos(theta + self.m * one_hot))
        return self.ce(logits, labels)


class PrototypicalLoss(nn.Module):
    def __init__(self, scale=32, margin=0.25):
        super().__init__()
        self.s, self.m = scale, margin
    
    def forward(self, anchor, positive, negatives):
        # anchor, positive: (B, D); negatives: (B, N, D)
        sim_pos = F.cosine_similarity(anchor, positive)
        sim_neg = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=2)
        logits = torch.cat([
            (self.s * (sim_pos - self.m)).unsqueeze(1),
            self.s * sim_neg
        ], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


def rejection_loss(anchor, positive, confuser, margin=0.4):
    sim_pos = F.cosine_similarity(anchor, positive)
    sim_neg = F.cosine_similarity(anchor, confuser)
    return F.relu(sim_neg - sim_pos + margin).mean()


class KDLoss(nn.Module):
    def __init__(self, temperature=4):
        super().__init__()
        self.T = temperature
    
    def forward(self, student_logits, teacher_logits):
        s = F.log_softmax(student_logits / self.T, dim=1)
        t = F.softmax(teacher_logits / self.T, dim=1)
        return self.T * self.T * F.kl_div(s, t, reduction='batchmean')


# --- training/scheduler.py ---

def grl_lambda_schedule(epoch, max_epochs):
    """Sigmoid ramp-up for GRL lambda."""
    p = epoch / max_epochs
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
```

---

## Day 5-6

### Sohini: Unified Model + Scorer (`models/disent_v2.py`, `models/scorer.py`)

```python
# --- models/scorer.py ---

class DualGateScorer(nn.Module):
    def __init__(self, embed_dim=192, w_kw=0.55, w_spk=0.45, ema_alpha=0.7):
        super().__init__()
        self.w_kw = nn.Parameter(torch.tensor(w_kw))
        self.w_spk = nn.Parameter(torch.tensor(w_spk))
        self.ema_alpha = ema_alpha
        self._ema_state = 0.0
    
    def forward(self, z_phn, z_spk, p_kw, p_spk):
        sim_kw = F.cosine_similarity(z_phn, p_kw)
        sim_spk = F.cosine_similarity(z_spk, p_spk)
        score = self.w_kw * sim_kw + self.w_spk * sim_spk
        return score, sim_kw, sim_spk
    
    def detect_streaming(self, score):
        """Apply EMA for streaming stability."""
        self._ema_state = self.ema_alpha * score + (1 - self.ema_alpha) * self._ema_state
        return self._ema_state
    
    def reset(self):
        self._ema_state = 0.0


# --- models/disent_v2.py ---

class DISENT_KWS_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BCResNet2()
        self.temporal = get_temporal_block(48)
        self.phn_head = PhoneticHead(48, 192)
        self.spk_head = SpeakerHead(48, 192)
        self.scorer = DualGateScorer()
    
    def forward(self, audio, p_spk=None, p_kw=None):
        """
        audio:  (B, 80, T) or (B, 1, 80, T)
        p_spk:  (B, 192) or None — enrolled speaker prototype
        p_kw:   (B, 192) or None — enrolled keyword prototype
        Returns: z_phn (B, 192), z_spk (B, 192)
        """
        h = self.encoder(audio)      # (B, 48, T')
        h = self.temporal(h)          # (B, 48, T')
        
        cond = None
        if p_spk is not None and p_kw is not None:
            cond = torch.cat([p_spk, p_kw], dim=-1)  # (B, 384)
        
        z_phn = self.phn_head(h, cond)
        z_spk = self.spk_head(h, cond)
        return z_phn, z_spk
    
    @torch.no_grad()
    def detect(self, audio, p_kw, p_spk):
        z_phn, z_spk = self.forward(audio, p_spk, p_kw)
        score, sim_kw, sim_spk = self.scorer(z_phn, z_spk, p_kw, p_spk)
        return score, sim_kw, sim_spk
    
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Total: {total:,} params ({total/1e6:.2f}M)")
        assert total < 3_000_000, f"OVER BUDGET: {total:,}"
        return total
```

### Swarnim: DSP Enrollment Augmentation (`data/synthetic.py`)

```python
def augment_enrollment(waveforms, n_total=30):
    """5 real → 30 total using DSP only. No XTTS needed."""
    variants = list(waveforms)
    
    pitch_shifts = [-2, -1, 1, 2]
    speed_factors = [0.9, 0.95, 1.05, 1.1]
    gain_db_options = [-3, -1.5, 1.5, 3]
    
    for wav in waveforms:
        for ps in pitch_shifts:
            variants.append(torchaudio.functional.pitch_shift(wav, 16000, ps))
    
    while len(variants) < n_total:
        base = random.choice(waveforms)
        ps = random.choice(pitch_shifts)
        sf = random.choice(speed_factors)
        gdb = random.choice(gain_db_options)
        aug = torchaudio.functional.pitch_shift(base, 16000, ps)
        aug, _ = torchaudio.functional.speed(aug, 16000, sf)
        target_len = base.shape[-1]
        aug = F.interpolate(aug.unsqueeze(0).float(), size=target_len).squeeze(0)
        aug = aug * (10 ** (gdb / 20))
        variants.append(aug)
    
    return variants[:n_total]
```

---

## Day 7 (Together, 3 hrs): 🔴 INTEGRATION CHECKPOINT

```python
# integration_test.py — RUN THIS TOGETHER ON DAY 7

from models.disent_v2 import DISENT_KWS_v2
from training.losses import AAMSoftmax, PrototypicalLoss, rejection_loss
from training.disentangle import DisentanglementLoss
from training.scheduler import grl_lambda_schedule

# 1. Model sanity
model = DISENT_KWS_v2()
model.count_params()  # must print < 3M

# 2. Forward pass
B = 8
audio = torch.randn(B, 80, 200)
z_phn, z_spk = model(audio)
assert z_phn.shape == (B, 192)
assert z_spk.shape == (B, 192)
print("✅ Forward pass OK")

# 3. Losses
aam = AAMSoftmax(192, 7205)
disent = DisentanglementLoss(192, 7205, 35)
spk_labels = torch.randint(0, 7205, (B,))
phn_labels = torch.randint(0, 35, (B,))

loss_spk = aam(z_spk, spk_labels)
loss_disent = disent(z_phn, z_spk, spk_labels, phn_labels, lambda_=0.5)
loss_reject = rejection_loss(z_phn, torch.randn(B, 192), torch.randn(B, 192))

total = loss_spk + 0.5 * loss_disent + 0.3 * loss_reject
total.backward()
print(f"✅ Backward pass OK, total loss: {total.item():.4f}")

# 4. Optimizer step
optimizer = torch.optim.AdamW(list(model.parameters()) + list(aam.parameters())
                               + list(disent.parameters()), lr=3e-4)
optimizer.step()
print("✅ Optimizer step OK")

# 5. Scorer
p_kw = torch.randn(1, 192)
p_spk = torch.randn(1, 192)
score, sim_kw, sim_spk = model.detect(audio[:1], p_kw, p_spk)
print(f"✅ Detection OK, score: {score.item():.4f}")

print("\n🎉 ALL INTEGRATION TESTS PASSED")
```

**If ANY test fails: fix it before Week 2. Do not proceed to training with broken integration.**

---

# WEEK 2: Train + Evaluate (Days 8-14)

---

## Day 8-10: Training on Kaggle

### Kaggle Notebook Setup

```python
# Cell 1: Clone repo
!git clone https://github.com/yourrepo/DISENT_KWS.git
%cd DISENT_KWS
!pip install -q mamba-ssm speechbrain pyroomacoustics wandb

# Cell 2: Load pre-trained ECAPA-TDNN for transfer learning
from speechbrain.pretrained import EncoderClassifier
teacher_spk = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_ecapa"
)
```

### Phase 1: Pre-training (20 epochs, ~5 hrs on T4)

```python
# Separate pre-training of each head
# Encoder + Phonetic head → GSC-v2 (keyword classification)
# Encoder + Speaker head → VoxCeleb (speaker classification)
# Loss: AAM-Softmax only, NO disentanglement
# lr=3e-4, AdamW, cosine annealing with 3-epoch warmup
# Save checkpoint every 5 epochs
```

### Phase 2: Joint Fine-tuning (20 epochs, ~8 hrs on T4)

```python
# All modules, all losses, disentanglement enabled
# Total loss:
#   L = L_kw + L_spk + 0.5*L_disent + 0.3*L_reject + 0.7*L_kd
# GRL λ ramp: grl_lambda_schedule(epoch, 20)
# lr=1e-4, AdamW, cosine annealing
# Hard negative mining from LibriPhrase-Hard
# Save checkpoint every 5 epochs + best by val loss
```

**Kaggle session strategy:**
```
Session 1 (~5hrs):  Phase 1 (20 epochs) → save checkpoint
Session 2 (~8hrs):  Load checkpoint → Phase 2 (20 epochs) → save final
```

### SpeechBrain ECAPA-TDNN Transfer (during Phase 1)

```python
# Load pre-trained weights into speaker head's first 2 blocks
# Freeze them — only train block 3 + pooling + projection
def load_pretrained_speaker_head(model, teacher):
    # Copy weights from teacher's first 2 SE-Res2Net blocks
    for i in range(2):
        model.spk_head.blocks[i].load_state_dict(
            extract_block_weights(teacher, i), strict=False
        )
        for param in model.spk_head.blocks[i].parameters():
            param.requires_grad = False
    print("✅ Loaded & froze first 2 ECAPA-TDNN blocks from SpeechBrain")
```

---

## Day 11-12: Evaluation (Swarnim) + Optimization (Sohini)

### Swarnim: Evaluation (`eval/benchmark.py`)

```python
def full_evaluation(model, test_data, musan_path, threshold):
    results = {}
    
    # 1. TA Clean
    results['ta_clean'] = compute_ta(model, test_data, noise=None)
    
    # 2. TA Noisy (per SNR)
    for snr in [-5, 0, 5, 10, 20, 30]:
        results[f'ta_snr_{snr}dB'] = compute_ta(model, test_data, snr_db=snr,
                                                  musan_path=musan_path)
    
    # 3. FA per hour
    results['fa_per_hr'] = compute_fa_rate(model, background_audio_path,
                                            duration_hrs=1.0, threshold=threshold)
    
    # 4. DET curve + optimal threshold
    fas, frs, thresholds = compute_det_curve(model, test_data)
    results['optimal_threshold'] = find_threshold_for_target_fa(fas, frs, thresholds,
                                                                 target_fa=1.0)
    # 5. Parameter count + latency
    results['params'] = sum(p.numel() for p in model.parameters())
    results['latency_ms'] = measure_latency(model, input_shape=(1, 80, 200), n_runs=100)
    
    return results
```

### Sohini: QAT + ONNX Export (`eval/export.py`)

```python
def quantize_and_export(model, dummy_input, save_dir):
    # 1. Quantization-Aware Training config
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    model_prep = torch.ao.quantization.prepare_qat(model.train())
    # ... fine-tune 5 more epochs with QAT ...
    model_int8 = torch.ao.quantization.convert(model_prep.eval())
    
    # 2. ONNX Export
    torch.onnx.export(model_int8, dummy_input,
                       os.path.join(save_dir, "disent_kws_v2.onnx"),
                       opset_version=17,
                       input_names=['audio'],
                       output_names=['z_phn', 'z_spk'],
                       dynamic_axes={'audio': {0: 'batch', 2: 'time'}})
    
    # 3. Verify ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(save_dir, "disent_kws_v2.onnx"))
    ort_out = sess.run(None, {'audio': dummy_input.numpy()})
    torch_out = model_int8(dummy_input)
    diff = abs(ort_out[0] - torch_out[0].detach().numpy()).max()
    print(f"Max diff PyTorch vs ONNX: {diff:.6f}")
    assert diff < 0.01, "ONNX export verification failed!"
    
    # 4. Measure ONNX latency
    import time
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {'audio': dummy_input.numpy()})
        times.append((time.perf_counter() - t0) * 1000)
    print(f"ONNX latency: {sum(times)/len(times):.1f}ms avg")
```

---

## Day 12 (Together): 🔴 MID-TRAINING EVALUATION

**This is the critical checkpoint with 9 days remaining to fix issues.**

Run `full_evaluation()` on the Phase 2 checkpoint. Check:

| Metric | Target | If below target → Action |
|:---|:---:|:---|
| TA Clean | ≥ 99% | Increase Phase 2 epochs to 30. Check if prototypical loss is working. |
| TA Noisy (0dB) | ≥ 90% | Increase noise aug probability to p=0.9. Add more MUSAN categories. |
| TA Noisy (-5dB) | ≥ 90% | This is the hardest. Try matched-condition fine-tuning: 5 epochs at SNR=-5 only. |
| FA/hr | < 1 | Lower threshold τ. If still high, add 3-consecutive-window requirement. |
| Params | < 3M | Should be ~1.99M. If over, prune channels. |

---

# WEEK 3: Ablation + Demo + Ship (Days 15-21)

---

## Day 15-16 (Swarnim): Ablation Study (`eval/ablation.py`)

Train 6 variant models (can run in parallel on Kaggle if needed):

```python
ABLATION_CONFIGS = {
    "full":           {},                                    # baseline
    "no_disent":      {"disable_grl": True, "disable_club": True},
    "no_film":        {"disable_film": True},
    "no_spk_gate":    {"disable_speaker_head": True},       # KWS only
    "no_augmentation":{"disable_augmentation": True},
    "no_kd":          {"disable_kd": True},
}

# For each config, train Phase 2 for 10 epochs (shorter) and evaluate
# Record: TA_clean, TA_noisy, FA/hr for each
```

**Expected results table:**

```
| Variant                | TA Clean | TA Noisy | FA/hr |
|:-----------------------|:--------:|:--------:|:-----:|
| Full DISENT-KWS v2     |  99.3%  |  94.1%  |  0.25 |
| − Disentanglement      |  98.1%  |  89.2%  |  1.2  |
| − FiLM conditioning    |  99.0%  |  92.5%  |  0.40 |
| − Speaker gate (KWS)   |  99.4%  |  94.5%  |  8.5  |
| − Noise augmentation   |  99.5%  |  72.3%  |  0.4  |
| − Knowledge distill.   |  98.5%  |  90.5%  |  0.45 |
```

## Day 15-16 (Sohini): Final QAT + Optimization

- Run full QAT (5 epochs on Phase 2 checkpoint)
- Structured pruning: 15% channels → 5 epoch recovery
- ONNX + TFLite export
- Latency profiling on CPU / available hardware

## Day 17-18 (Together): Enrollment + Demo

### Enrollment Pipeline (`enrollment/enroll.py`)

```python
def enroll_user(model, keyword_audio_paths, threshold_bg_audio=None):
    """Full enrollment: record → augment → extract → calibrate"""
    
    # 1. Load real recordings
    waveforms = [torchaudio.load(p)[0] for p in keyword_audio_paths]
    assert len(waveforms) >= 5, "Need at least 5 enrollment samples"
    
    # 2. DSP augment to 30 variants
    augmented = augment_enrollment(waveforms, n_total=30)
    
    # 3. Extract prototypes
    transform = LFBETransform()
    with torch.no_grad():
        embeddings = [model(transform(w).unsqueeze(0)) for w in augmented]
        z_phns = torch.stack([e[0].squeeze() for e in embeddings])
        z_spks = torch.stack([e[1].squeeze() for e in embeddings])
    
    p_kw = F.normalize(z_phns.mean(dim=0, keepdim=True), dim=-1)
    p_spk = F.normalize(z_spks.mean(dim=0, keepdim=True), dim=-1)
    
    # 4. Calibrate threshold (optional)
    threshold = 0.5  # default
    if threshold_bg_audio:
        threshold = calibrate_threshold(model, p_kw, p_spk,
                                         threshold_bg_audio, target_fa=1.0)
    
    return {"p_kw": p_kw, "p_spk": p_spk, "threshold": threshold}
```

### Demo Script (`demo.py`)

```python
import sounddevice as sd
import threading
import queue

class RealTimeDetector:
    def __init__(self, model, enrollment, device='cpu'):
        self.model = model.to(device).eval()
        self.p_kw = enrollment['p_kw'].to(device)
        self.p_spk = enrollment['p_spk'].to(device)
        self.threshold = enrollment['threshold']
        self.transform = LFBETransform()
        self.buffer = torch.zeros(1, 16000)  # 1 sec rolling buffer
        self.scorer = model.scorer
        self.scorer.reset()
    
    def audio_callback(self, indata, frames, time_info, status):
        chunk = torch.tensor(indata[:, 0]).unsqueeze(0)
        
        # Roll buffer: drop oldest, append newest
        self.buffer = torch.cat([self.buffer[:, frames:], chunk], dim=1)
        
        # Extract features from buffer
        features = self.transform(self.buffer).unsqueeze(0)
        
        # Detect
        with torch.no_grad():
            score, sim_kw, sim_spk = self.model.detect(features, self.p_kw, self.p_spk)
            smooth_score = self.scorer.detect_streaming(score.item())
        
        # Display
        bar = "█" * int(smooth_score * 50)
        status_icon = "🎯 DETECTED!" if smooth_score >= self.threshold else ""
        print(f"\rScore: [{bar:<50}] {smooth_score:.3f} {status_icon}", end="", flush=True)
    
    def run(self):
        print(f"🎤 Listening... (threshold={self.threshold:.3f})")
        print("Press Ctrl+C to stop\n")
        with sd.InputStream(callback=self.audio_callback, channels=1,
                             samplerate=16000, blocksize=2560):
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                print("\n\nStopped.")
```

## Day 19-20 (Together): Demo Scenarios + Results

**Record 5 demo scenarios:**

| # | Scenario | Expected | Actually |
|:-:|:---|:---:|:---:|
| 1 | Clean, target speaker, correct keyword | ✅ ACCEPT | ___ |
| 2 | Clean, wrong speaker, correct keyword | ❌ REJECT | ___ |
| 3 | Babble noise (10dB), target speaker, correct keyword | ✅ ACCEPT | ___ |
| 4 | Clean, target speaker, confuser word | ❌ REJECT | ___ |
| 5 | 60s background noise, no keyword | 0 triggers | ___ |

**Generate final KPI table:**

```markdown
| Metric              | Target  | Achieved | Status |
|:---------------------|:-------:|:--------:|:------:|
| TA (Clean)           | ≥ 99%  |   ___%   |  ✅/❌  |
| TA (Noisy, SNR≥0dB)  | ≥ 90%  |   ___%   |  ✅/❌  |
| TA (Noisy, SNR=-5dB) | ≥ 90%  |   ___%   |  ✅/❌  |
| FA                   | < 1/hr |   ___/hr |  ✅/❌  |
| Parameters           | < 3M   |  1.99M   |   ✅   |
| xRT                  | < 0.2s |   ___s   |  ✅/❌  |
| Model Size (INT8)    |   —    |   ___MB  |   —    |
```

## Day 21: Buffer + Package

- Fix any remaining issues
- Write README.md with:
  - Setup instructions
  - How to enroll a new user
  - How to run demo
  - Benchmark results
  - Architecture diagram
- Package deliverables:
  - `model_final.pt` (PyTorch checkpoint)
  - `model_final.onnx` (ONNX INT8)
  - `ablation_results.csv`
  - `demo.py` (working real-time demo)
  - All source code

---

## Critical Milestones Summary

| Day | Milestone | Owner | Hard deadline? |
|:---:|:---|:---|:---:|
| 0 | Repo + config.py + Kaggle setup | Both | ✅ |
| 3 | BC-ResNet + Temporal forward pass | Sohini | |
| 3 | Dataloaders output correct shapes | Swarnim | |
| 5 | Full model forward with dummy data | Sohini | |
| 5 | All losses backward with dummy data | Swarnim | |
| **7** | **Integration: 1 batch end-to-end** | **Both** | **🔴 YES** |
| 10 | Phase 1 pre-training complete | Both | |
| **12** | **Mid-training eval — identify gaps** | **Both** | **🔴 YES** |
| 14 | Phase 2 training complete | Both | |
| 16 | Ablation study complete | Swarnim | |
| 16 | QAT + ONNX export done | Sohini | |
| 18 | Demo working end-to-end | Both | |
| 20 | All results recorded, README done | Both | |
| **21** | **Submission ready** | **Both** | **🔴 YES** |

---

## Fallback Strategies

| Risk | Trigger | Fallback |
|:---|:---|:---|
| Mamba CUDA fails | Import error or compile error | Use DilatedConvTemporalBlock (already built) |
| Training doesn't converge | Loss plateau after 10 epochs | Disable L_disent, train L_kw + L_spk only first |
| Kaggle session timeout | 12hr limit hit | Resume from checkpoint (saved every 5 epochs) |
| CLUB causes NaN | NaN in loss | Set CLUB_WEIGHT=0, keep GRL only |
| VoxCeleb not on Kaggle | Dataset not found | Use SpeechBrain pre-trained ECAPA-TDNN (freeze entire speaker head) |
| TA noisy < 90% at Day 12 | Below target | Increase aug p=0.9, add matched-SNR fine-tuning for 5 epochs |
| FA > 1/hr at Day 12 | Above target | Lower τ, add 3-consecutive-window detection requirement |
| Conformer too slow | xRT > 0.2s | Replace with 2× DilatedConv blocks (same I/O shapes) |
| LibriPhrase hard to get | Download/format issues | Generate confusers synthetically from GSC using phoneme edit distance |
