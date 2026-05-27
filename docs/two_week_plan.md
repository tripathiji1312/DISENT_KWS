# DISENT-KWS v2 — Final 2-Week Sprint Plan (Updated)

## Key Changes from Previous Plan

| Change | Why |
|:---|:---|
| **Kaggle Notebooks** for training | VoxCeleb hosted natively, free T4 GPU, zero download |
| **1500 speakers** (not 500) | 500 is too few for AAM-Softmax angular margin learning |
| **Dilated DW-Conv1D** replaces Mamba | O(T) linear time, no CUDA build issues, architecturally consistent with BC-ResNet |
| **DSP augmentation** replaces XTTS | Pitch shift + speed perturb on 5 real samples → 25 variants in seconds |
| **Training starts Day 5** | Don't wait for Day 8 — start pre-training as soon as dataloaders work |
| **Checkpoint every 5 epochs** | Kaggle has 12hr session limits — must survive restarts |

---

## Infrastructure Setup

```
Local Machine:  Code development, git, debugging with dummy tensors
Kaggle:         Training + evaluation (T4 GPU, VoxCeleb hosted)
GitHub:         Sync between local and Kaggle via git clone
```

**Kaggle datasets to add to notebook:**
- `mozillaorg/common-voice` (optional, for accent diversity)
- `flozi00/voxceleb-2` or equivalent hosted VoxCeleb
- Google Speech Commands: download via torchaudio in notebook

---

## Updated Architecture: Dilated Conv1D replaces Mamba

```
                    PARAMETER BUDGET (REVISED)
┌────────────────────────────────────────────────────┐
│ Shared Encoder (BC-ResNet-2):         520K         │
│ Temporal Block (Dilated DW-Conv1D):   120K  (was 180K Mamba) │
│ Phonetic Head (Conformer + FiLM):     620K         │
│ Speaker Head (ECAPA-Lite + FiLM):     580K         │
│ Disentanglement (GRL + CLUB):         120K         │
│ Scorer + Projection:                   30K         │
│ ─────────────────────────────────────────          │
│ TOTAL:                               1.99M ✅      │
└────────────────────────────────────────────────────┘
```

**Dilated DW-Conv1D Temporal Block** (replaces Mamba):

```python
class TemporalBlock(nn.Module):
    """O(T) causal temporal modeling — Mamba-free fallback"""
    def __init__(self, channels=48):
        super().__init__()
        self.layers = nn.Sequential(
            # Layer 1: local context
            nn.Conv1d(channels, channels, kernel_size=3, dilation=1,
                      padding=2, groups=channels),  # causal: pad left only
            nn.BatchNorm1d(channels), nn.SiLU(),
            # Layer 2: medium context
            nn.Conv1d(channels, channels, kernel_size=5, dilation=2,
                      padding=8, groups=channels),
            nn.BatchNorm1d(channels), nn.SiLU(),
            # Layer 3: wide context (~60 frames = 600ms)
            nn.Conv1d(channels, channels, kernel_size=7, dilation=4,
                      padding=24, groups=channels),
            nn.BatchNorm1d(channels), nn.SiLU(),
            # Pointwise mix
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)  # residual connection
```

---

## Revised Day-by-Day Schedule

### Day 0 (Together, 2 hrs): Setup

**Both:**
- Create GitHub repo with folder structure
- Write `config.py` (all shared tensor shapes and hyperparams)
- Set up Kaggle notebook, verify VoxCeleb dataset is accessible
- `pip install` all deps locally for development

```python
# config.py — the contract
SAMPLE_RATE = 16000
N_MELS = 80
EMBED_DIM = 192
BATCH_SIZE = 128
MAX_AUDIO_SEC = 2.0
BC_CHANNELS = [16, 16, 32, 48]
NUM_SPEAKERS_TRAIN = 1500  # VoxCeleb subsample
AAM_SCALE = 30
AAM_MARGIN = 0.2
EMA_ALPHA = 0.7
SCORE_W_KW = 0.55
SCORE_W_SPK = 0.45
```

---

### WEEK 1: Build

---

#### Day 1-2 — Sohini: BC-ResNet-2 (`models/bc_resnet.py`)

Build the Broadcasted Residual Network shared encoder:

```
Input: (B, 1, 80, T)

Conv2D(1, 16, 5×5, stride=1, pad=2) + BN + ReLU

BC-ResBlock(16→16) × 2:
  ├── freq_branch: Conv2D(C, C, (3,1), pad=(1,0))  [frequency axis]
  ├── time_branch: Conv1D(C, C, 3, pad=1)            [time axis, mean over freq]
  └── output = freq_branch + broadcast(time_branch)   [broadcast add]

BC-ResBlock(16→32) × 2:  [stride 2 on freq axis for downsampling]
BC-ResBlock(32→48) × 2:  [stride 2 on freq axis]

Output: (B, 48, T')
```

**Verification:**
```python
model = BCResNet2()
x = torch.randn(4, 1, 80, 200)
out = model(x)
assert out.shape == (4, 48, 50)  # freq downsampled, time preserved
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")  # ~520K
```

#### Day 1-2 — Swarnim: Dataloaders (`data/datasets.py`)

**Google Speech Commands v2:**
```python
# torchaudio.datasets.SPEECHCOMMANDS — auto-downloads
# 35 words, ~105K utterances
# Output: (waveform, sample_rate, label, speaker_id, utterance_id)
# → Transform to LFBE: (B, 80, T)
```

**VoxCeleb (Kaggle-hosted):**
```python
# Subsample 1500 speakers randomly (seed=42 for reproducibility)
# For each speaker, load up to 50 utterances
# Output: (B, 80, T) + speaker_label
```

**LibriPhrase:**
```python
# HuggingFace datasets or manual download
# Hard split: anchor + positive (same word) + negative (confuser, edit dist ≤ 2)
# Easy split: anchor + positive + negative (random different word)
# Output: triplets of (B, 80, T)
```

All dataloaders return LFBE features via:
```python
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160,
    n_mels=80, f_min=20, f_max=7600
)
features = torch.log(mel_transform(waveform) + 1e-6)
```

---

#### Day 3 — Sohini: Temporal Block + FiLM (`models/mamba_block.py`, `models/film.py`)

**Temporal Block:** Dilated DW-Conv1D as shown above (~120K params)

**FiLM Conditioning:**
```python
class FiLM(nn.Module):
    def __init__(self, cond_dim=384, channels=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.ReLU(),
            nn.Linear(128, channels * 2)  # γ and β
        )
    
    def forward(self, x, cond):
        # x: (B, C, T), cond: (B, 384)
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(2)  # (B, C, 1)
        beta = beta.unsqueeze(2)
        return (1 + gamma) * x + beta
```

Verify: input `(B, 48, T')` + cond `(B, 384)` → output `(B, 48, T')`

#### Day 3-4 — Swarnim: Augmentation (`data/augmentations.py`)

```python
class AudioAugmentor:
    def __init__(self, musan_path, rir_generator):
        self.musan = load_musan(musan_path)
        self.rir_gen = rir_generator  # pyroomacoustics
    
    def __call__(self, waveform):
        # 1. RIR convolution (p=0.4)
        if random.random() < 0.4:
            rir = self.rir_gen.generate(
                room_dim=random.uniform(3,10),  # meters
                rt60=random.uniform(0.1, 1.0),
                distance=random.uniform(0.5, 5.0)
            )
            waveform = convolve(waveform, rir)
        
        # 2. Additive noise from MUSAN (p=0.7)
        if random.random() < 0.7:
            snr = random.uniform(-5, 30)  # full target range
            noise = random.choice(self.musan)
            waveform = add_noise(waveform, noise, snr_db=snr)
        
        # 3. Speed perturbation (p=0.3)
        if random.random() < 0.3:
            speed = random.choice([0.9, 0.95, 1.05, 1.1])
            waveform = torchaudio.functional.speed(waveform, orig_freq=16000, factor=speed)
        
        # 4. SpecAugment applied AFTER mel transform (in dataloader)
        return waveform
```

**SpecAugment** (applied to LFBE in dataloader):
```python
spec_augment = nn.Sequential(
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=25),
    torchaudio.transforms.TimeMasking(time_mask_param=25),
)
```

**MUSAN download:** `wget https://openslr.org/resources/17/musan.tar.gz`

---

#### Day 4-5 — Sohini: Heads (`models/heads.py`)

**Attentive Statistics Pooling** (used by both heads):
```python
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.Tanh(),
            nn.Linear(in_dim, 1)
        )
    
    def forward(self, x):
        # x: (B, C, T) → transpose to (B, T, C)
        x = x.transpose(1, 2)
        alpha = F.softmax(self.attention(x), dim=1)  # (B, T, 1)
        mu = (alpha * x).sum(dim=1)                   # (B, C)
        sigma = torch.sqrt(((alpha * (x - mu.unsqueeze(1))**2)).sum(dim=1) + 1e-6)
        return torch.cat([mu, sigma], dim=1)           # (B, 2C)
```

**Phonetic Head** (~620K):
```
FiLM(384, 48) → CausalConformer(d=192, heads=4, k=15) × 2
→ AttentiveStatsPool(192) → Linear(384→192)
```

**Speaker Head** (~580K):
```
FiLM(384, 48) → SE-DW-Res2Net(48, scale=4) × 3
→ AttentiveStatsPool(48) → Linear(96→192) → BN
```

#### Day 4-5 — Swarnim: Loss Functions (`training/losses.py`, `training/disentangle.py`)

**Gradient Reversal Layer:**
```python
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
```

**CLUB MI Estimator:**
```python
class CLUB(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.mu_net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.logvar_net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    
    def forward(self, z_spk, z_phn):
        mu = self.mu_net(z_spk)
        logvar = self.logvar_net(z_spk)
        # Positive: log q(z_phn | z_spk) for paired samples
        pos = -(mu - z_phn)**2 / (2 * logvar.exp()) - 0.5 * logvar
        # Negative: log q(z_phn | z_spk) for unpaired (shuffled)
        z_phn_shuffle = z_phn[torch.randperm(z_phn.size(0))]
        neg = -(mu - z_phn_shuffle)**2 / (2 * logvar.exp()) - 0.5 * logvar
        return (pos.sum(dim=-1).mean() - neg.sum(dim=-1).mean())
```

**AAM-Softmax:**
```python
class AAMSoftmax(nn.Module):
    def __init__(self, in_dim=192, n_classes=1500, scale=30, margin=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, embeddings, labels):
        # Normalize
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(embeddings, dim=1)
        cosine = F.linear(x, w)
        # Add angular margin to target class
        theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
        target_logits = torch.cos(theta[range(len(labels)), labels] + self.margin)
        cosine[range(len(labels)), labels] = target_logits
        return self.ce(self.scale * cosine, labels)
```

**Triplet Rejection Loss:**
```python
def rejection_loss(anchor, positive, confuser, margin=0.4):
    sim_pos = F.cosine_similarity(anchor, positive)
    sim_neg = F.cosine_similarity(anchor, confuser)
    return F.relu(sim_neg - sim_pos + margin).mean()
```

**Test all with dummy tensors:**
```python
z = torch.randn(128, 192)
labels = torch.randint(0, 1500, (128,))
loss = aam_softmax(z, labels)
loss.backward()  # must work
```

---

#### Day 5-6 — Sohini: Unified Model (`models/disent_v2.py`)

```python
class DISENT_KWS_v2(nn.Module):
    def __init__(self):
        self.encoder = BCResNet2()          # 520K
        self.temporal = TemporalBlock(48)   # 120K
        self.phn_head = PhoneticHead()      # 620K
        self.spk_head = SpeakerHead()       # 580K
        self.scorer = DualGateScorer()      # 30K
    
    def forward(self, audio, enroll_spk=None, enroll_kw=None):
        h = self.encoder(audio)
        h = self.temporal(h)
        
        cond = None
        if enroll_spk is not None and enroll_kw is not None:
            cond = torch.cat([enroll_spk, enroll_kw], dim=-1)  # (B, 384)
        
        z_phn = self.phn_head(h, cond)  # (B, 192)
        z_spk = self.spk_head(h, cond)  # (B, 192)
        
        return z_phn, z_spk
    
    def detect(self, audio, p_kw, p_spk):
        z_phn, z_spk = self.forward(audio, p_spk, p_kw)
        return self.scorer(z_phn, z_spk, p_kw, p_spk)
```

**Day 5 CRITICAL CHECK:**
```python
model = DISENT_KWS_v2()
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total:,}")  # MUST BE < 3,000,000
assert total < 3_000_000, f"OVER BUDGET: {total}"
# Target: ~1.99M
```

#### Day 5-6 — Swarnim: DSP Enrollment Augmentation (`data/synthetic.py`)

```python
def augment_enrollment(waveforms, n_variants=25):
    """Generate variants from 5 real enrollment samples using DSP only."""
    variants = list(waveforms)  # keep originals
    
    pitch_shifts = [-2, -1, 1, 2]          # semitones
    speed_factors = [0.9, 0.95, 1.05, 1.1]
    gain_db = [-3, -1.5, 1.5, 3]
    
    for wav in waveforms:
        for ps in pitch_shifts:
            shifted = torchaudio.functional.pitch_shift(wav, 16000, ps)
            variants.append(shifted)
        for sf in speed_factors:
            stretched = torchaudio.functional.speed(wav, 16000, sf)[0]
            # Resample back to original length
            stretched = F.interpolate(stretched.unsqueeze(0),
                                       size=wav.shape[-1]).squeeze(0)
            variants.append(stretched)
    
    # Randomly combine pitch+speed for remaining slots
    while len(variants) < n_variants + len(waveforms):
        base = random.choice(waveforms)
        ps = random.choice(pitch_shifts)
        sf = random.choice(speed_factors)
        g = random.choice(gain_db)
        aug = torchaudio.functional.pitch_shift(base, 16000, ps)
        aug = torchaudio.functional.speed(aug, 16000, sf)[0]
        aug = aug * (10 ** (g / 20))
        aug = F.interpolate(aug.unsqueeze(0), size=base.shape[-1]).squeeze(0)
        variants.append(aug)
    
    return variants[:n_variants + len(waveforms)]  # 30 total
```

---

#### Day 7 (Together, 3 hrs): 🔴 INTEGRATION CHECKPOINT

This is **non-negotiable**. By end of Day 7:

- [ ] Sohini's model accepts `(B, 1, 80, 200)` → outputs `z_phn, z_spk` each `(B, 192)`
- [ ] Swarnim's dataloader outputs correctly shaped batches
- [ ] Swarnim's augmentation pipeline runs without errors
- [ ] GRL plugs into model graph: `z_phn_rev = GRL.apply(z_phn, lambda_)` → backward works
- [ ] CLUB accepts `(z_spk, z_phn)` → returns scalar loss → backward works
- [ ] AAM-Softmax accepts `(z_spk, labels)` → returns scalar loss → backward works
- [ ] **ONE FULL BATCH flows: data → model → all losses → backward → optimizer.step()**

```python
# Integration test script
audio, spk_labels, kw_labels = next(iter(dataloader))
z_phn, z_spk = model(audio)
loss = (aam_softmax(z_spk, spk_labels)
        + proto_loss(z_phn, kw_labels)
        + 0.5 * (grl_spk_loss + grl_phn_loss + 0.1 * club(z_spk, z_phn))
        + 0.3 * rejection_loss(z_phn, positives, confusers))
loss.backward()
optimizer.step()
print("✅ Integration passed!")
```

---

### WEEK 2: Train, Evaluate, Ship

---

#### Day 8-9 (Together): Training on Kaggle

**Upload to Kaggle:**
```bash
git push origin main
# In Kaggle notebook:
!git clone https://github.com/yourrepo/DISENT_KWS.git
```

**Phase 1 — Pre-training (15 epochs, ~4 hrs on T4):**
```python
# Train encoder + phonetic head on GSC-v2
# Train encoder + speaker head on VoxCeleb (1500 speakers)
# Losses: AAM-Softmax only (no disentanglement yet)
# lr=3e-4, AdamW, cosine annealing
# Checkpoint every 5 epochs
```

**Phase 2 — Joint fine-tuning (15 epochs, ~6 hrs on T4):**
```python
# All modules, all losses, disentanglement enabled
# GRL λ ramp-up: λ(p) = 2/(1+exp(-10p)) - 1
# lr=1e-4, AdamW
# Hard negative mining from LibriPhrase-Hard
# Checkpoint every 5 epochs + save best by val loss
```

**Kaggle session strategy** (12hr limit):
- Session 1: Phase 1 (15 epochs) → save checkpoint → session ends
- Session 2: Load checkpoint → Phase 2 (15 epochs) → save final model

#### Day 10 — Swarnim: Evaluation (`eval/benchmark.py`)

```python
def evaluate(model, test_loader, thresholds):
    results = {}
    
    # TA Clean
    results['ta_clean'] = test_acceptance(model, test_loader, noise=None)
    
    # TA Noisy (per SNR)
    for snr in [-5, 0, 5, 10, 20, 30]:
        results[f'ta_snr_{snr}'] = test_acceptance(model, test_loader, snr_db=snr)
    
    # FA per hour
    results['fa_per_hr'] = count_false_accepts(model, background_audio_1hr)
    
    # DET curve for threshold calibration
    results['det_curve'] = compute_det(model, test_loader)
    results['optimal_threshold'] = find_threshold(results['det_curve'], target_fa=1.0)
    
    return results
```

#### Day 10 — Sohini: ONNX Export + Quantization

```python
# Quantization-Aware Training (if time permits, last 5 epochs of Phase 2)
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
model_prepared = torch.ao.quantization.prepare_qat(model)
# ... train 5 more epochs ...
model_int8 = torch.ao.quantization.convert(model_prepared)

# ONNX Export
dummy = torch.randn(1, 1, 80, 200)
torch.onnx.export(model, dummy, "disent_kws_v2.onnx", opset_version=17)

# Verify
import onnxruntime as ort
session = ort.InferenceSession("disent_kws_v2.onnx")
# Compare outputs, tolerance < 1%
```

#### Day 11-12 (Together): Enrollment + Demo

**Enrollment pipeline** (`enrollment/enroll.py`):
```python
def enroll(model, audio_files):
    """5 real recordings → extract prototypes"""
    waveforms = [torchaudio.load(f)[0] for f in audio_files]
    
    # DSP augmentation → 30 total variants
    augmented = augment_enrollment(waveforms, n_variants=25)
    
    # Extract embeddings
    features = [compute_lfbe(w) for w in augmented]
    with torch.no_grad():
        z_phns, z_spks = zip(*[model(f.unsqueeze(0)) for f in features])
    
    p_kw = torch.stack(z_phns).mean(dim=0)    # (1, 192)
    p_spk = torch.stack(z_spks).mean(dim=0)   # (1, 192)
    
    # Normalize
    p_kw = F.normalize(p_kw, dim=-1)
    p_spk = F.normalize(p_spk, dim=-1)
    
    return p_kw, p_spk  # 768 bytes total to store
```

**Demo script** (`demo.py`):
```python
import sounddevice as sd

def realtime_demo(model, p_kw, p_spk, threshold):
    buffer = RingBuffer(max_len=16000 * 0.64)  # 640ms
    ema_score = 0.0
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal ema_score
        buffer.append(indata[:, 0])
        
        if buffer.is_full():
            audio = buffer.get_tensor()
            features = compute_lfbe(audio).unsqueeze(0)
            
            with torch.no_grad():
                z_phn, z_spk = model(features, p_spk, p_kw)
            
            sim_kw = F.cosine_similarity(z_phn, p_kw)
            sim_spk = F.cosine_similarity(z_spk, p_spk)
            score = 0.55 * sim_kw + 0.45 * sim_spk
            ema_score = 0.7 * score.item() + 0.3 * ema_score
            
            if ema_score >= threshold:
                print(f"🎯 DETECTED! (score: {ema_score:.3f})")
    
    with sd.InputStream(callback=audio_callback, channels=1,
                         samplerate=16000, blocksize=2560):  # 160ms blocks
        print("Listening... Press Ctrl+C to stop")
        while True:
            sd.sleep(100)
```

#### Day 13 (Together): Demo Scenarios + Results Table

Test and record these 5 scenarios:
1. ✅ Clean room, target speaker says keyword → ACCEPT
2. ❌ Clean room, wrong speaker says keyword → REJECT
3. ✅ Noisy room (babble), target speaker says keyword → ACCEPT
4. ❌ Clean room, target speaker says confuser word → REJECT
5. ❌ Background noise only, no speech → no false trigger

**Generate results table:**

```
| Metric              | Target  | Achieved |
|:---------------------|:-------:|:--------:|
| TA (Clean)           | ≥ 99%  |   ___%   |
| TA (Noisy, SNR≥0dB)  | ≥ 90%  |   ___%   |
| TA (Noisy, SNR=-5dB) | ≥ 90%  |   ___%   |
| FA                   | < 1/hr |   ___/hr |
| Parameters           | < 3M   |  1.99M   |
| xRT                  | < 0.2s |   ___s   |
| Model Size (INT8)    |   —    |   ___MB  |
```

#### Day 14: Buffer + Package

- Fix any last bugs
- Write README with setup instructions
- Package: model.pt + model.onnx + demo.py + benchmark results
- Final latency measurement on target hardware

---

## Fallback Strategies

| Risk | Trigger | Fallback |
|:---|:---|:---|
| Training doesn't converge | Loss plateaus after 10 epochs | Disable L_disent, train with L_kw + L_spk only |
| Kaggle session timeout | Training interrupted | Resume from last checkpoint (saved every 5 epochs) |
| CLUB MI causes NaN | NaN in loss | Remove CLUB, keep GRL only (still provides disentanglement) |
| VoxCeleb not on Kaggle | Dataset missing | Use SpeechBrain pre-trained ECAPA-TDNN embeddings (freeze speaker head) |
| Conformer too slow | xRT > 0.2s | Replace with 2× DW-Conv1D blocks (same as temporal block) |
| TA noisy < 90% | Poor noise robustness | Increase augmentation probability to p=0.9 for noise, retrain 5 epochs |
| FA > 1/hr | Too many false triggers | Lower threshold τ (trade TA for FA), add 3-consecutive-window requirement |
