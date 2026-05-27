# DISENT-KWS v2 — Two-Week Implementation Plan

## Team: Sohini (Track A: Architecture) + Swarnim (Track B: Data/Training)

---

## Day 0 (Together, 2 hrs): The Contract

Create `config.py` with all shared constants:

```python
# config.py
SAMPLE_RATE = 16000
N_MELS = 80
WIN_MS = 25
HOP_MS = 10
EMBED_DIM = 192
BATCH_SIZE = 128
MAX_AUDIO_SEC = 2.0
N_CLASSES_GSC = 35
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
BC_CHANNELS = [16, 16, 32, 48]
NUM_SPEAKERS_VOXCELEB = 7205
```

Agree on tensor shapes:
- Audio features: `(B, 80, T)` where T ≈ 200 for 2s audio
- Shared encoder output: `(B, 48, T')`
- Embeddings: `(B, 192)`
- Prototypes: `(1, 192)`

Set up repo structure:

```
DISENT_KWS/
├── config.py
├── models/          # Sohini
│   ├── bc_resnet.py
│   ├── mamba_block.py
│   ├── film.py
│   ├── heads.py
│   ├── scorer.py
│   └── disent_v2.py
├── data/            # Swarnim
│   ├── datasets.py
│   ├── augmentations.py
│   └── synthetic.py
├── training/        # Swarnim
│   ├── losses.py
│   ├── disentangle.py
│   └── scheduler.py
├── eval/            # Swarnim
│   ├── benchmark.py
│   └── export.py
├── enrollment/      # Shared
│   └── enroll.py
└── train.py         # Shared
```

---

## Week 1: Independent Build

### Sohini (Track A) — `models/`

**Day 1-2: BC-ResNet-2 Shared Encoder** (`bc_resnet.py`)
- Implement BC-ResBlock: 2D freq-conv + 1D time-conv + broadcast add
- Stack: Conv2D(1,16,5×5) → 2×BC-Res(16→16) → 2×BC-Res(16→32) → 2×BC-Res(32→48)
- Add BatchNorm + ReLU after each block
- Verify: input `(B,1,80,T)` → output `(B,48,T')`, ~520K params
- Test with `torch.randn(4, 1, 80, 200)`

**Day 3: Mamba SSM Block** (`mamba_block.py`)
- `pip install mamba-ssm` (or implement minimal selective SSM)
- Wrapper: Linear(48→96) → Conv1D causal → SelectiveSSM(d_state=16) → Linear(96→48) + residual
- Verify: input `(B,48,T')` → output `(B,48,T')`, ~180K params

**Day 3-4: FiLM Conditioning** (`film.py`)
- Input: conditioning vector `(B, 384)` = concat(p_spk, p_kw)
- Output: γ, β each `(B, 48, 1)` for channel-wise modulation
- `x_out = (1 + γ) * x + β`
- ~30K params

**Day 4-5: Phonetic + Speaker Heads** (`heads.py`)
- **Phonetic Head**: FiLM → 2× CausalConformer(d=192, heads=4, conv_k=15) → AttentiveStatsPool → Linear(384→192)
- **Speaker Head**: FiLM → 3× SE-DW-Res2Net(48, scale=4) → AttentiveStatsPool → Linear → BN → 192-dim
- Implement AttentiveStatsPool: `α = softmax(v·tanh(Wh+b))`, `μ = Σα·h`, `σ = sqrt(Σα·(h-μ)²)`, output = Linear(concat(μ,σ))
- Phonetic: ~620K, Speaker: ~580K

**Day 5: Scorer + Unified Model** (`scorer.py`, `disent_v2.py`)
- Scorer: cosine similarities → weighted sum (w_kw=0.55, w_spk=0.45) → EMA smoothing → threshold
- Unified model: chain all modules, accept GRL hooks from training/
- **Total param check: assert sum < 3M** (target: 2.05M)

**Day 6-7: Debug + Integration Test**
- End-to-end forward pass with dummy data
- Profile: `torch.utils.benchmark` for latency
- Fix shape mismatches

### Swarnim (Track B) — `data/` + `training/`

**Day 1-2: Dataloaders** (`datasets.py`)
- Google Speech Commands v2: torchaudio download, 35-word classification labels
- VoxCeleb 1+2: SpeechBrain recipe for speaker IDs, segment loading
- LibriPhrase: HuggingFace datasets, hard/easy split, anchor/positive/negative triplets
- All output: `(B, 80, T)` LFBE features via `torchaudio.transforms.MelSpectrogram`

**Day 3-4: Augmentation Pipeline** (`augmentations.py`)
- RIR simulation: `pyroomacoustics` — rooms 3×3m to 10×10m, RT60 0.1-1.0s, distance 0.5-5m
- MUSAN noise injection: download MUSAN, mix at random SNR ∈ [-5, 30] dB
- SpecAugment: FrequencyMask(F=15, num=2) + TimeMask(T=25, num=2)
- Speed perturbation: torchaudio.functional, factors {0.9, 1.0, 1.1}
- Codec simulation: torchaudio sox effects (μ-law, GSM)
- Compose all with probabilities: RIR p=0.4, Noise p=0.7, Spec p=0.8, Speed p=0.3

**Day 4-5: Loss Functions** (`losses.py`, `disentangle.py`)
- **GRL**: `torch.autograd.Function` — forward passes through, backward reverses gradient × λ
- **CLUB MI**: variational q(z_phn|z_spk) = N(μ_θ, σ²_θ), upper bound on I(z_spk; z_phn)
- **AAM-Softmax**: `L = -log[exp(s·cos(θ_y+m)) / (exp(s·cos(θ_y+m)) + Σexp(s·cos(θ_j)))]`, s=30, m=0.2
- **Prototypical + AAM for keywords**: prototype centroid, angular margin
- **Triplet rejection loss**: `max(0, cos(anchor,confuser) - cos(anchor,target) + 0.4)`
- **KD loss**: `T²·KL(softmax(z_t/T) || softmax(z_s/T))`, T=4
- Lambda scheduler: `λ(p) = 2/(1+exp(-10p)) - 1`
- Test all with `torch.randn(128, 192)` dummy embeddings

**Day 5-6: Synthetic Data** (`synthetic.py`)
- XTTS v2 (Coqui TTS): clone speaker voice → generate keyword variations
- Quality filter: Whisper-tiny transcription, reject if WER > 10%
- Output: augmented enrollment set (5 real + 20 synthetic)

**Day 6-7: Integration Test with Sohini**
- Feed real dataloader output into model
- Verify loss backward passes work (especially GRL gradient reversal)
- One batch end-to-end: data → model → loss → backward → step

---

## Week 2: Train, Evaluate, Optimize

### Day 8-9 (Together): Training Loop (`train.py`)

```python
# Pseudocode
for epoch in range(total_epochs):
    for batch in dataloader:
        audio, spk_label, kw_label, enroll_proto = batch
        
        z_phn, z_spk = model(audio, enroll_proto)
        
        loss = (L_kw(z_phn, kw_label) 
              + L_spk(z_spk, spk_label) 
              + 0.5 * L_disent(z_phn, z_spk, spk_label, kw_label)
              + 0.3 * L_reject(z_phn, confusers)
              + 0.7 * L_kd(z_student, z_teacher))
        
        loss.backward()
        optimizer.step()
```

- Phase 1 (pre-train): 20 epochs on GSC + VoxCeleb separately
- Phase 2 (joint): 15 epochs with all losses + disentanglement
- Use W&B for logging

### Day 10-11 (Swarnim): Evaluation (`eval/`)

**benchmark.py:**
- TA (clean): test on LibriPhrase clean split
- TA (noisy): test with MUSAN noise at -5, 0, 5, 10, 20, 30 dB
- FA/hr: run model on 1hr of continuous non-target audio, count false triggers
- DET curve: sweep thresholds, plot FA vs FR, find optimal τ
- Per-SNR breakdown table

**Enrollment pipeline** (`enrollment/enroll.py`):
- Record 5-10 keyword utterances
- Extract p_kw = mean(f_phn(utterances)), p_spk = mean(f_spk(utterances))
- Optional XTTS augmentation
- Quality score + threshold calibration on 5-min background

### Day 10-11 (Sohini): Optimization

- Quantization-Aware Training: last 5 epochs with `torch.ao.quantization`
- Structured pruning: 15% channel pruning → 5 epoch recovery
- ONNX export: `torch.onnx.export(model, dummy, "model.onnx", opset_version=17)`
- Verify INT8 ONNX Runtime inference matches PyTorch output (tolerance < 1%)

### Day 12-13 (Together): Demo + Polish

**Build demo script** (`demo.py`):
- Real-time mic input → sliding window → model inference → detection output
- Visual: terminal display with confidence bars
- Test scenarios:
  1. Clean room, target speaker says keyword → ACCEPT
  2. Clean room, wrong speaker says keyword → REJECT
  3. Noisy room, target speaker says keyword → ACCEPT
  4. Clean room, target speaker says confuser word → REJECT
  5. Background babble, no keyword → silence (no false triggers)

**Prepare submission materials:**
- Model checkpoint (.pt + .onnx)
- Benchmark results table (all KPIs)
- Demo recording / live demo setup
- README with setup instructions

### Day 14: Buffer + Final Testing

- End-to-end stress test
- Fix any edge cases
- Final parameter count verification
- Final latency measurement
- Package everything

---

## Critical Milestones

| Day | Milestone | Owner |
|:---:|:---|:---|
| 0 | config.py + repo setup | Both |
| 3 | BC-ResNet + Mamba forward pass works | Sohini |
| 3 | Dataloaders output correct shapes | Swarnim |
| 5 | Full model forward pass with dummy data | Sohini |
| 5 | All loss functions tested with dummy embeddings | Swarnim |
| **7** | **🔴 Integration: 1 batch flows end-to-end** | **Both** |
| 9 | Training runs for 5+ epochs without crashing | Both |
| 11 | Benchmark numbers computed | Swarnim |
| 11 | ONNX export + INT8 working | Sohini |
| 13 | Demo working | Both |
| 14 | Submission ready | Both |

> [!WARNING]
> **Day 7 integration is NON-NEGOTIABLE.** If the model + data + losses don't connect by Day 7, you won't have enough time to train and evaluate. Treat Day 7 as a hard deadline.

---

## Dependencies to Install (Day 0)

```bash
pip install torch torchaudio speechbrain mamba-ssm
pip install pyroomacoustics torch-audiomentations
pip install TTS  # for XTTS/Coqui
pip install onnxruntime openai-whisper wandb
pip install librosa soundfile matplotlib
```

---

## What to Do If Training Doesn't Converge

Ordered troubleshooting:

1. **Disable disentanglement losses** → train with just L_kw + L_spk first
2. **Reduce learning rate** to 1e-5
3. **Check GRL λ schedule** — if ramping too fast, embeddings collapse
4. **Increase batch size** if GPU allows (256+)
5. **Pre-train speaker head longer** on VoxCeleb alone (10 more epochs)
6. **Verify augmentation isn't too aggressive** — test with p=0.3 for all augs first
