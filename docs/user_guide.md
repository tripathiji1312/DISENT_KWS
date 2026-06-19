# User & Developer Guide

This document describes how to train, enroll speakers, run the real-time demo, reproduce benchmark results, and generate final deliverables for the **DISENT-KWS** system.

---

## 1. Training Pipeline

Training consists of four phases: Phase 1 (softmax pre-training), Phase 2 (disentanglement & joint fine-tuning), Phase 3a (GE2E speaker refinement), and Phase 3b (hard-negative GE2E). After training, the scorer is calibrated via the benchmark script.

### Phase 1: Softmax Pre-training

Trains separate phonetic and speaker encoders using AAM-Softmax classification loss:

```bash
python src/train.py --phase 1 --epochs 20 --data-root /path/to/data_root --save-dir checkpoints
```

**What happens:**
- Shared encoder + phonetic head trained on Google Speech Commands v2 (35-word classification) using AAM-Softmax
- Shared encoder + speaker head trained on VoxCeleb1 (1,251 speaker classification) using AAM-Softmax
- Cosine annealing LR schedule with AdamW optimizer

**Expected training curve:**
```
Epoch 1/20 | Loss: 8.42 | KW Acc: 12.3% | Spk Acc: 0.8%
Epoch 5/20 | Loss: 3.15 | KW Acc: 78.5% | Spk Acc: 45.2%
Epoch 10/20| Loss: 1.87 | KW Acc: 92.1% | Spk Acc: 72.8%
Epoch 20/20| Loss: 1.02 | KW Acc: 97.3% | Spk Acc: 89.5%
```

### Phase 2: Joint Disentanglement & Triplet Training

Trains with GRL, CLUB MI estimation, and triplet rejection loss:

```bash
python src/train.py --phase 2 --epochs 20 --data-root /path/to/data_root \
    --save-dir checkpoints --resume checkpoints/phase1_best.pt
```

**What happens:**
- All modules trained jointly with the composite loss:
  $$L = L_{kw} + L_{spk} + 0.5L_{disent} + 0.3L_{reject}$$
- GRL λ ramps up sigmoidally: $\lambda(p) = 2/(1+e^{-10p}) - 1$
- CLUB MI estimator minimizes $I(z_{spk}; z_{phn})$
- Hard negatives mined from LibriPhrase

### Phase 3a: GE2E Speaker Fine-tuning

Refines the speaker head using Generalized End-to-End (GE2E) loss on VoxCeleb1 for improved speaker discriminability:

```bash
python src/train.py --phase 3a --epochs 20 --data-root /path/to/data_root \
    --save-dir checkpoints --resume checkpoints/phase2_best.pt
```

### Phase 3b: Hard-negative GE2E Fine-tuning

Continues GE2E training with hard-negative mining from LibriPhrase, forcing the speaker head to separate confusable speakers:

```bash
python src/train.py --phase 3b --epochs 20 --data-root /path/to/data_root \
    --save-dir checkpoints --resume checkpoints/phase3a_best.pt
```

### Scorer Calibration

After Phase 3b, calibrate the Dual-Gate Scorer via the benchmark script:

```bash
python src/eval/benchmark.py \
    --model-path checkpoints/phase3_hardneg_calibrated.pt \
    --data-root /path/to/data_root
```

**Scorer calibration output:**
```
🎯 OPTIMAL SCORER CALIBRATION
  Best w_kw       : 0.30
  Best w_spk      : 0.65
  Best threshold  : 0.2222
  Joint EER       : 23.47%
  Joint AUC       : 0.8425
```

---

## 2. Enrollment Pipeline

Before detecting a custom word from a specific user, you must enroll them. The enrollment script extracts reference prototypes from recordings.

### Steps to Enroll a User

Two options — live recording or pre-recorded files.

#### Option A: Live Microphone Recording (Recommended for Testing)

Records directly from your mic — just press Enter and speak:

```bash
python src/demo.py record \
    --model model_final.pt \
    --n-record 5 \
    --duration 2.0 \
    --out enrollment.pt
```

The script prompts you to say your keyword 5 times, then automatically extracts prototypes and saves the enrollment.

#### Option B: Pre-Recorded WAV Files

```bash
python src/demo.py enroll \
    --recordings /path/to/user/wavs/*.wav \
    --model model_final.pt \
    --out enrollment.pt
```

3. **What happens under the hood:**
   - **DSP Augmentation:** Each recording is pitch-shifted (±1, ±2 semitones), speed-perturbed (0.9×, 1.1×), and gain-jittered (±3 dB) to produce 30 diverse variants
   - **Prototype Extraction:** Audio variants pass through phonetic and speaker heads → compute mean embeddings
     $$p_{spk} = \frac{1}{N}\sum_{i=1}^N f_{spk}(x_i) \in \mathbb{R}^{192}$$
     $$p_{kw} = \frac{1}{N}\sum_{i=1}^N f_{phn}(x_i) \in \mathbb{R}^{192}$$
   - **Profile Saved:** 768 bytes total (2 × 192-dim FP32 vectors)

**Expected output:**
```
🔑 Processing 8 enrollment recordings...
  → DSP augmentation: 8 → 30 variants
🗣️ Extracting prototypes...
  → p_kw  ∈ ℝ¹⁹² ✓
  → p_spk ∈ ℝ¹⁹² ✓
💾 Profile saved → enrollment/profiles/activate.npy (768 bytes)
```

---

## 3. Real-Time Streaming Demo

The real-time streaming demo runs continuous audio input from the user's default microphone, sliding a 1-second buffer with 160 ms hops.

### How to Run the Demo

```bash
python src/demo.py \
    --profile enrollment/profiles/activate.npy \
    --threshold 0.2222
```

### Control Keys

| Key | Action |
|:---:|:---|
| **Ctrl+C** | Terminate the stream |
| **Enter** | Show current score summary |

### Expected Terminal Output

```
🎤 Listening... (threshold=0.222)
────────────────────────────────────────────────
Score: [█████████████░░░░░░░░░░░░░░░░░░░░░░░] 0.312
Score: [███████████████░░░░░░░░░░░░░░░░░░░░░] 0.345
Score: [█████████████████████████████████████] 0.887  🎯 DETECTED!
Score: [████████████████░░░░░░░░░░░░░░░░░░░░░] 0.423
────────────────────────────────────────────────
```

### Demo Scenario Results

These are the 5 validation scenarios we tested:

| # | Scenario | Expected | Result |
|:-:|:---|---|:---:|
| 1 | Clean room, target speaker says keyword | ✅ ACCEPT | ✅ |
| 2 | Clean room, wrong speaker says keyword | ❌ REJECT | ✅ |
| 3 | Babble noise (10 dB SNR), target speaker says keyword | ✅ ACCEPT | ✅ |
| 4 | Clean room, target speaker says confuser word ("activate" → "active") | ❌ REJECT | ✅ |
| 5 | 60 seconds background noise, no keyword spoken | 0 triggers | ✅ (0 triggers) |

---

## 4. Generating Final Submission Deliverables

To regenerate the ONNX model, DET evaluation curve, and ablation study reports from a final trained checkpoint:

```bash
python scripts/generate_final_artifacts.py \
    --checkpoint checkpoints/phase3_hardneg_calibrated.pt \
    --data-root /path/to/data_root \
    --wandb-project DISENT-KWS
```

### Script Arguments

| Argument | Default | Description |
|:---|---|:---|
| `--checkpoint` | required | Path to `.pt` model checkpoint |
| `--data-root` | required | Path to dataset folder |
| `--skip-ablation` | — | Skip long ablation fine-tuning loops |
| `--skip-onnx` | — | Skip ONNX export |
| `--skip-det` | — | Skip DET computation |
| `--device` | `cuda` | `cuda` or `cpu` |

### Generated Deliverables

```
📦 model_final.pt              — PyTorch checkpoint (1.806M params)
📦 model_final.onnx            — ONNX export (0.60 MB, INT8)
📊 ablation_results.csv         — Per-component ablation table
📈 docs/det_curve.png           — DET curve (FRR vs FAR)
📈 docs/param_budget.png        — Parameter distribution treemap
📈 docs/ablation_chart.png     — Ablation bar chart
📈 docs/snr_robustness.png     — SNR evaluation plot
📈 docs/training_phases.png    — Training pipeline diagram
```

---

## 5. Reproducing Benchmark Results

To verify the published KPIs from scratch:

```bash
# Step 1: Run full evaluation
python src/eval/benchmark.py \
    --model-path model_final.pt \
    --data-root /path/to/data_root

# Step 2: Run ablation study
python src/eval/ablation.py \
    --model-path model_final.pt \
    --data-root /path/to/data_root

# Step 3: Generate visuals
python scripts/generate_visuals.py \
    --ablation-csv ablation_results.csv \
    --output-dir docs/
```

**Expected KPI output:**
```
=================================================================
              🏆 DISENT-KWS — FINAL KPI REPORT 🏆
=================================================================
  Metric                           | Target     | Achieved     | Status
-----------------------------------------------------------------
  Parameters                       | < 3.0 M    |   1.806 M    | ✅
  CPU Latency                      | < 200 ms   |    26.4 ms   | ✅
  xRT Factor                       | < 0.20     |  0.0132      | ✅
  Keyword EER (standalone)         | low        |    4.69%     | ✅
  Speaker EER (standalone)         | low        |   17.86%     | ✅
  Joint EER                        | —          |   23.47%     | —
  Joint AUC                        | —          |  0.8425      | ✅
=================================================================
```

---

## 6. Testing

Run the full test suite to verify system integrity:

```bash
# All tests
make test

# Specific test files
make test-dataloaders

# With coverage
make test-cov

# Fast mode (skip slow tests)
pytest tests/ -m "not slow" -v
```

See [TESTING.md](../TESTING.md) for complete testing documentation.

---

## 7. Common Workflows

### Quick Start (Inference Only)

```bash
# 1. Set up environment
uv sync --all-extras

# 2. Download model from HuggingFace
huggingface-cli download tripathiji1312/DISENT-KWS model_final.pt --local-dir .

# 3. Enroll a speaker (live mic recording)
python src/demo.py record --model model_final.pt --out enrollment.pt

# 4. Run demo
python src/demo.py detect \
    --enrollment enrollment.pt \
    --model model_final.pt
```

### Full Training Pipeline

```bash
# Phase 1: Pre-train (T4 GPU, ~5 hours)
python src/train.py --phase 1 --epochs 20 \
    --data-root /data --save-dir checkpoints

# Phase 2: Joint fine-tune (T4 GPU, ~8 hours)
python src/train.py --phase 2 --epochs 20 \
    --data-root /data --save-dir checkpoints \
    --resume checkpoints/phase1_best.pt

# Phase 3a: GE2E speaker refinement (T4 GPU, ~6 hours)
python src/train.py --phase 3a --epochs 20 \
    --data-root /data --save-dir checkpoints \
    --resume checkpoints/phase2_best.pt

# Phase 3b: Hard-negative GE2E (T4 GPU, ~6 hours)
python src/train.py --phase 3b --epochs 20 \
    --data-root /data --save-dir checkpoints \
    --resume checkpoints/phase3a_best.pt

# Calibrate scorer & benchmark
python src/eval/benchmark.py \
    --model-path checkpoints/phase3_hardneg_calibrated.pt \
    --data-root /data

# Generate artifacts
python scripts/generate_final_artifacts.py \
    --checkpoint checkpoints/phase3_hardneg_calibrated.pt \
    --data-root /data
```

### ONNX Export Only

```bash
python -c "
from models.disent_v2 import DISENT_KWS_v2
import torch

model = DISENT_KWS_v2()
model.load_state_dict(torch.load('model_final.pt'))
model.eval()

dummy = torch.randn(1, 80, 200)
torch.onnx.export(model, dummy, 'model.onnx',
    input_names=['audio'], output_names=['z_phn', 'z_spk'],
    opset_version=17,
    dynamic_axes={'audio': {0: 'batch', 2: 'time'}})
print('✅ Exported to model.onnx')
"
```

---

## 8. Architecture Quick Reference

```
                            ┌───────────────────────────────┐
                            │  Audio Input (80×200 LFBE)    │
                            └──────────────┬────────────────┘
                                           ▼
                            ┌───────────────────────────────┐
                            │  BC-ResNet-2 Shared Encoder    │
                            │        1,806K  TOTAL           │
                            └──────────────┬────────────────┘
                                           ▼
                            ┌───────────────────────────────┐
                            │  Mamba / Dilated Conv1D        │
                            │  Temporal Block                │
                            └──────┬───────────────┬────────┘
                                   │               │
                         ┌─────────┘               └─────────┐
                         ▼                                   ▼
                  ┌───────────────┐                  ┌───────────────┐
                  │ Phonetic Head │                  │ Speaker Head  │
                  │  (Conformer)  │                  │ (ECAPA-Lite)  │
                  └───────┬───────┘                  └───────┬───────┘
                         ▼                                   ▼
                    z_phn ∈ ℝ¹⁹²                        z_spk ∈ ℝ¹⁹²
                         │                                   │
                         └───────────┐         ┌─────────────┘
                                     ▼         ▼
                              ┌──────────────────────┐
                              │   Dual-Gate Scorer    │
                              │   w_kw=0.30, w_spk=0.65│
                              │   τ=0.2222, EMA_α=0.7│
                              └──────────┬───────────┘
                                         ▼
                                   D ∈ {0, 1}
```
