# User & Developer Guide

This document describes how to train, enroll speakers, run the real-time demo, and generate final deliverables for the DISENT-KWS v2 system.

---

## 1. Training Pipeline

Training consists of three main stages: Phase 1 (pre-training), Phase 2 (disentanglement & joint fine-tuning), and Phase 3 (hard-negative training / calibration).

### Phase 1: Softmax Pre-training
Trains separate phonetic and speaker encoders using AAM-Softmax classification loss:
```bash
python train.py --phase 1 --epochs 20 --data-root /path/to/data_root --save-dir checkpoints
```

### Phase 2: Joint Disentanglement & Triplet Training
Trains with Gradient Reversal Layer (GRL), CLUB Mutual Information estimation, and triplet rejection loss to disentangle phonetic representations from speaker identities:
```bash
python train.py --phase 2 --epochs 20 --data-root /path/to/data_root --save-dir checkpoints --resume checkpoints/phase1_best.pt
```

### Phase 3: Hard-Negative Fine-Tuning & Scorer Calibration
To resolve edge cases (phonetically similar words or speaker variants), fine-tune on hard negatives and optimize the Dual-Gate Scorer weight ratios ($w_{kw}$ vs $w_{spk}$):
```bash
# Fine-tuning script handles building the hard neighbor graph and optimizing weights
python -m eval.benchmark --model-path checkpoints/phase3_hardneg_calibrated.pt --data-root /path/to/data_root
```

---

## 2. Enrollment Pipeline

Before detecting a custom word from a specific user, you must enroll them. The enrollment script extracts reference prototypes (embeddings) from enrollment recordings.

### Steps to Enroll a User

1. **Prepare recordings:** Record 5–10 short clips of the target user speaking their desired custom word (e.g. "activate"). Save them as `.wav` files (16kHz, 16-bit mono).
2. **Run the enrollment command:**
   ```bash
   python enrollment/enroll.py --audio-dir /path/to/user/wavs --output-dir enrollment/profiles --keyword "activate"
   ```
3. **What happens under the hood:**
   - **DSP Augmentations:** The script applies randomized speed perturbation, additive noise, and synthetic room impulse response (RIR) convolutions to expand the 5–10 recordings into 30 diverse variants.
   - **Prototype Extraction:** It passes the audio variants through the phonetic and speaker heads to compute the average speaker representation ($p_{spk} \in \mathbb{R}^{192}$) and phonetic representation ($p_{kw} \in \mathbb{R}^{192}$).
   - **Profile Saving:** The resulting profiles are saved as small NumPy/binary arrays in `enrollment/profiles/`.

---

## 3. Real-Time Streaming Demo

The real-time streaming demo runs on a continuous audio input from the user's default microphone, sliding a 1-second buffer with 160ms hops.

### How to Run the Demo

1. **Prerequisite:** Make sure you have enrolled a speaker/word profile.
2. **Execute the demo script:**
   ```bash
   python demo.py --profile enrollment/profiles/activate.npy --threshold 0.50
   ```
3. **Control Keys:**
   - Press **Ctrl+C** to terminate the stream.
4. **Output display:**
   The script prints a real-time smoothed similarity bar and alerts when both speaker identity and word phonetics cross the joint gate thresholds:
   ```
   Score: [████████████████████████████                      ] 0.524 🎯 DETECTED!
   ```

---

## 4. Re-generating Final Submission Deliverables

To re-generate the ONNX model, DET evaluation curve, and ablation study reports from a final trained checkpoint, run the helper script in the repository:

```bash
python scripts/generate_final_artifacts.py \
    --checkpoint checkpoints/phase3_hardneg_calibrated.pt \
    --data-root /path/to/data_root \
    --wandb-project DISENT-KWS-v2
```

### Script Arguments:
* `--checkpoint`: Path to the PyTorch checkpoint (`.pt` file).
* `--data-root`: Path to the folder containing datasets.
* `--skip-ablation`: Add this flag to skip the long ablation study fine-tuning loops.
* `--skip-onnx`: Add this flag to skip exporting the ONNX file.
* `--skip-det`: Add this flag to skip computing test scores and plotting the DET curve.

### Generated Deliverables:
- **`model_final.onnx`** (saved in the repository root)
- **`ablation_results.csv`** (saved in the repository root)
- **`docs/det_curve.png`** (saved in the `docs` folder)
