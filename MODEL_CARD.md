# Model Card: DISENT-KWS

## Model Description

**DISENT-KWS** is a speech disentanglement model for robust custom word detection. It uses a shared BC-ResNet-2 encoder with dual heads — a Causal Conformer (phonetic) and ECAPA-TDNN Lite (speaker) — to produce orthogonal latent embeddings. A Dual-Gate Scorer combines keyword and speaker similarity with calibrated weights for joint verification.

- **Total Parameters:** 1.806M
- **ONNX Size:** 0.60 MB (INT8 quantized)
- **Input:** 80-band log Mel filterbank energies, 200 frames (2 s @ 16 kHz)
- **Output:** `z_phn ∈ ℝ¹⁹²` (phonetic embedding), `z_spk ∈ ℝ¹⁹²` (speaker embedding)

## Intended Use

- **Custom keyword spotting** with speaker verification for access control, voice assistants, or wake-word systems
- **Real-time streaming** inference on CPU (26.4 ms per 2 s window)
- **Few-shot enrollment** (5+ utterances) for new users and custom words

## Architecture

| Component | Type | Params |
|---|---|---|
| Shared Encoder | BC-ResNet-2 | 33.8K |
| Temporal Block | Mamba SSM / Dilated Conv1D | 10.3K |
| Phonetic Head | Causal Conformer (4 heads, kernel 15) | 1,673K |
| Speaker Head | ECAPA-TDNN Lite (SE ratio 4, scale 4) | 88.8K |
| Scorer | Dual-Gate (w_kw=0.30, w_spk=0.65, EMA α=0.7) | — |

## Training Data

| Dataset | Usage | Samples |
|---|---|---|
| Google Speech Commands v2 | Keyword pre-training (35 classes) | 105K |
| VoxCeleb1 | Speaker verification (1,251 speakers) | 153K |
| LibriPhrase | Hard-negative triplet pairs | 3K triplets |
| MUSAN | Noise augmentation (babble, music, environmental) | 109 hrs |

## Training Phases

1. **Phase 1:** AAM-Softmax pre-training on keyword (GSC) and speaker (VoxCeleb) separately
2. **Phase 2:** Joint training with GRL adversarial reversal + CLUB MI minimization + triplet rejection loss
3. **Phase 3a:** GE2E speaker head refinement
4. **Phase 3b:** Hard-negative GE2E with LibriPhrase confusers

## Performance

| Metric | Value |
|---|---|
| Keyword EER | 4.69% |
| Speaker EER | 17.86% |
| Joint EER | 23.47% |
| Joint AUC | 0.8425 |
| CPU Latency | 26.43 ms (p95: 28.29 ms) |
| Real-Time Factor (xRT) | 0.0132 |
| Optimal Threshold (τ) | 0.2222 |

## Hardware & Runtime

- **Training:** NVIDIA GPU with 16 GB+ VRAM (tested on Tesla T4)
- **Inference:** CPU-only via ONNX Runtime; no GPU required
- **Memory:** ~200 MB RAM at runtime

## Known Limitations

- Requires clean enrollment recordings (SNR ≥ 10 dB recommended)
- Speaker EER (17.86%) is higher than keyword EER — joint verification mitigates this
- Mamba SSM fallback to Dilated Conv1D on platforms without CUDA (no performance loss)

## License

MIT
