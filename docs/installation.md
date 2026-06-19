# Installation & Setup Guide

This document outlines the steps required to set up the environment, install dependencies, and download datasets for reproducing the **DISENT-KWS** speech disentanglement system.

---

## 1. Prerequisites

| Requirement | Specification |
|:---|---|
| **OS** | Linux (Ubuntu 20.04/22.04 LTS recommended) or macOS |
| **Python** | 3.10 or 3.11 |
| **RAM** | Minimum 8 GB (16 GB+ recommended) |
| **GPU** | NVIDIA GPU with CUDA + 16 GB+ VRAM (for training); CPU-only works for inference |
| **Storage** | 50 GB+ free (for datasets) |
| **Package Manager** | `uv` (recommended) or `pip` |

---

## 2. Environment Setup

### Option A: Using `uv` (Recommended — 2× faster dependency resolution)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/<your-org>/DISENT_KWS.git
cd DISENT_KWS

# 3. Create virtual environment & install all dependencies
uv sync --all-extras
```

**Expected output:**
```
Resolved 85 packages in 1.2s
Installed 85 dependencies ✓
✅ Virtual environment ready at .venv/
```

### Option B: Using Standard `pip`

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/DISENT_KWS.git
cd DISENT_KWS

# 2. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> If `uv` is not installed, run `pip install uv` first, then use `uv sync --all-extras`.

---

## 3. Python Package Dependencies

| Category | Packages | Purpose |
|:---|---|:---|
| **Deep Learning** | `torch` ≥2.11.0, `torchaudio` ≥2.11.0 | Model definition, training, audio I/O |
| **DSP & Audio** | `librosa`, `soundfile`, `sounddevice`, `pyroomacoustics` | Audio processing, RIR simulation, microphone streaming |
| **Speaker Verification** | `speechbrain` | Pre-trained ECAPA-TDNN teacher model |
| **Inference & Export** | `onnx`, `onnxruntime` | Model export and optimized inference |
| **Evaluation** | `scikit-learn`, `pandas`, `matplotlib` | DET curves, ablation tables, visualizations |
| **Tracking** | `wandb` | Experiment logging and artifact sync |
| **Testing** | `pytest`, `pytest-cov` | Unit testing and coverage |
| **Linting** | `black`, `isort`, `flake8` | Code quality enforcement |

---

## 4. Dataset Setup

To train the models or run the full benchmarking pipeline, acquire these public datasets and organize them under a common `--data-root` directory.

### Directory Structure

```
<data-root>/
├── speech_commands/          # Google Speech Commands v2
│   ├── backward/
│   ├── bed/
│   ├── ...
│   └── validation_list.txt
├── voxceleb/                 # VoxCeleb 1 & 2
│   ├── wav/
│   │   ├── id10001/
│   │   ├── id10002/
│   │   └── ...
├── libriphrase/              # LibriPhrase
│   ├── libriphrase_easy.txt
│   └── libriphrase_hard.txt
└── musan/                    # MUSAN noise
    ├── noise/
    ├── music/
    └── speech/
```

### A. Google Speech Commands v2

```bash
# Auto-download via torchaudio (used by our dataloaders)
# Or manual download:
wget https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz -C <data-root>/speech_commands/
```

| Property | Value |
|:---|---|
| Size | ~2.3 GB |
| Utterances | 105,829 |
| Classes | 35 keywords |
| Format | 16 kHz, 16-bit, mono WAV |

### B. VoxCeleb 1 & 2

```bash
# Option 1: Kaggle-hosted (recommended — no download required on Kaggle)
# Add to Kaggle notebook: "voxceleb" dataset

# Option 2: Manual download
# Download from: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
# VoxCeleb1: ~6 GB | VoxCeleb2: ~80 GB
```

| Property | VoxCeleb 1 | VoxCeleb 2 |
|:---|---:|:---:|
| Speakers | 1,251 | 5,994 |
| Utterances | 153,516 | 1,092,009 |
| Size | ~6 GB | ~80 GB |

### C. LibriPhrase

```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech
# Extract LibriPhrase metadata from the dataset
# Copy triplet lists to <data-root>/libriphrase/
```

### D. MUSAN

```bash
wget https://openslr.org/resources/17/musan.tar.gz
tar -xzf musan.tar.gz -C <data-root>/musan/
```

| Subset | Duration | Files |
|:---|---:|:---:|
| Noise | 6 hrs | 931 |
| Music | 43 hrs | 422 |
| Speech | 60 hrs | 2,010 |

---

## 5. Hardware Acceleration Setup

### For Training (GPU)

```bash
# Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

**Expected:**
```
CUDA: True, Device: Tesla T4
```

### For Inference (CPU + ONNX Runtime)

```bash
# ONNX Runtime auto-detects CPU optimizations
python -c "import onnxruntime; print(f'Providers: {onnxruntime.get_available_providers()}')"
```

---

## 6. Verifying the Setup

Run the automated test suite to confirm everything is configured correctly:

```bash
make test
```

**Expected output (60+ tests):**
```
=========================================================
collected 64 items

tests/test_dataloaders.py .................... [ 78%]
tests/test_models.py .......................  [100%]
=========================================================
✅ 64 passed in 12.3s
```

> [!WARNING]
> Some tests require datasets to be present. Tests that fail due to missing data will skip gracefully with a clear message.

### Quick Verification (No Datasets Required)

```bash
python -c "
from models.disent_v2 import DISENT_KWS_v2
m = DISENT_KWS_v2()
total = sum(p.numel() for p in m.parameters())
print(f'✅ Model built: {total:,} params ({total/1e6:.2f}M)')
assert total < 3_000_000, 'OVER BUDGET!'
"
```

**Expected:**
```
✅ Model built: 1,806,068 params (1.81M)
```

---

## 7. Docker Setup (Optional)

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    sox libsndfile1-dev ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "demo.py", "--help"]
```

```bash
docker build -t disent-kws .
docker run --gpus all -it disent-kws python demo.py --help
```

---

## 8. Troubleshooting

| Problem | Likely Cause | Solution |
|:---|---|:---|
| `uv sync` fails | Missing system build dependencies | `apt-get install build-essential python3-dev` |
| `Mamba` import error | Mamba requires CUDA kernels | Automatic fallback to Dilated Conv1D — no action needed |
| `torchaudio` I/O error | Missing libsndfile | `apt-get install libsndfile1-dev` |
| OOM during training | Batch size too large for GPU | Reduce `BATCH_SIZE` in `config.py` (128 → 64) |
| Dataset not found | Wrong `--data-root` path | Verify directory structure (see §4) |
| Microphone not working | Missing ALSA/PulseAudio | `apt-get install portaudio19-dev` |
| No GPU detected | Missing CUDA drivers | `nvidia-smi` to verify; install CUDA toolkit if missing |

---

## 9. Next Steps

Once your environment is set up and tests pass:

1. **Read the [User Guide](user_guide.md)** — training, enrollment, and real-time demo instructions
2. **Read the [Solution Architecture](solution_architecture.md)** — mathematical foundations, architecture details, and ablation results
3. **Read the [Agentic AI Report](ax.md)** — how agentic AI tools were used in development
4. **Enroll a speaker** — `python src/enrollment/enroll.py --help`
5. **Run the demo** — `python src/demo.py --help`
