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
├── voxceleb/                 # VoxCeleb1
│   ├── id10001/
│   ├── id10002/
│   └── ...
├── libriphrase/              # LibriPhrase
│   ├── extracted/            # Extracted audio files
│   └── hard_triplets.csv     # Generated triplet CSV
└── musan/                    # MUSAN noise
    ├── noise/
    ├── music/
    └── speech/
```

### A. Google Speech Commands v2

```bash
# Download from Kaggle:
# https://www.kaggle.com/datasets/sylkaladin/speech-commands-v2
# Unzip to <data-root>/speech_commands/
```

Alternatively, use torchaudio's built-in downloader which pulls from TensorFlow:
```python
import torchaudio
torchaudio.datasets.SPEECHCOMMANDS(root="<data-root>", download=True)
```

| Property | Value |
|:---|---|
| Size | ~2.3 GB |
| Utterances | 105,829 |
| Classes | 35 keywords |
| Format | 16 kHz, 16-bit, mono WAV |

### B. VoxCeleb1

```bash
# Option 1: Kaggle-hosted (recommended — no download required on Kaggle)
# Add to Kaggle notebook: the "voxceleb" dataset
# Typical Kaggle path: /kaggle/input/voxceleb/wav/

# Option 2: Manual download
# Download from: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
# VoxCeleb1: ~6 GB | 1,251 speakers | 153,516 utterances
```

| Property | Value |
|:---:|---:|
| Speakers | 1,251 |
| Utterances | 153,516 |
| Size | ~6 GB |

### C. LibriPhrase

LibriPhrase is downloaded from HuggingFace and then we generate hard-triplet pairs using the included generation script:

```python
# Download from HuggingFace and build triplets
import os
import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download

DATA_ROOT     = "<data-root>/libriphrase"
EXTRACT_DIR   = os.path.join(DATA_ROOT, "extracted")
CSV_OUT       = os.path.join(DATA_ROOT, "hard_triplets.csv")
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Download
zip_path = hf_hub_download(
    repo_id="charsiu/libriphrase",
    filename="LibriPhrase_evalset.zip",
    repo_type="dataset"
)
meta_path = hf_hub_download(
    repo_id="charsiu/libriphrase",
    filename="libriphrase_diffspk_all_1word.csv",
    repo_type="dataset"
)

# Extract
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(EXTRACT_DIR)

# Load metadata
df = pd.read_csv(meta_path, encoding="latin-1")

# Build triplets from diffspk_positive (anchor-positive) and
# diffspk_hardneg (anchor-confuser) pairs
positives, negatives = {}, {}
for _, row in df.iterrows():
    label    = str(row["anchor_text"])
    type_lbl = str(row["type"])
    anchor_p  = os.path.join(EXTRACT_DIR, str(row["anchor"]))
    compare_p = os.path.join(EXTRACT_DIR, str(row["comparison"]))
    if type_lbl == "diffspk_positive":
        positives.setdefault(label, []).append((anchor_p, compare_p))
    elif type_lbl == "diffspk_hardneg":
        negatives.setdefault(label, []).append((anchor_p, compare_p))

# Write up to 3000 triplets
lines = []
for label in set(positives) & set(negatives):
    for a, p in positives[label][:len(negatives[label])]:
        if len(lines) >= 3000: break
        _, n = negatives[label][len(lines) % len(negatives[label])]
        if all(os.path.exists(x) for x in [a, p, n]):
            lines.append(f"{a},{p},{n},{label}\n")

with open(CSV_OUT, "w") as f:
    f.writelines(lines)

print(f"Created {len(lines)} triplets → {CSV_OUT}")
```

| Property | Value |
|:---|---|
| Source | [charsiu/libriphrase on HuggingFace](https://huggingface.co/datasets/charsiu/libriphrase) |
| Triplets Generated | up to 3,000 hard-negative pairs |
| Format | 16 kHz, mono WAV |

### D. MUSAN

```bash
# Download from Kaggle:
# https://www.kaggle.com/datasets/nhattruongdev/musan-noise
# Unzip to <data-root>/musan/
```

| Subset | Duration | Files |
|:---|---:|:---:|
| Noise | 6 hrs | 931 |
| Music | 43 hrs | 422 |
| Speech | 60 hrs | 2,010 |

---

## 5. Pre-trained Model Download (Inference Only)

If you only want to run inference/demo (not train from scratch), download the pre-trained model checkpoint from HuggingFace:

```bash
# Install huggingface-cli if not already
pip install huggingface_hub

# Download the final model checkpoint
huggingface-cli download tripathiji1312/DISENT-KWS model_final.pt --local-dir .

# Download ONNX export (for optimized CPU inference)
huggingface-cli download tripathiji1312/DISENT-KWS model_final.onnx --local-dir .
```

**Expected files:**
```
model_final.pt       — PyTorch checkpoint (1.806M params, ~7 MB)
model_final.onnx     — ONNX runtime export (0.60 MB)
```

> [!NOTE]
> The ONNX model works on CPU without any GPU dependencies — verify with `python -c "import onnxruntime as ort; print(ort.InferenceSession('model_final.onnx'))"`.

---

## 6. Hardware Acceleration Setup

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

## 7. Verifying the Setup

Run the automated test suite to confirm everything is configured correctly:

```bash
make test
```

**Expected output (60+ tests):**
```
=========================================================
collected 60 items

tests/test_dataloaders.py ...................... [ 50%]
tests/test_models.py .....................       [ 85%]
tests/test_training.py .........                [100%]
=========================================================
✅ 60 passed in 6.1s
```

> [!WARNING]
> Some tests require datasets to be present. Tests that fail due to missing data will skip gracefully with a clear message.

### Quick Verification (No Datasets Required)

```bash
python -c "
import sys; sys.path.insert(0, 'src')
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
