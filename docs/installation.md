# Installation & Setup Guide

This document outlines the steps required to set up the environment, install dependencies, and download datasets for reproducing the DISENT-KWS v2 speech disentanglement system.

---

## 1. Prerequisites

- **Operating System:** Linux (Ubuntu 20.04/22.04 LTS recommended) or macOS.
- **Python:** Python 3.10 or 3.11 is required.
- **Hardware:** 
  - Minimum: 8 GB RAM, modern CPU (for inference/demo).
  - Recommended: NVIDIA GPU with CUDA support and >= 16 GB VRAM (for training/ablation).
- **Package Manager:** `uv` is recommended for fast dependency resolution and virtual environment management. Alternatively, standard `pip` can be used.

---

## 2. Environment Setup

### Using `uv` (Recommended)

1. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Clone the repository:**
   ```bash
   git clone https://github.com/ennovatex-io/<your-repo-name>.git
   cd <your-repo-name>
   ```
3. **Synchronize dependencies & create virtual environment:**
   ```bash
   uv sync --all-extras
   ```
   This will automatically create a `.venv` directory, lock dependencies, and install both core and development/testing libraries.

### Using Standard `pip`

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ennovatex-io/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Python Package Dependencies

The main dependencies installed are:
* **Deep Learning Framework:** `torch` (>=2.11.0), `torchaudio` (>=2.11.0)
* **DSP & Audio Utilities:** `librosa`, `soundfile`, `sounddevice`, `pyroomacoustics`
* **Pre-trained Speaker Verification Model:** `speechbrain` (for loading pre-trained teacher model)
* **Model Inference/Export:** `onnx`, `onnxruntime`
* **Evaluation & Experiment Tracking:** `wandb`, `scikit-learn`, `pandas`, `matplotlib`
* **Development/Testing:** `pytest`, `pytest-cov`, `black`, `isort`, `flake8`

---

## 4. Dataset Setup

To train the models or run the full benchmarking and ablation study scripts, you must acquire the following public datasets and configure them under a common `--data-root` folder:

### A. Google Speech Commands (GSC) v2
* **Description:** Used for custom word (keyword) spotting.
* **Download:** [GSC v2 (12-class or full 35-class)](https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
* **Expected Directory Structure:**
  ```
  <data-root>/speech_commands/
  ├── backward/
  ├── bed/
  ├── ...
  └── validation_list.txt
  ```

### B. VoxCeleb 1 (and 2)
* **Description:** Large-scale speaker verification dataset used for speaker-specific embedding learning.
* **Download:** [VoxCeleb 1 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
* **Expected Directory Structure:**
  ```
  <data-root>/voxceleb/
  ├── wav/
  │   ├── id10001/
  │   ├── id10002/
  │   └── ...
  ```

### C. LibriPhrase (Optional, Fallback is GSC)
* **Description:** Used for keyword-level triplet mining and speaker rejection.
* **Download:** [LibriPhrase Dataset](https://github.com/PaddlePaddle/PaddleSpeech)
* **Expected Directory Structure:**
  ```
  <data-root>/libriphrase/
  ├── libriphrase_easy.txt
  ├── libriphrase_hard.txt
  └── ...
  ```

### D. MUSAN (Optional, for Noise Augmentation)
* **Description:** Background noise, babble, and music audio clips for SNR training.
* **Download:** [MUSAN Dataset](https://www.openslr.org/17/)
* **Expected Directory Structure:**
  ```
  <data-root>/musan/
  ├── noise/
  ├── music/
  └── speech/
  ```

---

## 5. Verifying the Setup

To verify that the project is successfully configured and all core modules are functioning correctly, run the automated test suite:

```bash
make test
```

If all 60 tests pass, your environment is ready for training, evaluation, and running the real-time demo.
