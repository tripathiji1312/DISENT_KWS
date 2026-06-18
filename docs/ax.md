# Agentic AI and Development Tools Report

This report explains the utilization of Agentic AI, large language model reasoning engines, cloud notebook environments, and automated workflows in implementing and optimizing the DISENT-KWS v2 system.

---

## 1. Agentic AI Setup and Developer Tools

The system was developed using a collaborative setup comprising a developer agent, an LLM reasoning backend, and local/cloud execution interfaces:

* **Antigravity (Developer Agent):** The primary agentic execution harness. Antigravity automates environment inspection, directory navigation, code modification, test suite execution, and file relocation. Equipped with filesystem read/write tools and shell execution capabilities, it translates architectural requirements into concrete source code.
* **Claude (Reasoning Engine):** The underlying foundation model driving the agent's reasoning. Claude resolves complex logical constraints, outlines implementation plans, proposes loss function formulations (such as the CLUB mutual information bounds and Gradient Reversal Layers), and debugs shape mismatches from error logs.
* **Kaggle & Google Colab:** Integrated cloud platform execution environments. Due to the high computational requirements of training on datasets like VoxCeleb (1,251 speakers) and Google Speech Commands (35 words), model training loops (Phase 1, 2, and 3) were offloaded to T4 and A100 GPU instances on Kaggle and Google Colab. Code updates were synced between the local workspace and these cloud instances via GitHub.
* **Weights & Biases (W&B):** Used for remote logging, experiment monitoring, and artifact synchronization. This enabled training on remote cloud nodes (Kaggle/Colab) while tracking convergence and syncing model checkpoints back to the developer workspace.

---

## 2. Agentic Workflows and Reasoning Pipelines

To ensure stability, correctness, and speed, we established a strict three-step reasoning pipeline:

```
[Reasoning & Design (Claude)]
            │
            ▼
[Execution & File IO (Antigravity)]
            │
            ▼
[Verification & Training (Kaggle / Colab / pytest)]
```

1. **Reasoning Phase:** Before writing code, Claude analysed the problem statement constraints (<3M parameters, <0.2s xRT, SNR -5dB to 30dB). It mathematically formulated the dual-head layout and selected BC-ResNet-2 as the shared backbone.
2. **Execution Phase:** Antigravity structured the codebase, created files, and modified relative import paths. To comply with the submission guidelines, Antigravity relocated the packages into a dedicated `src/` directory.
3. **Verification Phase:** The agent utilized the `run_command` tool to run the unit test suite (`make test`), checking shape matching, gradient flow, and parameter budgets. The training scripts were then run on Kaggle/Colab to produce the final model parameters, which were verified via ONNX runtime latency tests.

---

## 3. Tool Chaining and Automation

Chaining multiple specialized tools enabled faster and more efficient development:

* **Asynchronous Subagents:** For codebase exploration, research subagents were spawned to read long specification plans and inspect existing repositories without blocking the main agent's execution.
* **ONNX Verification Harness:** A validation pipeline was established where the agent exported the PyTorch model to ONNX, loaded it using ONNX Runtime, and compared outputs against the PyTorch reference tensor using randomized inputs, raising assertions if the numerical discrepancy exceeded $1\times10^{-5}$.
* **Artifact Synthesis Script:** Created `scripts/generate_final_artifacts.py` to automate exporting the final quantized model, running the ablation tests, and plotting the DET curve, eliminating manual command entry.

---

## 4. What Worked and What Did Not Work

### What Worked:
* **Decoupled Unit Testing:** Creating unit tests in the `tests/` directory allowed the agent to verify modules (like the Causal Conformer layers or RIR Simulator) before running full training runs, preventing runtime errors on remote GPU instances.
* **SSM Fallback Design:** Compilation of CUDA-specific selective scan kernels (Mamba) fails on general CPU systems. Setting up the Causal Dilated Conv1D block as an automatic fallback allowed the model to build and run unit tests on local CPU development environments while using Mamba on cloud GPUs.
* **Persistent Artifact Planning:** Using custom markdown plans (`implementation_plan.md`, `task.md`) preserved the context across chat boundaries and compactions, preventing loss of design state.

### What Did Not Work:
* **Local Dataset Loading:** Attempting to store and process large VoxCeleb audio files locally inside sandboxed environments led to memory limits and long download times. This was resolved by using Kaggle-hosted datasets and symlinking inputs locally.
* **Single-Phase Training:** Attempting to train the shared backbone and heads simultaneously with GRL and CLUB MI from random initialization led to representation collapse. We resolved this by training the model in phases (pre-training classification heads first, then introducing disentanglement constraints).
