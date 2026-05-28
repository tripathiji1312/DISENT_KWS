from __future__ import annotations
import os
import time
import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# ONNX runtime is optional — warn gracefully if missing
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    print("⚠️  onnxruntime not installed — ONNX verification will be skipped.")

import numpy as np

def prepare_qat(model: nn.Module, backend: str = "x86") -> nn.Module:
    model = copy.deepcopy(model)
    model.train()

    torch.backends.quantized.engine = backend
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    try:
        model = torch.ao.quantization.fuse_modules_qat(
            model,
            _find_fuseable_sequences(model),
            inplace=True,
        )
    except Exception as e:
        print(f"⚠️  Fusion skipped (non-critical): {e}")

    model = torch.ao.quantization.prepare_qat(model, inplace=True)
    print("✅  QAT preparation complete — fine-tune for 5 epochs then call finalize_qat()")
    return model


def finalize_qat(model_prep: nn.Module) -> nn.Module:
    """Convert the QAT-prepared + fine-tuned model to a static INT8 model.

    Must be called AFTER the QAT fine-tuning epochs are complete.
    The returned model runs on CPU only.

    Returns:
        model_int8 : quantized model (eval mode, CPU)
    """
    model_int8 = copy.deepcopy(model_prep)
    model_int8.eval()
    model_int8 = torch.ao.quantization.convert(model_int8, inplace=True)
    print("✅  QAT conversion complete — INT8 model ready")
    return model_int8


def _find_fuseable_sequences(model: nn.Module) -> list[list[str]]:
    """Walk the module tree and collect (Conv, BN, ReLU) / (Conv, BN) chains
    expressed as lists of dot-separated attribute paths."""
    sequences = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Sequential):
            children = list(mod.named_children())
            i = 0
            while i < len(children):
                child_name, child = children[i]
                if isinstance(child, (nn.Conv1d, nn.Conv2d)):
                    seq = [f"{name}.{child_name}" if name else child_name]
                    if i + 1 < len(children) and isinstance(
                        children[i + 1][1], (nn.BatchNorm1d, nn.BatchNorm2d)
                    ):
                        seq.append(
                            f"{name}.{children[i+1][0]}" if name else children[i + 1][0]
                        )
                        if i + 2 < len(children) and isinstance(
                            children[i + 2][1], (nn.ReLU, nn.SiLU)
                        ):
                            seq.append(
                                f"{name}.{children[i+2][0]}"
                                if name
                                else children[i + 2][0]
                            )
                            i += 3
                        else:
                            i += 2
                        if len(seq) >= 2:
                            sequences.append(seq)
                    else:
                        i += 1
                else:
                    i += 1
    return sequences

def export_onnx(
    model: nn.Module,
    save_path: str | os.PathLike,
    dummy_input: Optional[torch.Tensor] = None,
    opset: int = 17,
    verify: bool = True,
    tolerance: float = 1e-2,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = copy.deepcopy(model).cpu().eval()

    if dummy_input is None:
        dummy_input = torch.randn(1, config.N_MELS, config.MAX_FRAMES)

    print(f"📦  Exporting ONNX → {save_path} …")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            opset_version=opset,
            input_names=["audio"],
            output_names=["z_phn", "z_spk"],
            dynamic_axes={
                "audio": {0: "batch", 2: "time"},
                "z_phn": {0: "batch"},
                "z_spk": {0: "batch"},
            },
            do_constant_folding=True,
        )
    file_mb = save_path.stat().st_size / 1e6
    print(f"✅  ONNX export complete — {file_mb:.2f} MB")

    if verify and _ORT_AVAILABLE:
        _verify_onnx(model, str(save_path), dummy_input, tolerance)
    elif verify:
        print("⚠️  Skipping verification (onnxruntime not installed)")

    return str(save_path)


def _verify_onnx(
    torch_model: nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
    tolerance: float,
) -> None:
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress verbose ORT logs
    sess = ort.InferenceSession(onnx_path, sess_opts)

    with torch.no_grad():
        torch_out = torch_model(dummy_input)   # (z_phn, z_spk)

    ort_inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
    ort_out    = sess.run(None, ort_inputs)

    for i, (t_out, o_out, label) in enumerate(
        zip(torch_out, ort_out, ["z_phn", "z_spk"])
    ):
        diff = np.abs(t_out.numpy() - o_out).max()
        status = "✅" if diff < tolerance else "❌"
        print(f"  {status}  {label}  max |torch − ort| = {diff:.6f}  (tol={tolerance})")
        if diff >= tolerance:
            raise AssertionError(
                f"ONNX verification FAILED for {label}: diff={diff:.6f} > {tolerance}"
            )
    print("✅  ONNX numerical verification passed")

def profile_latency(
    model_or_onnx_path: nn.Module | str,
    input_shape: tuple[int, ...] = (1, config.N_MELS, config.MAX_FRAMES),
    n_warmup: int = 20,
    n_runs:   int = 100,
    device:   str = "cpu",
) -> dict[str, float]:
    dummy = torch.randn(*input_shape)
    times: list[float] = []

    if isinstance(model_or_onnx_path, (str, Path)):
        # ONNX Runtime path
        if not _ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed; cannot profile ONNX model")
        sess = ort.InferenceSession(str(model_or_onnx_path))
        inp_name = sess.get_inputs()[0].name
        inp_np   = dummy.numpy()
        runner   = lambda: sess.run(None, {inp_name: inp_np})  # noqa: E731
        label    = f"ONNX ({Path(model_or_onnx_path).name})"
    else:
        # PyTorch path
        model = model_or_onnx_path.to(device).eval()
        dummy = dummy.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        runner = lambda: model(dummy)  # noqa: E731
        label  = f"PyTorch [{device}]"

    # Warm-up
    for _ in range(n_warmup):
        runner()
    if device == "cuda" and not isinstance(model_or_onnx_path, (str, Path)):
        torch.cuda.synchronize()

    # Timed runs
    for _ in range(n_runs):
        t0 = time.perf_counter()
        runner()
        if device == "cuda" and not isinstance(model_or_onnx_path, (str, Path)):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)  # → ms

    arr = np.array(times)
    results = {
        "mean_ms": float(arr.mean()),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "min_ms":  float(arr.min()),
        "max_ms":  float(arr.max()),
    }
    print(f"\n⏱  Latency profile — {label}")
    print(f"   mean={results['mean_ms']:.1f} ms  "
          f"p50={results['p50_ms']:.1f} ms  "
          f"p95={results['p95_ms']:.1f} ms  "
          f"[min={results['min_ms']:.1f} / max={results['max_ms']:.1f}]")
    xrt = results["mean_ms"] / 1000.0 / (input_shape[-1] * config.HOP_LENGTH / config.SAMPLE_RATE)
    print(f"   xRT (real-time factor) = {xrt:.4f}  (target < 0.2)")
    return results
def full_export_pipeline(
    float_model:  nn.Module,
    save_dir:     str | os.PathLike,
    qat_loader=None,          # DataLoader for 5-epoch QAT fine-tune; None → skip QAT
    qat_epochs:   int = 5,
    qat_lr:       float = 1e-5,
    opset:        int = 17,
) -> dict[str, object]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}
    dummy = torch.randn(1, config.N_MELS, config.MAX_FRAMES)

    print("\n" + "=" * 55)
    print("  Step 1 — Float model latency baseline")
    print("=" * 55)
    float_model.cpu().eval()
    results["float_latency"] = profile_latency(float_model, device="cpu")

    export_model = float_model   # default: export the float model
    if qat_loader is not None:
        print("\n" + "=" * 55)
        print("  Step 2 — QAT fine-tuning")
        print("=" * 55)
        model_prep = prepare_qat(float_model)
        optimizer  = torch.optim.AdamW(model_prep.parameters(), lr=qat_lr)

        model_prep.train()
        for epoch in range(1, qat_epochs + 1):
            epoch_loss = 0.0
            for batch in qat_loader:
                audio, *labels = batch
                optimizer.zero_grad()
                z_phn, z_spk = model_prep(audio)
                loss = z_phn.pow(2).mean() + z_spk.pow(2).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"   QAT epoch {epoch}/{qat_epochs}  loss={epoch_loss:.4f}")

        export_model = finalize_qat(model_prep)
        print("\n  Step 2b — INT8 model latency")
        results["int8_latency"] = profile_latency(export_model, device="cpu")
    else:
        print("\n  Step 2 — QAT skipped (no qat_loader provided)")

    print("\n" + "=" * 55)
    print("  Step 3 — ONNX Export")
    print("=" * 55)
    onnx_path = save_dir / "disent_kws_v2.onnx"
    export_onnx(export_model, onnx_path, dummy_input=dummy, opset=opset)
    results["onnx_path"] = str(onnx_path)
    results["model_mb"]  = Path(onnx_path).stat().st_size / 1e6

    if _ORT_AVAILABLE:
        print("\n  Step 4 — ONNX Runtime latency")
        results["onnx_latency"] = profile_latency(str(onnx_path))
    else:
        print("\n  Step 4 — ONNX latency skipped (onnxruntime missing)")

    print("\n" + "=" * 55)
    print("  Export Summary")
    print("=" * 55)
    print(f"  ONNX file : {results.get('onnx_path')}")
    print(f"  Size      : {results.get('model_mb', 0):.2f} MB")
    fl = results.get("float_latency", {})
    print(f"  Float CPU : {fl.get('mean_ms', '—'):.1f} ms (mean)" if fl else "")
    il = results.get("int8_latency", {})
    print(f"  INT8 CPU  : {il.get('mean_ms', '—'):.1f} ms (mean)" if il else "")
    ol = results.get("onnx_latency", {})
    print(f"  ORT CPU   : {ol.get('mean_ms', '—'):.1f} ms (mean)" if ol else "")
    print("=" * 55)

    return results

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models.disent_v2 import DISENT_KWS_v2

    torch.manual_seed(0)
    model = DISENT_KWS_v2().eval()

    results = full_export_pipeline(
        float_model=model,
        save_dir="./artifacts",
        qat_loader=None,     # skip QAT in local test
    )
    print("\n✅  export.py self-test complete")
    print(f"   ONNX path : {results['onnx_path']}")
    print(f"   Size      : {results['model_mb']:.2f} MB")
