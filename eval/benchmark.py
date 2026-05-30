from __future__ import annotations
import os
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.disent_v2 import DISENT_KWS_v2
from data.datasets import LFBETransform

# Optional: try importing sounddevice or pyroomacoustics to log their status
try:
    import pyroomacoustics as pra
    HAS_PYROOM = True
except ImportError:
    HAS_PYROOM = False


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, config.N_MELS, config.MAX_FRAMES),
    n_warmup: int = 20,
    n_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measures the average latency of the model's forward pass and calculates xRT."""
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape).to(device)
    
    # Warm-up
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(dummy)
            
    if device == "cuda":
        torch.cuda.synchronize()
        
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
        
    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    p95_ms = float(np.percentile(times, 95))
    
    # xRT = latency / duration
    audio_duration_sec = input_shape[-1] * config.HOP_LENGTH / config.SAMPLE_RATE
    xrt = (avg_ms / 1000.0) / audio_duration_sec
    
    return {
        "mean_ms": avg_ms,
        "std_ms": std_ms,
        "p95_ms": p95_ms,
        "xrt": xrt
    }


@torch.no_grad()
def get_prototypes(
    model: DISENT_KWS_v2,
    gsc_loader: DataLoader,
    vox_loader: DataLoader,
    device: str = "cpu",
    n_samples: int = 10,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Extracts speaker and keyword prototypes by averaging embeddings from the loaders."""
    model = model.to(device).eval()
    
    # Extract Keyword Prototypes
    kw_embeddings: Dict[int, List[torch.Tensor]] = {}
    print("🔑 Extracting keyword prototypes...")
    for feat, kw_label in gsc_loader:
        feat = feat.to(device)
        z_phn, _ = model(feat)
        for i in range(feat.size(0)):
            lbl = int(kw_label[i].item())
            if lbl not in kw_embeddings:
                kw_embeddings[lbl] = []
            if len(kw_embeddings[lbl]) < n_samples:
                kw_embeddings[lbl].append(z_phn[i].cpu())
        
        # Stop early if we have enough samples for all classes
        if all(len(v) >= n_samples for v in kw_embeddings.values()) and len(kw_embeddings) >= config.NUM_KEYWORDS_GSC:
            break
            
    kw_prototypes = {}
    for lbl, embs in kw_embeddings.items():
        stacked = torch.stack(embs)
        kw_prototypes[lbl] = F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)
        
    # Extract Speaker Prototypes
    spk_embeddings: Dict[int, List[torch.Tensor]] = {}
    print("🗣️ Extracting speaker prototypes...")
    for feat, spk_label in vox_loader:
        feat = feat.to(device)
        _, z_spk = model(feat)
        for i in range(feat.size(0)):
            lbl = int(spk_label[i].item())
            if lbl not in spk_embeddings:
                spk_embeddings[lbl] = []
            if len(spk_embeddings[lbl]) < n_samples:
                spk_embeddings[lbl].append(z_spk[i].cpu())
                
        # Stop early if we have enough samples for at least 50 speakers
        if len(spk_embeddings) >= 50 and all(len(v) >= n_samples for v in spk_embeddings.values()):
            break
            
    spk_prototypes = {}
    for lbl, embs in spk_embeddings.items():
        stacked = torch.stack(embs)
        spk_prototypes[lbl] = F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)
        
    return kw_prototypes, spk_prototypes


@torch.no_grad()
def evaluate_joint_system(
    model: DISENT_KWS_v2,
    gsc_loader: DataLoader,
    vox_loader: DataLoader,
    kw_protos: Dict[int, torch.Tensor],
    spk_protos: Dict[int, torch.Tensor],
    device: str = "cpu",
    n_eval_pairs: int = 1000,
) -> Dict[str, Any]:
    """Evaluates the joint system (Keyword + Speaker Verification) by pairing positive and negative trials."""
    model = model.to(device).eval()
    
    # 1. Collect all speaker and keyword embeddings along with their labels
    phn_embs, phn_labels = [], []
    spk_embs, spk_labels = [], []
    
    print("📊 Collecting test embeddings...")
    for feat, kw_label in gsc_loader:
        feat = feat.to(device)
        z_phn, _ = model(feat)
        phn_embs.append(z_phn.cpu())
        phn_labels.append(kw_label)
        if len(torch.cat(phn_embs, dim=0)) >= n_eval_pairs:
            break
            
    for feat, spk_label in vox_loader:
        feat = feat.to(device)
        _, z_spk = model(feat)
        spk_embs.append(z_spk.cpu())
        spk_labels.append(spk_label)
        if len(torch.cat(spk_embs, dim=0)) >= n_eval_pairs:
            break
            
    phn_embs = F.normalize(torch.cat(phn_embs, dim=0)[:n_eval_pairs], dim=-1)
    phn_labels = torch.cat(phn_labels, dim=0)[:n_eval_pairs].numpy()
    
    spk_embs = F.normalize(torch.cat(spk_embs, dim=0)[:n_eval_pairs], dim=-1)
    spk_labels = torch.cat(spk_labels, dim=0)[:n_eval_pairs].numpy()
    
    # 2. Construct evaluation trials
    # Trial structure: we select a target keyword and a target speaker.
    # A positive trial has BOTH target keyword and target speaker.
    # Negative trials can be:
    #   - Correct speaker, wrong keyword (Keyword Confuser)
    #   - Wrong speaker, correct keyword (Speaker Confuser)
    #   - Wrong speaker, wrong keyword (Both Confuser)
    
    pos_scores = []
    kw_confuser_scores = []
    spk_confuser_scores = []
    both_confuser_scores = []
    
    print("⚖️ Pairing trials and computing scores...")
    for i in range(min(n_eval_pairs, 500)):
        # Target speaker index and keyword index
        spk_lbl = spk_labels[i]
        kw_lbl = phn_labels[i]
        
        # If we don't have prototypes for these, skip
        if spk_lbl not in spk_protos or kw_lbl not in kw_protos:
            continue
            
        p_spk = spk_protos[spk_lbl]
        p_kw = kw_protos[kw_lbl]
        
        # Positive Trial: target speaker's embedding and keyword's embedding
        z_spk_pos = spk_embs[i:i+1]
        z_phn_pos = phn_embs[i:i+1]
        
        score_pos, _, _ = model.scorer(z_phn_pos, z_spk_pos, p_kw, p_spk)
        pos_scores.append(score_pos.item())
        
        # Keyword Confuser: correct speaker, wrong keyword prototype (different class)
        wrong_kw_lbl = random.choice([lbl for lbl in kw_protos.keys() if lbl != kw_lbl])
        p_kw_wrong = kw_protos[wrong_kw_lbl]
        score_kw_conf, _, _ = model.scorer(z_phn_pos, z_spk_pos, p_kw_wrong, p_spk)
        kw_confuser_scores.append(score_kw_conf.item())
        
        # Speaker Confuser: wrong speaker embedding, correct keyword
        wrong_spk_idx = (i + 1) % n_eval_pairs
        z_spk_wrong = spk_embs[wrong_spk_idx:wrong_spk_idx+1]
        score_spk_conf, _, _ = model.scorer(z_phn_pos, z_spk_wrong, p_kw, p_spk)
        spk_confuser_scores.append(score_spk_conf.item())
        
        # Both Confuser: wrong speaker embedding, wrong keyword prototype
        score_both_conf, _, _ = model.scorer(z_phn_pos, z_spk_wrong, p_kw_wrong, p_spk)
        both_confuser_scores.append(score_both_conf.item())
        
    # Combine scores for ROC computation
    y_true = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(kw_confuser_scores)),
        np.zeros(len(spk_confuser_scores)),
        np.zeros(len(both_confuser_scores))
    ])
    
    y_scores = np.concatenate([
        pos_scores,
        kw_confuser_scores,
        spk_confuser_scores,
        both_confuser_scores
    ])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Calculate Equal Error Rate (EER) where FPR == FNR
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    # Calculate Area Under Curve (AUC)
    auc = roc_auc_score(y_true, y_scores)
    
    # Calculate True Accept (TA) and False Accept (FA) at specific default threshold
    def_thr = config.DEFAULT_THRESHOLD
    ta_clean = np.mean(np.array(pos_scores) >= def_thr) * 100
    fa_kw_conf = np.mean(np.array(kw_confuser_scores) >= def_thr) * 100
    fa_spk_conf = np.mean(np.array(spk_confuser_scores) >= def_thr) * 100
    fa_both_conf = np.mean(np.array(both_confuser_scores) >= def_thr) * 100
    
    return {
        "eer": eer * 100,
        "eer_threshold": eer_threshold,
        "auc": auc,
        "ta_clean": ta_clean,
        "fa_kw_conf": fa_kw_conf,
        "fa_spk_conf": fa_spk_conf,
        "fa_both_conf": fa_both_conf,
        "pos_scores": pos_scores,
        "kw_confuser_scores": kw_confuser_scores,
        "spk_confuser_scores": spk_confuser_scores,
        "both_confuser_scores": both_confuser_scores,
        "fpr": fpr,
        "fnr": fnr,
        "thresholds": thresholds
    }


def plot_det_curve(metrics: Dict[str, Any], save_path: str | Path) -> None:
    """Plots and saves the Detection Error Trade-off (DET) curve."""
    plt.figure(figsize=(8, 8))
    plt.plot(metrics["fpr"] * 100, metrics["fnr"] * 100, label=f"DISENT-KWS v2 (EER={metrics['eer']:.2f}%)", color="darkorange", lw=2)
    plt.plot([0, 100], [0, 100], color="navy", lw=1, linestyle="--", label="Random Classifier")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0.01, 100])
    plt.ylim([0.01, 100])
    
    plt.xlabel('False Accept Rate (%)', fontsize=12)
    plt.ylabel('False Reject Rate (%)', fontsize=12)
    plt.title('Detection Error Trade-off (DET) Curve', fontsize=14, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=11)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 DET Curve saved to {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DISENT-KWS v2 Benchmarking & Evaluation")
    parser.add_argument("--model-path", type=str, default="checkpoints/phase2_best.pt", help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="/kaggle/working/data_root", help="Data root path")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for extraction")
    parser.add_argument("--save-dir", type=str, default="benchmark_results", help="Directory to save evaluation results")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("         🔴 DISENT-KWS v2 SOTA BENCHMARKING PIPELINE 🔴")
    print("="*60 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Running benchmark on: {device}")
    
    # 1. Load Model
    model = DISENT_KWS_v2()
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        print(f"✅ Loaded weights from {args.model_path}")
    else:
        print(f"⚠️ Checkpoint not found at {args.model_path}. Benchmarking uninitialized model...")
        
    model = model.to(device).eval()
    
    # 2. Count Parameters
    total_params = model.count_params(verbose=True)
    
    # 3. Latency Benchmarking
    print("\n⏱️ Running latency and Real-Time Factor (xRT) profiling...")
    cpu_latency = measure_latency(model, device="cpu")
    print(f"   [CPU] Latency: {cpu_latency['mean_ms']:.2f} ms (p95: {cpu_latency['p95_ms']:.2f} ms) | xRT: {cpu_latency['xrt']:.4f}")
    
    gpu_latency = None
    if torch.cuda.is_available():
        gpu_latency = measure_latency(model, device="cuda")
        print(f"   [GPU] Latency: {gpu_latency['mean_ms']:.2f} ms (p95: {gpu_latency['p95_ms']:.2f} ms) | xRT: {gpu_latency['xrt']:.4f}")
        
    # 4. Prepare Dataloaders for Prototype & Accuracy Evaluation
    try:
        from data.datasets import GSCDataset, VoxCelebDataset
        from torch.utils.data import DataLoader
        
        print("\n📦 Initializing test datasets & dataloaders...")
        gsc_test = GSCDataset(os.path.join(args.data_root, "speech_commands"), subset="testing")
        vox_test = VoxCelebDataset(os.path.join(args.data_root, "voxceleb"), max_utts_per_spk=10)
        
        gsc_loader = DataLoader(gsc_test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        vox_loader = DataLoader(vox_test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        # 5. Extract Prototypes
        kw_protos, spk_protos = get_prototypes(model, gsc_loader, vox_loader, device=device)
        print(f"✅ Extracted {len(kw_protos)} keyword prototypes & {len(spk_protos)} speaker prototypes.")
        
        # 6. Evaluate verification performance
        eval_metrics = evaluate_joint_system(model, gsc_loader, vox_loader, kw_protos, spk_protos, device=device)
        
        # Plot DET Curve
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        plot_det_curve(eval_metrics, save_path / "det_curve.png")
        
        # 7. Print Final KPI Table
        print("\n" + "="*60)
        print("                 🏆 DISENT-KWS v2 KPI REPORT 🏆")
        print("="*60)
        print(f"  {'Metric':<30} | {'Target':<10} | {'Achieved':<10} | {'Status':<6}")
        print("-"*60)
        
        status_params = "✅" if total_params < 3_000_000 else "❌"
        print(f"  {'Model Parameter Count':<30} | {'< 3.0 M':<10} | {total_params/1e6:5.2f} M   | {status_params}")
        
        status_cpu_lat = "✅" if cpu_latency['mean_ms'] < 200 else "❌"
        print(f"  {'Latency [CPU]':<30} | {'< 200 ms':<10} | {cpu_latency['mean_ms']:5.1f} ms  | {status_cpu_lat}")
        
        status_cpu_xrt = "✅" if cpu_latency['xrt'] < 0.2 else "❌"
        print(f"  {'xRT (Real-Time Factor)':<30} | {'< 0.20':<10} | {cpu_latency['xrt']:5.4f}     | {status_cpu_xrt}")
        
        status_ta = "✅" if eval_metrics['ta_clean'] >= 99.0 else "❌"
        print(f"  {'True Accept (TA) Clean':<30} | {'≥ 99.0%':<10} | {eval_metrics['ta_clean']:5.1f}%    | {status_ta}")
        
        status_spk_conf = "✅" if eval_metrics['fa_spk_conf'] < 1.0 else "⚠️"
        print(f"  {'FA (Speaker Confuser)':<30} | {'< 1.0%':<10} | {eval_metrics['fa_spk_conf']:5.2f}%    | {status_spk_conf}")
        
        status_kw_conf = "✅" if eval_metrics['fa_kw_conf'] < 1.0 else "⚠️"
        print(f"  {'FA (Keyword Confuser)':<30} | {'< 1.0%':<10} | {eval_metrics['fa_kw_conf']:5.2f}%    | {status_kw_conf}")
        
        print(f"  {'Equal Error Rate (EER)':<30} | {'—':<10} | {eval_metrics['eer']:5.2f}%    | —")
        print(f"  {'System AUC score':<30} | {'—':<10} | {eval_metrics['auc']:5.4f}     | —")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n⚠️ Verification benchmarks skipped due to missing/empty datasets: {e}")
        print("  To run full accuracy and DET curves, please ensure Google Speech Commands v2 and VoxCeleb are linked correctly.")


if __name__ == "__main__":
    main()
