from __future__ import annotations
import os
import sys
import copy
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.disent_v2 import DISENT_KWS_v2
from train import train_phase2
from eval.benchmark import evaluate_joint_system, get_prototypes

# Define SOTA Ablation configs as described in the 3-Week Plan
ABLATION_CONFIGS = {
    "full": {},
    "no_disent": {
        "disable_grl": True,
        "disable_club": True
    },
    "no_film": {
        "disable_film": True
    },
    "no_spk_gate": {
        "disable_speaker_head": True
    },
    "no_augmentation": {
        "disable_augmentation": True
    },
    "no_kd": {
        "disable_kd": True
    }
}


def apply_ablation_config(model: nn.Module, cfg: dict) -> None:
    """Applies ablation rules dynamically to the model configuration or modules."""
    # 1. Disable FiLM Conditioning
    if cfg.get("disable_film", False):
        print("🔧 Ablation: Disabling FiLM conditioning in heads...")
        if hasattr(model.phn_head, "film"):
            # Replace FiLM with Identity behavior
            class IdentityFiLM(nn.Module):
                def forward(self, x, cond=None): return x
            model.phn_head.film = IdentityFiLM()
        if hasattr(model.spk_head, "film"):
            class IdentityFiLM(nn.Module):
                def forward(self, x, cond=None): return x
            model.spk_head.film = IdentityFiLM()

    # 2. Disable Speaker Gate (KWS keyword classification only)
    if cfg.get("disable_speaker_head", False):
        print("🔧 Ablation: Disabling Speaker Head (Keyword Verification Only)...")
        # Freeze or zero-out speaker head projections
        for p in model.spk_head.parameters():
            p.requires_grad = False
            nn.init.zeros_(p)


def run_ablation_study(
    data_root: str,
    epochs: int = 10,
    batch_size: int = 64,
    save_dir: str = "ablation_results"
) -> None:
    """Runs the ablation study systematically across all configurations."""
    os.makedirs(save_dir, exist_ok=True)
    results_list = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Systematic Ablation Study on {device}...")
    
    # Try importing datasets
    try:
        from data.datasets import GSCDataset, VoxCelebDataset, LibriPhraseDataset
        from data.augmentations import AudioAugmentor
        
        # Load core evaluation sets
        print("📦 Loading Datasets...")
        gsc_test = GSCDataset(os.path.join(data_root, "speech_commands"), subset="testing")
        vox_test = VoxCelebDataset(os.path.join(data_root, "voxceleb"), max_utts_per_spk=10)
        
        gsc_test_loader = DataLoader(gsc_test, batch_size=batch_size, shuffle=False, num_workers=2)
        vox_test_loader = DataLoader(vox_test, batch_size=batch_size, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        return

    for name, cfg in ABLATION_CONFIGS.items():
        print(f"\n==========================================")
        print(f"  Running Ablation: {name.upper()}")
        print(f"==========================================")
        
        # 1. Initialize clean model
        model = DISENT_KWS_v2()
        apply_ablation_config(model, cfg)
        
        # 2. Setup Augmentation based on config
        aug = None
        if not cfg.get("disable_augmentation", False):
            musan = os.path.join(data_root, "musan")
            aug = AudioAugmentor(musan_path=musan if os.path.exists(musan) else None)
            
        # Initialize training datasets
        gsc_train = GSCDataset(os.path.join(data_root, "speech_commands"), subset="training", augmentor=aug)
        vox_train = VoxCelebDataset(os.path.join(data_root, "voxceleb"), augmentor=aug, max_utts_per_spk=30)
        
        gsc_loader = DataLoader(gsc_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        vox_loader = DataLoader(vox_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        
        try:
            lp_train = LibriPhraseDataset(os.path.join(data_root, "libriphrase"), split="hard", augmentor=aug)
            lp_loader = DataLoader(lp_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        except Exception:
            lp_loader = gsc_loader
            
        # Load pre-trained phase 1 base if available
        p1_best = "checkpoints/phase1_best.pt"
        if os.path.exists(p1_best):
            ckpt = torch.load(p1_best, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)
            print(f"✅ Preloaded Phase 1 base from {p1_best}")
            
        model = model.to(device)
        
        # Override loss components if requested in config
        if cfg.get("disable_grl", False) or cfg.get("disable_club", False):
            # In our training script, GRL & CLUB live in disentangle loss weight.
            # We can modify config parameters dynamically:
            config.CLUB_WEIGHT = 0.0 if cfg.get("disable_club", False) else 0.1
            if cfg.get("disable_grl", False):
                config.GRL_MAX_LAMBDA = 0.0
                
        # 3. Fine-tune for requested epochs
        print(f"🔥 Fine-tuning {name} for {epochs} epochs...")
        train_phase2(
            model=model,
            gsc_loader=gsc_loader,
            vox_loader=vox_loader,
            libriphrase_loader=lp_loader,
            device=device,
            n_epochs=epochs,
            lr=1e-4,
            save_dir=os.path.join(save_dir, name)
        )
        
        # 4. Evaluate
        print(f"📊 Evaluating {name}...")
        model.eval()
        try:
            kw_protos, spk_protos = get_prototypes(model, gsc_test_loader, vox_test_loader, device=device)
            eval_metrics = evaluate_joint_system(model, gsc_test_loader, vox_test_loader, kw_protos, spk_protos, device=device)
            
            results_list.append({
                "Configuration": name,
                "TA Clean (%)": f"{eval_metrics['ta_clean']:.2f}%",
                "EER (%)": f"{eval_metrics['eer']:.2f}%",
                "AUC": f"{eval_metrics['auc']:.4f}",
                "FA (Keyword Confuser)": f"{eval_metrics['fa_kw_conf']:.2f}%",
                "FA (Speaker Confuser)": f"{eval_metrics['fa_spk_conf']:.2f}%"
            })
        except Exception as e:
            print(f"⚠️ Evaluation failed for {name}: {e}")
            results_list.append({
                "Configuration": name,
                "TA Clean (%)": "ERR",
                "EER (%)": "ERR",
                "AUC": "ERR",
                "FA (Keyword Confuser)": "ERR",
                "FA (Speaker Confuser)": "ERR"
            })
            
        # Restore configuration defaults
        config.CLUB_WEIGHT = 0.1
        config.GRL_MAX_LAMBDA = 1.0

    # 5. Generate final CSV and Markdown report
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(save_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Ablation Study complete! Saved CSV to {csv_path}")
    
    print("\n" + "="*70)
    print("                    📊 ABLATION STUDY RESULTS 📊")
    print("="*70)
    print(df.to_markdown(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DISENT-KWS v2 Ablation Runner")
    parser.add_argument("--data-root", type=str, default="/kaggle/working/data_root", help="Data root path")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs to train each variant")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="ablation_results", help="Save directory")
    args = parser.parse_args()
    
    run_ablation_study(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
