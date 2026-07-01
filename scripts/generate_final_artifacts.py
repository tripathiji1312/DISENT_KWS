#!/usr/bin/env python3
"""
generate_final_artifacts.py

This script automates the generation of the final model artifacts:
1. Exports the PyTorch model checkpoint to ONNX format (model_final.onnx in the repo root).
2. Generates the ablation study results (ablation_results.csv in the repo root).
3. Evaluates the model to produce the joint Detection Error Trade-off (DET) curve (docs/det_curve.png).

Optionally, it logs/uploads all generated files to Weights & Biases (W&B).
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# Set project root in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import config
from models.disent_v2 import DISENT_KWS_v2
from eval.export import export_onnx
from eval.benchmark import get_prototypes, evaluate_joint_system, plot_det_curve
from eval.ablation import run_ablation_study

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Generate final DISENT-KWS v2 deliverables")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/phase3_hardneg_calibrated.pt",
        help="Path to the calibrated/final PyTorch checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/kaggle/working/data_root",
        help="Path to the dataset root folder",
    )
    parser.add_argument(
        "--ablation-epochs",
        type=int,
        default=10,
        help="Number of epochs to train for each ablation variant",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation and training",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the long ablation study execution",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip the ONNX export",
    )
    parser.add_argument(
        "--skip-det",
        action="store_true",
        help="Skip the DET curve plotting",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="DISENT-KWS-v2",
        help="W&B Project name for artifact logging",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default="final-artifacts-generation",
        help="W&B run name",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Force upload files to wandb",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # -------------------------------------------------------------
    # 1. Load Model from Checkpoint
    # -------------------------------------------------------------
    print(f"📂 Loading model checkpoint from: {args.checkpoint}")
    model = DISENT_KWS_v2()
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        print("✅ Model weights loaded successfully!")
    else:
        print(f"⚠️  Checkpoint not found at {args.checkpoint}!")
        print("Proceeding with randomly initialized model weights (only for fallback/testing).")
        
    model = model.to(device).eval()
    
    # Track paths of successfully generated artifacts for W&B
    generated_artifacts = []

    # -------------------------------------------------------------
    # 2. Export to ONNX
    # -------------------------------------------------------------
    onnx_output_path = project_root / "model_final.onnx"
    if not args.skip_onnx:
        print("\n" + "="*60)
        print("📦  Step 1: Exporting Model to ONNX")
        print("="*60)
        
        # Verify and export
        try:
            dummy_input = torch.randn(1, config.N_MELS, config.MAX_FRAMES).to(device)
            export_onnx(
                model=model,
                save_path=onnx_output_path,
                dummy_input=dummy_input.cpu(),
                verify=True
            )
            print(f"✅ Saved model_final.onnx to: {onnx_output_path}")
            generated_artifacts.append(str(onnx_output_path))
        except Exception as e:
            print(f"❌ ONNX Export failed: {e}")
    else:
        print("⏩ Skipping ONNX Export step.")

    # -------------------------------------------------------------
    # 3. Evaluate and Plot DET Curve
    # -------------------------------------------------------------
    det_output_path = project_root / "docs" / "det_curve.png"
    if not args.skip_det:
        print("\n" + "="*60)
        print("📊  Step 2: Evaluating Model & Plotting DET Curve")
        print("="*60)
        
        try:
            from data.datasets import GSCDataset, VoxCelebDataset
            from torch.utils.data import DataLoader
            
            print(f"📂 Loading test datasets from root: {args.data_root}")
            gsc_test = GSCDataset(os.path.join(args.data_root, "speech_commands"), subset="testing")
            vox_test = VoxCelebDataset(os.path.join(args.data_root, "voxceleb"), max_utts_per_spk=10)
            
            gsc_loader = DataLoader(gsc_test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            vox_loader = DataLoader(vox_test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            
            # Extract prototypes
            kw_protos, spk_protos = get_prototypes(model, gsc_loader, vox_loader, device=device)
            
            # Evaluate joint system
            eval_metrics = evaluate_joint_system(model, gsc_loader, vox_loader, kw_protos, spk_protos, device=device)
            
            # Plot DET Curve
            plot_det_curve(eval_metrics, det_output_path)
            print(f"✅ Saved DET curve to: {det_output_path}")
            generated_artifacts.append(str(det_output_path))
            
            # Print Key Metrics
            print("\n🏆 ACHIEVED BENCHMARKS:")
            print(f"  True Accept (TA) Clean: {eval_metrics['ta_clean']:.2f}%")
            print(f"  False Accept (FA) Speaker Confuser: {eval_metrics['fa_spk_conf']:.2f}%")
            print(f"  False Accept (FA) Keyword Confuser: {eval_metrics['fa_kw_conf']:.2f}%")
            print(f"  Equal Error Rate (EER): {eval_metrics['eer']:.2f}%")
            print(f"  System AUC Score: {eval_metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"❌ DET Curve generation failed: {e}")
            print("Please check if your datasets are correctly located in the data-root.")
    else:
        print("⏩ Skipping DET Curve generation step.")

    # -------------------------------------------------------------
    # 4. Run Ablation Study
    # -------------------------------------------------------------
    ablation_output_path = project_root / "ablation_results.csv"
    if not args.skip_ablation:
        print("\n" + "="*60)
        print("🔬  Step 3: Running Systematic Ablation Study")
        print("="*60)
        print("Warning: This requires training multiple models and can take around 40-50 minutes on GPU.")
        
        try:
            # We redirect ablation output directory to root so it outputs ablation_results.csv
            temp_save_dir = project_root / "ablation_temp"
            run_ablation_study(
                data_root=args.data_root,
                epochs=args.ablation_epochs,
                batch_size=args.batch_size,
                save_dir=str(temp_save_dir)
            )
            
            # Move the generated ablation_results.csv to the project root
            source_csv = temp_save_dir / "ablation_results.csv"
            if source_csv.exists():
                import shutil
                shutil.move(str(source_csv), str(ablation_output_path))
                shutil.rmtree(str(temp_save_dir), ignore_errors=True)
                print(f"✅ Saved ablation_results.csv to: {ablation_output_path}")
                generated_artifacts.append(str(ablation_output_path))
            else:
                print("⚠️  Ablation CSV was not found in the temporary directory.")
        except Exception as e:
            print(f"❌ Ablation study execution failed: {e}")
    else:
        print("⏩ Skipping Ablation Study step.")

    # -------------------------------------------------------------
    # 5. Upload to W&B
    # -------------------------------------------------------------
    if (args.use_wandb or WANDB_AVAILABLE) and generated_artifacts:
        print("\n" + "="*60)
        print("☁️  Step 4: Registering / Uploading Artifacts to W&B")
        print("="*60)
        
        try:
            wandb.init(project=args.wandb_project, name=args.wandb_run, job_type="upload-artifacts")
            for artifact_path in generated_artifacts:
                if os.path.exists(artifact_path):
                    wandb.save(artifact_path)
                    print(f"Registered {Path(artifact_path).name} with W&B!")
            wandb.finish()
            print("✅ All artifacts successfully synced to Weights & Biases!")
        except Exception as e:
            print(f"⚠️  Could not upload to W&B: {e}")
            print("Please make sure you are logged in to W&B (run `wandb login`).")
            
    print("\n🎉 Artifact generation script finished.")


if __name__ == "__main__":
    main()
