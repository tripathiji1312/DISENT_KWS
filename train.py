from __future__ import annotations
import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from training import scheduler

sys.path.insert(0, os.path.dirname(__file__))
import config
from models.disent_v2 import DISENT_KWS_v2

try:
    from training.losses      import AAMSoftmax, PrototypicalLoss, rejection_loss, KDLoss
    from training.disentangle import DisentanglementLoss
    from training.scheduler   import grl_lambda_schedule
    _LOSSES_OK = True
except ImportError as e:
    print(f"⚠️  training modules not fully available: {e}")
    _LOSSES_OK = False

try:
    import wandb
    _WANDB_OK = bool(os.environ.get("WANDB_API_KEY"))
except ImportError:
    _WANDB_OK = False

def get_device() -> str:
    if torch.cuda.is_available():
        d = "cuda"
        print(f"🖥  GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = "cpu"
        print("⚠️  No GPU found — training on CPU (very slow)")
    return d


def save_checkpoint(state: dict, path: str | Path, tag: str = "", upload_unique: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Save the standard file locally (e.g., checkpoints/phase1_best.pt)
    torch.save(state, path)
    print(f"💾  Checkpoint saved locally → {path}  {tag}")
    
    if _WANDB_OK:
        if upload_unique:
            # 2. Generate a unique name with the epoch number for W&B
            epoch = state.get("epoch", 0)
            base_name = Path(path).stem          # e.g., "phase1_best"
            ext = Path(path).suffix              # e.g., ".pt"
            
            unique_path = Path(path).parent / f"{base_name}_epoch{epoch:02d}{ext}"
            
            # Save the unique copy
            torch.save(state, unique_path)
            
            # Tell W&B to save the unique file (it will upload immediately)
            wandb.save(str(unique_path))
            print(f"☁️  Unique copy registered with W&B for instant upload → {unique_path}")
        else:
            # Normal periodic files (like "phase1_epoch10.pt") are already unique
            wandb.save(str(path))


def load_checkpoint(path, model, optimizer=None, device="cpu",
                    aam_kw=None, aam_spk=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as e:
            print(f"⚠️  Optimizer state skipped (cross-phase resume): {e}")
            print("    Optimizer starts fresh — expected when crossing Phase 1 → 2")

    if aam_kw and "aam_kw" in ckpt:
        aam_kw.load_state_dict(ckpt["aam_kw"])
    if aam_spk and "aam_spk" in ckpt:
        aam_spk.load_state_dict(ckpt["aam_spk"])

    if scheduler and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"⚠️  Scheduler state skipped: {e}")

    epoch = ckpt.get("epoch", 0)
    print(f"📂  Resumed from {path}  (epoch {epoch})")
    return epoch


def log(metrics: dict, step: int) -> None:
    """Log to W&B if available, always print."""
    msg = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items())
    print(f"  [{step:4d}] {msg}")
    if _WANDB_OK:
        wandb.log(metrics, step=step)
def train_phase1(
    model:        DISENT_KWS_v2,
    gsc_loader,
    vox_loader,
    device:       str,
    n_epochs:     int  = 20,
    lr:           float = 3e-4,
    save_dir:     str   = "checkpoints",
    resume_from:  str | None = None,
) -> None:
    if not _LOSSES_OK:
        raise RuntimeError("training.losses not available — run `git pull` from Swarnim first")

    model = model.to(device).train()

    aam_kw  = AAMSoftmax(config.EMBED_DIM, config.NUM_KEYWORDS_GSC,
                          scale=config.AAM_SCALE, margin=config.AAM_MARGIN).to(device)
    aam_spk = AAMSoftmax(config.EMBED_DIM, config.NUM_SPEAKERS_VOXCELEB,
                          scale=config.AAM_SCALE, margin=config.AAM_MARGIN).to(device)

    params = (list(model.parameters())
              + list(aam_kw.parameters())
              + list(aam_spk.parameters()))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        start_epoch = load_checkpoint(resume_from, model, optimizer, device,
                                    aam_kw=aam_kw, aam_spk=aam_spk)

    if _WANDB_OK:
        _cfg = {
            k: v for k, v in vars(config).items()
            if k.isupper()
            and isinstance(v, (int, float, str, bool, list, tuple))
        }
        wandb.init(project="DISENT-KWS-v2", name="phase1", config=_cfg)

    print(f"\n{'='*55}")
    print(f"  Phase 1 — Pre-training  ({n_epochs} epochs)")
    print(f"{'='*55}\n")

    global_step = 0
    best_loss   = float("inf")

    for epoch in range(start_epoch, n_epochs):
        epoch_loss_kw  = 0.0
        epoch_loss_spk = 0.0
        n_batches      = 0

        for i, (feat, kw_label) in enumerate(gsc_loader):
            feat, kw_label = feat.to(device), kw_label.to(device)
            optimizer.zero_grad()
            z_phn, _ = model(feat)
            loss_kw  = aam_kw(z_phn, kw_label)
            loss_kw.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            epoch_loss_kw += loss_kw.item()
            global_step   += 1
            n_batches     += 1
            if (i + 1) % 50 == 0:
                print(f"  Ep{epoch+1} GSC {i+1}/{len(gsc_loader)} loss={loss_kw.item():.4f}", flush=True)


        for i, (feat, spk_label) in enumerate(vox_loader):
            feat, spk_label = feat.to(device), spk_label.to(device)
            optimizer.zero_grad()
            _, z_spk = model(feat)
            loss_spk = aam_spk(z_spk, spk_label)
            loss_spk.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            epoch_loss_spk += loss_spk.item()
            global_step    += 1
            n_batches      += 1
            if (i + 1) % 50 == 0:
                print(f"  Ep{epoch+1} VoxCeleb {i+1}/{len(vox_loader)} loss={loss_spk.item():.4f}", flush=True)

        scheduler.step()
        avg_kw  = epoch_loss_kw  / max(len(gsc_loader), 1)
        avg_spk = epoch_loss_spk / max(len(vox_loader), 1)
        total   = avg_kw + avg_spk

        log({"epoch": epoch + 1, "loss_kw": avg_kw,
             "loss_spk": avg_spk, "total": total,
             "lr": scheduler.get_last_lr()[0]}, global_step)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "aam_kw": aam_kw.state_dict(), "aam_spk": aam_spk.state_dict(), "scheduler": scheduler.state_dict()},
                f"{save_dir}/phase1_epoch{epoch+1:02d}.pt",
                tag=f"loss={total:.4f}"
            )
# Change this block inside train_phase1:
        if total < best_loss:
            best_loss = total
            save_checkpoint(
                {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "aam_kw": aam_kw.state_dict(), "aam_spk": aam_spk.state_dict(), "scheduler": scheduler.state_dict()},
                f"{save_dir}/phase1_best.pt",
                tag="(best)",
                upload_unique=True  # <--- ADD THIS
            )

    print(f"\n✅  Phase 1 complete — best loss: {best_loss:.4f}\n")

def train_phase2(
    model:        DISENT_KWS_v2,
    gsc_loader,
    vox_loader,
    libriphrase_loader,
    device:       str,
    n_epochs:     int   = 20,
    lr:           float = 1e-4,
    save_dir:     str   = "checkpoints",
    resume_from:  str | None = None,
) -> None:
    if not _LOSSES_OK:
        raise RuntimeError("training.losses not available — run `git pull` from Swarnim first")

    model = model.to(device).train()

    aam_kw    = AAMSoftmax(config.EMBED_DIM, config.NUM_KEYWORDS_GSC,
                            scale=config.AAM_SCALE, margin=config.AAM_MARGIN).to(device)
    aam_spk   = AAMSoftmax(config.EMBED_DIM, config.NUM_SPEAKERS_VOXCELEB,
                            scale=config.AAM_SCALE, margin=config.AAM_MARGIN).to(device)
    proto     = PrototypicalLoss(config.PROTO_SCALE, config.PROTO_MARGIN).to(device)
    disent    = DisentanglementLoss(
                    config.EMBED_DIM,
                    config.NUM_SPEAKERS_VOXCELEB,
                    config.NUM_KEYWORDS_GSC).to(device)
    kd_loss   = KDLoss(config.KD_TEMPERATURE)

    params = (list(model.parameters())
              + list(aam_kw.parameters())
              + list(aam_spk.parameters())
              + list(disent.parameters()))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        loaded_epoch = load_checkpoint(
            resume_from, model, optimizer, device,
            aam_kw=aam_kw, aam_spk=aam_spk,   # warm-start softmax heads
        )
        # Cross-phase guard: detect whether this is a Phase 1 checkpoint
        # (epoch counter from Phase 1 must NOT carry into Phase 2).
        # A checkpoint is treated as a mid-Phase-2 resume ONLY if the file
        # path explicitly contains "phase2" — otherwise always start fresh.
        is_phase2_resume = "phase2" in str(resume_from)
        if is_phase2_resume and loaded_epoch < n_epochs:
            start_epoch = loaded_epoch
            print(f"↩️   Mid-Phase-2 resume from epoch {loaded_epoch}")
        else:
            start_epoch = 0
            print(f"🔄  Cross-phase resume (Phase 1 → Phase 2): "
                  f"loaded weights from epoch {loaded_epoch}, "
                  f"Phase 2 epoch counter reset to 0")

    if _WANDB_OK:
        _cfg = {
            k: v for k, v in vars(config).items()
            if k.isupper()
            and isinstance(v, (int, float, str, bool, list, tuple))
        }
        wandb.init(project="DISENT-KWS-v2", name="phase2", config=_cfg)

    print(f"\n{'='*55}")
    print(f"  Phase 2 — Joint Fine-tuning  ({n_epochs} epochs)")
    print(f"{'='*55}\n")

    global_step = 0
    best_loss   = float("inf")

    for epoch in range(start_epoch, n_epochs):
        lam = grl_lambda_schedule(epoch, n_epochs)   # GRL lambda ramp
        epoch_losses: dict[str, float] = {k: 0.0 for k in
                         ["kw", "spk", "disent", "reject", "total"]}

        for (feat_gsc, kw_label), (feat_vox, spk_label), (anchor, pos, neg, _) in zip(
        gsc_loader, vox_loader, libriphrase_loader):
            feat_gsc  = feat_gsc.to(device)
            feat_vox  = feat_vox.to(device)
            kw_label  = kw_label.to(device)
            spk_label = spk_label.to(device)
            anchor    = anchor.to(device)
            pos       = pos.to(device)
            neg       = neg.to(device)

            # ── Step A: classification + disentanglement (2 forward passes) ───
            optimizer.zero_grad()
            z_phn_g, z_spk_g = model(feat_gsc)
            loss_kw = aam_kw(z_phn_g, kw_label)

            z_phn_v, z_spk_v = model(feat_vox)
            loss_spk    = aam_spk(z_spk_v, spk_label)
            loss_disent = disent(z_phn_v, z_spk_v, spk_label, kw_label, lam)

            loss_A = loss_kw + loss_spk + 0.5 * loss_disent
            loss_A.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            # Explicitly free Step A's activations before Step B
            del z_phn_g, z_spk_g, z_phn_v, z_spk_v
            torch.cuda.empty_cache()

            # ── Step B: rejection / triplet loss (3 forward passes) ──────────
            optimizer.zero_grad()
            anc_phn, _ = model(anchor)
            pos_phn, _ = model(pos)
            neg_phn, _ = model(neg)
            loss_reject = rejection_loss(anc_phn, pos_phn, neg_phn,
                                        config.REJECTION_MARGIN)
            (0.3 * loss_reject).backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            # ── Logging (detach so no graph is kept) ─────────────────────────
            total = loss_A.detach() + 0.3 * loss_reject.detach()
            for k, v in [("kw",     loss_kw),
                        ("spk",    loss_spk),
                        ("disent", loss_disent),
                        ("reject", loss_reject),
                        ("total",  total)]:
                epoch_losses[k] += v.item()
            global_step += 1

        scheduler.step()
        n = max(len(gsc_loader), 1)
        avg = {k: v / n for k, v in epoch_losses.items()}

        log({"epoch": epoch + 1, "grl_lambda": lam,
             "lr": scheduler.get_last_lr()[0], **avg}, global_step)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                f"{save_dir}/phase2_epoch{epoch+1:02d}.pt",
                tag=f"total={avg['total']:.4f}"
            )
# Change this block inside train_phase2:
        if avg["total"] < best_loss:
            best_loss = avg["total"]
            save_checkpoint(
                {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                f"{save_dir}/phase2_best.pt",
                tag="(best)",
                upload_unique=True  # <--- ADD THIS
            )

    print(f"\n✅  Phase 2 complete — best loss: {best_loss:.4f}\n")
    if _WANDB_OK:
        wandb.finish()


# ===========================================================================
#  CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="DISENT-KWS v2 Training")
    parser.add_argument("--phase",      type=int, required=True, choices=[1, 2],
                        help="Training phase: 1=pre-train, 2=joint fine-tune")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--resume",     default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--save-dir",   default="checkpoints")
    parser.add_argument("--data-root",  default="/kaggle/input",
                        help="Root directory where datasets are mounted")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader workers. Use 0 to debug silent crashes.")
    args = parser.parse_args()

    device = get_device()
    model  = DISENT_KWS_v2()
    model.count_params()

    # Build data loaders (Swarnim's modules — must be available for phase runs)
    try:
        from data.datasets      import GSCDataset, VoxCelebDataset, LibriPhraseDataset, LFBETransform as DLFBETransform
        from data.augmentations import AudioAugmentor, SpecAugment
        from torch.utils.data   import DataLoader

        augmentor = AudioAugmentor(
            musan_path=os.path.join(args.data_root, "musan") if os.path.exists(
                os.path.join(args.data_root, "musan")) else None
        )
        xform = DLFBETransform()

        gsc_ds  = GSCDataset(os.path.join(args.data_root, "speech_commands"),
                              subset="training", augmentor=augmentor)
        vox_ds  = VoxCelebDataset(os.path.join(args.data_root, "voxceleb"),
                                   augmentor=augmentor)

        nw = args.num_workers
        pp = {"prefetch_factor": 2} if nw > 0 else {}
        pw = nw > 0

        gsc_loader = DataLoader(gsc_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=nw, pin_memory=True,
                        persistent_workers=pw, **pp)
        vox_loader = DataLoader(vox_ds, batch_size=args.batch_size,
                                shuffle=True, num_workers=nw, pin_memory=True,
                                persistent_workers=pw, **pp)

    except Exception as e:
        print(f"⚠️  DataLoader setup failed: {e}")
        print("    Ensure Swarnim's data/ modules are merged before running training.")
        sys.exit(1)

    if args.phase == 1:
        n = args.epochs or 20
        train_phase1(model, gsc_loader, vox_loader, device,
                     n_epochs=n, save_dir=args.save_dir, resume_from=args.resume)

    elif args.phase == 2:
        try:
            lp_ds = LibriPhraseDataset(
                os.path.join(args.data_root, "libriphrase"), split="hard")
            lp_loader = DataLoader(lp_ds, batch_size=args.batch_size,
                                    shuffle=True, num_workers=4, pin_memory=True)
        except Exception as e:
            print(f"⚠️  LibriPhrase not found ({e}) — using GSC as confuser fallback")
            lp_loader = gsc_loader   # placeholder

        n = args.epochs or 20
        best_ckpt = os.path.join(args.save_dir, "phase1_best.pt")
        resume    = args.resume or (best_ckpt if os.path.exists(best_ckpt) else None)
        train_phase2(model, gsc_loader, vox_loader, lp_loader, device,
                     n_epochs=n, save_dir=args.save_dir, resume_from=resume)


if __name__ == "__main__":
    main()