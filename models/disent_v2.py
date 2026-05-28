from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from models.bc_resnet  import BCResNet2
from models.temporal   import get_temporal_block
from models.heads      import PhoneticHead, SpeakerHead
from models.scorer     import DualGateScorer


class DISENT_KWS_v2(nn.Module):

    def __init__(
        self,
        embed_dim:   int = config.EMBED_DIM,          # 192
        temporal_ch: int = config.TEMPORAL_CHANNELS,  # 48
    ):
        super().__init__()

        # Shared feature extraction backbone
        self.encoder  = BCResNet2()
        self.temporal = get_temporal_block(temporal_ch)

        # Task-specific embedding heads
        self.phn_head = PhoneticHead(in_ch=temporal_ch, embed_dim=embed_dim)
        self.spk_head = SpeakerHead( in_ch=temporal_ch, embed_dim=embed_dim)

        # Scorer (used only at inference time; trained implicitly via losses)
        self.scorer = DualGateScorer(embed_dim=embed_dim)


    def forward(
        self,
        audio: torch.Tensor,                     # (B, 80, T) or (B, 1, 80, T)
        p_spk: torch.Tensor | None = None,       # (B, 192)   enrolled speaker proto
        p_kw:  torch.Tensor | None = None,       # (B, 192)   enrolled keyword proto
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared encoder
        h = self.encoder(audio)      # (B, 48, T)
        h = self.temporal(h)         # (B, 48, T)

        # Build conditioning vector if both prototypes are available
        cond: torch.Tensor | None = None
        if p_spk is not None and p_kw is not None:
            cond = torch.cat([p_spk, p_kw], dim=-1)   # (B, 384)

        z_phn = self.phn_head(h, cond)   # (B, 192)
        z_spk = self.spk_head(h, cond)   # (B, 192)
        return z_phn, z_spk

    @torch.no_grad()
    def detect(
        self,
        audio: torch.Tensor,          # (B, 80, T)
        p_kw:  torch.Tensor,          # (B, 192) or (1, 192)
        p_spk: torch.Tensor,          # (B, 192) or (1, 192)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """End-to-end detection in a single call.

        Returns:
            score   : (B,)
            sim_kw  : (B,)
            sim_spk : (B,)
        """
        z_phn, z_spk = self.forward(audio, p_spk, p_kw)
        return self.scorer(z_phn, z_spk, p_kw, p_spk)

    def count_params(self, verbose: bool = True) -> int:
        modules = {
            "encoder  (BCResNet2)":   self.encoder,
            "temporal (block)":       self.temporal,
            "phn_head (Conformer)":   self.phn_head,
            "spk_head (ECAPA-Lite)":  self.spk_head,
            "scorer   (DualGate)":    self.scorer,
        }
        total = 0
        if verbose:
            print("─" * 42)
            print(f"{'Module':<28} {'Params':>10}")
            print("─" * 42)
        for name, mod in modules.items():
            n = sum(p.numel() for p in mod.parameters())
            total += n
            if verbose:
                print(f"{name:<28} {n:>10,}")
        if verbose:
            print("─" * 42)
            print(f"{'TOTAL':<28} {total:>10,}  ({total/1e6:.3f}M)")
            print("─" * 42)
        assert total < 3_000_000, (
            f"⛔  OVER BUDGET: {total:,} params  (limit 3,000,000)"
        )
        if verbose:
            print("✅  Within 3 M param budget")
        return total

    def load_pretrained_speaker_blocks(
        self, teacher, n_freeze: int = 2
    ) -> None:
        try:
            teacher_sd = teacher.mods.embedding_model.state_dict()
            for i in range(min(n_freeze, len(self.spk_head.blocks))):
                result = self.spk_head.blocks[i].load_state_dict(
                    {k.split(".", 1)[1]: v for k, v in teacher_sd.items()
                     if k.startswith(f"blocks.{i}.")},
                    strict=False,
                )
                print(f"  block {i}: loaded {len(result.matched_keys)} / "
                      f"{sum(1 for _ in self.spk_head.blocks[i].parameters())} tensors")
                for param in self.spk_head.blocks[i].parameters():
                    param.requires_grad = False
            print(f"✅  Loaded & froze first {n_freeze} ECAPA blocks from SpeechBrain")
        except Exception as exc:
            print(f"⚠️   SpeechBrain transfer skipped: {exc}")

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    print("\n=== DISENT-KWS v2 sanity check ===\n")

    model = DISENT_KWS_v2()
    model.count_params()

    B = 8
    audio = torch.randn(B, config.N_MELS, config.MAX_FRAMES)

    z_phn, z_spk = model(audio)
    assert z_phn.shape == (B, config.EMBED_DIM), z_phn.shape
    assert z_spk.shape == (B, config.EMBED_DIM), z_spk.shape
    print("✅  Forward (no cond) OK")

    p_kw  = torch.randn(B, config.EMBED_DIM)
    p_spk = torch.randn(B, config.EMBED_DIM)
    z_phn2, z_spk2 = model(audio, p_spk, p_kw)
    assert z_phn2.shape == (B, config.EMBED_DIM)
    print("✅  Forward (with cond) OK")

    score, sim_kw, sim_spk = model.detect(audio, p_kw, p_spk)
    assert score.shape == (B,)
    print(f"✅  Detect OK — sample score: {score[0].item():.4f}")

    dummy_loss = z_phn.mean() + z_spk.mean()
    dummy_loss.backward()
    print("✅  Backward pass OK")

    print("\n🎉  All model checks passed.\n")
