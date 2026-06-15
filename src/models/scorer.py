from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class DualGateScorer(nn.Module):
    def __init__(
        self,
        embed_dim:  int   = config.EMBED_DIM,   # 192
        w_kw:       float = config.SCORE_W_KW,  # 0.55
        w_spk:      float = config.SCORE_W_SPK, # 0.45
        ema_alpha:  float = config.EMA_ALPHA,   # 0.70
    ):
        super().__init__()
        # Learnable gate weights — constrained to be positive via softplus
        self._w_kw_raw  = nn.Parameter(torch.tensor(w_kw))
        self._w_spk_raw = nn.Parameter(torch.tensor(w_spk))
        self.ema_alpha  = ema_alpha

        self.register_buffer("_ema_state", torch.zeros(1))
    @property
    def w_kw(self) -> torch.Tensor:
        return F.softplus(self._w_kw_raw)

    @property
    def w_spk(self) -> torch.Tensor:
        return F.softplus(self._w_spk_raw)
    def forward(
        self,
        z_phn: torch.Tensor,   # (B, D)
        z_spk: torch.Tensor,   # (B, D)
        p_kw:  torch.Tensor,   # (B, D)  or  (1, D)
        p_spk: torch.Tensor,   # (B, D)  or  (1, D)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_phn_n = F.normalize(z_phn, dim=-1)
        z_spk_n = F.normalize(z_spk, dim=-1)
        p_kw_n  = F.normalize(p_kw,  dim=-1)
        p_spk_n = F.normalize(p_spk, dim=-1)

        sim_kw  = F.cosine_similarity(z_phn_n, p_kw_n,  dim=-1)  # (B,)
        sim_spk = F.cosine_similarity(z_spk_n, p_spk_n, dim=-1)  # (B,)

        score = self.w_kw * sim_kw + self.w_spk * sim_spk         # (B,)
        return score, sim_kw, sim_spk
    @torch.no_grad()
    def detect_streaming(
        self,
        z_phn: torch.Tensor,   # (1, D)
        z_spk: torch.Tensor,   # (1, D)
        p_kw:  torch.Tensor,   # (1, D)
        p_spk: torch.Tensor,   # (1, D)
        threshold: float = config.DEFAULT_THRESHOLD,
    ) -> tuple[float, bool]:
        score, _, _ = self.forward(z_phn, z_spk, p_kw, p_spk)
        raw = score.item()
        smooth = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._ema_state.item()
        self._ema_state.fill_(smooth)
        return smooth, smooth >= threshold

    def reset(self) -> None:
        """Reset EMA state — call between utterances / new enrollment sessions."""
        self._ema_state.zero_()
    def extra_repr(self) -> str:
        return (
            f"w_kw={self.w_kw.item():.3f}, "
            f"w_spk={self.w_spk.item():.3f}, "
            f"ema_alpha={self.ema_alpha}"
        )

if __name__ == "__main__":
    torch.manual_seed(0)

    scorer = DualGateScorer()
    n = sum(p.numel() for p in scorer.parameters())
    print(f"DualGateScorer params : {n}  (just the 2 gate scalars)")
    print(scorer)

    B, D = 4, config.EMBED_DIM
    z_phn = torch.randn(B, D)
    z_spk = torch.randn(B, D)
    p_kw  = torch.randn(1, D)
    p_spk = torch.randn(1, D)

    score, sim_kw, sim_spk = scorer(z_phn, z_spk, p_kw, p_spk)
    assert score.shape  == (B,)
    assert sim_kw.shape == (B,)
    print(f"✅  Batch scorer OK — scores: {score.detach().numpy().round(3)}")

    scorer.reset()
    for i in range(5):
        s, triggered = scorer.detect_streaming(
            z_phn[:1], z_spk[:1], p_kw, p_spk
        )
        print(f"   frame {i}: smooth={s:.4f}  triggered={triggered}")
    print("✅  Streaming scorer OK")
