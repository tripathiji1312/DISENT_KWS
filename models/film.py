from __future__ import annotations
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class FiLM(nn.Module):
    def __init__(
        self,
        cond_dim: int = config.EMBED_DIM * 2,   # 384
        channels: int = config.TEMPORAL_CHANNELS, # 48
        hidden:   int = 128,
    ):
        super().__init__()
        self.channels = channels

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels * 2),  # outputs [gamma | beta]
        )

        nn.init.zeros_(self.mlp[-1].weight)
        # gamma bias → 1, beta bias → 0
        with torch.no_grad():
            self.mlp[-1].bias[:channels]  = 1.0  # gamma
            self.mlp[-1].bias[channels:]  = 0.0  # beta

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.mlp(cond)                          # (B, 2C)
        gamma, beta = params.chunk(2, dim=-1)            # each (B, C)
        gamma = gamma.unsqueeze(-1)                      # (B, C, 1)
        beta  = beta.unsqueeze(-1)                       # (B, C, 1)
        return gamma * x + beta                          # broadcast over T
if __name__ == "__main__":
    torch.manual_seed(0)

    film = FiLM(cond_dim=config.EMBED_DIM * 2, channels=config.TEMPORAL_CHANNELS)
    n = sum(p.numel() for p in film.parameters())
    print(f"FiLM params: {n:,}")

    B, C, T = 4, config.TEMPORAL_CHANNELS, config.MAX_FRAMES
    x    = torch.randn(B, C, T)
    cond = torch.randn(B, config.EMBED_DIM * 2)

    out = film(x, cond)
    assert out.shape == (B, C, T), f"Expected ({B},{C},{T}), got {out.shape}"
    print(f"✅  FiLM forward OK — output: {tuple(out.shape)}")

    # Identity behaviour at init: output ≈ input  (gamma≈1, beta≈0)
    film_untrained = FiLM(cond_dim=config.EMBED_DIM * 2, channels=config.TEMPORAL_CHANNELS)
    out_id = film_untrained(x, torch.zeros(B, config.EMBED_DIM * 2))
    err = (out_id - x).abs().max().item()
    print(f"   Near-identity error at init: {err:.6f}  (expect < 0.01)")
