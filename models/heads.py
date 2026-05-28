from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from models.film import FiLM

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)                                  # (B, T, C)
        alpha = F.softmax(self.attn(xt), dim=1)                 # (B, T, 1)

        mu    = (alpha * xt).sum(dim=1)                         # (B, C)
        var   = (alpha * (xt - mu.unsqueeze(1)) ** 2).sum(dim=1)
        sigma = torch.sqrt(var + 1e-6)                          # (B, C)
        return torch.cat([mu, sigma], dim=1)                    # (B, 2C)


class CausalConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = config.EMBED_DIM,          # 192
        n_heads: int = config.CONFORMER_HEADS,    # 4
        conv_k:  int = config.CONFORMER_CONV_K,   # 15
        dropout: float = 0.1,
    ):
        super().__init__()

        # Feed-forward 1 (Macaron)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Multi-head causal self-attention
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Depthwise causal conv  (k-1 left-pad, trim right)
        self.norm3   = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=conv_k,
            padding=conv_k - 1,   # will trim right side manually
            groups=d_model,
            bias=False,
        )
        self.conv_bn  = nn.BatchNorm1d(d_model)
        self.conv_act = nn.SiLU()

        # Feed-forward 2 (Macaron)
        self.norm4 = nn.LayerNorm(d_model)
        self.ff2   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)       # (B, T, C)
        T = x.size(1)

        # 1. FFN 1 (Macaron ×0.5)
        x = x + 0.5 * self.ff1(self.norm1(x))

        # 2. Causal self-attention
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_out, _ = self.attn(
            self.norm2(x), self.norm2(x), self.norm2(x),
            attn_mask=causal_mask,
        )
        x = x + attn_out

        conv_in  = self.norm3(x).transpose(1, 2)   # (B, C, T)
        conv_out = self.dw_conv(conv_in)            # (B, C, T + k-1)
        conv_out = conv_out[..., :T]                # trim right → causal
        conv_out = self.conv_act(self.conv_bn(conv_out))
        x = x + conv_out.transpose(1, 2)           # back to (B, T, C)

        x = x + 0.5 * self.ff2(self.norm4(x))

        x = self.final_norm(x)
        return x.transpose(1, 2)    # (B, C, T)

class SEDWRes2NetBlock(nn.Module):
    def __init__(
        self,
        channels:  int = config.TEMPORAL_CHANNELS,  # 48
        scale:     int = config.ECAPA_SCALE,         # 4
        se_ratio:  int = config.ECAPA_SE_RATIO,      # 4
    ):
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        width = channels // scale

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, kernel_size=3, padding=1,
                          groups=width, bias=False),       # depth-wise
                nn.Conv1d(width, width, kernel_size=1, bias=False),  # point-wise
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            )
            for _ in range(scale - 1)
        ])

        # Squeeze-Excitation gate
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                          # (B, C, 1)
            nn.Flatten(),                                     # (B, C)
            nn.Linear(channels, channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // se_ratio, channels),
            nn.Sigmoid(),
        )

        self.bn      = nn.BatchNorm1d(channels)
        self.scale   = scale
        self.width   = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        splits   = x.chunk(self.scale, dim=1)   # list of (B, width, T)

        outputs = [splits[0]]
        for i, conv in enumerate(self.convs):
            # Each branch sees current split + accumulated output
            inp = splits[i + 1] + outputs[-1] if i > 0 else splits[i + 1]
            outputs.append(conv(inp))

        x = torch.cat(outputs, dim=1)           # (B, C, T)
        x = self.bn(x)

        se_weight = self.se(x).unsqueeze(-1)    # (B, C, 1)
        x = x * se_weight

        return residual + x

class PhoneticHead(nn.Module):
    def __init__(
        self,
        in_ch:     int = config.TEMPORAL_CHANNELS,  # 48
        embed_dim: int = config.EMBED_DIM,           # 192
    ):
        super().__init__()
        self.film       = FiLM(cond_dim=embed_dim * 2, channels=in_ch)
        self.proj_up    = nn.Sequential(
            nn.Conv1d(in_ch, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self.conformer1 = CausalConformerBlock(embed_dim)
        self.conformer2 = CausalConformerBlock(embed_dim)
        self.pool       = AttentiveStatsPool(embed_dim)
        self.proj_out   = nn.Linear(embed_dim * 2, embed_dim)
        self.norm_out   = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x:    torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is not None:
            x = self.film(x, cond)                # (B, 48, T)
        x = self.proj_up(x)                       # (B, 192, T)
        x = self.conformer1(x)                    # (B, 192, T)
        x = self.conformer2(x)                    # (B, 192, T)
        x = self.pool(x)                          # (B, 384)
        x = self.proj_out(x)                      # (B, 192)
        return self.norm_out(x)                   # (B, 192)

class SpeakerHead(nn.Module):
    def __init__(
        self,
        in_ch:     int = config.TEMPORAL_CHANNELS,  # 48
        embed_dim: int = config.EMBED_DIM,           # 192
    ):
        super().__init__()
        self.film   = FiLM(cond_dim=embed_dim * 2, channels=in_ch)
        self.blocks = nn.Sequential(
            SEDWRes2NetBlock(in_ch),
            SEDWRes2NetBlock(in_ch),
            SEDWRes2NetBlock(in_ch),
        )
        self.pool   = AttentiveStatsPool(in_ch)
        self.proj   = nn.Linear(in_ch * 2, embed_dim)  # 96 → 192
        self.bn     = nn.BatchNorm1d(embed_dim)

    def forward(
        self,
        x:    torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is not None:
            x = self.film(x, cond)          # (B, 48, T)
        x = self.blocks(x)                  # (B, 48, T)
        x = self.pool(x)                    # (B, 96)
        x = self.proj(x)                    # (B, 192)
        return self.bn(x)                   # (B, 192)
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C_in, T = 4, config.TEMPORAL_CHANNELS, config.MAX_FRAMES
    D = config.EMBED_DIM

    x    = torch.randn(B, C_in, T)
    cond = torch.randn(B, D * 2)

    phn = PhoneticHead(C_in, D)
    n_phn = sum(p.numel() for p in phn.parameters())
    print(f"PhoneticHead params : {n_phn:,}")
    z_phn = phn(x, cond)
    assert z_phn.shape == (B, D), f"Expected ({B},{D}), got {z_phn.shape}"
    print(f"✅  PhoneticHead forward OK — output: {tuple(z_phn.shape)}")

    z_phn_nc = phn(x, cond=None)
    assert z_phn_nc.shape == (B, D)
    print("✅  PhoneticHead (no cond) OK")

    spk = SpeakerHead(C_in, D)
    n_spk = sum(p.numel() for p in spk.parameters())
    print(f"\nSpeakerHead params  : {n_spk:,}")
    z_spk = spk(x, cond)
    assert z_spk.shape == (B, D), f"Expected ({B},{D}), got {z_spk.shape}"
    print(f"✅  SpeakerHead forward OK — output: {tuple(z_spk.shape)}")

    print(f"\nCombined head params: {n_phn + n_spk:,}")
