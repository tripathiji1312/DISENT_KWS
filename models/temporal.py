from __future__ import annotations
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
USE_MAMBA = False
try:
    from mamba_ssm import Mamba  # type: ignore
    USE_MAMBA = True
    print("✅  Mamba SSM available — using MambaTemporalBlock")
except Exception:
    print("⚠️   Mamba unavailable — using DilatedConvTemporalBlock fallback")
class MambaTemporalBlock(nn.Module):
    def __init__(
        self,
        d_model: int = config.TEMPORAL_CHANNELS,  # 48
        d_state: int = config.MAMBA_D_STATE,       # 16
        d_conv:  int = config.MAMBA_D_CONV,        # 4
        expand:  int = 2,
    ):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # Mamba expects (B, T, C)
        x = x.transpose(1, 2)          # (B, T, C)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.transpose(1, 2)          # (B, C, T)
        return x + residual
class _CausalDWBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = nn.ConstantPad1d(((kernel - 1) * dilation, 0), 0.0)
        self.dw  = nn.Conv1d(
            channels, channels,
            kernel_size=kernel,
            dilation=dilation,
            groups=channels,    # depth-wise
            bias=False,
        )
        self.pw  = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(self.pad(x))
        x = self.pw(x)
        x = self.act(self.bn(x))
        return x


class DilatedConvTemporalBlock(nn.Module):
    def __init__(self, channels: int = config.TEMPORAL_CHANNELS):
        super().__init__()
        self.layers = nn.ModuleList([
            _CausalDWBlock(channels, kernel=3, dilation=1),
            _CausalDWBlock(channels, kernel=5, dilation=2),
            _CausalDWBlock(channels, kernel=7, dilation=4),
        ])
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm      = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            x = layer(x)
        x = self.norm(self.pointwise(x))
        return x + residual

def get_temporal_block(channels: int = config.TEMPORAL_CHANNELS) -> nn.Module:
    if USE_MAMBA:
        return MambaTemporalBlock(d_model=channels)
    return DilatedConvTemporalBlock(channels=channels)
if __name__ == "__main__":
    torch.manual_seed(0)

    block = get_temporal_block(48)
    print(f"Temporal block type : {block.__class__.__name__}")
    n = sum(p.numel() for p in block.parameters())
    print(f"Params              : {n:,}")

    dummy = torch.randn(4, 48, config.MAX_FRAMES)
    out   = block(dummy)
    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape}"
    print(f"✅  Temporal block forward OK — output: {tuple(out.shape)}")
