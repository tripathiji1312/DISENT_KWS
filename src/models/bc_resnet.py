import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class BCResBlock(nn.Module):
    """Broadcasted Residual Block — the core BC-ResNet innovation.

    Args:
        channels     : number of input AND output channels (no change here).
        stride_freq  : stride along the frequency axis for the 2-D branch.
                       Set to 2 to halve the frequency dimension; 1 otherwise.
    """

    def __init__(self, channels: int, stride_freq: int = 1):
        super().__init__()

        self.freq_conv = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=(3, 1),
                stride=(stride_freq, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.time_conv = nn.Sequential(
            nn.Conv1d(
                channels, channels,
                kernel_size=3,
                padding=1,
                groups=channels,   # depth-wise → tiny param count
                bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),  # point-wise
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        if stride_freq == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.AvgPool2d(kernel_size=(stride_freq, 1), stride=(stride_freq, 1)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, F, T)
        Returns:
            out : (B, C, F', T)   where F' = ceil(F / stride_freq)
        """
        residual = self.skip(x)                   # (B, C, F', T)

        # Frequency branch
        freq_out = self.freq_conv(x)               # (B, C, F', T)

        # Temporal branch: average over freq → 1-D conv → broadcast back
        time_in  = x.mean(dim=2)                   # (B, C, T)
        time_out = self.time_conv(time_in)          # (B, C, T)
        time_out = time_out.unsqueeze(2)            # (B, C, 1, T)

        # Broadcast-add (time_out expands over F' via broadcasting)
        out = residual + freq_out + time_out        # (B, C, F', T)
        return out



class BCResNet2(nn.Module):

    def __init__(self):
        super().__init__()

        # Stem: single conv, no downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Block group 1  (16 channels)
        self.group1 = nn.Sequential(
            BCResBlock(16, stride_freq=1),
            BCResBlock(16, stride_freq=1),
            BCResBlock(16, stride_freq=2),   # F/2
        )

        # Channel expansion 16 → 32
        self.expand1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Block group 2  (32 channels)
        self.group2 = nn.Sequential(
            BCResBlock(32, stride_freq=1),
            BCResBlock(32, stride_freq=2),   # F/4
        )

        # Channel expansion 32 → 48
        self.expand2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Block group 3  (48 channels)
        self.group3 = nn.Sequential(
            BCResBlock(48, stride_freq=1),
            BCResBlock(48, stride_freq=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 80, T)  or  (B, 1, 80, T)
        Returns:
            h : (B, 48, T)  — shared frame-level representation
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)           # (B, 1, 80, T)

        x = self.stem(x)                 # (B, 16, 80, T)
        x = self.group1(x)               # (B, 16, 20, T)
        x = self.expand1(x)              # (B, 32, 20, T)
        x = self.group2(x)               # (B, 32,  5, T)
        x = self.expand2(x)              # (B, 48,  5, T)
        x = self.group3(x)               # (B, 48,  5, T)
        x = x.mean(dim=2)               # (B, 48,  T)  — freq mean-pool
        return x

    def count_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        print(f"BCResNet2 params: {n:,}  ({n/1e6:.3f}M)")
        return n


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BCResNet2()
    model.count_params()

    # Shape contract
    B, F, T = 4, config.N_MELS, config.MAX_FRAMES
    dummy = torch.randn(B, F, T)
    out   = model(dummy)

    assert out.shape == (B, 48, T), f"Expected (4,48,200), got {out.shape}"
    print(f"✅  BCResNet2 forward pass OK — output: {tuple(out.shape)}")

    # Also accept 4-D input
    dummy4d = torch.randn(B, 1, F, T)
    out4d   = model(dummy4d)
    assert out4d.shape == (B, 48, T)
    print("✅  BCResNet2 4-D input OK")
