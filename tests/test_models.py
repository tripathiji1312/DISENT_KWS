"""
Unit tests for DISENT-KWS v2 model components.
Run: pytest tests/test_models.py -v
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config

from models.bc_resnet import BCResNet2, BCResBlock
from models.temporal import get_temporal_block, DilatedConvTemporalBlock
from models.film import FiLM
from models.heads import AttentiveStatsPool, CausalConformerBlock, SEDWRes2NetBlock, PhoneticHead, SpeakerHead
from models.scorer import DualGateScorer
from models.disent_v2 import DISENT_KWS_v2


# Set random seed for determinism across all tests
@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


class TestBCResNet2:
    """Test BCResNet2 encoder and BCResBlock."""

    def test_bc_res_block_shapes(self):
        """Test BCResBlock shape preservation and downsampling."""
        # 1. Stride 1 (shape preservation)
        block = BCResBlock(channels=16, stride_freq=1)
        x = torch.randn(4, 16, 20, 100)
        out = block(x)
        assert out.shape == (4, 16, 20, 100)

        # 2. Stride 2 (halve frequency dimension)
        block_down = BCResBlock(channels=16, stride_freq=2)
        out_down = block_down(x)
        assert out_down.shape == (4, 16, 10, 100)

    def test_bc_resnet2_shapes(self):
        """Test BCResNet2 with 3D and 4D input shapes."""
        model = BCResNet2()
        B, F, T = 4, config.N_MELS, config.MAX_FRAMES  # 4, 80, 200

        # 3D input: (B, F, T)
        x3d = torch.randn(B, F, T)
        out3d = model(x3d)
        assert out3d.shape == (B, 48, T)
        assert out3d.dtype == torch.float32

        # 4D input: (B, 1, F, T)
        x4d = torch.randn(B, 1, F, T)
        out4d = model(x4d)
        assert out4d.shape == (B, 48, T)

    def test_bc_resnet2_params(self):
        """Verify BCResNet2 parameter count is < 100K (lightweight contract)."""
        model = BCResNet2()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"BCResNet2 params: {params}")
        # Note: BC-ResNet-2 is a lightweight architecture. Ensure it's reasonably small.
        assert params < 100000, f"BCResNet2 params exceed 100K: {params}"

    def test_bc_resnet2_batch_independence(self):
        """Verify that samples in a batch do not influence each other's outputs."""
        model = model = BCResNet2().eval()
        x1 = torch.randn(1, config.N_MELS, config.MAX_FRAMES)
        x2 = torch.randn(1, config.N_MELS, config.MAX_FRAMES)
        
        batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            out_batch = model(batch)
            out_indep1 = model(x1)
            
        assert torch.allclose(out_batch[:1], out_indep1, atol=1e-4)


class TestTemporalBlock:
    """Test DilatedConvTemporalBlock and factory function."""

    def test_factory_returns_block(self):
        """Verify factory returns a valid PyTorch module."""
        block = get_temporal_block(channels=48)
        assert isinstance(block, nn.Module)

    def test_dilated_conv_temporal_block_shapes(self):
        """Test shape preservation and residual connection."""
        block = DilatedConvTemporalBlock(channels=48)
        x = torch.randn(4, 48, 100)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Verify that gradients flow perfectly through the temporal block."""
        block = DilatedConvTemporalBlock(channels=48)
        x = torch.randn(4, 48, 100, requires_grad=True)
        out = block(x)
        loss = out.pow(2).mean()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestFiLM:
    """Test FiLM conditioning layer."""

    def test_film_shapes(self):
        """Verify shape of output matches input."""
        film = FiLM(cond_dim=384, channels=48)
        x = torch.randn(4, 48, 100)
        cond = torch.randn(4, 384)
        out = film(x, cond)
        
        assert out.shape == (4, 48, 100)

    def test_film_identity_at_init(self):
        """FiLM should approximate identity mapping (gamma ≈ 1, beta ≈ 0) at initialization when cond=zeros."""
        film = FiLM(cond_dim=384, channels=48)
        x = torch.randn(4, 48, 100)
        cond = torch.zeros(4, 384)
        out = film(x, cond)
        
        # Output should be extremely close to input
        max_diff = (out - x).abs().max().item()
        assert max_diff < 1e-4, f"FiLM is not identity at init, max_diff={max_diff}"

    def test_film_gradients(self):
        """Verify gradients propagate to both input and conditioning tensors."""
        film = FiLM(cond_dim=384, channels=48)
        x = torch.randn(4, 48, 100, requires_grad=True)
        cond = torch.randn(4, 384, requires_grad=True)
        out = film(x, cond)
        loss = out.pow(2).mean()
        loss.backward()
        
        assert x.grad is not None
        assert cond.grad is not None
        assert not torch.isnan(cond.grad).any()


class TestHeads:
    """Test all specific model heads and blocks."""

    def test_attentive_stats_pool(self):
        """Test AttentiveStatsPool reduces time dimension and doubles channels (mu + sigma)."""
        pool = AttentiveStatsPool(in_dim=48)
        x = torch.randn(4, 48, 100)
        out = pool(x)
        assert out.shape == (4, 96)  # (B, 2 * in_dim)

    def test_causal_conformer_block(self):
        """Test CausalConformerBlock shape preservation."""
        block = CausalConformerBlock(d_model=192)
        x = torch.randn(4, 192, 100)
        out = block(x)
        assert out.shape == (4, 192, 100)

    def test_sedw_res2net_block(self):
        """Test SEDWRes2NetBlock shape preservation."""
        block = SEDWRes2NetBlock(channels=48)
        x = torch.randn(4, 48, 100)
        out = block(x)
        assert out.shape == (4, 48, 100)

    def test_phonetic_head(self):
        """Test PhoneticHead output shape and optional conditioning."""
        head = PhoneticHead(in_ch=48, embed_dim=192)
        x = torch.randn(4, 48, 100)
        cond = torch.randn(4, 384)

        # 1. With conditioning
        out_cond = head(x, cond)
        assert out_cond.shape == (4, 192)

        # 2. Without conditioning
        out_nocond = head(x, cond=None)
        assert out_nocond.shape == (4, 192)

    def test_speaker_head(self):
        """Test SpeakerHead output shape and optional conditioning."""
        head = SpeakerHead(in_ch=48, embed_dim=192)
        x = torch.randn(4, 48, 100)
        cond = torch.randn(4, 384)

        # 1. With conditioning
        out_cond = head(x, cond)
        assert out_cond.shape == (4, 192)

        # 2. Without conditioning
        out_nocond = head(x, cond=None)
        assert out_nocond.shape == (4, 192)


class TestDualGateScorer:
    """Test DualGateScorer and streaming behavior."""

    def test_batch_scorer_shapes(self):
        """Verify output shapes of the gate scorer."""
        scorer = DualGateScorer(embed_dim=192)
        B = 4
        z_phn = torch.randn(B, 192)
        z_spk = torch.randn(B, 192)
        p_kw = torch.randn(B, 192)
        p_spk = torch.randn(B, 192)

        score, sim_kw, sim_spk = scorer(z_phn, z_spk, p_kw, p_spk)
        assert score.shape == (B,)
        assert sim_kw.shape == (B,)
        assert sim_spk.shape == (B,)

    def test_streaming_ema_monotonicity_and_reset(self):
        """Verify EMA convergence, monotonicity, and resetting state."""
        scorer = DualGateScorer(embed_dim=192, ema_alpha=0.7)
        z_phn = torch.randn(1, 192)
        z_spk = torch.randn(1, 192)
        p_kw = torch.randn(1, 192)
        p_spk = torch.randn(1, 192)

        # Initial EMA state is 0.0
        assert scorer._ema_state.item() == 0.0

        # Step streaming multiple times with identical inputs
        scores = []
        for _ in range(5):
            smooth, _ = scorer.detect_streaming(z_phn, z_spk, p_kw, p_spk)
            scores.append(smooth)

        # Verify smoothing effect (should converge to the static score value)
        static_score, _, _ = scorer(z_phn, z_spk, p_kw, p_spk)
        static_val = static_score.item()
        
        # Distances to the final target value should shrink
        diffs = [abs(s - static_val) for s in scores]
        for i in range(len(diffs) - 1):
            assert diffs[i+1] <= diffs[i], "EMA should converge toward static value"

        # Test reset
        scorer.reset()
        assert scorer._ema_state.item() == 0.0


class TestDISENTKWSV2:
    """Test unified model DISENT_KWS_v2."""

    def test_full_model_forward(self):
        """Verify forward shapes with/without conditioning."""
        model = DISENT_KWS_v2()
        B, F, T = 4, config.N_MELS, config.MAX_FRAMES
        audio = torch.randn(B, F, T)

        # 1. No conditioning
        z_phn, z_spk = model(audio)
        assert z_phn.shape == (B, config.EMBED_DIM)
        assert z_spk.shape == (B, config.EMBED_DIM)

        # 2. With conditioning
        p_kw = torch.randn(B, config.EMBED_DIM)
        p_spk = torch.randn(B, config.EMBED_DIM)
        z_phn_c, z_spk_c = model(audio, p_spk=p_spk, p_kw=p_kw)
        assert z_phn_c.shape == (B, config.EMBED_DIM)
        assert z_spk_c.shape == (B, config.EMBED_DIM)

    def test_full_model_detect(self):
        """Verify detect helper method."""
        model = DISENT_KWS_v2()
        B, F, T = 4, config.N_MELS, config.MAX_FRAMES
        audio = torch.randn(B, F, T)
        p_kw = torch.randn(B, config.EMBED_DIM)
        p_spk = torch.randn(B, config.EMBED_DIM)

        score, sim_kw, sim_spk = model.detect(audio, p_kw, p_spk)
        assert score.shape == (B,)
        assert sim_kw.shape == (B,)
        assert sim_spk.shape == (B,)

    def test_full_model_backward(self):
        """Verify full backpropagation pass without NaN/inf gradients."""
        model = DISENT_KWS_v2()
        B, F, T = 4, config.N_MELS, config.MAX_FRAMES
        audio = torch.randn(B, F, T, requires_grad=True)
        
        z_phn, z_spk = model(audio)
        loss = z_phn.pow(2).sum() + z_spk.pow(2).sum()
        loss.backward()

        assert audio.grad is not None
        assert not torch.isnan(audio.grad).any()
        assert not torch.isinf(audio.grad).any()

    def test_param_count(self):
        """Verify parameter count is under 3M constraint."""
        model = DISENT_KWS_v2()
        n = model.count_params()
        assert n < 3000000, f"Unified model has too many parameters: {n}"
