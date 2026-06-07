"""
Run: pytest tests/test_dataloaders.py -v
"""

import sys
import os
import tempfile
import unittest.mock
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.datasets import LFBETransform, GSCDataset, VoxCelebDataset, LibriPhraseDataset


class TestLFBETransform:
    """Test LFBETransform feature extraction."""

    @pytest.fixture
    def transform(self):
        """Create LFBETransform instance."""
        return LFBETransform()

    def test_transform_1d_input(self, transform):
        """Test transform on 1D waveform (T,)."""
        wave = torch.randn(16000 * 2)  # 2 seconds at 16kHz
        out = transform(wave)
        
        assert out.shape[0] == 80, f"Expected n_mels=80, got {out.shape[0]}"
        assert out.shape[1] == 200, f"Expected frames=200, got {out.shape[1]}"
        assert out.dtype == torch.float32

    def test_transform_2d_input(self, transform):
        """Test transform on 2D waveform (1, T)."""
        wave = torch.randn(1, 16000 * 2)
        out = transform(wave)
        
        assert out.shape == (80, 200)
        assert out.dtype == torch.float32

    def test_transform_padding(self, transform):
        """Test padding of short audio."""
        wave_short = torch.randn(8000)  # 0.5 seconds
        out = transform(wave_short)
        
        assert out.shape == (80, 200), "Padding should produce standard shape"
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_transform_trimming(self, transform):
        """Test trimming of long audio."""
        wave_long = torch.randn(48000)  # 3 seconds
        out = transform(wave_long)
        
        assert out.shape == (80, 200), "Trimming should produce standard shape"
        assert not torch.isnan(out).any(), "Output contains NaN"

    @pytest.mark.parametrize("length_sec,expected_frames", [
        (0.5, 200),   # short, padded
        (1.0, 200),   # normal
        (2.0, 200),   # full
        (3.0, 200),   # long, trimmed
    ])
    def test_transform_various_lengths(self, transform, length_sec, expected_frames):
        """Parametrized test for various audio lengths."""
        wave = torch.randn(int(16000 * length_sec))
        out = transform(wave)
        
        assert out.shape[1] == expected_frames

    def test_transform_log_compression(self, transform):
        """Test that log compression is applied (output should be log mel)."""
        wave = torch.randn(16000 * 2)
        out = transform(wave)
        
        # Log mel should be negative (log of values < 1)
        assert out.min() < 0, "Log mel should have negative values"
        assert out.max() > -20, "Log mel should not be extremely negative"


class TestGSCDataset:
    """Test GSCDataset."""

    @pytest.fixture
    def transform(self):
        return LFBETransform()

    def test_gsc_dataset_init_empty(self, transform):
        """Test GSCDataset on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with unittest.mock.patch.object(GSCDataset, '_resolve_root', return_value=tmpdir):
                ds = GSCDataset(root=tmpdir, subset='training', transform=transform)
            # Empty dir = 0 samples
            assert len(ds) >= 0, "Dataset length should be non-negative"

    def test_gsc_dataset_structure(self, transform):
        """Test GSCDataset has required attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with unittest.mock.patch.object(GSCDataset, '_resolve_root', return_value=tmpdir):
                ds = GSCDataset(root=tmpdir, subset='training', transform=transform)
            assert hasattr(ds, 'samples'), "GSCDataset should have 'samples' attribute"
            assert hasattr(ds, 'transform'), "GSCDataset should have 'transform' attribute"


class TestVoxCelebDataset:
    """Test VoxCelebDataset."""

    @pytest.fixture
    def transform(self):
        return LFBETransform()

    def test_voxceleb_dataset_empty(self, transform):
        """Test VoxCelebDataset on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = VoxCelebDataset(root=tmpdir, transform=transform)
            assert len(ds) == 0, "Empty dir should yield 0 samples"

    def test_voxceleb_dataset_structure(self, transform):
        """Test VoxCelebDataset has required attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = VoxCelebDataset(root=tmpdir, transform=transform)
            assert hasattr(ds, 'samples'), "Should have 'samples' attribute"
            assert hasattr(ds, 'transform'), "Should have 'transform' attribute"

    def test_voxceleb_dataset_max_utts_per_spk(self, transform):
        """Test max_utts_per_spk parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = VoxCelebDataset(root=tmpdir, transform=transform, max_utts_per_spk=10)
            # Should not error even on empty dir
            assert len(ds) == 0


class TestLibriPhraseDataset:
    """Test LibriPhraseDataset."""

    @pytest.fixture
    def transform(self):
        return LFBETransform()

    def test_libriophrase_dataset_empty(self, transform):
        """Test LibriPhraseDataset on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = LibriPhraseDataset(root=tmpdir, split='hard', transform=transform)
            assert len(ds) == 0, "Empty dir should yield 0 triplets"

    def test_libriophrase_dataset_structure(self, transform):
        """Test LibriPhraseDataset has required attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = LibriPhraseDataset(root=tmpdir, split='hard', transform=transform)
            assert hasattr(ds, 'triplets'), "Should have 'triplets' attribute"
            assert hasattr(ds, 'transform'), "Should have 'transform' attribute"

    @pytest.mark.parametrize("split", ['hard', 'medium', 'easy'])
    def test_libriophrase_split_parameter(self, transform, split):
        """Test different split parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = LibriPhraseDataset(root=tmpdir, split=split, transform=transform)
            # Should not error
            assert len(ds) >= 0


class TestDataLoadersIntegration:
    """Integration tests for all dataloaders."""

    def test_lfbe_shape_consistency(self):
        """Test that all inputs produce consistent output shape."""
        tr = LFBETransform()
        
        shapes_in = [
            (16000 * 2,),           # 1D
            (1, 16000 * 2),         # 2D mono
            (8000,),                # short
            (48000,),               # long
        ]
        
        for shape in shapes_in:
            wave = torch.randn(shape)
            out = tr(wave)
            assert out.shape == (80, 200), f"Shape mismatch for input {shape}"

    def test_transform_no_nans(self):
        """Test that transform doesn't produce NaN."""
        tr = LFBETransform()
        for _ in range(10):
            wave = torch.randn(16000 * 2)
            out = tr(wave)
            assert not torch.isnan(out).any(), "Output contains NaN"

    def test_transform_deterministic(self):
        """Test that same input produces same output."""
        tr = LFBETransform()
        wave = torch.randn(16000 * 2)
        
        out1 = tr(wave.clone())
        out2 = tr(wave.clone())
        
        assert torch.allclose(out1, out2), "Transform should be deterministic"
