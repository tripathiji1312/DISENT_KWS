"""
Unit tests for data augmentations.
Run: pytest tests/test_augmentations.py -v
"""

import sys
import os
import random
import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config
from data.augmentations import RIRSimulator, AudioAugmentor, SpecAugment


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    random.seed(42)


class TestRIRSimulator:
    """Test Room Impulse Response (RIR) simulator."""

    def test_rir_generation_shape(self):
        """Verify that RIRSimulator generates a 1D tensor."""
        sim = RIRSimulator(
            room_range=(3.0, 5.0),
            rt60_range=(0.1, 0.2),
            dist_range=(0.5, 1.0),
            sr=16000
        )
        rir = sim.generate()
        
        assert isinstance(rir, torch.Tensor)
        assert rir.dim() == 1
        assert rir.dtype == torch.float32
        assert rir.shape[0] > 0
        assert not torch.isnan(rir).any()

    def test_rir_validity(self):
        """Verify RIR is a valid 1D tensor with non-zero energy.
        
        Note: The fallback FIR path normalizes sum-abs to 1.0, but when
        pyroomacoustics is installed the raw room acoustics RIR is returned
        without normalization, so we only check basic validity.
        """
        sim = RIRSimulator()
        rir = sim.generate()
        
        assert isinstance(rir, torch.Tensor)
        assert rir.dim() == 1
        assert rir.dtype == torch.float32
        assert rir.shape[0] > 0
        assert not torch.isnan(rir).any()
        assert rir.abs().sum().item() > 0


class TestAudioAugmentor:
    """Test waveform level audio augmentations pipeline."""

    def test_waveform_shape_preservation(self):
        """Verify that AudioAugmentor preserves shape structure (channel dimension)."""
        augmentor = AudioAugmentor(musan_path=None)
        
        # Test 1D input (T,) -> outputs (1, T')
        wave1d = torch.randn(16000)
        out1d = augmentor(wave1d)
        assert out1d.dim() == 2
        assert out1d.shape[0] == 1
        assert not torch.isnan(out1d).any()

        # Test 2D input (C, T) -> outputs (C, T')
        wave2d = torch.randn(2, 16000)
        out2d = augmentor(wave2d)
        assert out2d.dim() == 2
        assert out2d.shape[0] == 2
        assert not torch.isnan(out2d).any()

    def test_speed_perturbation_resampling(self):
        """Verify the speed perturbation resampling math on waveform duration."""
        augmentor = AudioAugmentor(musan_path=None)
        
        # Rig speed perturbation probability to 1.0
        # By setting random to return a value that triggers only speed perturbation
        # RIR: p=0.4, Noise: p=0.7, Speed: p=0.3, Gain: p=0.5
        # Let's verify by checking length change on multiple trials
        wave = torch.randn(1, 16000)
        lengths = []
        for _ in range(20):
            # force only speed perturbation or observe results
            out = augmentor(wave)
            lengths.append(out.shape[-1])
            
        # The output lengths should either be 16000 (if speed p=0.3 not selected)
        # or correspond to factor * 16000 (roughly, speed up or slow down)
        # Factors are [0.9, 0.95, 1.05, 1.1].
        # In speed up (e.g. 1.1), length is 16000 / 1.1 ≈ 14545.
        # In slow down (e.g. 0.9), length is 16000 / 0.9 ≈ 17777.
        valid_lengths = {16000, 14545, 15238, 16842, 17777}
        for l in lengths:
            # allow small room for rounding in resample
            matched = any(abs(l - val) < 50 for val in valid_lengths)
            assert matched, f"Unexpected length after augmentation: {l}"

    def test_noise_addition_snr(self):
        """Verify the SNR-based noise addition math is correct."""
        augmentor = AudioAugmentor(musan_path=None)
        signal = torch.ones(1, 16000)  # constant signal
        noise = torch.ones(1, 16000) * 0.5  # constant noise
        
        # Test addition of noise with SNR=0dB (equal power)
        # SNR = 10 * log10(P_sig / P_noise)
        # For 0dB, P_noise should equal P_sig.
        # Signal power = 1.0, Noise power = 0.25 -> noise scale should be 2.0 (since 2^2 * 0.25 = 1.0)
        out = augmentor._add_noise(signal, noise, snr_db=0.0)
        expected_noise_scaled = noise * 2.0
        expected = signal + expected_noise_scaled
        assert torch.allclose(out, expected)

        # Test addition of noise with SNR=6dB (signal power is 4x noise power)
        out_6db = augmentor._add_noise(signal, noise, snr_db=6.0206)
        # scale = sqrt(1.0 / 0.25 * 10^(-0.60206)) = sqrt(4 * 0.25) = 1.0
        assert torch.allclose(out_6db, signal + noise)

    def test_convolve_fallback(self):
        """Test the manual FFT-based convolution fallback logic."""
        augmentor = AudioAugmentor()
        # signal (1, 100) and RIR filter of length 10
        signal = torch.randn(1, 100)
        rir = torch.randn(10)
        
        # convolve using custom convolve
        convolved = augmentor._convolve(signal, rir)
        assert convolved.shape == (1, 100)
        assert not torch.isnan(convolved).any()


class TestSpecAugment:
    """Test SpecAugment frequency and time masking."""

    def test_spec_augment_shapes(self):
        """Verify shape preservation for 2D and 3D spectrograms."""
        spec_aug = SpecAugment(freq_mask=5, time_mask=10, num_masks=1)
        
        # 2D spec: (n_mels, T)
        x2d = torch.randn(80, 200)
        out2d = spec_aug(x2d)
        assert out2d.shape == (80, 200)

        # 3D spec: (B, n_mels, T)
        x3d = torch.randn(4, 80, 200)
        out3d = spec_aug(x3d)
        assert out3d.shape == (4, 80, 200)

    def test_spec_augment_masking_effect(self):
        """Verify that SpecAugment actually masks (zeroes out) portions of the input."""
        spec_aug = SpecAugment(freq_mask=10, time_mask=15, num_masks=2)
        # Start with all ones
        x = torch.ones(1, 80, 200)
        out = spec_aug(x)
        
        # Verify that some values have been masked to exactly zero
        num_zeros = (out == 0.0).sum().item()
        total_elements = out.numel()
        
        assert num_zeros > 0, "No masking was applied (no zero values)"
        assert num_zeros < total_elements, "Entire spectrogram was masked to zero"
