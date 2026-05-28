from __future__ import annotations
import sys
import os
import random
import torch
import torch.nn as nn
import torchaudio

# Add project root to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Try to import pyroomacoustics
HAS_PYROOM = False
try:
    import pyroomacoustics as pra
    HAS_PYROOM = True
except ImportError:
    pass


class RIRSimulator:
    """Simulate room impulse responses at target distances.
    
    Falls back to a high-quality decaying-noise FIR impulse response
    when `pyroomacoustics` is not available.
    """
    def __init__(
        self,
        room_range: tuple[float, float] = config.ROOM_DIM_RANGE,  # (3.0, 10.0)
        rt60_range: tuple[float, float] = config.RT60_RANGE,      # (0.1, 1.0)
        dist_range: tuple[float, float] = config.DISTANCE_RANGE,  # (0.5, 5.0)
        sr: int = config.SAMPLE_RATE,                             # 16000
    ):
        self.room_range = room_range
        self.rt60_range = rt60_range
        self.dist_range = dist_range
        self.sr = sr

    def generate(self) -> torch.Tensor:
        """Generates a 1D room impulse response tensor."""
        room_dim = random.uniform(*self.room_range)
        rt60 = random.uniform(*self.rt60_range)
        dist = random.uniform(*self.dist_range)

        if HAS_PYROOM:
            try:
                width, depth, height = room_dim, room_dim, 3.0
                # Ensure the microphone distance is valid within the room
                if dist >= width:
                    dist = width - 0.5

                # Construct ShoeBox room using Sabine's formula for absorption
                absorption = pra.inverse_sabine(rt60, [width, depth, height])
                # Limit absorption to physically realistic values [0.01, 0.99]
                absorption = max(0.01, min(0.99, absorption))
                materials = pra.Material(absorption)
                
                room = pra.ShoeBox(
                    [width, depth, height],
                    fs=self.sr,
                    materials=materials,
                    max_order=10
                )

                # Source and mic placed at height 1.5m
                source_pos = [width / 2.0, depth / 2.0, 1.5]
                mic_pos = [width / 2.0 + dist, depth / 2.0, 1.5]

                room.add_source(source_pos)
                room.add_microphone(mic_pos)
                room.compute_rir()

                # Get room impulse response (mic 0, source 0)
                rir_np = room.rir[0][0]
                return torch.tensor(rir_np, dtype=torch.float32)
            except Exception:
                # If room acoustics simulation errors, fall back to FIR method
                pass

        # Robust Fallback FIR RIR Method
        # Delay based on speed of sound (343 m/s)
        speed_of_sound = 343.0
        delay_sec = dist / speed_of_sound
        delay_samples = int(self.sr * delay_sec)

        # Decay based on RT60 (reverb time is duration for amplitude to decay by 60 dB, i.e., 10^-3 factor)
        num_samples = int(self.sr * rt60)
        if num_samples < 1:
            num_samples = 1
        t = torch.linspace(0, rt60, num_samples)
        # exponential decay envelope: e^(-6.9078) ≈ 0.001
        decay = torch.exp(-6.9078 * t / rt60)
        noise = torch.randn(num_samples)
        rir_decay = noise * decay

        # Prepend silence to simulate direct-path delay
        if delay_samples > 0:
            rir = torch.cat([torch.zeros(delay_samples), rir_decay])
        else:
            rir = rir_decay

        # Normalize the RIR filter to prevent overall gain shifts
        rir = rir / (rir.abs().sum() + 1e-8)
        return rir


class AudioAugmentor:
    """Waveform-level augmentation pipeline applied BEFORE Mel Spectrogram.
    
    Consists of four pipeline stages, each with independent probability:
      1. RIR convolution (p=0.4)
      2. MUSAN additive noise (p=0.7)
      3. Speed perturbation (p=0.3)
      4. Gain jitter (p=0.5)
    """
    def __init__(self, musan_path: str | None = None, sr: int = config.SAMPLE_RATE):
        self.sr = sr
        self.rir_sim = RIRSimulator(
            room_range=config.ROOM_DIM_RANGE,
            rt60_range=config.RT60_RANGE,
            dist_range=config.DISTANCE_RANGE,
            sr=sr,
        )
        self.musan_path = musan_path
        self.noise_paths = []

        if musan_path and os.path.exists(musan_path):
            for root, _, files in os.walk(musan_path):
                for file in files:
                    if file.endswith(".wav"):
                        self.noise_paths.append(os.path.join(root, file))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Input waveform should be (C, T) or (T,)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 1. RIR Convolution (p=0.4)
        if random.random() < 0.4:
            rir = self.rir_sim.generate()
            waveform = self._convolve(waveform, rir)

        # 2. Additive Noise (p=0.7)
        if random.random() < 0.7 and (self.noise_paths or self.musan_path is not None):
            snr_db = random.uniform(*config.SNR_RANGE)
            noise = self._get_noise(waveform.shape[-1])
            # Match channel size if necessary
            if noise.shape[0] != waveform.shape[0]:
                noise = noise.expand(waveform.shape[0], -1)
            waveform = self._add_noise(waveform, noise, snr_db)

        # 3. Speed Perturbation (p=0.3)
        if random.random() < 0.3:
            factor = random.choice([0.9, 0.95, 1.05, 1.1])
            # Speed perturbation implemented via torchaudio resampling
            target_sr = int(self.sr * factor)
            waveform = torchaudio.functional.resample(waveform, target_sr, self.sr)

        # 4. Gain Jitter (p=0.5)
        if random.random() < 0.5:
            gain_db = random.uniform(-6.0, 6.0)
            factor = 10.0 ** (gain_db / 20.0)
            waveform = waveform * factor

        return waveform

    def _convolve(self, waveform: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
        C, T = waveform.shape
        K = rir.shape[0]
        rir = rir.to(waveform.device, dtype=torch.float32)

        # Try to use fast torchaudio fftconvolve
        try:
            from torchaudio.functional import fftconvolve
            # fftconvolve expects (..., T)
            out = fftconvolve(waveform, rir.unsqueeze(0), mode="full")
            return out[..., :T]
        except Exception:
            # Robust fallback FFT-based 1D convolution
            N = 2 ** ((T + K - 1).bit_length())
            waveform_fft = torch.fft.rfft(waveform, n=N)
            rir_fft = torch.fft.rfft(rir, n=N)
            out_fft = waveform_fft * rir_fft.unsqueeze(0)
            out = torch.fft.irfft(out_fft, n=N)
            return out[..., :T]

    def _get_noise(self, target_len: int) -> torch.Tensor:
        """Helper to get a random noise waveform segment or fallback to synthetic noise."""
        if not self.noise_paths:
            # Fallback to white noise if no musan files are present
            return torch.randn(1, target_len)

        path = random.choice(self.noise_paths)
        try:
            noise, sr = torchaudio.load(path)
            if sr != self.sr:
                noise = torchaudio.functional.resample(noise, sr, self.sr)
            return self._match_length(noise, target_len)
        except Exception:
            # Fallback to gaussian noise if loading fails
            return torch.randn(1, target_len)

    def _match_length(self, noise: torch.Tensor, target_len: int) -> torch.Tensor:
        T = noise.shape[-1]
        if T >= target_len:
            start = random.randint(0, T - target_len)
            return noise[..., start : start + target_len]
        else:
            repeats = (target_len + T - 1) // T
            noise_repeat = noise.repeat(1, repeats)
            return noise_repeat[..., :target_len]

    def _add_noise(self, signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        s_power = signal.pow(2).mean()
        n_power = noise.pow(2).mean()
        if n_power == 0:
            return signal
        scale = torch.sqrt(s_power / (n_power + 1e-8) * (10.0 ** (-snr_db / 10.0)))
        return signal + scale * noise


class SpecAugment(nn.Module):
    """SpecAugment applied on Mel Spectrogram features (frequency & time masking)."""
    def __init__(
        self,
        freq_mask: int = config.SPEC_FREQ_MASK,  # 15
        time_mask: int = config.SPEC_TIME_MASK,  # 25
        num_masks: int = config.SPEC_NUM_MASKS,  # 2
    ):
        super().__init__()
        import torchaudio.transforms as T
        self.freq_masks = nn.ModuleList([T.FrequencyMasking(freq_mask) for _ in range(num_masks)])
        self.time_masks = nn.ModuleList([T.TimeMasking(time_mask) for _ in range(num_masks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_mels, T) or (B, n_mels, T)
        for f_mask in self.freq_masks:
            x = f_mask(x)
        for t_mask in self.time_masks:
            x = t_mask(x)
        return x


if __name__ == "__main__":
    print("=== Testing Augmentation Classes ===")
    torch.manual_seed(42)
    random.seed(42)

    # 1. Test RIRSimulator
    rir_sim = RIRSimulator()
    rir = rir_sim.generate()
    print(f"RIR Simulator Fallback: shape={rir.shape}, sum={rir.sum():.4f}")
    assert rir.dim() == 1, "RIR must be 1D"

    # 2. Test AudioAugmentor
    augmentor = AudioAugmentor()  # no musan path -> uses synthetic fallback
    dummy_wav = torch.randn(1, 32000)  # 2 seconds at 16kHz
    augmented = augmentor(dummy_wav)
    print(f"AudioAugmentor: input={dummy_wav.shape}, output={augmented.shape}")
    assert augmented.dim() == 2, "Output must be (C, T)"
    assert augmented.shape[0] == 1, "Channel shape must be preserved"

    # 3. Test SpecAugment
    spec_aug = SpecAugment()
    dummy_spec = torch.randn(80, 200)  # (n_mels, frames)
    masked_spec = spec_aug(dummy_spec)
    print(f"SpecAugment: input={dummy_spec.shape}, output={masked_spec.shape}")
    assert masked_spec.shape == dummy_spec.shape

    print("🎉 All augmentations self-tests passed!")
