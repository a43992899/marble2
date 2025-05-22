# ssl_benchmark/transforms/audio/waveform.py
import torch
import random
import torchaudio

from benchmark.core.base_transform import BaseTransform
from benchmark.core.registry import register

@register("transforms", "random_crop")
class RandomCrop(BaseTransform):
    def __init__(self, crop_size: int):
        """
        crop_size: 欲裁剪到的样本长度（以采样点计）
        """
        self.crop_size = crop_size

    def __call__(self, sample: dict) -> dict:
        waveform = sample["waveform"]  # Tensor [C, T]
        T = waveform.size(-1)
        if T <= self.crop_size:
            # pad if 太短
            pad = self.crop_size - T
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            start = random.randint(0, T - self.crop_size)
            waveform = waveform[..., start:start + self.crop_size]
        sample["waveform"] = waveform
        return sample

@register("transforms", "random_resample")
class AddNoise(BaseTransform):
    def __init__(self, snr_db: float):
        self.snr = snr_db

    def __call__(self, sample):
        waveform = sample["waveform"]
        rms = waveform.pow(2).mean().sqrt()
        noise_std = rms / (10 ** (self.snr / 20))
        noise = torch.randn_like(waveform) * noise_std
        sample["waveform"] = waveform + noise
        return sample
