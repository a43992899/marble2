# marble/modules/transforms.py
import random
import re
from typing import Sequence, Dict, Optional, Union, Tuple, List

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from marble.core.base_transform import BaseEmbTransform, BaseAudioTransform


############################## Audio Transforms ##############################
class AudioTransformDataset(torch.utils.data.Dataset):
    """Sequentially apply BaseAudioTransform instances on raw waveforms."""
    def __init__(self, base_dataset, transforms: list[BaseAudioTransform]):
        self.base = base_dataset
        self.transforms = transforms
        # assume base_dataset has sample_rate attribute
        self.sample_rate = getattr(base_dataset, "sample_rate", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # base[idx] returns:
        #   waveform: Tensor of shape [C, T] (or [1, T] for mono)
        #   label: any (e.g. int)
        #   path: str
        waveform, label, path = self.base[idx]

        # ensure waveform is [C, T]
        assert waveform.ndim == 2 and waveform.shape[0] > 0, \
            f"Expected waveform shape [C, T], got {waveform.shape}"

        sample = {
            "waveform": waveform,            # Tensor [C, T]
            "sampling_rate": self.sample_rate  # int
        }

        # apply each transform in sequence
        for t in self.transforms:
            sample = t(sample)

        # final waveform
        new_wav = sample["waveform"]         # Tensor [C, T] or [T] (for mert)
        return new_wav, label, path


class AudioLayerNorm(BaseAudioTransform):
    """
    Normalize each channel to zero‐mean, unit‐variance over time.

    Args:
        eps (float): to avoid div by zero.
        affine (bool): if True, learn scale & bias per channel.
    """
    def __init__(self, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # gamma, beta: each [1, 1] (broadcast to [C, T])
            self.gamma = nn.Parameter(torch.ones(1, 1))
            self.beta  = nn.Parameter(torch.zeros(1, 1))

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # w: [C, T]
        w = sample["waveform"]
        mean = w.mean(dim=-1, keepdim=True)  # [C, 1]
        std  = w.std(dim=-1, keepdim=True)   # [C, 1]
        # normalized: [C, T]
        w_norm = (w - mean) / (std + self.eps)
        if self.affine:
            # broadcast gamma, beta to [C, T]
            w_norm = w_norm * self.gamma + self.beta
        sample["waveform"] = w_norm          # [C, T]
        return sample
    

class RandomCrop(BaseAudioTransform):
    def __init__(self, crop_size: int):
        """
        Args:
            crop_size (int): target length in samples (T_out).
        """
        super().__init__()
        self.crop_size = crop_size

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # waveform: [C, T]
        waveform = sample["waveform"]
        C, T = waveform.shape
        if T <= self.crop_size:
            pad = self.crop_size - T
            # pad to [C, crop_size]
            waveform = F.pad(waveform, (0, pad))
        else:
            start = random.randint(0, T - self.crop_size)
            # crop to [C, crop_size]
            waveform = waveform[:, start : start + self.crop_size]
        sample["waveform"] = waveform       # [C, crop_size]
        return sample


class AddNoise(BaseAudioTransform):
    """
    Adds random Gaussian noise to the waveform based on a random SNR."""
    def __init__(self, snr_min: float = 5.0, snr_max: float = 20.0): 
        super().__init__()
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, sample):
        # waveform: [C, T]
        waveform = sample["waveform"]
        # 随机采样一个 SNR
        snr = torch.empty(1).uniform_(self.snr_min, self.snr_max).item() # scalar
        rms = waveform.pow(2).mean().sqrt() # scalar
        # noise: [C, T]
        noise_std = rms / (10 ** (snr / 20))
        noise = torch.randn_like(waveform) * noise_std
        sample["waveform"] = waveform + noise
        return sample


class Resample(BaseAudioTransform):
    def __init__(self, orig_freq: int, new_freq: int):
        """
        Args:
            orig_freq (int): original sampling rate.
            new_freq  (int): desired sampling rate.
        """
        super().__init__()
        self.resampler = torchaudio.transforms.Resample(orig_freq, new_freq)

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input waveform: [C, T]
        out = self.resampler(sample["waveform"])
        # output waveform: [C, T_new]
        sample["waveform"] = out
        return sample


class Spectrogram(BaseAudioTransform):
    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        power: float = 2.0,
    ):
        """
        Args:
            n_fft (int): FFT window size.
            win_length (int): window length.
            hop_length (int): hop length between frames.
            power (float): exponent for magnitude.
        """
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length or n_fft,
            hop_length=hop_length or (win_length or n_fft)//2,
            power=power,
        )

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input waveform: [C, T]
        S = self.spec(sample["waveform"])
        # spectrogram: [C, F, T']
        sample["spectrogram"] = S
        return sample


class MelSpectrogram(BaseAudioTransform):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 400,
        n_mels: int = 80,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
    ):
        """
        Args:
            sample_rate (int): sampling rate.
            n_fft (int): FFT window size.
            n_mels (int): number of Mel bins.
            win_length (int): window length.
            hop_length (int): hop between frames.
        """
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length or n_fft,
            hop_length=hop_length or (win_length or n_fft)//2,
            n_mels=n_mels,
        )

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input waveform: [C, T]
        M = self.melspec(sample["waveform"])
        # mel spectrogram: [C, n_mels, T']
        sample["mel_spectrogram"] = M
        return sample


############################## Embedding Transforms ##############################


class LayerSelector(BaseEmbTransform):
    """
    Selects a subset of hidden‐state layers.
    支持整型列表，也支持形如 "start..end" 的字符串范围。
    """
    RANGE_RE = re.compile(r"^(\d+)\.\.(\d+)$")

    def __init__(self, layers: Sequence[Union[int, str]]):
        super().__init__()
        self.layers = self._parse_layers(layers)
        print(f"LayerSelector initialized with layers: {self.layers}")

    def _parse_layers(self, layers):
        parsed = []
        for x in layers:
            if isinstance(x, str):
                m = self.RANGE_RE.match(x.strip())
                if m:
                    start, end = map(int, m.groups())
                    if end < start:
                        raise ValueError(f"Range end ({end}) < start ({start})")
                    parsed.extend(range(start, end+1))
                else:
                    # 如果不是范围，就尝试转成单个 int
                    parsed.append(int(x))
            else:
                parsed.append(int(x))
        return parsed

    def forward(self, hidden_states: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
        selected = [hidden_states[i] for i in self.layers]
        stacked = torch.stack(selected, dim=1)
        assert stacked.ndim == 4, \
            f"Expected 4D tensor after stacking, got {stacked.ndim}D"
        return stacked


class LayerWeightedSum(BaseEmbTransform):
    """
    Learns a weighted sum over L layers via a 1×1 Conv1d.
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=num_layers, out_channels=1, kernel_size=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Weighted sum over layers, of shape
                (batch_size, 1, seq_len, hidden_size).
        """
        x_flat = rearrange(x, 'b l t h -> b l (t h)')
        y = self.conv(x_flat)
        return rearrange(y, 'b 1 (t h) -> b 1 t h', h=x.size(-1))


class MLPReduce(BaseEmbTransform):
    """
    Flattens layers & hidden dims and reduces via an MLP.
    """
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(num_layers * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Reduced representation of shape
                (batch_size, 1, seq_len, hidden_size).
        """
        xt = rearrange(x, 'b l t h -> (b t) (l h)')
        y = self.fc(xt)
        return rearrange(y, '(b t) h -> b 1 t h', t=x.size(2))


class TimeAdaptivePool(BaseEmbTransform):
    """
    Applies adaptive average pooling over time to a fixed length.
    """
    def __init__(self, target_frames: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(target_frames)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Time‐pooled tensor of shape
                (batch_size, num_layers, target_frames, hidden_size).
        """
        x2 = rearrange(x, 'b l t h -> (b l) h t')
        y = self.pool(x2)
        return rearrange(y, '(b l) h t -> b l t h', b=x.size(0), l=x.size(1))


class TimeAvgPool(BaseEmbTransform):
    """
    Computes simple average pooling over the time dimension.
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Time‐averaged tensor of shape
                (batch_size, num_layers, 1, hidden_size).
        """
        return reduce(x, 'b l t h -> b l 1 h', 'mean')


class TimeInterpolation(BaseEmbTransform):
    """
    Interpolates the time dimension to a new fixed length.
    """
    def __init__(self, target_frames: int, mode: str = "linear", align_corners: bool = False):
        super().__init__()
        self.target_frames = target_frames
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Interpolated tensor of shape
                (batch_size, num_layers, target_frames, hidden_size).
        """
        x2 = rearrange(x, 'b l t h -> (b l) h t')
        y = F.interpolate(
            x2,
            size=self.target_frames,
            mode=self.mode,
            align_corners=self.align_corners if self.mode in ("linear", "bilinear", "trilinear") else None
        )
        return rearrange(y, '(b l) h t -> b l t h', b=x.size(0), l=x.size(1))
