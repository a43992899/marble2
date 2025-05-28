# marble/tasks/EMO/datamodule.py

import json
import random
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule


LABEL2IDX = {
    'arousal': 0, 
    'valence': 1
}

EXAMPLE_JSONL = {
    "audio_path": "data/EMO/emomusic/wav/0005.wav", 
    "label": [-0.2568980419200396, 0.25994786922319774], 
    "duration": 45.01004535147392, 
    "sample_rate": 44100, 
    "num_samples": 1984943, 
    "bit_depth": 16, 
    "channels": 1
}

class _EMOAudioBase(Dataset):
    """
    Base dataset for EMO audio:
    - Splits each audio file into non-overlapping clips of length `clip_seconds` (last clip zero-padded).
    - This is a regression task with two labels: arousal and valence.
    """
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.channel_mode = channel_mode
        if channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.min_clip_ratio = min_clip_ratio

        # 读取元数据
        with open(jsonl, 'r') as f:
            self.meta = [json.loads(line) for line in f]

        # Build index map: (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
        self.index_map: List[Tuple[int, int, int, int, int]] = []
        self.resamplers = {}
        for file_idx, info in enumerate(self.meta):
            orig_sr = info['sample_rate']
            # Prepare resampler if needed
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

            orig_clip_frames = int(self.clip_seconds * orig_sr)
            orig_channels = info['channels']
            total_samples = info['num_samples']

            # Number of full clips and remainder
            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            # Decide whether to keep the last shorter clip
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full

            for slice_idx in range(n_slices):
                self.index_map.append(
                    (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
                )


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Load and return one audio clip and its label.

        Inputs:
            idx: int - index of the clip

        Returns:
            waveform: torch.Tensor, shape (self.channels, self.clip_len_target)
            label: int
            path: str
        """
        # Unpack mapping info
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path = info['audio_path']
        label = torch.from_numpy(np.array(info['label'], dtype=np.float32))  # (2,)

        # Compute frame offset and load clip
        offset = slice_idx * orig_clip
        waveform, _ = torchaudio.load(
            path,
            frame_offset=offset,
            num_frames=orig_clip
        )  # (orig_channels, orig_clip)

        # Channel alignment / downmixing
        if orig_channels >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                elif self.channel_mode == "random":
                    # 将 mix 作为一个选项
                    choice = torch.randint(0, orig_channels + 1, (1,)).item()
                    if choice == orig_channels:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        waveform = waveform[choice:choice+1]
                else:
                    raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
            else:
                waveform = waveform[:self.channels]
        else:
            # Repeat last channel to pad to desired channels
            last = waveform[-1:].repeat(self.channels - orig_channels, 1)
            waveform = torch.cat([waveform, last], dim=0)

        # Resample if needed
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # Pad to target length if short
        if waveform.size(1) < self.clip_len_target:
            pad = self.clip_len_target - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))
        
        # Final shape: (self.channels, self.clip_len_target)
        return waveform, label, path


class EMOAudioTrain(_EMOAudioBase):
    """
    训练集：DataModule 中设置 shuffle=True。
    """
    pass


class EMOAudioVal(_EMOAudioBase):
    """
    验证集：DataModule 中设置 shuffle=False。
    """
    pass


class EMOAudioTest(EMOAudioVal):
    """
    测试集：同验证集逻辑。
    """
    pass


class EMODataModule(BaseDataModule):
    pass
