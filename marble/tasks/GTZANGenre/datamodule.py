# marble/tasks/GTZANGenre/datamodule.py

import json
import random
from typing import List, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.utils import instantiate_from_config
from marble.modules.transforms import AudioTransformDataset


LABEL2IDX = {
    'blues': 0, 'classical': 1, 'country': 2, 'disco': 3,
    'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7,
    'reggae': 8, 'rock': 9
}
EXAMPLE_JSONL = {
    "audio_path": "data/GTZAN/genres/blues/blues.00012.wav",
    "label": "blues",
    "duration": 30.013333333333332,
    "sample_rate": 22050,
    "num_samples": 661794,
    "bit_depth": 16,
    "channels": 1
}

class _GTZANGenreAudioBase(Dataset):
    """
    Base dataset for GTZAN genre audio:
    - Splits each audio file into non-overlapping clips of length `clip_seconds` (last clip zero-padded).
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
        label = LABEL2IDX[info['label']]

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


class GTZANGenreAudioTrain(_GTZANGenreAudioBase):
    """
    训练集：DataModule 中设置 shuffle=True。
    """
    pass


class GTZANGenreAudioVal(_GTZANGenreAudioBase):
    """
    验证集：DataModule 中设置 shuffle=False。
    """
    pass


class GTZANGenreAudioTest(GTZANGenreAudioVal):
    """
    测试集：同验证集逻辑。
    """
    pass


class GTZANGenreDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train: dict,
        val: dict,
        test: dict,
        audio_transforms: dict | None = None,  # 改成 dict
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # audio_transforms 是 dict，包含 train/val/test keys
        self.audio_transforms = audio_transforms or {"train": [], "val": [], "test": []}

        self.train_config = train
        self.val_config   = val
        self.test_config  = test

    def _wrap(self, dataset: Dataset, stage: str) -> Dataset:
        """根据 stage 选对应的 transforms 列表来 wrap Dataset"""
        transforms = [
            instantiate_from_config(cfg) 
            for cfg in self.audio_transforms.get(stage, [])
        ]
        if transforms:
            return AudioTransformDataset(dataset, transforms)
        return dataset

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            # 原始 train/val dataset
            train_ds = instantiate_from_config(self.train_config)
            val_ds   = instantiate_from_config(self.val_config)
            # 分别 wrap
            self.train_dataset = self._wrap(train_ds, "train")
            self.val_dataset   = self._wrap(val_ds,   "val")
        if stage in (None, "test"):
            test_ds = instantiate_from_config(self.test_config)
            self.test_dataset = self._wrap(test_ds, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
