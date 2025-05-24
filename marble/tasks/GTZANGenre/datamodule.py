# marble/tasks/GTZANGenre/datamodule.py

import json
import math
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
    基类：将每个文件切成若干不重叠的 clip_seconds 片段，
    ceil保证最后一段也能取到（会被补零）；并做重采样 & 通道对齐。
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

        # 构造 (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels) 的映射
        self.index_map: List[Tuple[int, int, int, int, int]] = []
        self.resamplers = {}
        for file_idx, info in enumerate(self.meta):
            orig_sr = info['sample_rate']
            # 复用同采样率的 resampler
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

            orig_clip_frames = int(self.clip_seconds * orig_sr)
            orig_channels = info['channels']
            total_samples = info['num_samples']

            # 计算整片数量和残片
            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            # 根据 min_clip_ratio 决定是否保留最后残片
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
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path = info['audio_path']
        label = LABEL2IDX[info['label']]

        # 计算偏移并加载
        offset = slice_idx * orig_clip
        waveform, _ = torchaudio.load(
            path,
            frame_offset=offset,
            num_frames=orig_clip
        )  # (orig_channels, orig_clip)

        # 通道对齐 & 单声道处理
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
            last = waveform[-1:].repeat(self.channels - orig_channels, 1)
            waveform = torch.cat([waveform, last], dim=0)

        # 重采样
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # 补齐到目标长度
        if waveform.size(1) < self.clip_len_target:
            pad = self.clip_len_target - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))

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
    """
    LightningDataModule：根据 stage 自动装配 train/val/test Dataset 并返回对应 DataLoader。
    """
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train: dict,
        val: dict,
        test: dict,
        audio_transforms: list | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if audio_transforms is not None:
            self.audio_transforms = [
                instantiate_from_config(cfg) for cfg in audio_transforms
            ]
        else:
            self.audio_transforms = []
        
        self.train_config = train
        self.val_config = val
        self.test_config = test

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            train_ds = instantiate_from_config(self.train_config)
            val_ds   = instantiate_from_config(self.val_config)
            if self.audio_transforms:
                train_ds = AudioTransformDataset(train_ds, self.audio_transforms)
                val_ds   = AudioTransformDataset(val_ds,   self.audio_transforms)
            self.train_dataset = train_ds
            self.val_dataset   = val_ds
        if stage in (None, "test"):
            test_ds = instantiate_from_config(self.test_config)
            if self.audio_transforms:
                test_ds = AudioTransformDataset(test_ds, self.audio_transforms)
            self.test_dataset = test_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,            # 打乱训练片段
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,           # 按序评估
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,           # 按序评估
            num_workers=self.num_workers,
            pin_memory=True,
        )