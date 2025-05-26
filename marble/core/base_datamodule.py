# marble/core/base_datamodule.pyimport torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from typing import Sequence, Union, Dict

from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.utils import instantiate_from_config
from marble.modules.transforms import AudioTransformDataset


class BaseDataModule(pl.LightningDataModule, metaclass=ABCMeta):
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
        print(f"No transforms for stage '{stage}', using original dataset.")
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
