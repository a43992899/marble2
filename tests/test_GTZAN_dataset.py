import json
import math
import torch
import pytest

import torchaudio
import torch.nn.functional as F
from marble.tasks.GTZANGenre.datamodule import (
    _GTZANGenreAudioBase,
    GTZANGenreAudioTrain,
    GTZANGenreAudioVal,
    GTZANGenreAudioTest,
    GTZANGenreDataModule,
    LABEL2IDX,
)
from marble.core.utils import instantiate_from_config

@pytest.fixture(autouse=True)
def dummy_meta(tmp_path, monkeypatch):
    # Create a dummy JSONL metadata file
    jsonl_path = tmp_path / "meta.jsonl"
    entries = [
        {"audio_path": "dummy.wav", "label": "blues", "duration": 1.0, "sample_rate": 10, "num_samples": 25, "bit_depth": 16, "channels": 2}
    ]
    with open(jsonl_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    # Monkeypatch torchaudio.load to return a deterministic waveform
    def dummy_load(path, frame_offset, num_frames):
        # waveform shape: (2 channels, num_frames)
        waveform = torch.stack([torch.arange(num_frames), torch.arange(num_frames)], dim=0).float()
        return waveform, 10
    monkeypatch.setattr(torchaudio, "load", dummy_load)
    # Monkeypatch torchaudio.transforms.Resample to identity
    monkeypatch.setattr(torchaudio.transforms, "Resample", lambda orig_sr, sr: (lambda x: x))
    return str(jsonl_path)


def test_base_dataset_length_and_indexing(dummy_meta):
    sample_rate = 10
    channels = 1
    clip_seconds = 2.5  # 2.5 * 10 = 25 frames per slice
    dataset = _GTZANGenreAudioBase(
        sample_rate=sample_rate,
        channels=channels,
        clip_seconds=clip_seconds,
        jsonl=dummy_meta,
        channel_mode="first",
    )
    # Only one slice: ceil(25/25)=1
    assert len(dataset) == 1
    waveform, label, path = dataset[0]
    # waveform shape: (channels, clip_seconds * sample_rate)
    expected_frames = int(clip_seconds * sample_rate)
    assert waveform.shape == (channels, expected_frames)
    # label mapping
    assert label == LABEL2IDX["blues"]
    assert path == "dummy.wav"


def test_channel_modes(dummy_meta):
    sample_rate = 10
    channels = 1
    clip_seconds = 1.0  # 10 frames
    for mode in ["first", "mix", "random"]:
        dataset = _GTZANGenreAudioBase(
            sample_rate=sample_rate,
            channels=channels,
            clip_seconds=clip_seconds,
            jsonl=dummy_meta,
            channel_mode=mode,
        )
        waveform, _, _ = dataset[0]
        # Should still match expected length
        assert waveform.shape == (channels, int(clip_seconds * sample_rate))


def test_padding_short_clip(dummy_meta, monkeypatch):
    # Simulate file shorter than clip length
    def short_load(path, frame_offset, num_frames):
        # return fewer frames than requested
        waveform = torch.ones((2, num_frames - 5))
        return waveform, 10
    monkeypatch.setattr(torchaudio, "load", short_load)
    sample_rate = 10
    channels = 2
    clip_seconds = 3.0  # 30 frames target
    dataset = _GTZANGenreAudioBase(
        sample_rate=sample_rate,
        channels=channels,
        clip_seconds=clip_seconds,
        jsonl=dummy_meta,
    )
    waveform, _, _ = dataset[0]
    assert waveform.shape == (channels, int(clip_seconds * sample_rate))


def test_datamodule_and_dataloaders(dummy_meta, monkeypatch):
    # Monkeypatch instantiate_from_config to return a simple train dataset
    def dummy_inst(cfg):
        # ignore cfg, return a train dataset instance
        return GTZANGenreAudioTrain(
            sample_rate=10,
            channels=1,
            clip_seconds=2.5,
            jsonl=dummy_meta,
        )
    monkeypatch.setattr("marble.tasks.GTZANGenre.datamodule.instantiate_from_config", dummy_inst)
    # Create DataModule without additional transforms
    dm = GTZANGenreDataModule(
        batch_size=2,
        num_workers=0,
        train={"class_path": "", "init_args": {}},
        val={"class_path": "", "init_args": {}},
        test={"class_path": "", "init_args": {}},
        audio_transforms=None,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    waveform, label, path = batch
    # Since dataset length is 1, batch size will be 1
    assert waveform.shape == (1, 1, int(2.5 * 10))
    assert label.shape == (1,)
    assert isinstance(path, list) and len(path) == 1
