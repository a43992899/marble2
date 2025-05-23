# marble/utils/io_utils.py

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union, Literal

import torch
import torchaudio


def list_audio_files(
    root: Union[str, Path],
    extensions: Tuple[str, ...] = ('.wav', '.flac', '.mp3', '.webm', '.mp4'),
    recursive: bool = True,
) -> List[Path]:
    """
    列出目录下所有指定后缀的音频文件（Path 列表）。
    """
    root = Path(root)
    if recursive:
        files = [p for p in root.rglob('*') if p.suffix.lower() in extensions]
    else:
        files = [p for p in root.iterdir() if p.suffix.lower() in extensions]
    return sorted(files)


def load_audio(
    file_path: Union[str, Path],
    target_sr: int,
    mono: bool = True,
    normalize: bool = False,
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, int]:
    """
    读取音频并重采样到 target_sr（若不同）。
    返回 (waveform, sample_rate)，waveform 维度为 (1, n_samples)。
    """
    wav, sr = torchaudio.load(str(file_path))  # shape: (C, T)
    if mono and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if normalize:
        wav = wav / wav.abs().max()
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr).to(device)
        wav = resampler(wav.to(device)).cpu()
        sr = target_sr
    return wav, sr


def crop_waveform(
    wav: torch.Tensor,
    sr: int,
    length_sec: Optional[float] = None,
    length_samples: Optional[int] = None,
    random_crop: bool = False,
    pad: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    将 wav 裁剪到指定长度（秒 或 采样点），返回 (cropped_wav, start_idx)。
    - random_crop=True ：在可能范围内随机起点；
    - pad=True        ：不足时在末尾 zero-pad；
    """
    assert not (length_sec and length_samples), "请只指定 length_sec 或 length_samples"
    if length_sec:
        length = int(sr * length_sec)
    elif length_samples:
        length = length_samples
    else:
        return wav, 0

    total = wav.size(-1)
    if total >= length:
        max_start = total - length
        start = random.randint(0, max_start) if random_crop else 0
        cropped = wav[..., start:start + length]
    else:
        start = 0
        if pad:
            pad_amount = length - total
            cropped = torch.nn.functional.pad(wav, (0, pad_amount))
        else:
            cropped = wav
    return cropped, start


def load_and_crop(
    file_path: Union[str, Path],
    target_sr: int,
    crop_sec: Optional[float] = None,
    random_crop: bool = False,
    pad: bool = False,
    mono: bool = True,
    normalize: bool = False,
    device: torch.device = torch.device('cpu'),
    return_start: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """
    一步完成：load_audio → crop_waveform → resample。
    """
    wav, sr = load_audio(file_path, target_sr, mono, normalize, device)
    if crop_sec or pad:
        wav, start = crop_waveform(
            wav, sr,
            length_sec=crop_sec,
            random_crop=random_crop,
            pad=pad
        )
    else:
        start = 0
    return (wav, start) if return_start else wav

AlignMode = Literal['overlap', 'pad', 'discard']


def chunk_waveform(
    wav: torch.Tensor,
    sr: int,
    window_sec: float,
    overlap_pct: float = 0.0,
    align: AlignMode = 'overlap',
) -> List[torch.Tensor]:
    """
    将 [1, T] 的 wav 按滑动窗口切分成固定长度列表。
    - window_sec：窗口长（秒）
    - overlap_pct：重叠率（0–100）
    - align：最后一块的对齐策略：
        'overlap'：和上一块重叠填充
        'pad'    ：末尾 zero-pad
        'discard'：丢弃
    返回 List[Tensor], 每块 shape 都为 (1, sr * window_sec).
    """
    assert 0 <= overlap_pct < 100, "overlap_pct 需在 [0,100) 内"
    win_len = int(sr * window_sec)
    step = int(win_len * (1 - overlap_pct / 100))
    total = wav.size(-1)
    chunks = []

    # 切分主块
    for start in range(0, total, step):
        end = start + win_len
        chunk = wav[..., start:end]
        chunks.append(chunk)

    # 处理末尾
    last = chunks[-1]
    if last.size(-1) < win_len:
        deficit = win_len - last.size(-1)
        if align == 'overlap' and len(chunks) >= 2:
            prev = chunks[-2]
            tail = prev[..., -deficit:]
            chunks[-1] = torch.cat([tail, last], dim=-1)
        elif align == 'pad':
            chunks[-1] = torch.nn.functional.pad(last, (0, deficit))
        elif align == 'discard':
            chunks.pop()
        else:
            raise ValueError(f"未知 align 模式：{align}")

    # 最终确保长度一致
    for c in chunks:
        assert c.size(-1) == win_len

    return chunks
