from abc import ABC, abstractmethod

class BaseTransform(ABC):
    """所有 Transform 必须继承此类，并实现 __call__。"""
    @abstractmethod
    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample: 包含至少 "waveform" (Tensor[n_channels, n_samples])
        Returns:
            也要是一个 dict，通常添加或替换 sample["waveform"]/["spec"] 等字段
        """
        pass