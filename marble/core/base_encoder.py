import torch
from abc import ABCMeta, abstractmethod
from typing import Optional

class BaseEncoder(torch.nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for encoders. Subclasses need to implement the forward method to encode raw audio or spectrogram 
    into a feature representation.
    
    Output shape: [batch, time_steps, feature_dim]
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Optionally initialize layers or parameters based on kwargs

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward method to map the input (audio, spectrogram, or mel-spectrogram) to high-dimensional features.

        Args:
            input_tensor: Tensor of shape [batch, time] for audio, or [batch, time_steps, feature_dim] for spectrograms
            input_len: Optional tensor of sequence lengths for each input in the batch.

        Returns:
            Tensor of shape [batch, time_steps, feature_dim]
        """
        raise NotImplementedError("Subclasses must implement this method")
