import torch
from abc import ABCMeta, abstractmethod
from typing import Optional

class BaseDecoder(torch.nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for decoders. Subclasses need to implement the forward method to decode feature representations 
    back to task-specific outputs (e.g., logits for classification, continuous vectors for regression).
    """
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Optionally initialize layers based on input/output dimensions

    @abstractmethod
    def forward(self, emb: torch.Tensor, emb_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward method to map the embeddings and their lengths to task-specific outputs.

        Args:
            emb: Tensor of shape [batch, time_steps, in_dim]
            emb_len: Optional tensor of sequence lengths for each input in the batch.

        Returns:
            Tensor of shape [batch, time_steps, out_dim] or [batch, out_dim] for task outputs
        """
        raise NotImplementedError("Subclasses must implement this method")
