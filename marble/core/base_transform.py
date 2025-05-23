# marble/core/base_transform.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Sequence, Union, Dict


class BaseEmbTransform(nn.Module, ABC):
    """
    Abstract base class for post‐processing transformer outputs.
    Safely intercepts ModelOutput objects to extract `.hidden_states`
    while preserving all of PyTorch's hook, no_grad, and JIT behavior.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        outputs: Union[Sequence[torch.Tensor], object],
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Safely override nn.Module.__call__ to:
          1. Extract `hidden_states` if `outputs` has that attribute.
          2. Delegate into the original nn.Module.__call__, which
             will handle hooks, no_grad, tracing, etc., then call forward().

        Args:
            outputs: Either
                - A tuple/list of Tensors, each of shape (B, T, H), or
                - A model‐output object (e.g. BaseModelOutput) with `.hidden_states`.
            *args, **kwargs: Passed through to forward().

        Returns:
            Tensor: Whatever your forward() returns.
        """
        # 1. Normalize to a Sequence[Tensor]
        hidden_states = (
            outputs.hidden_states
            if hasattr(outputs, "hidden_states")
            else outputs
        )
        # 2. Call nn.Module.__call__, which invokes hooks and then forward()
        return super().__call__(hidden_states, *args, **kwargs)

    @abstractmethod
    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Core transform logic. You must implement this in subclasses.

        Args:
            hidden_states (Sequence[Tensor]): List/tuple of N tensors,
                each of shape (batch_size, seq_len, hidden_size).

        Returns:
            Tensor: Transformed output; shape is up to the subclass.
        """
        raise NotImplementedError("Subclasses must implement forward()")


class BaseAudioTransform(nn.Module, ABC):
    """
    Base class for dict‐based audio transforms.
    Inherit from nn.Module so that __call__() → forward() is wired up automatically.
    """

    @abstractmethod
    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            sample (dict): must contain at least "waveform": Tensor[C, T].
        Returns:
            sample (dict): with same keys (possibly modified in place).
        """
        raise NotImplementedError("Subclasses must implement forward()")
