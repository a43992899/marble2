import torch
from marble.core.base_encoder import BaseEncoder

class IdentityEncoder(BaseEncoder):
    """
    A no-op encoder that returns its input unchanged.
    Used when your DataModule already yields features.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # simply pass through
        return x
