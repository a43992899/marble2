import torch
from marble.core.registry import register
from marble.core.base_encoder import BaseEncoder

@register("encoder", "identity")
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
