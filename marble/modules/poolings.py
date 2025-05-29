import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMaxPool1D(nn.Module):
    """
    Global Max Pooling over time dimension.
    Args:
        x: Tensor of shape (batch, time, channels)
    Returns:
        Tensor of shape (batch, channels)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # max over time axis (dim=1)
        return torch.max(x, dim=1).values


class GlobalAvgPool1D(nn.Module):
    """
    Global Average Pooling over time dimension.
    Args:
        x: Tensor of shape (batch, time, channels)
    Returns:
        Tensor of shape (batch, channels)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)


class MaxAvgPool1D(nn.Module):
    """
    Combines max and average pooling: \alpha * max + (1-\alpha) * avg
    \alpha can be a fixed float or a learnable parameter.
    """
    def __init__(self, alpha: float = 0.5, learnable: bool = False):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = torch.max(x, dim=1).values
        avg_pool = torch.mean(x, dim=1)
        if isinstance(self.alpha, nn.Parameter):
            alpha = torch.sigmoid(self.alpha)
        else:
            alpha = self.alpha
        return alpha * max_pool + (1 - alpha) * avg_pool


class AttentionPooling1D(nn.Module):
    """
    Dot-Product Attention Pooling over time.
    Learns a linear scoring for each frame.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        scores = self.attention(x).squeeze(-1)        # (batch, time)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, time, 1)
        return torch.sum(weights * x, dim=1)         # (batch, channels)


class GatedAttentionPooling1D(nn.Module):
    """
    Gated Attention Pooling: adds a sigmoid gate to the attention mechanism.
    """
    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.V = nn.Linear(in_features, hidden_size)
        self.U = nn.Linear(in_features, hidden_size)
        self.context = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        H = torch.tanh(self.V(x))              # (batch, time, hidden)
        Gate = torch.sigmoid(self.U(x))        # (batch, time, hidden)
        H = H * Gate                            # gated representation
        scores = self.context(H).squeeze(-1)   # (batch, time)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)   # (batch, channels)


class AutoPool1D(nn.Module):
    """
    Auto-Pooling: learns a temperature \alpha to interpolate between
    max and average pooling via log-sum-exp.
    """
    def __init__(self, in_features: int, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        B, T, C = x.shape
        # scale by alpha and compute log-sum-exp across time
        x_scaled = x * self.alpha                # (batch, time, channels)
        lse = torch.logsumexp(x_scaled, dim=1)   # (batch, channels)
        # normalize and return
        return (lse - torch.log(torch.tensor(T, dtype=x.dtype, device=x.device))) / self.alpha
