# tasks/gtzan_genre/decoder.py
from typing import Optional, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from marble.core.base_decoder import BaseDecoder
from marble.core.utils import instantiate_from_config


class MLPDecoder(BaseDecoder):
    """
    MLP Decoder with customizable layers, optional activation functions, and dropout.
    Supports input tensors of shape [B, L, T, H], where H is the embedding dimension.
    Uses einops for pooling operations.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 10,
        hidden_layers: list = [512],
        activation_fn: Optional[Dict] = None, # e.g. {"class_path": "torch.nn.ReLU"}
        dropout: float = 0.5
    ):
        super().__init__(in_dim, out_dim)

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation_fn is not None:
                activation_fn = instantiate_from_config(activation_fn)
                layers.append(activation_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, emb, *_):
        # emb: [B, L, T, H] -> mean-pool across L and T -> [B, H]
        assert emb.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {emb.dim()}D tensor"
        emb = reduce(emb, 'b l t h -> b h', 'mean')
        return self.net(emb)
    

class LinearDecoder(BaseDecoder):
    """
    Linear Decoder supporting input tensors of shape [B, L, T, H].
    Uses einops for pooling operations.
    """
    def __init__(self, in_dim: int, out_dim: int = 10):
        super().__init__(in_dim, out_dim)
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, emb, *_):
        # emb: [B, L, T, H] -> mean-pool across L and T -> [B, H]
        assert emb.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {emb.dim()}D tensor"
        emb = reduce(emb, 'b l t h -> b h', 'mean')
        return self.net(emb)


class LSTMDecoder(BaseDecoder):
    """
    LSTM Decoder for 4D sequence data.
    Supports input tensors of shape [B, L, T, H].
    Uses einops for reshaping.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2
    ):
        super().__init__(in_dim, out_dim)
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, emb, *_):
        # emb: [B, L, T, H] -> flatten to [B, L*T, H]
        assert emb.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {emb.dim()}D tensor"
        emb_flat = rearrange(emb, 'b l t h -> b (l t) h')
        lstm_out, _ = self.lstm(emb_flat)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


# Attempt to import Flash Attention implementation; fallback to native PyTorch scaled_dot_product_attention
try:
    from flash_attn.modules.mha import FlashMHA
except ImportError:
    FlashMHA = None

if FlashMHA is None:
    class FlashMHA(nn.Module):
        """
        Fallback multi-head attention using PyTorch's scaled_dot_product_attention.
        """
        def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
            self.dropout = dropout
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, query, key, value, attn_mask=None):
            B, Tq, D = query.shape
            _, Tk, _ = key.shape
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
            k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
            v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout, is_causal=False)
            out = rearrange(out, 'b h t d -> b t (h d)')
            return self.out_proj(out)


class FlashTransformerDecoderLayer(nn.Module):
    """
    Single layer of Transformer decoder using Flash Attention (or fallback), supports 4D sequence inputs.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = FlashMHA(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = FlashMHA(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.linear2 = nn.Linear(ff_hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x,           # [B, seq_len, H]
        memory,      # [B, mem_seq_len, H]
        tgt_mask=None,
        memory_mask=None
    ):
        residual = x
        x2 = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = residual + self.dropout1(x2)
        x = self.norm1(x)
        residual = x
        x2 = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = residual + self.dropout2(x2)
        x = self.norm2(x)
        residual = x
        x2 = self.linear2(self.dropout3(F.relu(self.linear1(x))))
        x = residual + self.dropout3(x2)
        x = self.norm3(x)
        return x


class TransformerDecoder(BaseDecoder):
    """
    Transformer Decoder with Flash Attention (or fallback), supports input tensors of shape [B, L, T, H].
    Utilizes einops for reshaping operations.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_hidden_dim: int = 2048,
        max_seq_len: int = 500,
        dropout: float = 0.1
    ):
        super().__init__(in_dim, out_dim)
        self.embed_dim = in_dim
        self.pos_emb = nn.Embedding(max_seq_len, in_dim)
        self.layers = nn.ModuleList([
            FlashTransformerDecoderLayer(
                embed_dim=in_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        tgt: torch.Tensor,     # [B, L, T, H]
        memory: torch.Tensor,  # [B, Lm, Tm, H]
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ):
        assert tgt.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {tgt.dim()}D tensor"
        assert memory.dim() == 4, f"Expected 4D tensor [B, Lm, Tm, H], got {memory.dim()}D tensor"
        B, L, T, H = tgt.shape
        # Flatten target and memory sequences
        seq_len = L * T
        tgt_flat = rearrange(tgt, 'b l t h -> b (l t) h')
        pos_ids = torch.arange(seq_len, device=tgt.device).unsqueeze(0).expand(B, seq_len)
        x = tgt_flat + self.pos_emb(pos_ids)
        Lm, Tm = memory.shape[1], memory.shape[2]
        memory_flat = rearrange(memory, 'b lm tm h -> b (lm tm) h')
        for layer in self.layers:
            x = layer(x, memory_flat, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits_flat = self.fc_out(x)
        # Reshape back to [B, L, T, out_dim]
        return rearrange(logits_flat, 'b (l t) d -> b l t d', l=L, t=T)
