# tasks/gtzan_genre/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from marble.core.registry import register
from marble.core.base_decoder import BaseDecoder


class MLPDecoder(BaseDecoder):
    def __init__(self, in_dim: int, out_dim: int = 10, hidden_layers: list = [512], activation_fn: nn.Module = nn.ReLU):
        """
        MLP Decoder with customizable layers and activation functions.

        Args:
            in_dim (int): The input dimension (e.g., size of embedding).
            out_dim (int): Number of output classes.
            hidden_layers (list): List of integers specifying the number of neurons in each hidden layer.
            activation_fn (nn.Module): Activation function to use (default is ReLU).
        """
        super().__init__(in_dim, out_dim)
        
        layers = []
        prev_dim = in_dim
        
        # Create hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, out_dim))
        
        # Create the sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, emb, *_):
        # emb: [B, T', D] → mean-pool → [B, D]
        emb = emb.mean(dim=1)
        return self.net(emb)
    
    
class LinearDecoder(BaseDecoder):
    def __init__(self, in_dim: int, out_dim: int = 10):
        """
        Linear Decoder.

        Args:
            in_dim (int): The input dimension (e.g., size of embedding).
            out_dim (int): Number of output classes.
        """
        super().__init__(in_dim, out_dim)
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, emb, *_):
        # emb: [B, T', D] → mean-pool → [B, D]
        emb = emb.mean(dim=1)
        return self.net(emb)


class LSTMDecoder(BaseDecoder):
    def __init__(self, in_dim: int, out_dim: int = 10, hidden_size: int = 128, num_layers: int = 2):
        """
        LSTM Decoder for sequence data.

        Args:
            in_dim (int): The input dimension (e.g., size of embedding).
            out_dim (int): Number of output classes.
            hidden_size (int): Size of LSTM hidden state.
            num_layers (int): Number of LSTM layers.
        """
        super().__init__(in_dim, out_dim)
        
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, emb, *_):
        # emb: [B, T', D]
        lstm_out, _ = self.lstm(emb)  # LSTM output: [B, T', hidden_size]
        last_hidden = lstm_out[:, -1, :]  # Take the last time-step's hidden state
        return self.fc(last_hidden)



# 尝试导入 Flash Attention 的实现，若不可用则回退到 PyTorch 原生 scaled_dot_product_attention
try:
    from flash_attn.modules.mha import FlashMHA
except ImportError:
    FlashMHA = None

# 如果没有安装 flash_attn，实现一个简单的 FlashMHA 回退版本
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
            # 线性投影
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            # 划分多头
            q = q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
            # 计算 scaled dot-product attention
            out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout, is_causal=False)
            # 合并多头
            out = out.transpose(1, 2).contiguous().view(B, Tq, D)
            # 输出映射
            return self.out_proj(out)


class FlashTransformerDecoderLayer(nn.Module):
    """
    Single layer of Transformer decoder using Flash Attention for both self- and cross-attention.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        # Flash self-attention (或回退)
        self.self_attn = FlashMHA(embed_dim, num_heads, dropout=dropout)
        # Flash cross-attention (或回退)
        self.cross_attn = FlashMHA(embed_dim, num_heads, dropout=dropout)
        # 前馈网络
        self.linear1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.linear2 = nn.Linear(ff_hidden_dim, embed_dim)
        # 层归一化和 dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None):
        # x: [B, T_tgt, D], memory: [B, T_src, D]
        # 1) Self-attention
        residual = x
        x2 = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = residual + self.dropout1(x2)
        x = self.norm1(x)
        # 2) Cross-attention
        residual = x
        x2 = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = residual + self.dropout2(x2)
        x = self.norm2(x)
        # 3) 前馈
        residual = x
        x2 = self.linear2(self.dropout3(F.relu(self.linear1(x))))
        x = residual + self.dropout3(x2)
        x = self.norm3(x)
        return x


class TransformerDecoder(BaseDecoder):
    """
    Transformer Decoder with Flash Attention (或回退)，用于 ASR 任务。
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_hidden_dim: int = 2048,
                 max_seq_len: int = 500,
                 dropout: float = 0.1):
        super().__init__(in_dim, out_dim)
        self.embed_dim = in_dim
        # 可学习位置编码
        self.pos_emb = nn.Embedding(max_seq_len, in_dim)
        # 解码器层堆叠
        layers = []
        for _ in range(num_layers):
            layers.append(FlashTransformerDecoderLayer(
                embed_dim=in_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout))
        self.layers = nn.ModuleList(layers)
        # 输出分类
        self.fc_out = nn.Linear(in_dim, out_dim)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None):
        B, T_tgt, _ = tgt.size()
        # 添加位置编码
        pos_ids = torch.arange(T_tgt, device=tgt.device).unsqueeze(0).expand(B, T_tgt)
        x = tgt + self.pos_emb(pos_ids)
        # 逐层处理
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        # 输出 logits
        logits = self.fc_out(x)
        return logits
