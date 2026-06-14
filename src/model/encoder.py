import torch
import torch.nn as nn
from .positional import SinusoidalPositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        max_len,
        depth: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        pad_id: int = 0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            dim,
            padding_idx=pad_id
        )
        
        self.pos_embedding = SinusoidalPositionalEncoding(dim=dim, max_len=max_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(dim, n_heads, ff_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, input_ids, key_padding_mask=None):
        B, L = input_ids.size()
        device = input_ids.device

        x = self.embedding(input_ids)
        x = self.pos_embedding(x)
          
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            
        return x