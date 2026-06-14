import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor):
        T = x.size(1)
        
        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_len {self.max_len}"
            )

        
        return x + self.pe[:T].unsqueeze(0)
        