import torch
import torch.nn as nn
from .positional import SinusoidalPositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("_cached_mask", None, persistent=False)
    
    def _causal_mask(self, L, device):
        if self._cached_mask is None or self._cached_mask.size(0) < L:
            mask = torch.triu(
                torch.ones(L, L, device=device), diagonal=1
            ).bool()
            self._cached_mask = mask
        return self._cached_mask[:L, :L]
        
    def forward(
        self,
        x,
        encoder_out,
        enc_key_padding_mask=None,
        dec_key_padding_mask=None
    ):
        L = x.size(1)
        
        attn_mask = self._causal_mask(L, x.device)
        
        self_attn_out, _ = self.self_attn(
            x, x, x, 
            attn_mask = attn_mask,
            key_padding_mask=dec_key_padding_mask
        )
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out, _ = self.cross_attn(
            x, encoder_out, encoder_out,
            key_padding_mask=enc_key_padding_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class AlignDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
        pad_id: int
    ):
        super().__init__()
        
        self.pad_id = pad_id
        
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_embedding = SinusoidalPositionalEncoding(dim=dim, max_len=max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(dim, n_heads, ff_dim, dropout)
            for _ in range(depth)
        ])

        self.proj = nn.Linear(dim, vocab_size)

    def forward(
        self,
        encoder_out,
        decoder_input,
        enc_padding_mask=None
    ):
        
        dec_padding_mask = (decoder_input == self.pad_id)
        
        B, L = decoder_input.size()
        device = decoder_input.device
        
        x = self.token_embed(decoder_input)
        x = self.pos_embedding(x)
                
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                enc_key_padding_mask=enc_padding_mask,
                dec_key_padding_mask=dec_padding_mask
            )
        align_logits = self.proj(x)
        return align_logits
    
class GapDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
        pad_id: int
    ):
        super().__init__()
        
        self.pad_id = pad_id
        
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_embedding = SinusoidalPositionalEncoding(dim=dim, max_len=max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(dim, n_heads, ff_dim, dropout)
            for _ in range(depth)
        ])

        self.proj = nn.Linear(dim, vocab_size)

    def forward(
        self,
        encoder_out,
        decoder_input,
        enc_padding_mask=None
    ):
        
        dec_padding_mask = (decoder_input == self.pad_id)
        
        B, L = decoder_input.size()
        device = decoder_input.device
        
        x = self.token_embed(decoder_input)
        x = self.pos_embedding(x)
                
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                enc_key_padding_mask=enc_padding_mask,
                dec_key_padding_mask=dec_padding_mask
            )
        gap_logits = self.proj(x)
        return gap_logits
