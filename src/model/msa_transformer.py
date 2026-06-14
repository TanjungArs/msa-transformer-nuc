import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import AlignDecoder, GapDecoder


class MSATransformer(nn.Module):
    def __init__(
        self,
        align_vocab_size: int,
        gap_vocab_size: int,
        dim: int,
        enc_depth: int,
        dec_depth: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        max_len,
        align_pad_id: int,
        gap_pad_id: int
    ):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=align_vocab_size,
            dim=dim,
            max_len=max_len,
            depth=enc_depth,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            pad_id=align_pad_id
        )
        
        self.decoder_align = AlignDecoder(
            vocab_size=align_vocab_size,
            dim=dim,
            depth=dec_depth,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            pad_id=align_pad_id,
            max_len=max_len
        )
        
        self.decoder_gap = GapDecoder(
            vocab_size=gap_vocab_size,
            dim=dim,
            depth=dec_depth,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            pad_id=gap_pad_id,
            max_len=max_len
        )

    def forward(
        self,
        input_ids,
        key_padding_mask,
        decoder_input_align,
        decoder_input_gap
    ):
                
        encoder_out = self.encoder(
            input_ids=input_ids,
            key_padding_mask=key_padding_mask
        )
        
        logits_align = self.decoder_align(
            encoder_out=encoder_out,
            decoder_input=decoder_input_align,
            enc_padding_mask=key_padding_mask
        )
        
        logits_gap = self.decoder_gap(
            encoder_out=encoder_out,
            decoder_input=decoder_input_gap,
            enc_padding_mask=key_padding_mask
        )
        
        return logits_align, logits_gap
