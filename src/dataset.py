import torch
import json
from torch.utils.data import Dataset
from src.tokenizer import AlignTokenizer, GapTokenizer

class MSADataset(Dataset):
    def __init__(
        self,
        path: str,
        align_tokenizer: AlignTokenizer | None = None,
        gap_tokenizer: GapTokenizer | None = None,
        max_samples: int | None = None,
        max_len: int | None = None,
    ):
        self.align_tokenizer = align_tokenizer or AlignTokenizer()
        self.gap_tokenizer = gap_tokenizer or GapTokenizer()
        self.max_len = max_len
        self.samples = []
        
        with open(path) as f:
            for line in f:
                sample = json.loads(line)
                
                target_align = self.align_tokenizer.encode(sample["aligned_string"], add_eos=True)
                
                if self.max_len is None or len(target_align) <= self.max_len:
                    self.samples.append(sample)
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx:int):
        sample = self.samples[idx]
        input_ids = self.align_tokenizer.encode(sample["unalign_string"], add_eos=False)
        target_align = self.align_tokenizer.encode(sample["aligned_string"], add_eos=True)
        target_gap = self.gap_tokenizer.encode(sample["gap_string"], add_eos=True)
        
        return {
            "input_ids": input_ids,
            "target_align": target_align,
            "target_gap": target_gap,
        }

    def collate_fn(self, batch: list[dict]):
        align_pad = self.align_tokenizer.pad_id
        align_sos = self.align_tokenizer.sos_id
        
        gap_pad = self.gap_tokenizer.pad_id
        gap_sos = self.gap_tokenizer.sos_id

        def pad_seq(seqs, pad_id):
            maxlen = max(len(x) for x in seqs)
            return torch.tensor(
                [x + [pad_id] * (maxlen - len(x)) for x in seqs],
                dtype=torch.long
            )

        input_ids = pad_seq([b["input_ids"] for b in batch], align_pad)
        key_padding_mask = (input_ids == align_pad)

        target_align = [b["target_align"] for b in batch]
        labels_align = pad_seq(target_align, align_pad)
        decoder_input_align = pad_seq([[align_sos] + t[:-1] for t in target_align], align_pad)        
        
        target_gap = [b["target_gap"] for b in batch]
        labels_gap = pad_seq(target_gap, gap_pad)
        decoder_input_gap = pad_seq([[gap_sos] + t[:-1] for t in target_gap], gap_pad)        
        
        return {
            "input_ids": input_ids,
            "key_padding_mask": key_padding_mask,
            "decoder_input_align": decoder_input_align,
            "labels_align": labels_align,
            "decoder_input_gap": decoder_input_gap,
            "labels_gap": labels_gap
        }
