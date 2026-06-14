import torch.nn as nn
import torch.nn.functional as F

class MSALoss(nn.Module):
    def __init__(self, align_pad_id: int, gap_pad_id):
        super().__init__()
        self.align_pad_id = align_pad_id
        self.gap_pad_id = gap_pad_id
        

    def forward(self, logits_align, labels_align, logits_gap, labels_gap):
        B, T, V_align = logits_align.shape
        _, _, V_gap = logits_gap.shape
        

        align_loss = F.cross_entropy(
            logits_align.view(B * T, V_align),
            labels_align.view(B * T),
            ignore_index=self.align_pad_id
        )
        
        gap_loss = F.cross_entropy(
            logits_gap.view(B * T, V_gap),
            labels_gap.view(B * T),
            ignore_index=self.gap_pad_id
        )
        
        loss = align_loss + gap_loss

        return loss, align_loss, gap_loss
