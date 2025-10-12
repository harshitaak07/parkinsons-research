import torch
import torch.nn as nn

class IntermediateFusion(nn.Module):
    def __init__(self, mask_missing=False):
        super().__init__()
        self.mask_missing = mask_missing

    def forward(self, gait_emb, non_motor_emb, time_emb=None):
        """
        gait_emb: [batch, gait_dim]
        non_motor_emb: [batch, non_motor_dim]
        time_emb: [batch, time_dim] or None
        """
        embeddings = [gait_emb, non_motor_emb]
        if time_emb is not None:
            embeddings.append(time_emb)

        fused = torch.cat(embeddings, dim=-1)

        if self.mask_missing:
            fused = fused

        return fused
