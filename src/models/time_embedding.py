import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def _init_(self, embedding_dim=16):
        super()._init_()
        self.embedding = nn.Linear(1, embedding_dim)
        
    def forward(self, delta_t):
        return self.embedding(delta_t)