import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Linear(1, embedding_dim)
        
    def forward(self, delta_t):
        return self.embedding(delta_t)