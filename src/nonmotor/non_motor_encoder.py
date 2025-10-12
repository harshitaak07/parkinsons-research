import torch
import torch.nn as nn

class NonMotorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.model(x)
