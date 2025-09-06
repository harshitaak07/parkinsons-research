import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2, hidden_dim=64):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, 1)  

    def forward(self, x):
        x = self.transformer(x)  
        x = x[:, 0, :]  
        return self.classifier(x)
