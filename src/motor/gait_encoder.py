# -*- coding: utf-8 -*-
"""
Representation learning module for motor (gait) data:
- Defines neural encoders for transforming gait features into embeddings
- Supports exporting embeddings for downstream multimodal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaitEncoder(nn.Module):
    """
    Simple feedforward encoder to obtain low-dimensional gait embeddings.
    
    Args:
        input_dim (int): Number of input gait features.
        hidden_dim (int, optional): Dimension of hidden layer. Default is 64.
        embedding_dim (int, optional): Dimension of output embedding. Default is 32.
        dropout (float, optional): Dropout probability for regularization. Default is 0.1.
    """

    def __init__(self, input_dim, hidden_dim=64, embedding_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.encoder(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for generating embeddings without requiring gradients.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def get_gait_embeddings(df, feature_cols, model=None, device='cpu'):
    """
    Generate gait embeddings from a DataFrame using a trained encoder.
    
    Args:
        df (pd.DataFrame): Input gait data.
        feature_cols (list): List of feature column names.
        model (nn.Module): Trained GaitEncoder model.
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        np.ndarray: Array of learned embeddings.
    """
    import numpy as np
    import pandas as pd
    if model is None:
        raise ValueError("A trained GaitEncoder model must be provided.")

    model.eval()
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
    with torch.no_grad():
        embeddings = model(X).cpu().numpy()

    # Return as DataFrame for consistency
    emb_cols = [f'emb_{i+1}' for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=emb_cols)
