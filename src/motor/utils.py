import os
import torch

def save_embeddings(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)
