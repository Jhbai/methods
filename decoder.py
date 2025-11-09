import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WeakDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (n_batch, n_series, input_dim)
        # out: (n_batch, n_series, output_dim), it's reconstruction
        return self.decoder(src)
