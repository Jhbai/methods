import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_batch, n_series, hid_dim)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
    
class MEMTO_Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, input_dim: int, latent_dim: int, n_heads: int, n_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim

        # ----- Embedded Layer ----- #
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # ----- Positional Encoder ----- #
        self.pos_encoder = PositionalEncoding(latent_dim, dropout)

        # ----- Transformer Encoder Layers ----- #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # x: (n_batch, n_series, n_features)
        src = self.input_proj(src) * math.sqrt(self.latent_dim)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src)
        return out
