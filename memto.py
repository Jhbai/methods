import torch
import torch.nn as nn
from encoder import *
from decoder import *
from memory_gate import *

class MEMTO(nn.Module):
    """MEMTO Structure"""
    def __init__(self, input_dim: int, latent_dim: int, n_heads: int, n_layers: int, dim_feedforward: int, num_memory_items: int, decoder_hidden_dim: int, mem_temp: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.encoder = MEMTO_Encoder(input_dim=input_dim, latent_dim=latent_dim, n_heads=n_heads, n_layers=n_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.memory_module = GatedMemoryModule(num_items=num_memory_items, latent_dim=latent_dim, temperature=mem_temp)

        # input dim is the latent concated with query latent, output dim is the input dim
        self.decoder = WeakDecoder(input_dim=2 * latent_dim, hidden_dim=decoder_hidden_dim, output_dim=input_dim)

    def mode(self, training: bool):
        self.training = training

    def forward(self, src: torch.tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (n_batch, n_series, n_features)
        out: [Reconstruction: torch.tensor, attention_weights: torch.tensor] : [(n_batch, n_series, n_features), (n_batch, n_series, num_memory_items)]
        """

        queries = self.encoder(src)
        if self.training:
            self.memory_module.gated_memory_update(queries)
        updated_queries, attention_weights = self.memory_module(queries)
        reconstructed_output = self.decoder(updated_queries)
        return reconstructed_output, attention_weights
