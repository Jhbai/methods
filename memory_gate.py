import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedMemoryModule(nn.Module):
    def __init__(self, num_items: int, latent_dim: int, temperature: float = 0.1):
        super().__init__()
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.temperature = temperature

        self.memory = nn.Parameter(torch.Tensor(num_items, latent_dim))

        self.U_psi = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_psi = nn.Linear(latent_dim, latent_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.memory)
        nn.init.xavier_uniform_(self.U_psi.weight)
        nn.init.xavier_uniform_(self.W_psi.weight)

    def gated_memory_update(self, queries: torch.Tensor):
        # queries: (n_batch, n_series, latent_dim)
        attn_scores_v = torch.matmul(queries, self.memory.t())
        attn_v = F.softmax(attn_scores_v / self.temperature, dim=1).permute(0, 2, 1)
        weighted_queries = torch.matmul(attn_v, queries)
        psi = torch.sigmoid(self.U_psi(self.memory) + self.W_psi(weighted_queries))
        new_memory_batch = (1 - psi) * self.memory.unsqueeze(0) + psi * weighted_queries
        self.memory.data = new_memory_batch.mean(dim=0).data

    def query_update(self, queries: torch.Tensor) -> torch.Tensor:
        # queries: (n_batch, n_series, latent_dim)
        attn_scores_w = torch.matmul(queries, self.memory.t())
        attn_w = F.softmax(attn_scores_w / self.temperature, dim=2)
        retrieved_memory = torch.matmul(attn_w, self.memory)
        updated_queries = torch.cat([queries, retrieved_memory], dim=2)
        return updated_queries, attn_w
    
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return self.query_update(queries)
