import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import periodogram, sawtooth
from scipy.stats import entropy, wasserstein_distance, shapiro
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerConditioner(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, input_dim * 2)
        
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x) # 先從input_dim投影到d_model維度
        h = self.pos_enc(h) # 做positional encoding
        h = self.transformer(h) # 經過transformer encoder
        params = self.output_proj(h) # 映射回 input_dim * 2 維度，表示s和t
        s, t = params.chunk(2, dim=-1) # 切兩半
        s = torch.tanh(s) # 限制s的範圍，避免exp(s)過大
        return s, t

class TimeWiseAffineCoupling(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, parity: int = 0):
        super().__init__()
        self.parity = parity
        self.conditioner = TransformerConditioner(input_dim, d_model, nhead, num_layers, dropout=0.05)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        # ----- 用parity去決定mask的交錯方式，0表示偶數位置為1，1表示奇數位置為1 ----- #
        mask = torch.zeros(T, device=x.device)
        if self.parity == 0:
            mask[0::2] = 1.0 
        else:
            mask[1::2] = 1.0
        mask = mask.view(1, T, 1)
        
        # ----- x_cond是x經過mask後的輸入 ----- #
        x_cond = x * mask

        # ----- 從被mask的x_cond中計算s和t ----- #
        s, t = self.conditioner(x_cond)
        s = s * (1 - mask)
        t = t * (1 - mask)
        
        # ----- 看是要forward還是backward ----- #
        if not reverse:
            z = mask * x + (1 - mask) * (x * torch.exp(s) + t)
            log_det = s.sum(dim=[1, 2])
        else:
            z_in = x 
            x_out = mask * z_in + (1 - mask) * ((z_in - t) * torch.exp(-s))
            log_det = -s.sum(dim=[1, 2])
            z = x_out

        return z, log_det

class TransformerNormalizingFlow(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_flow_layers: int = 4, num_transformer_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.flows = nn.ModuleList()

        # ----- 建立多層的Flow ----- #
        for i in range(num_flow_layers):
            self.flows.append(TimeWiseAffineCoupling(input_dim, d_model, nhead, num_transformer_layers, parity=(i % 2)))
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        # ----- 正向通過所有flow，每一層輸出維度不變 ----- #
        for flow in self.flows:
            z, log_det = flow(z, reverse=False)
            log_det_total += log_det # Jacobian determinant 的累加
        
        # ----- 計算標準高斯的log probability ----- #
        log_prob_z = -0.5 * (z.pow(2).sum(dim=[1, 2]) + np.prod(z.shape[1:]) * math.log(2 * math.pi)) # p(z)
        log_prob_x = log_prob_z + log_det_total # p(x) = p(z) + log|det(dz/dx)|
        
        return z, log_prob_x

    def sample(self, num_samples: int, seq_len: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, seq_len, self.input_dim, device=device)
        for flow in reversed(self.flows):
            z, _ = flow(z, reverse=True)
        return z
