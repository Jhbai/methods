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

# ----- 合成數據生成函數 ----- #
def generate_synthetic_data(n_samples=2000, seq_len=40):
    # ----- 請AI生成的數據生成函數 ----- #
    x = []
    for _ in range(n_samples):
        base_freq_curve = np.random.uniform(0.3, 0.8, size=seq_len)
        kernel_size = 5
        smooth_freq = np.convolve(base_freq_curve, np.ones(kernel_size)/kernel_size, mode='same')
        
        phase = np.cumsum(smooth_freq)
        phase += np.random.uniform(0, 0.5*np.pi)
        
        signal = sawtooth(phase, width=1)
        amp = np.random.uniform(0.8, 1.2)
        
        start_level = np.random.uniform(-0.5, 0.5)
        end_level = np.random.uniform(-0.5, 0.5)
        trend = np.linspace(start_level, end_level, seq_len)
        
        noise = np.random.normal(0, 0.05, size=seq_len)
        
        final_wave = (signal * amp) + trend + noise
        x.append(final_wave)
        
    return torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(-1)

# ----- 計算硬體設定 ----- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ----- 超參數設定 ----- #
SEQ_LEN = 40
INPUT_DIM = 1
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100

# ----- 資料準備 ----- #
full_data = generate_synthetic_data(n_samples=2000, seq_len=SEQ_LEN)
train_size = int(0.8 * len(full_data))
train_data = full_data[:train_size].to(DEVICE) # 訓練資料，維度為 (n_batch, seq_len, input_dim)
test_data = full_data[train_size:].to(DEVICE) # 測試資料，維度為 (n_batch, seq_len, input_dim)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)

# ----- 模型建立 ----- #
model = TransformerNormalizingFlow(input_dim=INPUT_DIM, d_model=64, nhead=4, num_flow_layers=6, num_transformer_layers=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----- 訓練迴圈 ----- #
print("\nStarting Training...")
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    
    for x_batch, in train_loader:
        optimizer.zero_grad()
        _, log_prob = model(x_batch)
        loss = -log_prob.mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN loss detected")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        _, val_log_prob = model(test_data)
        val_loss = -val_log_prob.mean().item()
    
    avg_train_loss = np.mean(batch_losses)
    train_losses.append(avg_train_loss)
    test_losses.append(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
