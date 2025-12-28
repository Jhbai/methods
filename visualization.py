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

print("\nGenerating visualizations...")
model.eval()
with torch.no_grad():
    n_viz_samples = 500
    x_test = test_data[:n_viz_samples]
    z_test, log_prob_test = model(x_test)
    x_generated = model.sample(num_samples=n_viz_samples, seq_len=SEQ_LEN, device=DEVICE)

x_test_np = x_test.cpu().numpy()
z_test_np = z_test.cpu().numpy()
x_generated_np = x_generated.cpu().numpy()
log_prob_np = log_prob_test.cpu().numpy()

fig = plt.figure(figsize=(20, 12))

# Row 1
ax1 = plt.subplot(3, 4, 1)
ax1.plot(train_losses, label='Train', color='#1f77b4')
ax1.plot(test_losses, label='Test', color='#ff7f0e')
ax1.set_title('Training Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 4, 2)
for i in range(min(10, len(x_test_np))):
    ax2.plot(x_test_np[i, :, 0], alpha=0.5)
ax2.set_title('Real Data')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 4, 3)
for i in range(min(10, len(x_generated_np))):
    ax3.plot(x_generated_np[i, :, 0], alpha=0.5)
ax3.set_title('Generated Data')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 4, 4)
time_idx = SEQ_LEN // 2
ax4.hist(x_test_np[:, time_idx, 0], bins=30, alpha=0.5, density=True, label='Real')
ax4.hist(x_generated_np[:, time_idx, 0], bins=30, alpha=0.5, density=True, label='Gen')
ax4.set_title(f'Marginal Dist (t={time_idx})')
ax4.legend()

# Row 2
ax5 = plt.subplot(3, 4, 5)
scatter = ax5.scatter(z_test_np[:, 0, 0], z_test_np[:, 1, 0], c=np.arange(len(z_test_np)), s=10, alpha=0.6)
ax5.set_title('Latent Space (t=0 vs t=1)')
ax5.set_xlim(-4, 4); ax5.set_ylim(-4, 4)

ax6 = plt.subplot(3, 4, 6)
z_std = np.random.randn(n_viz_samples, 2)
ax6.scatter(z_std[:, 0], z_std[:, 1], s=10, color='gray', alpha=0.6)
ax6.set_title('Target Gaussian')
ax6.set_xlim(-4, 4); ax6.set_ylim(-4, 4)

ax7 = plt.subplot(3, 4, 7)
for t in [0, 10, 20, 30]:
    if t < SEQ_LEN:
        ax7.hist(z_test_np[:, t, 0], bins=30, density=True, alpha=0.3, label=f't={t}')
x_r = np.linspace(-4, 4, 100)
ax7.plot(x_r, stats.norm.pdf(x_r), 'k--')
ax7.set_title('Latent Marginals')
ax7.legend(fontsize=8)

ax8 = plt.subplot(3, 4, 8)
stats.probplot(z_test_np.flatten(), dist="norm", plot=ax8)
ax8.set_title('Q-Q Plot (All Latents)')

# Row 3
ax9 = plt.subplot(3, 4, 9)
steps = np.arange(SEQ_LEN)
ax9.plot(steps, x_test_np.mean(0).flatten(), label='Real Mean')
ax9.plot(steps, x_generated_np.mean(0).flatten(), '--', label='Gen Mean')
ax9.set_title('Time Statistics')
ax9.legend()

ax10 = plt.subplot(3, 4, 10)
# Simple Autocorrelation
def simple_acf(x):
    # x shape: [N, T]
    N, T = x.shape
    acfs = []
    for i in range(N):
        series = x[i] - np.mean(x[i])
        var = np.var(x[i])
        if var > 1e-6:
            r = np.correlate(series, series, mode='full')
            r = r[T-1:] / (var * T)
            acfs.append(r[:T//2])
    return np.mean(acfs, axis=0)

real_acf = simple_acf(x_test_np[:, :, 0])
gen_acf = simple_acf(x_generated_np[:, :, 0])
ax10.plot(real_acf, label='Real')
ax10.plot(gen_acf, '--', label='Gen')
ax10.set_title('Autocorrelation')
ax10.legend()

# ----------------- FIXED SECTION STARTS HERE -----------------
ax11 = plt.subplot(3, 4, 11)
# Vectorized periodogram returns:
# freqs: [K] (1D array of frequencies)
# psd:   [N, K] (Power spectral density for each sample)
freqs_r, psd_r = periodogram(x_test_np[:, :, 0], axis=1)
freqs_g, psd_g = periodogram(x_generated_np[:, :, 0], axis=1)

# Fix: Do not use [0] on freqs, it is already the frequency axis
ax11.semilogy(freqs_r, psd_r.mean(axis=0), label='Real', color='#1f77b4')
ax11.semilogy(freqs_g, psd_g.mean(axis=0), '--', label='Gen', color='#ff7f0e')
ax11.set_title('Power Spectral Density')
ax11.legend()
ax11.grid(True, alpha=0.3)
# ----------------- FIXED SECTION ENDS HERE -----------------

ax12 = plt.subplot(3, 4, 12)
ax12.hist(log_prob_np, bins=30, color='green', alpha=0.6)
ax12.set_title('Log Likelihood')

plt.tight_layout()
plt.savefig('flow_viz_final.png', dpi=150)
print("Saved flow_viz_final.png")
plt.show()

# Metrics
print("\nEvaluation:")
wd = np.mean([wasserstein_distance(x_test_np[:,t,0], x_generated_np[:,t,0]) for t in range(SEQ_LEN)])
print(f"Avg Wasserstein Dist: {wd:.4f}")
print("Done.")
