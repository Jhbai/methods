import numpy as np
import torch

# ----- Data Generation ----- #
def generate_multivariate_timeseries(n_samples: int, n_timesteps: int, anomaly_fraction: float = 0.0, noise_level: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    n_features = 3
    random_seed = 42
    np.random.seed(random_seed)
    time = np.linspace(0, 10 * np.pi, n_timesteps)

    # Generate
    data = np.zeros((n_samples, n_timesteps, n_features))
    labels = np.zeros((n_samples, n_timesteps))

    for i in range(n_samples):
        f0 = np.sin(time + np.random.uniform(-0.5, 0.5))
        f1 = (1 + 0.5 * f0) * np.cos(0.5 * time + np.random.uniform(-0.5, 0.5))
        f2 = 0.6 * f0 + 0.4 * f1
        sample = np.stack([f0, f1, f2], axis=1)

        noise = np.random.normal(0, noise_level, sample.shape)
        data[i, :, :] = sample + noise
    
    f1 = (1 + 0.5 * f0) * np.cos(0.5 * time + np.random.uniform(-0.5, 0.5))

    # Generate Anomalies
    if anomaly_fraction > 0:
        n_total_points = n_samples * n_timesteps
        n_anomalies = int(n_total_points * anomaly_fraction)
        
        for _ in range(n_anomalies):
            sample_idx = np.random.randint(0, n_samples)
            time_idx = np.random.randint(0, n_timesteps)
            feature_idx = np.random.randint(0, n_features)

            if labels[sample_idx, time_idx] == 0:
                anomaly_magnitude = np.random.choice([-1, 1]) * np.random.uniform(3, 5) * data[sample_idx, time_idx, :].std()
                data[sample_idx, time_idx, feature_idx] += anomaly_magnitude
                labels[sample_idx, time_idx] = 1
                
    return data, labels

# ----- Training ----- #
# Init parameters
N_SAMPLES = 500
N_TIMESTEPS = 100
N_FEATURES = 3
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model hyperparameters
LATENT_DIM = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 128
NUM_MEMORY_ITEMS = 10 # Paper suggestion
DECODER_HIDDEN = 128

# Training hyperparameters
LR = 1e-4
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 5
LAMBDA_ENTROPY = 0.01

print(f"Using device: {DEVICE}")

# Generating data, according to the thesis, only use normal samples for training
print("Generating normal training data...")
train_data, _ = generate_multivariate_timeseries(
    n_samples=N_SAMPLES,
    n_timesteps=N_TIMESTEPS,
    anomaly_fraction=0.0 # training set contains no anomalies
)

# DataLoader
train_tensor = torch.tensor(train_data, dtype=torch.float32)
train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Prepare a subset of loader for K-means(10% of data by paper suggestion)
kmeans_subset_size = int(N_SAMPLES * 0.1)
kmeans_indices = torch.randperm(len(train_dataset))[:kmeans_subset_size]
kmeans_subset = torch.utils.data.Subset(train_dataset, kmeans_indices)
kmeans_loader = DataLoader(kmeans_subset, batch_size=BATCH_SIZE)

# initialize model
model = MEMTO(
    input_dim=N_FEATURES,
    latent_dim=LATENT_DIM,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    num_memory_items=NUM_MEMORY_ITEMS,
    decoder_hidden_dim=DECODER_HIDDEN
).to(DEVICE)
model.mode(1)

loss_fn = MEMTOLoss(lambda_entropy=LAMBDA_ENTROPY)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Phase 1 training
print("\n--- Starting Training Phase 1: Encoder Pre-training ---")
phase1_initial_training(model, train_loader, optimizer, PHASE1_EPOCHS, DEVICE)

print("\n--- Initializing Memory with K-means ---")
initialize_memory_with_kmeans(model, kmeans_loader, NUM_MEMORY_ITEMS, DEVICE)

# Phase 2 training
optimizer = optim.Adam(model.parameters(), lr=LR)
print("\n--- Starting Training Phase 2: Full Model Training ---")
phase2_full_training(model, train_loader, optimizer, loss_fn, PHASE2_EPOCHS, DEVICE)

print("\nTraining process completed successfully.")


# ----- Inference ----- #
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def calculate_anomaly_scores(model, data_tensor, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        data_tensor = data_tensor.to(device)

        reconstructed_output, _ = model(data_tensor)
        queries = model.encoder(data_tensor)
        memory = model.memory_module.memory.data

        # ----- Calculate ISD (Input Space Deviation) ----- #
        isd = torch.linalg.norm(data_tensor - reconstructed_output, dim=2).squeeze(0)

        # ----- Calculate LSD (Latent Space Deviation) ----- #
        L, C = queries.shape[1], queries.shape[2]
        M = memory.shape[0]
        dist_matrix = torch.linalg.norm(queries.unsqueeze(2) - memory.unsqueeze(0).unsqueeze(0), dim=3)
        lsd = dist_matrix.min(dim=2).values.squeeze(0)

        # ----- Combine scores ----- #
        lsd_softmax = F.softmax(lsd, dim=0)
        final_scores = lsd_softmax * isd

    return (
        reconstructed_output.squeeze(0).cpu().numpy(),
        isd.cpu().numpy(),
        lsd.cpu().numpy(),
        final_scores.cpu().numpy()
    )

def plot_inference_results(original_ts, reconstructed_ts, anomaly_scores, ts_labels, sample_idx):
    n_features = original_ts.shape[1]
    
    fig, axes = plt.subplots(n_features + 1, 1, figsize=(18, 4 * (n_features + 1)), sharex=True)
    fig.suptitle(f'Inference Results for Sample {sample_idx}', fontsize=16)

    for i in range(n_features):
        ax = axes[i]
        ax.plot(original_ts[:, i], label='Original Data', color='blue', alpha=0.8)
        ax.plot(reconstructed_ts[:, i], label='Reconstructed Data', color='red', linestyle='--')
        ax.set_ylabel(f'Feature {i}')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        # Highlight true anomaly regions
        anomaly_indices = np.where(ts_labels == 1)[0]
        for start in anomaly_indices:
             ax.axvspan(start - 0.5, start + 0.5, color='orange', alpha=0.3, lw=0)

    # Add a unified legend for the anomaly highlight
    patch = patches.Patch(color='orange', alpha=0.3, label='True Anomaly Region')
    axes[0].legend(handles=[
        plt.Line2D([0], [0], color='blue', label='Original Data'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Reconstructed Data'),
        patch
    ], loc='upper right')


    # Plot anomaly scores
    ax_score = axes[n_features]
    ax_score.plot(anomaly_scores, label='Anomaly Score', color='green')
    ax_score.set_xlabel('Time Step')
    ax_score.set_ylabel('Score')
    ax_score.legend(loc='upper right')
    ax_score.grid(True, linestyle='--', alpha=0.6)
    
    # Also highlight anomalies on the score plot
    for start in anomaly_indices:
        ax_score.axvspan(start - 0.5, start + 0.5, color='orange', alpha=0.3, lw=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

# (Assuming the training code from the previous response has just finished)
print("\n--- Starting Inference Phase ---")
model.mode(0)

# 5. Generate a new test dataset WITH anomalies
print("Generating test data with anomalies...")
test_data, test_labels = generate_multivariate_timeseries(
    n_samples=10, 
    n_timesteps=N_TIMESTEPS,
    anomaly_fraction=0.05 # 5% of points are anomalies
)

test_tensor = torch.tensor(test_data, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# 6. Select a single sample to inspect and run inference
SAMPLE_TO_INSPECT = 7 # You can change this index
print(f"\nRunning inference on sample #{SAMPLE_TO_INSPECT}...")

# Prepare the single sample for the model (needs a batch dimension)
sample_data = test_tensor[SAMPLE_TO_INSPECT].unsqueeze(0)
sample_labels = test_labels_tensor[SAMPLE_TO_INSPECT].numpy()

# 7. Get numerical results
reconstructed, isd, lsd, final_scores = calculate_anomaly_scores(
    model, 
    sample_data, 
    DEVICE
)

print("\n--- Numerical Results ---")
print(f"Shape of original data: {sample_data.squeeze(0).shape}")
print(f"Shape of reconstructed data: {reconstructed.shape}")
print(f"Average Anomaly Score (ISD): {isd.mean():.4f}")
print(f"Average Anomaly Score (LSD): {lsd.mean():.4f}")
print(f"Max Final Anomaly Score: {final_scores.max():.4f}")

# 8. Visualize the results
print("\n--- Generating Visualization ---")
plot_inference_results(
    original_ts=sample_data.squeeze(0).cpu().numpy(),
    reconstructed_ts=reconstructed,
    anomaly_scores=final_scores,
    ts_labels=sample_labels,
    sample_idx=SAMPLE_TO_INSPECT
)
