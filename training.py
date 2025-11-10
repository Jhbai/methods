import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

def phase1_initial_training(model, train_loader, optimizer, epochs, device):
    """Using reconstruction error for first phase training to get a good encoder"""
    model.to(device)

    model.train()
    for epoch in range(1, epochs+1):
        loss_val = 0.0
        for batch in train_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data)

            loss = nn.MSELoss()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        _loss = round(loss_val / len(train_loader), 4)
        print(f"Phase 1 | Epoch [{epoch}/{epochs}], Reconstruction Loss: {_loss}")

def initialize_memory_with_kmeans(model, data_loader, num_clusters, device):
    model.to(device)
    model.eval()

    queries_list = []
    with torch.no_grad():
        for batch in data_loader:
            data = batch[0].to(device)
            queries = model.encoder(data) # get queries from encoder
            queries_list.append(queries.reshape(-1, model.encoder.latent_dim)) # turns (n_batch, n_series, hid_dim) into (n_batch*n_series, hid_dim)
        all_queries = torch.cat(queries_list, dim=0).cpu().numpy()

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(all_queries)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        model.memory_module.memory.data = centroids

def phase2_full_training(model, train_loader, optimizer, loss_fn, epochs, device):
    model.to(device)
    for epoch in range(1, 1+epochs):
        model.train()
        loss_val, rec_val, entr_val = 0, 0, 0
        for batch in train_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed, attention_weights = model(data)
            loss, l_rec, l_entr = loss_fn(reconstructed, data, attention_weights)
            
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item()
            rec_val += l_rec.item()
            entr_val += l_entr.item()
            
        _loss = round(loss_val / len(train_loader), 4)
        _rec = round(rec_val / len(train_loader), 4)
        _entr = round(entr_val / len(train_loader), 4)
        print(f"Phase 2 | Epoch [{epoch}/{epochs}], Total Loss: {_loss} | L_rec: {_rec} | L_entr: {_entr}")
