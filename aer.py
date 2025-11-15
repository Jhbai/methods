import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# 參照論文: AER: Auto-Encoder with Regression for Time Series Anomaly Detection
class AnomalyDetector(torch.nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Using device:", self.device)
        self.encoder_lstm = torch.nn.LSTM(input_size=1, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.relu = torch.nn.ReLU()

        self.decoder_lstm = None
        self.output_dense = None

        self.dense = None
        self.win_size = None
        self.to(self.device)

    def forward(self, x):
        _, (hidden, cell) = self.encoder_lstm(x)
        hidden_cat = hidden.permute(1, 0, 2).contiguous().view(x.size(0), 1, -1)
        decoder_input = hidden_cat.repeat(1, self.win_size + 2, 1) # repeat vector

        decoder_output, _ = self.decoder_lstm(decoder_input)
        decoder_output = self.relu(decoder_output)

        output_sequence = self.output_dense(decoder_output)
        bac_prediction = output_sequence[:, 0, :].squeeze(1)
        for_prediction = output_sequence[:, -1, :].squeeze(1)
        reconstruction = output_sequence[:, 1:-1, :].squeeze(2)

        return reconstruction, bac_prediction, for_prediction

    def fit(self, data, win_size, epochs=10, batch_size=128, learning_rate=0.001):
        self.win_size = win_size
        encoder_hidden_size = 128
        encoder_output_size = encoder_hidden_size * 2

        self.decoder_lstm = torch.nn.LSTM(input_size=encoder_output_size, hidden_size=encoder_hidden_size, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
        self.output_dense = torch.nn.Linear(encoder_output_size, 1).to(self.device)

        X, y = [], []
        for i in range(1, len(data) - self.win_size):
            window = data[i : i + self.win_size]
            target = data[i - 1 : i + self.win_size + 1]
            X.append(window)
            y.append(target)

        X = np.array(X).reshape(-1, self.win_size, 1)
        y = np.array(y)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            loss_val = 0.0
            self.train()
            for X_tensor, y_tensor in dataloader:
              X_tensor = X_tensor.to(self.device)
              y_tensor = y_tensor.to(self.device)
              reconstruction, bac_prediction, for_prediction = self.forward(X_tensor)

              loss_recon = loss_fn(reconstruction, X_tensor.squeeze(2))
              loss_pred_past = loss_fn(bac_prediction, y_tensor[:, 0])
              loss_pred_future = loss_fn(for_prediction, y_tensor[:, -1])

              loss = 0.5*loss_recon + 0.25*loss_pred_past + 0.25*loss_pred_future
              loss_val += loss.item()

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_val/len(dataloader):.4f}")

    def decision_function(self, data, batch_size=128):
        T = len(data)

        X, y = [], []
        for i in range(1, len(data) - self.win_size):
            window = data[i : i + self.win_size]
            target = data[i - 1 : i + self.win_size + 1]
            X.append(window)
            y.append(target)

        X = np.array(X).reshape(-1, self.win_size, 1)
        y = np.array(y)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        recon_errors_list = list()
        forward_pred_errors_list = list()
        backward_pred_errors_list = list()

        scores = list()
        self.eval()
        with torch.no_grad():
          for X_tensor, y_tensor in dataloader:
            X_tensor = X_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
            reconstruction, bac_prediction, for_prediction = self.forward(X_tensor)
            bac_prediction = bac_prediction.reshape(-1, )
            for_prediction = for_prediction.reshape(-1, )

            loss_fn_none = torch.nn.MSELoss(reduction='none')

            recon_errors = loss_fn_none(reconstruction, X_tensor.squeeze(2)).mean(axis=1).cpu()
            recon_errors_list.extend(list(recon_errors.cpu().numpy()))

            backward_pred_errors = loss_fn_none(bac_prediction, y_tensor[:, 0])
            backward_pred_errors_list.extend(list(backward_pred_errors.cpu().numpy()))

            forward_pred_errors = loss_fn_none(for_prediction, y_tensor[:, -1])
            forward_pred_errors_list.extend(list(forward_pred_errors.cpu().numpy()))
          recon_errors = np.array(recon_errors_list)
          forward_pred_errors = np.array(forward_pred_errors_list)
          backward_pred_errors = np.array(backward_pred_errors_list)

          ap_f_full = np.zeros(T)
          ap_r_full = np.zeros(T)

          f_start_index = self.win_size + 1
          ap_f_full[f_start_index: f_start_index+len(forward_pred_errors)] = forward_pred_errors

          r_start_index = 0
          ap_r_full[r_start_index: r_start_index+len(backward_pred_errors)] = backward_pred_errors

          bidirectional_pred_errors = np.zeros(T)
          overlap_mask = (ap_f_full > 0) & (ap_r_full > 0)
          bidirectional_pred_errors[overlap_mask] = (ap_f_full[overlap_mask] + ap_r_full[overlap_mask]) / 2


          non_overlap_mask = ~overlap_mask
          bidirectional_pred_errors[non_overlap_mask] = np.maximum(ap_f_full[non_overlap_mask], ap_r_full[non_overlap_mask])

          final_recon_errors = np.zeros(T)
          final_recon_errors[r_start_index : r_start_index + len(recon_errors)] = recon_errors

          final_scores = final_recon_errors * bidirectional_pred_errors
          return list(final_scores)

    def predict(self, data):
        anomaly_scores = np.array(self.decision_function(data))
        full_scores = np.zeros(len(data))
        full_scores[1:] = anomaly_scores[:-1]

        # ----- masking ----- #
        T = len(data)
        masking_fraction = 0.01
        mask_size = int(masking_fraction * T)
        if mask_size > 0:
          mask_value = np.min(full_scores)
          full_scores[:mask_size] = mask_value

        # ----- HL Logic ----- #
        win_size = len(data) // 3
        step_size = max(1, win_size // 10)
        suspected_indices = set()

        for i in range(0, len(full_scores) - win_size + 1, step_size):
            window = full_scores[i : i + win_size]
            mean = np.mean(window)
            std = np.std(window)
            if std == 0: continue
            threshold = mean + 4 * std

            anomalies_in_window = np.where(window > threshold)[0] + i
            suspected_indices.update(anomalies_in_window)

        if not suspected_indices:
            return np.zeros(len(data))
        sorted_indices = sorted(list(suspected_indices))
        sequences = []
        current_sequence = [sorted_indices[0]]
        for i in range(1, len(sorted_indices)):
          if sorted_indices[i] == sorted_indices[i-1] + 1:
              current_sequence.append(sorted_indices[i])
          else:
              sequences.append(current_sequence)
              current_sequence = [sorted_indices[i]]
        sequences.append(current_sequence)


        scored_sequences = []
        for seq in sequences:
          max_score = np.max(full_scores[seq])
          scored_sequences.append({'max_score': max_score, 'indices': seq})
        sorted_sequences = sorted(scored_sequences, key=lambda x: x['max_score'], reverse=True)

        final_sequences = []
        if len(sorted_sequences) > 0:
          final_sequences.append(sorted_sequences[0])
          last_max_score = sorted_sequences[0]['max_score']
          for i in range(1, len(sorted_sequences)):
            current_max_score = sorted_sequences[i]['max_score']
            if last_max_score > 0:
              percent_drop = (last_max_score - current_max_score) / last_max_score
            else:
              percent_drop = 0
            if percent_drop <= 0.13:
              break
            final_sequences.append(sorted_sequences[i])
            last_max_score = current_max_score
        predictions = np.zeros(len(data))
        for seq_info in final_sequences:
          predictions[seq_info['indices']] = 1
        return predictions

"""
[example]
x = np.linspace(0, 10*np.pi, 10081)
val = np.sin(x) + np.random.normal(0, 0.1, size = x.shape)
aidx = np.random.choice(np.arange(x.shape[0]//2, x.shape[0]), 20).astype(int)
val[aidx] = np.random.normal(5, 10, size = aidx.shape)
arr = val[:]
arr = (arr-np.min(arr))/(np.max(arr) - np.min(arr) + 1e-5)
model = AnomalyDetector()
model.fit(arr[:arr.shape[0]//2], 100, epochs=35)

scores = model.decision_function(arr)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize = (24, 4))
ax[0].plot(arr, color = "blue")
ax[1].plot(scores, color = "red")
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (24, 2))
idxs = np.where(model.predict(arr[arr.shape[0]//2:]))[0]
for idx in idxs:
  ax.axvline(idx, color = "red")

ax.plot(arr[arr.shape[0]//2:], color = "blue")
plt.show()
"""
