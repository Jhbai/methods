import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class ReRe(nn.Module):
    def __init__(self):
        super(ReRe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.b = 3
        self.hidden_units = 10
        self.epochs = 50
        self.lr = 0.15

        self.model1 = self._build_model()
        self.model2 = self._build_model()

        self.history = []
        self.aare_history1 = []
        self.aare_history2 = []

        self.is_ready = False
        self.to(self.device)

    def _build_model(self):
        return nn.Sequential(
            nn.LSTM(input_size=1, hidden_size=self.hidden_units, batch_first=True),
            nn.Linear(in_features=self.hidden_units, out_features=1)
        )

    def forward(self, x):
        x = x.view(1, self.b, 1)
        lstm_out, _ = self.model1[0](x)
        last_hidden_state = lstm_out[:, -1, :]
        prediction = self.model1[1](last_hidden_state)
        return prediction.squeeze()

    def fit(self, x):
        if len(x) != 2 * self.b + 1:
            raise ValueError(f"Input for fit must have length 2*b+1 = {2 * self.b + 1}")

        self.history = list(x)

        train_x = torch.tensor(self.history[self.b : 2 * self.b], dtype=torch.float32).to(self.device)
        train_y = torch.tensor(self.history[2 * self.b], dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(self.model1.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        patience = 5
        counter = 0
        best_loss = float("inf")
        for epoch in range(self.epochs):
            self.model1.train()
            optimizer.zero_grad()
            output = self.forward(train_x)
            loss = criterion(output, train_y)
            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break
            loss.backward()
            optimizer.step()


        self.model2.load_state_dict(self.model1.state_dict())
        self.model1.eval()
        self.model2.eval()

        with torch.no_grad():
            for t in range(2 * self.b - 1, 2 * self.b + 1):
                errors = []
                for y in range(t - self.b + 1, t + 1):
                    input_seq = torch.tensor(self.history[y - self.b : y], dtype=torch.float32).to(self.device)
                    pred = self.forward(input_seq).item()
                    actual = self.history[y]
                    error = np.abs(pred - actual) / (np.abs(actual) + 1e-8)
                    errors.append(error)
                aare = np.mean(errors)
                self.aare_history1.append(aare)
                self.aare_history2.append(aare)

        self.is_ready = True

    def predict(self, arr):
        if not self.is_ready:
            raise RuntimeError("Model is not ready. Call .fit() first.")

        results = np.zeros(len(arr), dtype=int)

        for i, v_t in enumerate(arr):
            t = len(self.history)

            # Detector 1
            is_abnormal1 = False
            self.model1.eval()
            with torch.no_grad():
                errors1 = []
                for y in range(t - self.b + 1, t + 1):
                    hist_y = self.history + [v_t] if y == t else self.history
                    input_seq = torch.tensor(hist_y[y - self.b : y], dtype=torch.float32).to(self.device)

                    lstm_out, _ = self.model1[0](input_seq.view(1, self.b, 1))
                    last_hidden_state = lstm_out[:, -1, :]
                    pred_tensor = self.model1[1](last_hidden_state)
                    pred = pred_tensor.item()

                    actual = hist_y[y]
                    errors1.append(np.abs(pred - actual) / (np.abs(actual) + 1e-8))
                aare1_t = np.mean(errors1)

            mu1 = np.mean(self.aare_history1)
            sigma1 = np.std(self.aare_history1)
            thd1 = mu1 + 3 * sigma1

            if aare1_t > thd1:
                temp_model1 = self._build_model().to(self.device)
                temp_model1.train()
                optimizer = torch.optim.Adam(temp_model1.parameters(), lr=self.lr)
                criterion = nn.MSELoss()
                train_x = torch.tensor(self.history[t - self.b : t], dtype=torch.float32).to(self.device)
                train_y = torch.tensor(v_t, dtype=torch.float32).to(self.device)

                patiences = 5
                counter = 0
                best = float("inf")
                for epoch in range(self.epochs):
                    optimizer.zero_grad()

                    lstm_out_temp, _ = temp_model1[0](train_x.view(1, self.b, 1))
                    last_hidden_state_temp = lstm_out_temp[:, -1, :]
                    output_tensor = temp_model1[1](last_hidden_state_temp)
                    output = output_tensor.squeeze()

                    loss = criterion(output, train_y)
                    if loss.item() < best:
                        best = loss.item()
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patiences:
                            break
                    loss.backward()
                    optimizer.step()

                temp_model1.eval()
                with torch.no_grad():
                    errors1_new = []
                    for y in range(t - self.b + 1, t + 1):
                        hist_y = self.history + [v_t] if y == t else self.history
                        input_seq = torch.tensor(hist_y[y - self.b : y], dtype=torch.float32).to(self.device)

                        lstm_out_temp, _ = temp_model1[0](input_seq.view(1, self.b, 1))
                        last_hidden_state_temp = lstm_out_temp[:, -1, :]
                        pred_tensor = temp_model1[1](last_hidden_state_temp)
                        pred = pred_tensor.item()

                        actual = hist_y[y]
                        errors1_new.append(np.abs(pred - actual) / (np.abs(actual) + 1e-8))
                    aare1_t_new = np.mean(errors1_new)

                if aare1_t_new > thd1:
                    is_abnormal1 = True
                else:
                    self.model1.load_state_dict(temp_model1.state_dict())
                    aare1_t = aare1_t_new

            self.aare_history1.append(aare1_t)

            # Detector 2
            is_abnormal2 = False
            is_normal_for_hist2 = True
            self.model2.eval()
            with torch.no_grad():
                errors2 = []
                for y in range(t - self.b + 1, t + 1):
                    hist_y = self.history + [v_t] if y == t else self.history
                    input_seq = torch.tensor(hist_y[y - self.b : y], dtype=torch.float32).to(self.device)

                    lstm_out, _ = self.model2[0](input_seq.view(1, self.b, 1))
                    last_hidden_state = lstm_out[:, -1, :]
                    pred_tensor = self.model2[1](last_hidden_state)
                    pred = pred_tensor.item()

                    actual = hist_y[y]
                    errors2.append(np.abs(pred - actual) / (np.abs(actual) + 1e-8))
                aare2_t = np.mean(errors2)

            mu2 = np.mean(self.aare_history2)
            sigma2 = np.std(self.aare_history2)
            thd2 = mu2 + 3 * sigma2

            if aare2_t > thd2:
                temp_model2 = self._build_model().to(self.device)
                temp_model2.train()
                optimizer = torch.optim.Adam(temp_model2.parameters(), lr=self.lr)
                criterion = nn.MSELoss()
                train_x = torch.tensor(self.history[t - self.b : t], dtype=torch.float32).to(self.device)
                train_y = torch.tensor(v_t, dtype=torch.float32).to(self.device)

                patience = 5
                counter = 0
                best = float("inf")
                for epoch in range(self.epochs):
                    optimizer.zero_grad()

                    lstm_out_temp, _ = temp_model2[0](train_x.view(1, self.b, 1))
                    last_hidden_state_temp = lstm_out_temp[:, -1, :]
                    output_tensor = temp_model2[1](last_hidden_state_temp)
                    output = output_tensor.squeeze()

                    loss = criterion(output, train_y)
                    if loss.item() < best:
                        best = loss.item()
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            break
                    loss.backward()
                    optimizer.step()

                temp_model2.eval()
                with torch.no_grad():
                    errors2_new = []
                    for y in range(t - self.b + 1, t + 1):
                        hist_y = self.history + [v_t] if y == t else self.history
                        input_seq = torch.tensor(hist_y[y - self.b : y], dtype=torch.float32).to(self.device)

                        lstm_out_temp, _ = temp_model2[0](input_seq.view(1, self.b, 1))
                        last_hidden_state_temp = lstm_out_temp[:, -1, :]
                        pred_tensor = temp_model2[1](last_hidden_state_temp)
                        pred = pred_tensor.item()

                        actual = hist_y[y]
                        errors2_new.append(np.abs(pred - actual) / (np.abs(actual) + 1e-8))
                    aare2_t_new = np.mean(errors2_new)

                if aare2_t_new > thd2:
                    is_abnormal2 = True
                    is_normal_for_hist2 = False
                else:
                    self.model2.load_state_dict(temp_model2.state_dict())
                    aare2_t = aare2_t_new

            if is_normal_for_hist2:
                self.aare_history2.append(aare2_t)

            if is_abnormal1 and is_abnormal2:
                results[i] = 1

            self.history.append(v_t)

        return results

"""
[Example]
x = np.linspace(0, 10*np.pi, 10081)
val = np.sin(x) + np.random.normal(0, 0.1, size = x.shape)
aidx = np.random.choice(np.arange(x.shape[0]//2, x.shape[0]), 4).astype(int)
val[aidx] = np.random.normal(5, 10, size = aidx.shape)
arr = val[:]
arr = (arr-np.min(arr))/(np.max(arr) - np.min(arr) + 1e-5)
model = ReRe()
model.fit(arr[:7])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,1,figsize=(24, 4))
x = arr[7:]
ano = model.predict(x)
ax[0].plot(x, color = "blue")
ax[1].plot(ano, color="red")
ax[0].grid(color = "gray", linestyle = "--", alpha = .4)
ax[1].grid(color = "gray", linestyle = "--", alpha = .4)
plt.show()
"""
