import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class SR_CNN(nn.Module):
    def __init__(self, window_size=64):
        super(SR_CNN, self).__init__()
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----- Layer defined follow the paper ----- #
        self.layer1 = nn.Conv1d(window_size, window_size, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv1d(window_size, 2 * window_size, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2 * window_size, 4 * window_size)
        self.fc2 = nn.Linear(4 * window_size, window_size)
        
        self.relu = nn.ReLU(inplace=True) # ReLU will be used after 1st CNN and 3rd Dense
        self.sigmoid = nn.Sigmoid() # Ouptut is the probability of class

        # ----- Loss function ----- #
        self.criterion = nn.BCELoss()

        # ----- Edge Effect from FFT, linear padding in the end of sequennces ----- #
        self.estimated_points_number = 5
        
        # -----
        self.to(self.device)

    def _spectral_residual_transform(self, values):
        # ----- sequences length and padding length ----- #
        n = len(values)
        m = self.estimated_points_number
        
        # ----- 1. Gradient extension for padding ------ #
        if n > m:
            grads = [(values[-1] - values[-m-1+i]) / (m - i + 1) for i in range(1, m + 1)]
            avg_g = np.mean(grads)
            ext_point = values[-1] + avg_g * m
            extension = [ext_point] * m
            values_ext = np.concatenate([values, extension])
        else:
            values_ext = values

        # ----- 2. FFT ----- #
        trans = np.fft.fft(values_ext)
        mag = np.abs(trans)
        phase = np.angle(trans)

        # ----- 3. Log Amplitude & Average ----- #
        eps = 1e-8
        log_amp = np.log(mag + eps)
        q = 3
        if len(log_amp) >= q:
            avg_log_amp = np.convolve(log_amp, np.ones(q)/q, mode='same')
        else:
            avg_log_amp = log_amp

        # ----- 4. Spectral Residual ----- #
        spectral_residual = log_amp - avg_log_amp

        # ----- 5. Inverse FFT -> Saliency Map ----- #
        sr_complex = np.exp(spectral_residual + 1j * phase)
        spatial_sr = np.abs(np.fft.ifft(sr_complex))

        return spatial_sr[:n]

    def _inject_anomaly(self, values):
        """
        宣告 control 變數確保特定位置(win_size - 6)在長時間未被選中時會被強制injected
        """

        # ----- Using hasattr to check this attr exists or not ----- #
        if not hasattr(self, 'control'):
            self.control = 0
        # ----- Accoding to the paper, it's about five points in the series ----- #
        max_inject_nums = 5 

        # ----- Internal Helper Functions ----- #
        def normalize(a):
            amin = np.min(a)
            amax = np.max(a)
            a = (a - amin) / (amax - amin + 1e-5)
            return 3 * a

        def average_filter(a, win=5):
            kernel = np.ones(win) / win
            return np.convolve(a, kernel, mode='same')

        # ----- transform to double then normalize ----- #
        data = np.array(values).astype(np.float64)
        data = normalize(data)

        # ----- Always injected (num >= 1) ----- #
        num = np.random.randint(1, max_inject_nums)
        
        # ----- Random select the indices ----- #
        ids = np.random.choice(self.window_size, num, replace=False)
        lbs = np.zeros(self.window_size, dtype=np.int64)

        # ----- Check whether 6th point is selected or not, if not, accumulate the control value ----- #
        if (self.window_size - 6) not in ids:
            self.control += np.random.random()
        else:
            self.control = 0
            
        # ----- injected ----- #
        if self.control > 100:
            target_pos = self.window_size - 6
            if target_pos >= 0: # 確保索引合法
                ids[0] = target_pos
            self.control = 0

        # ----- Injection Formula from paper ----- #
        mean = np.mean(data)
        dataavg = average_filter(data)
        var = np.var(data)
        for id in ids:
            data[id] += (dataavg[id] + mean) * np.random.randn() * min((1 + var), 10)
            lbs[id] = 1

        is_anomaly = 1

        return data, 1

    def forward_cnn(self, x):
        """
        Forward pass using the architecture from Reference.py
        x shape: (Batch, Window) -> Adapted from MyCode's (Batch, 1, Window)
        """
        # Reference.py expects: x.view(x.size(0), self.window, 1)
        # This implies Input x is (Batch, Window)
        
        x = x.view(x.size(0), self.window_size, 1) # (Batch, Window, 1)
        
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x) # Output: (Batch, 2*Window, 1)
        
        x = x.view(x.size(0), -1) # Flatten -> (Batch, 2*Window)
        x = self.relu(x)
        
        x = self.fc1(x) # -> (Batch, 4*Window)
        x = self.relu(x)
        
        x = self.fc2(x) # -> (Batch, Window)
        
        return self.sigmoid(x)

    def fit(self, x_series, n_epochs=10, batch_size=32):
        """
        Training loop adapted to handle Reference.py output shape.
        x_series: List or Array of time series data.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        print(f"熊熊 Note: Starting training with Reference architecture...")
        
        # --- Data Generation (On-the-fly for demo) ---
        X_train = []
        y_train = []
        samples_count = 1000 

        # Creating synthetic training samples from the input series
        # Note: In production, do this more efficiently using datasets
        series = x_series[0] if isinstance(x_series, list) else x_series[0, :]
        
        for _ in range(samples_count//2):
            start_idx = np.random.randint(0, len(series) - self.window_size)
            segment = series[start_idx : start_idx + self.window_size]

            sr_map = self._spectral_residual_transform(segment)
            X_train.append(sr_map.copy())
            y_train.append(0)

            injected_seg, label = self._inject_anomaly(segment)
            sr_map = self._spectral_residual_transform(injected_seg)
            
            X_train.append(sr_map.copy())
            y_train.append(label)

        X_tensor = torch.FloatTensor(np.array(X_train)).to(self.device) # (N, W)
        y_tensor = torch.FloatTensor(np.array(y_train)).unsqueeze(1).to(self.device) # (N, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_history = []
        
        for epoch in range(n_epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward
                # batch_x shape is (Batch, Window)
                outputs = self.forward_cnn(batch_x) # Output: (Batch, Window)
                
                # --- CRITICAL ADAPTATION ---
                # Reference.py outputs a score for the WHOLE window (size W).
                # MyCode.py training labels are for the LAST point (size 1).
                # We take the last element of the output vector to match the label.
                prediction_for_last_point = outputs[:, -1].unsqueeze(1) 
                
                loss = self.criterion(prediction_for_last_point, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            loss_history.append(avg_loss)
            if (epoch+1) % 2 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
        
        return loss_history

    def predict(self, x):
        """
        Inference: Returns anomaly score for the last point of the input series.
        """
        self.eval()
        if len(x) < self.window_size:
            return 0.0
            
        target_window = x[-self.window_size:]
        
        # 1. SR Transform
        sr_map = self._spectral_residual_transform(target_window)
        
        # 2. To Tensor
        sr_tensor = torch.FloatTensor(sr_map).unsqueeze(0).to(self.device) # (1, W)
        
        # 3. Model Inference
        with torch.no_grad():
            output = self.forward_cnn(sr_tensor) # (1, W)
            
        # Return score of the last point
        return output[0, :]

if __name__ == "__main__":
    # ----- Data Preparation ----- #
    t = np.linspace(0, 100, 2000)
    data = np.sin(t) + np.random.normal(0, 0.1, 2000)
    plt.figure(figsize = (24, 2))
    plt.plot(data, color = "black")
    plt.show()
    
    # ----- Add a real anomaly at the end for testing ----- #
    data[-1] = 5.0 
    
    # ----- Reshape for the model (N_series, Length) ----- #
    data_input = data.reshape(1, -1)

    print(f"Data shape: {data_input.shape}")

    # ----- Model training ----- #
    window_size = 128
    model = SR_CNN(window_size=window_size)
    print("\n--- Training Start ---")
    losses = model.fit(data_input, n_epochs=35, batch_size=32)
    print("--- Training Done ---")

    # ----- Inference ----- #
    print("\n--- Testing Inference ---")
    fig, ax = plt.subplots(3, 1, figsize = (12, 6))
    
    """Case A: Normal"""
    normal_segment = data[0:window_size]
    score_normal = model.predict(normal_segment)
    ax[0].plot(normal_segment)
    _ax = ax[0].twinx()
    _ax.axhline(0.95, color = "gray", linestyle = "--")
    _ax.plot(score_normal.detach().cpu().numpy(), color = "red")
    
    """Case B: Anomaly"""
    anomaly_segment = data[-window_size:]
    score_anomaly = model.predict(anomaly_segment)
    ax[1].plot(anomaly_segment)
    _ax = ax[1].twinx()
    _ax.axhline(0.95, color = "gray", linestyle = "--")
    _ax.plot(score_anomaly.detach().cpu().numpy(), color = "red")
    
    # ----- Plot Loss -----
    ax[2].plot(losses)
    ax[0].set_title("Training Loss (Reference Arch + SR)")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Loss")
    plt.show()
