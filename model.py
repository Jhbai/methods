from model import JumpStarter
import numpy as np

print("=" * 70)
print("JumpStarter 測試")
print("=" * 70)

# 生成測試數據
np.random.seed(42)
n_samples = 500
n_features = 10

print(f"\n生成測試數據: {n_samples} × {n_features}")

t = np.linspace(0, 10 * np.pi, n_samples)
X = np.zeros((n_samples, n_features))

for i in range(n_features):
    X[:, i] = np.sin(t * (i + 1) / 3) + 0.1 * np.random.randn(n_samples)

# 注入異常
print("注入異常 [200:220]")
X[200:220, :] += 3.0

# 創建檢測器
detector = JumpStarter(contamination=0.02)

# 訓練
print("\n" + "=" * 70)
detector.fit(X)
predictions = detector.predict(X)
print("=" * 70)

# 分析
anomaly_mask = predictions == -1
n_anomalies = anomaly_mask.sum()

print(f"\n檢測結果:")
print(f"  異常點數: {n_anomalies}")
print(f"  異常率: {n_anomalies / n_samples * 100:.2f}%")

if n_anomalies > 0:
    anomaly_indices = np.where(anomaly_mask)[0]
    print(f"  位置（前10）: {anomaly_indices[:10]}")
    
    detected = [i for i in anomaly_indices if 200 <= i < 220]
    recall = len(detected) / 20 * 100
    print(f"  捕獲率: {recall:.1f}% ({len(detected)}/20)")

print(f"  閾值: {detector.threshold_:.4f}")
