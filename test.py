import numpy as np
import machine_learning

# --- 1. Test Mann-Kendall ---
print("--- Mann-Kendall Test ---")
data_trend = np.random.normal(size=(7, ))
data_trend = data_trend.astype(np.float32)
s, var_s, z = machine_learning.mk_test(data_trend)
print(f"S statistic: {s}")
print(f"Variance of S: {var_s:.2f}")
print(f"Z score: {z:.2f}\n")

# --- 2. Test Theil-Sen ---
print("--- Theil-Sen Estimator ---")
x = np.arange(10, dtype=np.float32)
y = np.array(2 * x + 1.5 + np.random.randn(10) * 0.5, dtype=np.float32)
intercept, slope = machine_learning.ts_estimator(x, y)
print(f"Estimated Intercept (alpha): {intercept:.2f} (True: ~1.5)")
print(f"Estimated Slope (beta): {slope:.2f} (True: 2.0)\n")

# --- 3. Test Theil-Sen Window ---
print("--- Theil-Sen Window Estimator ---")
y_window = np.concatenate([
    np.linspace(0, 5, 20),
    np.linspace(5, 2, 20),
    np.linspace(2, 8, 20)
]).astype(np.float32)
baseline = machine_learning.ts_window_estimator(y_window, win_size=20)
print(f"Input shape: {y_window.shape}")
print(f"Returned baseline shape: {baseline.shape}\n")
# You can plot 'y_window' and 'baseline' to see the result.

# --- 4. Test PELT Change Point Detection ---
print("--- PELT Change Point Detection ---")
signal = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(5, 1, 100),
    np.random.normal(2, 1.5, 100)
]).astype(np.float32)
# Penalty is a crucial hyperparameter to tune
changepoints = machine_learning.pelt_change_point(signal, penalty=10.0)
print(f"Detected change points at indices: {changepoints}")
print("(Expecting points around 100 and 200)\n")


# --- 5. Test DTW Subsequence Alignment ---
print("--- DTW Subsequence Alignment ---")
src_pattern = np.array([0, 1, 2, 1, 0], dtype=np.float32)
trg_sequence = np.array([0,0,0, -0.2, 1.1, 2.3, 0.9, 0.1, -0.1, 0, 0], dtype=np.float32)
result = machine_learning.dtw_subsequence_alignment(src_pattern, trg_sequence, sakoe_chiba_band=0.5)

if result:
    print(f"Pattern found from index {result['start']} to {result['end']}")
    print(f"DTW distance: {result['distance']:.2f}")
else:
    print("No alignment found.")