---

# 高效能時間序列分析函式庫 (High-Performance Time Series Analysis Library)

這是一個使用 C 語言編寫，並透過 Cython 封裝給 Python 使用的高效能時間序列分析工具庫。本函式庫的核心演算法利用了 **AVX (Advanced Vector Extensions)** 指令集進行了底層最佳化，以實現對 NumPy 陣列的快速處理。

## 功能特性

本函式庫提供了以下幾種常用的時間序列分析演算法：

*   **Mann-Kendall 趨勢檢定 (Mann-Kendall Test)**: 一種非參數統計檢定方法，用於判斷時間序列資料中是否存在單調趨勢。
*   **Theil-Sen 估計器 (Theil-Sen Estimator)**: 一種穩健的線性迴歸方法，用於估計斜率和截距，對離群值不敏感。同時提供滑動視窗版本，用於計算局部趨勢基線。
*   **PELT 變點偵測 (Pruned Exact Linear Time - PELT)**: 一種快速準確的演算法，用於在時間序列中偵測平均值或變異數發生變化的時間點 (變點)。
*   **DTW 子序列對齊 (Dynamic Time Warping Subsequence Alignment)**: 一種用於尋找一個較短的查詢序列 (pattern) 在一個較長的目標序列中最佳匹配位置的演算法，對時間軸上的伸縮和扭曲具有穩健性。

## 環境需求

*   C 編譯器 (例如 GCC 或 Clang)
*   Python 3.x
*   NumPy
*   Cython
*   支援 **AVX** 指令集的 CPU

## 安裝與建置

您可以透過 `setup.py` 腳本來編譯 C 原始碼，並產生可供 Python 匯入的模組。請在專案根目錄下執行以下指令：

```bash
python setup.py build_ext --inplace
```

成功執行後，目錄下會產生一個名為 `machine_learning.so` (Linux/macOS) 或 `machine_learning.pyd` (Windows) 的檔案，這就是可以被 Python 使用的二進位模組。

## API 使用指南

以下是每個模組的詳細使用方法。所有函式都接收 `numpy.ndarray` 作為輸入，並建議使用 `np.float32` 型別以獲得最佳效能。

### 1. Mann-Kendall 趨勢檢定 (`mk_test`)

此函式用於檢測時間序列中的單調趨勢。

**函式簽名**:
`mk_test(data: np.ndarray) -> (int, float, float)`

**參數**:
*   `data` (np.ndarray): 一維的 `float32` NumPy 陣列。

**回傳**:
一個包含三個值的元組：
*   `S` (int): Mann-Kendall 的 S 統計量。
*   `var_S` (float): S 的變異數。
*   `Z` (float): 標準化的 Z 分數統計量。

**範例**:
```python
import numpy as np
import machine_learning

# 產生一組隨機數據
data_trend = np.random.normal(size=(7, )).astype(np.float32)

# 進行 Mann-Kendall 檢定
s, var_s, z = machine_learning.mk_test(data_trend)

print("--- Mann-Kendall Test ---")
print(f"S statistic: {s}")
print(f"Variance of S: {var_s:.2f}")
print(f"Z score: {z:.2f}")
```

### 2. Theil-Sen 估計器 (`ts_estimator`)

估計一組 (x, y) 資料點的穩健線性迴歸線。

**函式簽名**:
`ts_estimator(x: np.ndarray, y: np.ndarray) -> (float, float)`

**參數**:
*   `x` (np.ndarray): 一維的 `float32` NumPy 陣列。
*   `y` (np.ndarray): 一維的 `float32` NumPy 陣列，長度需與 `x` 相同。

**回傳**:
一個包含兩個值的元組：
*   `intercept` (float): 估計的截距 (alpha)。
*   `slope` (float): 估計的斜率 (beta)。

**範例**:
```python
import numpy as np
import machine_learning

print("\n--- Theil-Sen Estimator ---")
x = np.arange(10, dtype=np.float32)
# y = 2x + 1.5 + noise
y = np.array(2 * x + 1.5 + np.random.randn(10) * 0.5, dtype=np.float32)

intercept, slope = machine_learning.ts_estimator(x, y)
print(f"Estimated Intercept (alpha): {intercept:.2f} (True: ~1.5)")
print(f"Estimated Slope (beta): {slope:.2f} (True: 2.0)")
```

### 3. 滑動視窗 Theil-Sen 估計器 (`ts_window_estimator`)

對時間序列進行滑動視窗 Theil-Sen 估計，常用於計算訊號的局部趨勢基線。

**函式簽名**:
`ts_window_estimator(y: np.ndarray, win_size: int) -> np.ndarray`

**參數**:
*   `y` (np.ndarray): 一維的 `float32` 時間序列 NumPy 陣列。
*   `win_size` (int): 滑動視窗的大小。

**回傳**:
一個與 `y` 等長的一維 NumPy 陣列，代表每個點的基線值。

**範例**:
```python
import numpy as np
import machine_learning

print("\n--- Theil-Sen Window Estimator ---")
# 產生一個有分段趨勢的訊號
y_window = np.concatenate([
    np.linspace(0, 5, 20),
    np.linspace(5, 2, 20),
    np.linspace(2, 8, 20)
]).astype(np.float32)

baseline = machine_learning.ts_window_estimator(y_window, win_size=20)
print(f"Input shape: {y_window.shape}")
print(f"Returned baseline shape: {baseline.shape}")
# 可以將 'y_window' 和 'baseline' 進行繪圖比較
```

### 4. PELT 變點偵測 (`pelt_change_point`)

偵測時間序列中統計特性發生變化的點。

**函式簽名**:
`pelt_change_point(signal: np.ndarray, penalty: float) -> list`

**參數**:
*   `signal` (np.ndarray): 一維的 `float32` 時間序列 NumPy 陣列。
*   `penalty` (float): 懲罰值，一個關鍵的超參數。值越高，偵測到的變點越少。需要根據資料特性進行調整。

**回傳**:
一個包含所有偵測到的變點索引位置的 Python 列表。

**範例**:
```python
import numpy as np
import machine_learning

print("\n--- PELT Change Point Detection ---")
# 產生一個包含兩個變點的訊號
signal = np.concatenate([
    np.random.normal(0, 1, 100),   # 均值為 0
    np.random.normal(5, 1, 100),   # 均值為 5
    np.random.normal(2, 1.5, 100) # 均值為 2
]).astype(np.float32)

# 懲罰值是需要調整的關鍵參數
changepoints = machine_learning.pelt_change_point(signal, penalty=10.0)
print(f"Detected change points at indices: {changepoints}")
print("(Expecting points around 100 and 200)")
```

### 5. DTW 子序列對齊 (`dtw_subsequence_alignment`)

在一個長序列中尋找一個短模式序列的最佳匹配段。

**函式簽名**:
`dtw_subsequence_alignment(src: np.ndarray, trg: np.ndarray, sakoe_chiba_band: float = 0.0, threshold: float = 0.0) -> dict or None`

**參數**:
*   `src` (np.ndarray): 一維 `float32` NumPy 陣列，代表查詢的模式 (pattern)。
*   `trg` (np.ndarray): 一維 `float32` NumPy 陣列，代表被搜尋的長序列。
*   `sakoe_chiba_band` (float): Sakoe-Chiba 約束帶的寬度比例，用於加速計算。設為 0 表示不使用。通常設為 0.1 到 0.5 之間。
*   `threshold` (float): 剪枝閾值。如果部分對齊的成本超過此值，則提前中止計算。設為 0 表示不使用。

**回傳**:
*   如果找到對齊，回傳一個包含以下鍵的字典：
    *   `start` (int): 匹配段在目標序列 (`trg`) 中的起始索引。
    *   `end` (int): 匹配段在目標序列 (`trg`) 中的結束索引。
    *   `distance` (float): 正規化的 DTW 距離。
    *   `path` (tuple): 一個包含兩個 ndarray (路徑 i, 路徑 j) 的元組，代表詳細的對齊路徑。
*   如果沒有找到有效的對齊，回傳 `None`。

**範例**:
```python
import numpy as np
import machine_learning

print("\n--- DTW Subsequence Alignment ---")
src_pattern = np.array([0, 1, 2, 1, 0], dtype=np.float32)
trg_sequence = np.array([0,0,0, -0.2, 1.1, 2.3, 0.9, 0.1, -0.1, 0, 0], dtype=np.float32)

# 使用 50% 的 Sakoe-Chiba band
result = machine_learning.dtw_subsequence_alignment(src_pattern, trg_sequence, sakoe_chiba_band=0.5)

if result:
    print(f"Pattern found from index {result['start']} to {result['end']}")
    print(f"DTW distance: {result['distance']:.2f}")
else:
    print("No alignment found.")
```
