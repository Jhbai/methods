使用一組多變量時間序列（`X`，維度為 K）來預測另一組多變量時間序列（`y`，維度為 M）是一個常見的應用場景，特別是對於像 `TinyTimeMixerForecaster` 這類基於深度學習的模型。
根據 `sktime` 的文件，可以使用 K 個變量的時間序列作為特徵 (`X`) 來預測 M 個變量的時間序列 (`y`)
### 核心概念

在 `sktime` 的框架中，這個問題被定義為**帶有外生變數的多變量預測 (Multivariate Forecasting with Exogenous Variables)**。

1.  **目標序列 `y` (Endogenous)**：這是您想要預測的 M 個時間序列。在 `sktime` 中，它應該是一個 `pandas.DataFrame`，其中包含 M 個欄位，每個欄位代表一個目標時間序列。
2.  **外生特徵 `X` (Exogenous)**：這是您用來輔助預測的 K 個時間序列。它也應該是一個 `pandas.DataFrame`，包含 K 個欄位。`TinyTimeMixerForecaster` 和許多其他現代預測器都支援使用外生變數來提高預測準確性。
3.  **時間對齊**：最關鍵的一點是，`y` 和 `X` 的 `pandas.DatetimeIndex` 或 `pandas.PeriodIndex` 必須完全對齊。模型需要根據同一時間點的 `X` 和 `y` 來學習它們之間的關係。
4.  **未來 `X` 的需求**：當您要預測未來時，您必須提供**未來時間點對應的 `X` 值**。 這是因為模型在訓練時學會了依賴 `X` 來預測 `y`，所以在預測時也需要未來的 `X` 作為輸入。如果未來的 `X` 未知，您可能需要先對 `X` 本身進行預測。

### `TinyTimeMixerForecaster` 的角色

`TinyTimeMixerForecaster` 是一個預訓練的深度學習模型，非常適合處理這種多變量問題。 它的設計使其能夠捕捉多個序列之間以及外生變數與目標序列之間的複雜動態關係。

### 實作步驟與程式碼範例

以下是一個完整的範例，展示如何準備數據、訓練模型並進行預測。

假設我們要用 5 個變數 (K=5) 來預測 2 個變數 (M=2)。

```python
import pandas as pd
import numpy as np
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.utils.plotting import plot_series

# 1. 參數設定
M = 2  # 目標序列的維度 (y)
K = 5  # 特徵序列的維度 (X)
n_timepoints_train = 100  # 訓練數據的時間點數量
n_timepoints_future = 12  # 要預測的未來時間點數量

# 2. 數據準備 (y 和 X)
# 創建一個時間索引
train_idx = pd.date_range(start="2024-01-01", periods=n_timepoints_train, freq="h")
future_idx = pd.date_range(start=train_idx[-1] + pd.Timedelta(hours=1), periods=n_timepoints_future, freq="h")
full_idx = train_idx.union(future_idx)

# --- 創建目標序列 y (M=2 維) ---
# 為了讓範例更真實，我們創建一些有規律的數據加上噪聲
y_data = {
    f'target_{i}': np.sin(np.arange(n_timepoints_train) / (10 + i*5)) + np.random.rand(n_timepoints_train) * 0.2
    for i in range(M)
}
y_train = pd.DataFrame(y_data, index=train_idx)

# --- 創建外生特徵序列 X (K=5 維) ---
# X 必須包含訓練期間和預測期間的數據
X_data = {
    f'feature_{j}': np.random.rand(n_timepoints_train + n_timepoints_future)
    for j in range(K)
}
X_full = pd.DataFrame(X_data, index=full_idx)

# 將 X 分為訓練部分和預測部分
X_train = X_full.loc[train_idx]
X_future = X_full.loc[future_idx]

print("--- 數據形狀 ---")
print(f"y_train shape: {y_train.shape}") # (100, 2)
print(f"X_train shape: {X_train.shape}") # (100, 5)
print(f"X_future shape: {X_future.shape}") # (12, 5)

# 3. 模型初始化
# TinyTimeMixerForecaster 是一個深度學習模型，可能需要一些時間來訓練
# 為了快速演示，我們使用較少的 epoch
forecaster = TinyTimeMixerForecaster(
    model_path="ibm/TTM",  # 使用 Hugging Face 上的預訓練模型
    training_args={"num_train_epochs": 5} # 設置訓練參數，例如訓練週期
)

# 4. 模型訓練
# 我們將 y_train 和 X_train 傳入 fit 方法
# sktime 會自動處理多變量 y 和多變量 X
print("\n--- 開始訓練模型 ---")
forecaster.fit(y=y_train, X=X_train)
print("--- 模型訓練完成 ---")


# 5. 進行預測
# 創建預測範圍 (Forecasting Horizon)
fh = np.arange(1, n_timepoints_future + 1)

# 進行預測時，必須提供未來的外生變數 X_future
print("\n--- 開始進行預測 ---")
y_pred = forecaster.predict(fh=fh, X=X_future)
print("--- 預測完成 ---")

# 6. 結果檢視
print("\n--- 預測結果 ---")
print(y_pred)
print(f"\n預測結果的形狀: {y_pred.shape}") # 應該是 (12, 2)

# 視覺化其中一個目標變數的預測結果
# plot_series 需要 y_test，這裡我們沒有，所以只畫出訓練數據和預測結果
fig, ax = plot_series(y_train['target_0'], y_pred['target_0'], labels=["y_train (target_0)", "y_pred (target_0)"])
ax.set_title("多變量預測 (其中一個變數的結果)")
```

### 關於 `update` 方法

最初連結到的 `update` 方法用於**增量學習**。當您有新的數據點進來時（例如，新的一小時或一天的數據），您可以使用 `forecaster.update(y_new, X_new)` 來更新模型，而無需從頭開始重新訓練 `fit`。這在線上學習或需要頻繁更新模型的場景中非常高效。

使用 `update` 的流程如下：
1.  `forecaster.fit(y_train, X_train)`
2.  收到新數據 `y_new`, `X_new`
3.  `forecaster.update(y_new, X_new)`
4.  `y_pred = forecaster.predict(fh, X=X_future)`

### 總結

要在 `sktime` 中使用 K-變量序列預測 M-變量序列，需要：
1.  將目標序列（M 維）和特徵序列（K 維）分別準備成兩個具有對齊時間索引的 `pandas.DataFrame`。
2.  選擇一個支援外生變數的預測器，例如 `TinyTimeMixerForecaster`。
3.  使用 `fit(y_train, X_train)` 進行訓練。
4.  在呼叫 `predict(fh, X=X_future)` 時，務必提供與預測範圍 `fh` 對應的未來外生變數 `X_future`。

這個流程在 `sktime` 中是標準化的，適用於所有支援多變量輸入和外生變數的預測器。
