在 sktime 裡，只要把  
• 目標序列 y 做成 pandas DataFrame（M 欄、DatetimeIndex）  
• 外生變數 X 也做成 DataFrame（K 欄，與 y 同一時間軸，K 可以遠大於 M）  
就可以直接把 X 當成 features 送進任何「支援 X」且「支援多變量 y」的預測器，例如 TinyTimeMixerForecaster、MultioutputTimeSeriesRegression-Forecaster 等。

─────────────────────────────  
1. 資料準備  
```python
# y: M 個 target，每一欄一個變數
y = pd.DataFrame(
        data = {...},          # shape = (T, M)
        index = pd.date_range(...),
        columns = ["y1", "y2", ...])

# X: K 個外生特徵
X = pd.DataFrame(
        data = {...},          # shape = (T, K)
        index = y.index)       # 時間軸一定要對齊
```

2. 以 TinyTimeMixer 為例  
TinyTimeMixer 同時支援  
• 多變量 y  
• `fit(y, X=None, fh=None)`、`predict(fh, X)` 這種帶 X 的 API。  
```python
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.forecasting.model_selection import temporal_train_test_split

# 切出最後 H 步當測試集
H = 24                      # 要預測的步數 / fh
y_train, y_test = temporal_train_test_split(y, test_size=H)
X_train, X_test = temporal_train_test_split(X, test_size=H)

fh = list(range(1, H+1))    # [1, 2, …, H]

forecaster = TinyTimeMixerForecaster(
              config={"context_length": 96, "prediction_length": H},
              training_args={"num_train_epochs": 5})

forecaster.fit(y_train, X=X_train, fh=fh)       # ← K≫M 也沒關係
y_pred = forecaster.predict(fh=fh, X=X_test)    # 帶入未來 X
```

3. 其他做法  
如果你想把任何 sklearn 回歸器拿來一次預測 M 條序列，可用 sktime 的「多輸出-化簡器」：  
```python
from sktime.forecasting.compose import MultioutputTimeSeriesRegressionForecaster
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, random_state=0)
forecaster = MultioutputTimeSeriesRegressionForecaster(
                estimator=rf, window_length=24)  # 用過去 24 步做特徵
forecaster.fit(y_train, X=X_train, fh=fh)
y_pred = forecaster.predict(fh=fh, X=X_test)
```

4. 必備細節與陷阱  
• 索引對齊：大部分預測器帶有 `X-y-must-have-same-index` 的 tag，索引不對會報錯。更新 (update) 時一樣要把同索引的 X 傳進去。  
• 預測時的 X：如果未來的外生特徵無法先知道，需自行外插或用最後一次觀測值填補。  
• 資料維度：  
  - 「多變量單一實例」→ 2-D DataFrame (行是時間，列是變數)。  
  - 如果有多實例(panel/hierarchical)，改用 MultiIndex。  
• 想先做差分、標準化…可用 TransformedTargetForecaster 或 ForecastingPipeline 把轉換器包住。  
• Hyper-parameter tuning：ForecastingGridSearchCV 與 sktime 的 CV splitters 一樣支援帶 X。

這樣就完成了「K 維時間序列特徵 ➜ M 維時間序列目標」的工作流程；K > M 完全沒問題，只要把 X、y 依規定的 sktime 介面餵給模型即可。
