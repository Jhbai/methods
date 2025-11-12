--- 
# PyOD: Python 異常檢測庫

[![PyPI version](https://badge.fury.io/py/pyod.svg)](https://badge.fury.io/py/pyod)
[![Downloads](https://pepy.tech/badge/pyod)](https://pepy.tech/project/pyod)

**PyOD** 是一個功能強大且全面的 Python 異常檢測（Outlier Detection）工具包。它匯集了數十種經典和新興的異常檢測算法，並提供了一致的 API 接口，讓使用者可以輕鬆地進行模型訓練、評估和預測。

## 主要特色

*   **全面的算法庫**: 包含超過 50 種主流的異常檢測算法，涵蓋從傳統統計方法到最新的深度學習模型。
*   **統一的 API**: 所有算法都遵循 Scikit-Learn 風格的 API，包括 `fit()`、`predict()`、`decision_function()` 和 `predict_proba()` 等方法，易於學習和使用。
*   **高效能與可擴展性**: 底層使用 JIT (Just-In-Time) 編譯和並行計算進行優化，能夠處理大規模數據集。
*   **詳細的文件與範例**: 提供清晰的官方文件和豐富的範例，幫助使用者快速上手。

## 安裝

你可以透過 pip 來安裝 PyOD：

```bash
pip install pyod
```

若要安裝包含所有依賴項的完整版本（例如，若要使用基於深度學習的模型），請使用：

```bash
pip install pyod[full]
```

## 快速入門

使用 PyOD 進行異常檢測的流程非常簡單，基本上包含以下幾個步驟：

1.  **導入模型**：從 `pyod.models` 中選擇並導入你想要使用的算法。
2.  **生成或載入數據**：準備你的訓練數據。
3.  **初始化並訓練模型**：建立模型實例並使用 `.fit()` 方法進行訓練。
4.  **獲取結果**：
    *   使用 `.predict()` 獲取每個數據點的預測標籤（0 表示正常，1 表示異常）。
    *   使用 `.decision_function()` 獲取每個數據點的原始異常分數。分數越高，代表該點越可能是異常。

### 範例：使用 KNN 檢測器

以下是一個使用 `KNN` (K-Nearest Neighbors) 模型的簡單範例：

```python
# 1. 導入必要的函式庫
import numpy as np
from scipy import stats
from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers

# 2. 生成範例數據
# X_train 將包含 200 個樣本和 2 個特徵
# y_train 是對應的標籤 (0: 正常, 1: 異常)
# contamination 代表數據集中異常點的比例
X_train, y_train = generate_data(n_train=200, n_features=2, contamination=0.1, random_state=42)

# 3. 初始化並訓練 KNN 模型
# n_neighbors 是 k-NN 算法中的 'k'
clf_name = 'KNN'
clf = KNN(n_neighbors=5)
clf.fit(X_train)

# 4. 獲取訓練數據的預測結果
y_train_pred = clf.labels_  # 預測標籤
y_train_scores = clf.decision_scores_  # 原始異常分數

# 也可以在新數據上進行預測
X_test, y_test = generate_data(n_test=100, n_features=2, contamination=0.1, random_state=42)

y_test_pred = clf.predict(X_test)  # 預測標籤
y_test_scores = clf.decision_function(X_test)  # 原始異常分數

# 5. 評估結果
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score

print(f"\n在訓練數據上的評估結果:")
print(f"ROC AUC 分數: {roc_auc_score(y_train, y_train_scores):.4f}")
print(f"Precision@n 分數: {precision_n_scores(y_train, y_train_scores):.4f}")

print(f"\n在測試數據上的評估結果:")
print(f"ROC AUC 分數: {roc_auc_score(y_test, y_test_scores):.4f}")
print(f"Precision@n 分數: {precision_n_scores(y_test, y_test_scores):.4f}")

```

## 支援的算法

PyOD 支援多種類型的異常檢測算法，包括：

*   **線性模型**: `PCA`, `mcd`, `OCSVM`
*   **基於鄰近度的模型**: `LOF`, `COF`, `KNN`, `ABOD`
*   **基於集成的方法**: `IsolationForest` (孤立森林), `FeatureBagging`
*   **基於統計的方法**: `HBOS`
*   **基於圖的方法**: `LUNAR`
*   **神經網絡與深度學習模型**: `AutoEncoder`, `VAE`, `DeepSVDD`, `MO_GAAL`
*   **其他**: `COPOD`, `ECOD`

完整的算法列表請參考官方文件。

## 資源連結

*   **官方文件**: [https://pyod.readthedocs.io/](https://pyod.readthedocs.io/)
*   **GitHub 原始碼**: [https://github.com/yzhao062/pyod](https://github.com/yzhao062/pyod)
*   **範例與教學**: [https://github.com/yzhao062/pyod/tree/master/notebooks](https://github.com/yzhao062/pyod/tree/master/notebooks)

## 如何貢獻

PyOD 是一個開源專案，歡迎任何形式的貢獻，包括但不限於：

*   回報 Bug
*   提出新功能建議
*   貢獻程式碼 (Pull Requests)
*   改善文件

詳細資訊請參考專案的貢獻指南。

## 引用

如果您在學術研究中使用了 PyOD，請引用以下論文：

```
@article{zhao2019pyod,
    author  = {Zhao, Yue and Nasrullah, Zain and Li, Zhekai},
    title   = {PyOD: A Python Toolbox for Scalable Outlier Detection},
    journal = {Journal of Machine Learning Research},
    year    = {2019},
    volume  = {20},
    number  = {96},
    pages   = {1-7},
    url     = {http://jmlr.org/papers/v20/19-011.html}
}
```
