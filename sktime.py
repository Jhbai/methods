# !pip install sktime
# !pip install wandb

# ----- 避免 warnings ----- #
import warnings
warnings.filterwarnings('ignore')

# ----- sktime 套件引入 ----- #
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.datasets import load_airline

# ----- 視覺化套件 ----- #
import matplotlib.pyplot as plt

# ----- 使用套件本身的Data ----- #
y = load_airline()

# ----- 模型設定 ----- #
"""
config和training_args一定要設定，預設設定為
(1) context_length = 512, 
(2) num_train_epochs = 10, 
(3) learning_rate = 1e-4
未必符合各種資料集
"""
model = TinyTimeMixerForecaster(
    config={
        "context_length": 20,
        "prediction_length": 10,
    },
    training_args={
        "num_train_epochs": 100,
        "output_dir": "test_output",
        "per_device_train_batch_size": 32,
        "report_to": "none",
        "learning_rate": 1e-2,
    },
)

# ----- 模型訓練
model.fit(y, fh=[i+1 for i in range(10)]) 
y_pred = model.predict() 

arr = y.tolist()
plt.plot(arr, color = "blue")
pred_arr = y_pred.tolist()
plt.plot([i for i in range(len(arr)-1, len(arr)+len(pred_arr))], arr[-1:] + pred_arr, color = "green", linestyle = '--')
plt.show()
