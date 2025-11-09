---

# LLM 本地推理框架

這是一個高效、易於擴展的大型語言模型（LLM）本地推理框架，專為在個人電腦上運行如 Gemma 3 和 Qwen 3 等先進模型而設計。此框架通過整合 4-bit 量化、KV 快取管理和流式生成等技術，旨在以更低的硬體資源需求，提供流暢的對話式 AI 體驗。

## 核心功能

*   **高效的記憶體利用**：採用 `BitsAndBytes` NF4 進行 4-bit 量化，大幅降低模型載入所需的 VRAM，讓消費級顯卡也能運行數十億參數的大型模型。
*   **對話加速**：內建基於使用者 ID (`uid`) 的 KV 快取（Key-Value Cache）管理機制。在連續對話中，能重複利用先前的計算結果，顯著提升後續回應的生成速度。
*   **流式生成（Streaming）**：生成的回應以 token-by-token 的方式即時輸出，提供如 ChatGPT 般的打字機效果，優化使用者體驗。
*   **統一模型載入器**：透過 `LLMLoader.py`，使用者僅需提供模型路徑，即可自動辨識並載入對應的模型處理物件（Gemma 3 或 Qwen 3），簡化了呼叫流程。
*   **易於擴展**：框架結構清晰，可以輕易地仿照 `gemma3.py` 或 `qwen3.py` 的結構，新增對其他模型的支援。

## 環境需求

請先確保您的環境已安裝 Python，並具備 NVIDIA GPU 及對應的 CUDA Toolkit。

**必要的 Python 套件：**
*   `torch`: 核心深度學習框架。
*   `transformers`: Hugging Face 的模型和分詞器函式庫。
*   `bitsandbytes`: 用於模型量化。
*   `accelerate`: 輔助 PyTorch 進行多設備（GPU）執行的函式庫。

您可以使用 pip 來安裝所有必要的套件：
```bash
pip install torch transformers bitsandbytes accelerate
```

## 快速開始

### 1. 下載模型權重

首先，您需要從 Hugging Face 或其他來源下載您想要使用的模型權重，並將它們存放在本地資料夾中。

例如，您可以將 Gemma 3 模型存放在 `C:/Users/user/LLM/gemma3`，將 Qwen 模型存放在 `./Qwen3-8B`。

### 2. 使用範例

以下是如何使用此框架進行模型推理的範例。

#### 方法一：使用統一載入器 `LLMLoader` (建議)

`LLMLoader` 會根據您提供的路徑名稱自動判斷應載入 `Gemma3Object` 還是 `Qwen3Object`。

```python
import sys
from LLMLoader import LoadModel

# --- 參數設定 ---
# 根據您的模型路徑修改
MODEL_PATH = "C:/Users/user/LLM/gemma3"  # 或者 "./Qwen3-8B"
USER_PROMPT = "你好，請你用中文做一個自我介紹"
USER_ID = "user-001"  # 用於區分不同使用者的對話歷史

# --- 載入模型 ---
try:
    print("正在載入模型...")
    model = LoadModel(MODEL_PATH)
    print("模型載入成功！")
except Exception as e:
    print(f"模型載入失敗: {e}")
    sys.exit()

# --- Qwen 模型特有的系統提示詞初始化 (可選) ---
# 如果您載入的是 Qwen 模型，可以先設定一個系統提示詞
if "qwen" in MODEL_PATH.lower():
    print("正在初始化系統提示詞...")
    system_prompt = "你是一個由 Qwen 開發的有用的人工智慧助理。"
    model.initialize(system_prompt, USER_ID)
    print("初始化完成。")

# --- 開始生成對話 ---
print(f"\n使用者: {USER_PROMPT}")
print("\n模型回應: ")

full_response = ""
# predict 方法是一個生成器，會逐字返回結果
for word in model.predict(prompt=USER_PROMPT, uid=USER_ID):
    print(word, end="", flush=True)
    full_response += word

print("\n\n--- 對話生成結束 ---")

# 您可以繼續與模型對話，它會記得先前的內容
# next_prompt = "請你總結一下你剛剛說了什麼"
# for word in model.predict(prompt=next_prompt, uid=USER_ID):
#     print(word, end="", flush=True)

```

#### 方法二：直接使用模型物件

如果您想明確地使用某個模型的物件，也可以直接匯入並實例化它。

**Gemma 3 範例:**
```python
from gemma3 import Gemma3Object

model_path = "C:/Users/user/LLM/gemma3"
gemma_model = Gemma3Object(path=model_path)

prompt = "Google 開發過哪些知名的 AI 模型？"
user_id = "user-gemma"

print("Gemma 3 回應: ")
for word in gemma_model.predict(prompt=prompt, uid=user_id):
    print(word, end="", flush=True)
print()
```

**Qwen 3 範例:**
```python
from qwen3 import Qwen3Object

model_path = "./Qwen3-8B"
qwen_model = Qwen3Object(path=model_path)
user_id = "user-qwen"

# 1. (可選) 初始化系統提示詞
system_prompt = "你是一個樂於助人的 AI 助理。"
qwen_model.initialize(sys_prmt=system_prompt, uid=user_id)

# 2. 進行對話
prompt = "什麼是大型語言模型？"

print("Qwen 3 回應: ")
for word in qwen_model.predict(prompt=prompt, uid=user_id):
    print(word, end="", flush=True)
print()
```

## 程式碼結構說明

*   **`LLMLoader.py`**
    *   `LoadModel(path)`: 核心的工廠函數。它接收一個模型路徑，並透過檢查路徑字串中是否包含 `"gemma3"` 來決定實例化 `Gemma3Object` 還是 `Qwen3Object`。這是與框架互動最簡單的入口點。

*   **`gemma3.py` / `qwen3.py`**
    *   這兩個檔案的結構高度相似，分別定義了 `Gemma3Object` 和 `Qwen3Object` 類別。
    *   `__init__(self, path, history_len)`: 建構函式。負責從給定路徑載入模型和分詞器，並設定 4-bit 量化。同時初始化 `CacheObject` 用於管理對話歷史。
    *   `predict(self, prompt, uid)`: 主要的推理函數。它接收使用者輸入 (`prompt`) 和使用者ID (`uid`)，處理模板、生成 token，並以生成器 (generator) 的形式逐個 `yield` 回應的文字片段。
    *   `initialize(self, sys_prmt, uid)` (`qwen3.py` 專有): 用於設定系統提示詞。Qwen 模型通常建議先輸入一個系統層級的指令來定義 AI 的角色，此函數專門處理這個預填 (prefill) 過程。

*   **`CacheObject` 內部類別**
    *   定義於 `gemma3.py` 和 `qwen3.py` 中。
    *   這是一個簡單的快取管理器，用一個字典 (`self.history`) 儲存每個 `uid` 對應的 `DynamicCache` 物件。
    *   它會限制儲存的對話歷史數量 (`history_len`)，當超過上限時，會自動刪除最舊的對話紀錄，避免記憶體無限增長。
