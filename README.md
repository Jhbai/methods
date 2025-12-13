## `main.py` 檔案功能說明

### 函式列表與說明

#### `execute_command(command: str, timeout: int = 60) -> str`
- **參數**:
  - `command` (str): 要執行的 shell 命令。
  - `timeout` (int, 預設值 60): 命令執行的超時時間（秒）。
- **功能**: 執行一個 shell 命令並捕獲其標準輸出和錯誤輸出。如果命令超時或執行失敗，則返回錯誤訊息。

#### `read_file(path: str) -> str`
- **參數**:
  - `path` (str): 要讀取的檔案路徑。
- **功能**: 讀取指定路徑的檔案內容，並在每行前面加上行號。如果讀取失敗，則返回錯誤訊息。

#### `write_file(path: str, content: str) -> str`
- **參數**:
  - `path` (str): 要寫入的檔案路徑。
  - `content` (str): 要寫入檔案的內容。
- **功能**: 將內容寫入指定路徑的檔案。如果檔案不存在，則會建立；如果檔案存在，則會覆蓋。如果寫入失敗，則返回錯誤訊息。

#### `edit_file(path: str, start_line: str, end_line: str, new_content: str) -> str`
- **參數**:
  - `path` (str): 要編輯的檔案路徑。
  - `start_line` (str): 編輯的起始行號（從 1 開始）。
  - `end_line` (str): 編輯的結束行號。
  - `new_content` (str): 用於替換指定行範圍的新內容。
- **功能**: 編輯指定路徑的檔案，用 `new_content` 替換從 `start_line` 到 `end_line` 的內容。如果編輯失敗，則返回錯誤訊息。

#### `list_files(directory: str = ".", pattern: str = "*") -> str`
- **參數**:
  - `directory` (str, 預設值 "."): 要列出檔案的目錄路徑。
  - `pattern` (str, 預設值 "*"): 用於過濾檔案的 glob 模式。
- **功能**: 列出指定目錄下符合模式的檔案。會遞迴搜尋子目錄，並排除以 `.` 開頭的目錄。返回最多 200 個結果。如果列出失敗，則返回錯誤訊息。

#### `search_files(directory: str, regex: str) -> str`
- **參數**:
  - `directory` (str): 要搜尋的目錄路徑。
  - `regex` (str): 要搜尋的正規表達式模式。
- **功能**: 在指定目錄下的檔案中搜尋符合正規表達式模式的內容。返回最多 100 個匹配項，每個匹配項會顯示檔案路徑、行號和匹配行的部分內容。如果搜尋失敗，則返回錯誤訊息。

#### `create_directory(path: str) -> str`
- **參數**:
  - `path` (str): 要建立的目錄路徑。
- **功能**: 建立指定路徑的目錄。如果目錄已存在，則不會有任何操作。如果建立失敗，則返回錯誤訊息。

#### `delete_file(path: str) -> str`
- **參數**:
  - `path` (str): 要刪除的檔案路徑。
- **功能**: 刪除指定路徑的檔案。不支援刪除目錄，如果路徑是目錄則會返回錯誤訊息。如果刪除失敗或檔案不存在，則返回錯誤訊息。

#### `parse_tool_calls(text: str) -> List[Dict]`
- **參數**:
  - `text` (str): 包含工具呼叫的文字內容。
- **功能**: 解析文字中 `<tool_call>...</tool_call>` 格式的工具呼叫，並將其轉換為字典列表。每個字典包含工具名稱和參數。

#### `execute_tool(tool_call: Dict) -> str`
- **參數**:
  - `tool_call` (Dict): 包含工具名稱和參數的字典。
- **功能**: 根據 `tool_call` 的內容執行對應的工具函式（如 `execute_command`, `read_file` 等），並返回工具執行的結果。如果工具名稱未知，則返回錯誤訊息。

#### `clear_vram()`
- **功能**: 清理 GPU 記憶體，包括執行 Python 垃圾回收和 PyTorch CUDA 快取清理。

#### `get_vram_usage() -> float`
- **功能**: 返回當前已分配的 GPU 記憶體使用量（GB）。如果 CUDA 不可用，則返回 0.0。

#### `get_vram_ratio() -> float`
- **功能**: 返回當前 GPU 記憶體分配量佔總記憶體的比例。如果 CUDA 不可用，則返回 0.0。

#### `check_memory_pressure() -> bool`
- **功能**: 檢查 GPU 記憶體使用比例是否超過預設閾值 `VRAM_THRESHOLD` (0.85)，用於判斷是否存在記憶體壓力。

#### `cleanup_old_sessions()`
- **功能**: 清理超過 `SESSION_TIMEOUT` (3600 秒) 的舊會話，並在清理後呼叫 `clear_vram()` 釋放記憶體。

#### `get_or_create_session(uid: str) -> Session`
- **參數**:
  - `uid` (str): 用戶的唯一識別符。
- **功能**: 獲取或建立一個新的會話。如果會話已存在，則更新其最後訪問時間。返回 `Session` 物件。

#### `load_model()`
- **功能**: 加載 Gemma3ForConditionalGeneration 模型及其處理器。模型以 4 位元量化方式加載，並自動配置到可用設備。此函式會設定 `global model` 和 `processor` 變數。

#### `decode_image(image_data: str) -> Image.Image`
- **參數**:
  - `image_data` (str): Base64 編碼的圖片資料字串（可包含 `data:` 前綴）。
- **功能**: 將 Base64 編碼的圖片資料解碼並轉換為 PIL `Image.Image` 物件。

#### `ContinuousBatchProcessor` (類別)
- **功能**: 處理多個並發請求的批次處理器，用於管理和排隊模型推斷請求。
  - `__init__()`: 初始化批次處理器，包括請求列表、鎖和事件。
  - `add_request(request: PendingRequest)`: 將一個 `PendingRequest` 加入到處理佇列中。
  - `get_batch() -> List[PendingRequest]`: 從佇列中獲取一個批次的請求，最多 `MAX_BATCH_SIZE` 個。
  - `run()`: 批次處理器的主要運行迴圈，不斷獲取和處理請求。在處理前會檢查記憶體壓力並清理舊會話。
  - `process_request(request: PendingRequest)`: 異步處理單個請求，呼叫 `generate_response` 並處理可能發生的錯誤。
  - `generate_response(request: PendingRequest)`: 根據請求中的訊息和圖片生成模型回應，並將生成的 token 流式傳輸到請求的回應佇列中。支援處理多模態輸入和工具呼叫。

#### `stream_with_tools(uid: str, prompt: str, images: Optional[List[str]] = None)`
- **參數**:
  - `uid` (str): 用戶的唯一識別符。
  - `prompt` (str): 用戶輸入的提示訊息。
  - `images` (Optional[List[str]]): 可選的 Base64 編碼圖片列表。
- **功能**: 處理用戶的聊天請求，包括管理會話、與模型互動以及執行工具呼叫。它以異步方式生成回應，並支援多輪對話和工具的使用。會話中會維護 `SYSTEM_PROMPT` 以引導模型使用工具。

#### `lifespan(app: FastAPI)` (異步上下文管理器)
- **參數**:
  - `app` (FastAPI): FastAPI 應用實例。
- **功能**: 定義 FastAPI 應用程序的生命週期事件。在應用啟動時加載模型和初始化 `ContinuousBatchProcessor`，在應用關閉時停止批次處理器。

#### `@app.post("/chat")` `async def chat(request: ChatRequest)`
- **參數**:
  - `request` (ChatRequest): 包含 `uid`, `prompt` 和 `images` 的聊天請求物件。
- **功能**: 處理聊天請求的 API 端點。它接收聊天請求，並使用 `stream_with_tools` 異步生成回應，以流式傳輸的方式返回給客戶端。

#### `@app.delete("/session/{uid}")` `async def delete_session(uid: str)`
- **參數**:
  - `uid` (str): 要刪除的會話的唯一識別符。
- **功能**: 處理刪除會話的 API 端點。它從 `sessions` 中刪除指定 `uid` 的會話，並清理 GPU 記憶體。返回刪除狀態。

#### `@app.get("/health")` `async def health()`
- **功能**: 健康檢查 API 端點。返回應用程序的運行狀態，包括 GPU 記憶體使用量、記憶體比例和當前活動會話數量。

#### `@app.post("/clear_memory")` `async def clear_memory()`
- **功能**: 手動清理記憶體的 API 端點。它呼叫 `cleanup_old_sessions()` 和 `clear_vram()` 來釋放記憶體。返回清理後的記憶體狀態。

## 操作範例: `predict.py` 的操作邏輯與標準作業程序 (SOP)

`predict.py` 提供了一個 `predict` 函式，用於透過 HTTP POST 請求與 `main.py` 中運行的 `/chat` API 進行互動。它將使用者提供的提示 (prompt) 和唯一使用者識別碼 (UID) 發送到 API，並可選地包含 Base64 編碼的圖片資料。`predict` 函式設計為一個生成器，能夠以串流方式處理來自 API 的回應，即時返回生成的文字片段。

### 標準作業程序 (SOP)

1.  **確認 API 端點設定**: `predict` 函式預設 `main.py` 的 `/chat` API 運行在 `http://localhost:8000/chat`。若您的 `main.py` 服務部署在其他位址或埠號，請修改 `predict.py` 中的 `api_url` 變數以符合實際情況。

2.  **準備請求參數**: 在呼叫 `predict` 函式時，您需要提供一個字典作為參數，其中必須包含以下鍵值：
    *   `"prompt"` (str): 這是您希望模型處理的文字提示。
    *   `"uid"` (str): 這是用於識別使用者會話的唯一識別碼。每個使用者或會話應使用不同的 UID。
    
    此外，您還可以選擇提供以下參數來實現多模態輸入：
    *   `"image"` (str 或 list[str]): 這是可選的 Base64 編碼圖片資料。您可以提供單個 Base64 字串或 Base64 字串的列表。`predict` 函式會自動處理將單一圖片字串轉換為 API 預期的列表格式。

3.  **執行 `predict` 函式並處理串流回應**: 
    `predict` 函式是一個生成器。呼叫它後，您可以迭代其返回的結果來獲取模型生成的每個文字片段。這使得您可以即時地顯示或處理模型的回應。

    **範例程式碼片段**:
    ```python
    from predict import predict

    example_params = {
        "uid": "your_unique_user_id", # 請替換為您的實際 UID
        "prompt": "請告訴我關於人工智慧的最新發展。",
        # "image": "your_base64_encoded_image_string_here" # 可選：如果需要發送圖片，請替換為實際的 Base64 圖片字串
    }
        
    print("模型回應：")
    try:
        for part in predict(example_params):
            print(part, end="") # 即時印出每個文字片段
    except ValueError as e:
        print(f"參數錯誤: {e}")
    print("\n回應結束。")
    ```

4.  **錯誤處理**: 在 `predict` 函式內部，已經包含了對 API 請求失敗 (如網路問題、HTTP 錯誤狀態碼) 和其他未知錯誤的處理。如果發生錯誤，`predict` 函式將會 `yield` 一個包含錯誤訊息的字串，以便您可以在應用程式中捕獲並顯示這些錯誤。
