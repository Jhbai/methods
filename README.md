<img width="1500" height="950" alt="異步 + 串流" src="https://github.com/user-attachments/assets/7bde86e9-e553-44d1-9b09-b038fbf4d5ab" />

FastAPI 異步串流與 StreamResponse 實作範例
本專案旨在透過一個簡單的範例，深度解析如何在 Python FastAPI 框架中，利用 asyncio 與 StreamResponse 實現異步（Asynchronous）的串流資料傳輸。
這個技術對於需要長時間運行的背景任務特別有用，例如：
- (1) 大型語言模型（LLM）的即時推理（Inference）。
- (2) 分塊（Chunking）處理大型資料集。
- (3) 與速度較慢的外部 API 進行通訊。
- (4) 透過串流響應，客戶端無需等待整個任務完成，而是可以即時接收到部分生成的資料，大幅提升了應用的互動性與使用者體驗。


下方展示了本專案中兩種不同 API 端點的處理流程：一個是標準的同步響應，另一個則是異步的串流響應:
[核心概念解析]
- Event Loop (事件循環): asyncio 的核心，負責調度所有異步任務。當一個任務（例如 await heavy_io()）進入等待 I/O 操作的狀態時，事件循環會暫停該任務，並切換到另一個就緒的任務，從而實現非阻塞式（Non-blocking）的並行處理。
- StreamResponse (串流響應): FastAPI 允許響應一個「異步生成器」（Asynchronous Generator）。StreamResponse 會迭代這個生成器，每當生成器 yield 一個資料塊時，就立即將其發送給客戶端，直到生成器執行完畢。
- async def with yield (異步生成器): 這是實現串流內容的關鍵。函數使用 async def 定義，並在其中使用 yield 來回傳資料。每次 yield 之後，函數的執行狀態會被保存，並將控制權交還給事件循環。當客戶端準備好接收下一個資料塊時，事件循環會從上次暫停的地方繼續執行該函數。
- run_in_executor: 在異步環境中，絕對不能直接運行會阻塞的同步程式碼（例如 CPU 密集型計算或傳統的阻塞 I/O）。loop.run_in_executor 方法可以將這類阻塞任務提交到一個獨立的線程池（Thread Pool）中執行，從而避免主事件循環被阻塞。在圖中，heavy_io() 就是一個模擬的阻塞任務。

[API 端點說明]
- (1) /infer/home - 標準同步端點
  - 行為: 這是一個標準的 FastAPI 異步端點。當請求到達時，它會立即處理並一次性返回完整的響應 "OK"。
  - 流程: 請求 -> 立即處理 -> 返回完整響應

- (2) /infer/predict - 異步串流端點
  - 行為: 這個端點會返回一個 StreamResponse。當客戶端連接後，伺服器會逐步執行 fake_stream 生成器，每 yield 一次資料，就將該資料塊傳送給客戶端。
  - 流程: 客戶端發起請求。FastAPI 開始迭代 fake_stream 生成器。在 for 迴圈中，heavy_io() 被提交到線程池執行，主事件循環不會被阻塞，每 heavy_io() 完成一次後，進行 yield 並被發送給客戶端。重複此過程，直到生成器執行完畢。

[如何運行]
安裝依賴:
pip install "fastapi[all]"

[啟動伺服器]
python -B -m main
伺服器將在 http://127.0.0.1:8000 上運行。

[測試輸出]
您可以使用 jupyter 工具編輯 requests.get(url) 來觀察串流的效果，或者透過 threading.Thread(funcs, **args) 觀察異步效果。
