# 自製 ASGI 伺服器與應用程式解析

這份文件將深入解析一個簡單的 ASGI (Asynchronous Server Gateway Interface) 伺服器 (`MyUvicorn.py`) 及其所服務的 ASGI 應用程式 (`MyApi.py`) 的實作細節，從底層邏輯開始，逐步說明其運作原理與相關知識點。

## 一、底層邏輯概述：ASGI 簡介

ASGI 是一個 Python 異步 Web 伺服器與應用程式之間的標準介面，旨在為異步 Python Web 框架和伺服器提供一個通用的通信方式。它類似於 WSGI (Web Server Gateway Interface)，但專為異步操作設計。

一個 ASGI 應用程式是一個可調用對象 (Callable)，它接受三個參數：
1.  `scope`: 一個字典，包含有關特定連線的詳細資訊（例如請求路徑、方法、協議等）。
2.  `receive`: 一個異步可調用對象 (async callable)，用於從伺服器接收事件（例如請求主體資料）。
3.  `send`: 一個異步可調用對象 (async callable)，用於向伺服器傳送事件（例如響應頭、響應主體）。

在這個專案中，我們簡化了 `receive` 的概念，直接在伺服器端解析請求主體並將其放入 `scope` 中，因此 `MyAPI` 函式只接受 `scope` 和 `send`。

## 二、`MyUvicorn.py`：自製 ASGI 伺服器

`MyUvicorn.py` 檔案負責建立一個基本的異步 TCP 伺服器，模擬 ASGI 伺服器的核心功能，接收 HTTP 請求，將其轉換為 ASGI `scope` 格式，然後呼叫 ASGI 應用程式，並將應用程式產生的 ASGI 響應訊息轉換回 HTTP 響應傳送給客戶端。

### 實作知識點：

1.  **`asyncio` 異步程式設計**：
    *   `MyUvicorn.py` 大量使用了 Python 的 `asyncio` 庫來實現非阻塞的 I/O 操作，特別是網路連線的處理。
    *   `asyncio.start_server(handle_connection, host, port)` 用於啟動一個 TCP 伺服器，它會監聽指定的主機和埠，並為每個傳入的連線呼叫 `handle_connection` 協程。
    *   `async with server: await server.serve_forever()` 確保伺服器持續運行直到被中斷。

2.  **TCP 連線處理 (`handle_connection`)**：
    *   `reader, writer = await asyncio.start_server(...)` 中的 `reader` 和 `writer` 是 `asyncio.StreamReader` 和 `asyncio.StreamWriter` 物件，提供了方便的異步讀寫資料介面。
    *   `data = await reader.read(4096)`：從客戶端讀取原始 TCP 資料。這裡設定每次讀取最大 4096 位元組。

3.  **HTTP 請求解析**：
    *   伺服器會手動解析收到的原始 HTTP 請求資料。一個典型的 HTTP 請求由請求行、請求頭和可選的請求主體組成，各部分之間用 `\r\n` (CRLF) 分隔，請求頭和請求主體之間用 `\r\n\r\n` 分隔。
    *   `method, path, _ = headers_part.decode().split(" ")`：從請求行中解析方法、路徑。
    *   **處理 POST 請求的 Body (考慮分段接收)**：
        *   伺服器會嘗試從 HTTP 請求頭中尋找 `Content-Length`，以確定請求主體的完整長度。
        *   找到請求主體的開始位置 `body_start_index`。
        *   如果 `Content-Length` 大於已讀取主體部分，則繼續從 `reader` 讀取剩餘的位元組，確保獲取完整的請求主體。
        *   最後將解析出的 `method`, `path`, `body` 放入 `scope` 字典中。

4.  **`send` 函數實作**：
    *   `MyUvicorn.py` 內部定義了一個 `send` 異步函數，作為傳遞給 ASGI 應用程式的參數。這個 `send` 函數負責將 ASGI 應用程式傳送的標準 ASGI 響應訊息轉換成實際的 HTTP 響應並寫入到 TCP 連線中。
    *   當收到 `{"type": "http.response.start", "status": ..., "headers": ...}` 訊息時，它會構建 HTTP 狀態行和響應頭，然後寫入到 `writer`。
    *   當收到 `{"type": "http.response.body", "body": ...}` 訊息時，它會將響應主體寫入 `writer`，並呼叫 `await writer.drain()` 確保資料被送出，然後 `writer.close()` 關閉連線。

5.  **呼叫 ASGI 應用程式**：
    *   `await app(scope, send)`：在解析完 HTTP 請求並準備好 `scope` 和 `send` 函數後，伺服器會呼叫傳入的 ASGI 應用程式 (`MyAPI`)。這就是伺服器與應用程式之間的橋樑。

## 三、`MyApi.py`：ASGI 應用程式範例

`MyApi.py` 是一個簡單的 ASGI 應用程式，它根據請求的路徑和方法來產生不同的 HTTP 響應。

### 實作知識點：

1.  **`MyAPI(scope, send)` 函數簽名**：
    *   符合 ASGI 應用程式的標準，接受 `scope` (請求上下文) 和 `send` (響應傳送器) 參數。

2.  **請求資訊提取**：
    *   `path = scope["path"]` 和 `method = scope["method"]`：從 `scope` 字典中提取請求的路徑和 HTTP 方法，用於路由判斷。

3.  **路由與響應邏輯**：
    *   **`/home` (GET 請求)**：當收到 `/home` 的 GET 請求時，應用程式會設定狀態碼為 `200`，並回應 `b"Welcome to JHBai's Home Page!"`。
    *   **`/echo` (POST 請求)**：當收到 `/echo` 的 POST 請求時，應用程式會從 `scope["body"]` 中獲取請求主體，並將其作為回應內容，例如 `b"Receive {payload}"`，狀態碼為 `200`。

4.  **ASGI 響應訊息構建與傳送**：
    *   應用程式會構建兩個標準的 ASGI 響應訊息並透過 `send` 函數傳送：
        *   `{"type": "http.response.start", "status": status_code, "headers": [...]}`：包含 HTTP 狀態碼和響應頭。在這裡設定 `Content-Type` 為 `text/plain` 和 `Content-Length`。
        *   `{"type": "http.response.body", "body": body}`：包含實際的響應主體內容。
    *   `await send(headers)` 和 `await send(response)`：異步地將這些響應訊息傳送給伺服器，由伺服器負責轉換為原始 HTTP 響應並送回客戶端。

## 四、總結

透過 `MyUvicorn.py` 和 `MyApi.py` 這兩個檔案，我們實作了一個迷你的 ASGI 伺服器和一個簡單的 ASGI 應用程式。這個專案展示了 ASGI 的基本工作原理，包括伺服器如何解析 HTTP 請求並構建 `scope`，應用程式如何處理 `scope` 並透過 `send` 發送響應訊息，以及伺服器如何將 ASGI 響應訊息轉換回標準 HTTP 響應傳送給客戶端。這是一個理解現代異步 Python Web 開發基礎的良好範例。

```python
if __name__ == "__main__":
    try:
        asyncio.run(MyServer(MyAPI))
    except KeyboardInterrupt:
        print("Server stopped.")
```
這段程式碼是 `MyUvicorn.py` 的入口點。它使用 `asyncio.run()` 來啟動 `MyServer` 協程，並將 `MyAPI` 作為 ASGI 應用程式傳遞給它。當程式被 `KeyboardInterrupt` (例如按下 Ctrl+C) 中斷時，會印出 "Server stopped."。
