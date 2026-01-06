import asyncio

async def MyAPI(scope, send):
    # ----- 初始化 ----- #
    path = scope["path"]
    method = scope["method"]

    # ----- 主要邏輯 ----- #
    if path == "/home" and method == "GET":
        body = b"Welcome to JHBai's Home Page!"
        status_code = 200
    
    elif path == "/echo" and method == "POST":
        payload = scope["body"]
        body = f"Receive {payload}".encode()
        status_code = 200

    
    # ----- 回傳處理 ----- #
    headers = {
        "type": "http.response.start",
        "status": status_code,
        "headers": [
            (b"Content-Type", b"text/plain"),
            (b"Content-Length", str(len(body)).encode("utf-8")),
        ]
    }
    
    response = {
        "type": "http.response.body",
        "body": body,
    }

    await send(headers)
    await send(response)
