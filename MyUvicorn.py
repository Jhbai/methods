import asyncio
from MyApi import MyAPI

async def MyServer(app, host='127.0.0.1', port=8000):
    async def handle_connection(reader, writer):
        # ----- 處理接收 ----- #
        data = await reader.read(4096)
        print(data)
        try:
            content_length = int(data.split(b"\r\n")[6].split(b" ")[-1].decode())
            body_part = data.split(b"\r\n\r\n")[-1]
            if len(body_part) < content_length: # 處理 body 分段接收
                data += await reader.read(content_length - len(body_part))
        except:
            pass

        # ----- 解析請求 ----- #
        headers_part = data.split(b"\r\n")[0]
        body_part = data.split(b"\r\n\r\n")[-1]

        """
        規格為 GET /Method/1.1\r\n ... ...
        """
        method, path, _ = headers_part.decode().split(" ")
        if method == "POST":
            scope = {
                "method": method,
                "path": path,
                "body": body_part.decode(),
            }
        else:
            scope = {
                "method": method,
                "path": path,
            }

        async def send(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                status_line = f"HTTP/1.1 {status_code} OK\r\n"
                writer.write(status_line.encode())

                for k, v in message["headers"]:
                    header_line = f"{k.decode()}: {v.decode()}\r\n"
                    writer.write(header_line.encode())
                writer.write(b"\r\n")
                
            if message["type"] == "http.response.body":
                writer.write(message["body"])
                await writer.drain()
                writer.close()
        await app(scope, send)
    server = await asyncio.start_server(handle_connection, host, port)
    print("Start MyServer... ...")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(MyServer(MyAPI))
    except KeyboardInterrupt:
        print("Server stopped.")
