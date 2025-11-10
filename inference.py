import time
import queue
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

q = queue.Queue()
router = APIRouter(prefix="/infer", tags=["Inference"])

# 模擬Heavy IO
IO_data = ["你好", "這是", "測試", "Stream", "輸出", "並且", "要求", "異步", "避免", "阻塞"]
def heavy_io(i):
    time.sleep(1)
    return IO_data[i]

# 異部串流輸出
@router.get("/predict")
async def stream_response():
    # 異步 + 串流
    loop = asyncio.get_running_loop()
    async def fake_stream():
        for i in range(len(IO_data)):
            result = await loop.run_in_executor(None, lambda: heavy_io(i))
            yield f"{result}\n"
    return StreamingResponse(fake_stream(), media_type="text/plain")

@router.get("/home")
async def stream_response():
    return "OK"
