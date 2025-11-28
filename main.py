import time
import json
import uuid
import asyncio
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 引入你的模型物件
from model import Gemma3Object

# ----- 初始化 FastAPI ----- #
app = FastAPI(title="Gemma 3 OpenAI Compatible API", version="1.0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- 全域變數存放模型 ----- #
model_instance: Optional[Gemma3Object] = None

@app.on_event("startup")
async def startup_event():
    global model_instance
    print("正在載入 Gemma 3 模型，請稍候...")
    try:
        model_instance = Gemma3Object() 
        print("模型載入完成！")
    except Exception as e:
        print(f"模型載入失敗: {e}")
        raise e

# ----- Pydantic Models ----- #

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = "default_user"
    
    class Config:
        extra = "ignore"

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# ----- API Endpoints ----- #

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma-3",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "bear_engineer",
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model_instance
    if not model_instance:
        raise HTTPException(status_code=500, detail="Model not initialized")

    # 1. 解析輸入
    last_user_message = ""
    for msg in request.messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                last_user_message = msg.content
            break
    last_user_message += "\n"
    for msg in reversed(request.messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                last_user_message += msg.content
            elif isinstance(msg.content, list):
                text_parts = []
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                last_user_message += "".join(text_parts)
            break
    
    if not last_user_message:
        last_user_message = "Hello"

    uid = request.user if request.user else "cline_default_session"
    print(f"收到請求 - UID: {uid}, Prompt長度: {len(last_user_message)}")
    print(f"收到Prompt為: {last_user_message[:50]}...")

    # 2. 處理 Streaming Response
    if request.stream:
        # 注意: 這裡加上 media_type="text/event-stream; charset=utf-8" 非常關鍵
        return StreamingResponse(
            stream_generator(model_instance, last_user_message, uid),
            media_type="text/event-stream; charset=utf-8"
        )
    
    # 3. 處理 Non-Streaming Response
    else:
        full_response = ""
        generator = model_instance.chat(last_user_message, uid=uid)
        for token in generator:
            full_response += token
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=full_response),
                    finish_reason="stop"
                )
            ]
        )

# ----- Generator for SSE ----- #
def stream_generator(model, prompt, uid):
    req_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    print(f"--- 開始串流生成 (UID: {uid}) ---")
    
    try:
        token_generator = model.chat(prompt, uid=uid)
        
        token_count = 0
        for token in token_generator:
            print(token, end='', flush=True)
            token_count += 1
            # DEBUG: 每 10 個 token 印一次，確認模型有在動
            if token_count % 10 == 1:
                print(f"生成中... ({token_count} tokens): {token[:10]}...")

            chunk_data = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "gemma-3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }
                ]
            }
            # 確保 JSON 轉換正確且不使用 ASCII escape (支援中文)
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            
            # 重要：Local環境生成太快有時會導致 client buffer 溢位，稍微睡一下讓 buffer flush
            time.sleep(0.005) 
            
    except Exception as e:
        print(f"!!! 生成過程發生錯誤 !!!: {e}")
        error_chunk = {
             "id": req_id,
             "object": "chat.completion.chunk",
             "created": created_time,
             "model": "gemma-3",
             "choices": [{"index": 0, "delta": {"content": f"\n[Error: {str(e)}]"}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
    
    print(f"--- 生成結束，總共 {token_count} tokens ---")

    end_chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": "gemma-3",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    # log_level="info" 可以讓你看得更清楚
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
