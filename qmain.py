from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Any # 加入 Optional, Any
import uvicorn
import uuid
import time # 導入 time 模組
from model import Qwen3Object

# 初始化 Qwen3Object
qwen_model = Qwen3Object()

app = FastAPI()

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-8b",
                "object": "model",
                "created": 1677649200, 
                "owned_by": "self",
            }
        ],
    }

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-8b" # 修改模型預設值為 qwen3-8b
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

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    user_id = request.user if request.user else str(uuid.uuid4())
    system_prompt = ""
    user_message = ""

    # 解析訊息
    for message in request.messages:
        if message.role == "system":
            system_prompt = message.content
        elif message.role == "user":
            if isinstance(message.content, str):
                user_message = message.content
            elif isinstance(message.content, list):
                # 處理多模態內容，提取文本部分
                text_parts = []
                for item in message.content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                user_message = " ".join(text_parts)
            # 其他類型的 content 可能需要進一步處理或忽略

    # 如果需要處理 stop token，可以在這裡傳入 model，目前先保持原樣
    if system_prompt:
        qwen_model.initialize(system_prompt, user_id)

    if request.stream:
        async def generate_response():
            # 這裡不需要 full_response_content 累積，因為串流是即時發送
            for chunk in qwen_model.predict(user_message, user_id):
                response_data = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()), # 動態時間戳
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                # JSON 序列化時要注意格式
                import json
                yield f"data: {json.dumps(response_data)}\n\n"
            
            # 發送結束訊號
            final_response_data = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()), # 動態時間戳
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_response_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_response(), media_type="text/event-stream")
    else:
        # 非串流模式
        full_response_content = ""
        for chunk in qwen_model.predict(user_message, user_id):
            full_response_content += chunk

        response_data = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()), # 動態時間戳
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_response_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0, 
                "completion_tokens": 0, 
                "total_tokens": 0 
            }
        }
        return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
