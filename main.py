import torch
import asyncio
import subprocess
import os
import re
import json
import base64
import gc
import time
import fnmatch
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from io import BytesIO
from PIL import Image
from threading import Lock
import traceback

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from transformers import (
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    AutoProcessor,
    DynamicCache,
)

MODEL_ID = "D:/LLM/gemma/gemma3_4b"
BATCH_CHUNK_SIZE = 1024
MAX_BATCH_SIZE = 8
MAX_NEW_TOKENS = 4096
SESSION_TIMEOUT = 3600
VRAM_THRESHOLD = 0.85

model = None
processor = None
sessions: Dict[str, "Session"] = {}
request_queue: asyncio.Queue = None
processing_lock = asyncio.Lock()
batch_processor = None


@dataclass
class Session:
    messages: List[Dict] = field(default_factory=list)
    last_access: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ChatRequest(BaseModel):
    uid: str
    prompt: str
    images: Optional[List[str]] = None


@dataclass
class PendingRequest:
    uid: str
    messages: List[Dict]
    images: Optional[List[Image.Image]]
    response_queue: asyncio.Queue
    session: Session


SYSTEM_PROMPT = """You are an AI assistant with the ability to execute tools. When you need to use a tool, output it in the following XML format:

<tool_call>
<name>tool_name</name>
<param name="param_name">param_value</param>
</tool_call>

Available tools:
1. execute_command - Execute a shell command
   Parameters: command (string)
   
2. read_file - Read the contents of a file
   Parameters: path (string)
   
3. write_file - Write content to a file
   Parameters: path (string), content (string)
   
4. edit_file - Edit a file by replacing content between line numbers
   Parameters: path (string), start_line (string), end_line (string), new_content (string)
   
5. list_files - List files in a directory
   Parameters: directory (string, default "."), pattern (string, default "*")
   
6. search_files - Search for a regex pattern in files
   Parameters: directory (string), regex (string)

7. create_directory - Create a new directory
   Parameters: path (string)

8. delete_file - Delete a file
   Parameters: path (string)

After receiving tool results, analyze them and continue. When finished, respond normally without tool calls."""


def execute_command(command: str, timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        output = result.stdout + result.stderr
        return output.strip() if output.strip() else "Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out."
    except Exception as e:
        return f"Error: {str(e)}"


def read_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.split('\n')
        numbered_lines = [f"{i+1}|{line}" for i, line in enumerate(lines)]
        return '\n'.join(numbered_lines)
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(path: str, content: str) -> str:
    try:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def edit_file(path: str, start_line: str, end_line: str, new_content: str) -> str:
    try:
        start = int(start_line)
        end = int(end_line)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        new_lines = new_content.split('\n')
        new_lines = [line + '\n' if not line.endswith('\n') else line for line in new_lines]
        if new_lines and new_lines[-1].endswith('\n') and (end >= len(lines) or not lines[end-1].endswith('\n')):
            new_lines[-1] = new_lines[-1].rstrip('\n')
        lines[start-1:end] = new_lines
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return f"Successfully edited {path} from line {start} to {end}"
    except Exception as e:
        return f"Error editing file: {str(e)}"


def list_files(directory: str = ".", pattern: str = "*") -> str:
    try:
        files = []
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for filename in filenames:
                if fnmatch.fnmatch(filename, pattern):
                    rel_path = os.path.relpath(os.path.join(root, filename), directory)
                    files.append(rel_path)
        return "\n".join(sorted(files)[:200]) if files else "No files found."
    except Exception as e:
        return f"Error listing files: {str(e)}"


def search_files(directory: str, regex: str) -> str:
    try:
        matches = []
        pattern = re.compile(regex)
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for filename in filenames:
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            if pattern.search(line):
                                matches.append(f"{filepath}:{i}: {line.strip()[:100]}")
                                if len(matches) >= 100:
                                    return "\n".join(matches) + "\n... (truncated)"
                except:
                    pass
        return "\n".join(matches) if matches else "No matches found."
    except Exception as e:
        return f"Error searching: {str(e)}"


def create_directory(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        return f"Successfully created directory: {path}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"


def delete_file(path: str) -> str:
    try:
        if os.path.isfile(path):
            os.remove(path)
            return f"Successfully deleted: {path}"
        elif os.path.isdir(path):
            return "Error: Path is a directory. Use execute_command with 'rm -rf' for directories."
        else:
            return f"Error: Path does not exist: {path}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"


def parse_tool_calls(text: str) -> List[Dict]:
    tool_calls = []
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        name_match = re.search(r'<name>(.*?)</name>', match, re.DOTALL)
        if name_match:
            tool_name = name_match.group(1).strip()
            params = {}
            param_pattern = r'<param name="([^"]+)">(.*?)</param>'
            param_matches = re.findall(param_pattern, match, re.DOTALL)
            for param_name, param_value in param_matches:
                params[param_name] = param_value
            tool_calls.append({"name": tool_name, "params": params})
    
    return tool_calls


def execute_tool(tool_call: Dict) -> str:
    name = tool_call["name"]
    params = tool_call["params"]
    
    if name == "execute_command":
        return execute_command(params.get("command", ""))
    elif name == "read_file":
        return read_file(params.get("path", ""))
    elif name == "write_file":
        return write_file(params.get("path", ""), params.get("content", ""))
    elif name == "edit_file":
        return edit_file(
            params.get("path", ""),
            params.get("start_line", "1"),
            params.get("end_line", "1"),
            params.get("new_content", "")
        )
    elif name == "list_files":
        return list_files(params.get("directory", "."), params.get("pattern", "*"))
    elif name == "search_files":
        return search_files(params.get("directory", ""), params.get("regex", ""))
    elif name == "create_directory":
        return create_directory(params.get("path", ""))
    elif name == "delete_file":
        return delete_file(params.get("path", ""))
    else:
        return f"Unknown tool: {name}"


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_vram_usage() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def get_vram_ratio() -> float:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return allocated / total
    return 0.0


def check_memory_pressure() -> bool:
    return get_vram_ratio() > VRAM_THRESHOLD


def cleanup_old_sessions():
    current_time = time.time()
    expired = [uid for uid, session in sessions.items() 
               if current_time - session.last_access > SESSION_TIMEOUT]
    for uid in expired:
        del sessions[uid]
    if expired:
        clear_vram()


def get_or_create_session(uid: str) -> Session:
    if uid not in sessions:
        sessions[uid] = Session()
    sessions[uid].last_access = time.time()
    return sessions[uid]


def load_model():
    global model, processor
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()


def decode_image(image_data: str) -> Image.Image:
    if image_data.startswith("data:"):
        image_data = image_data.split(",")[1]
    img_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


class ContinuousBatchProcessor:
    def __init__(self):
        self.pending: List[PendingRequest] = []
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.running = True
        
    async def add_request(self, request: PendingRequest):
        async with self.lock:
            self.pending.append(request)
            self.event.set()
    
    async def get_batch(self) -> List[PendingRequest]:
        try:
            await asyncio.wait_for(self.event.wait(), timeout=0.02)
        except asyncio.TimeoutError:
            pass
        
        async with self.lock:
            if not self.pending:
                self.event.clear()
                return []
            batch = self.pending[:MAX_BATCH_SIZE]
            self.pending = self.pending[MAX_BATCH_SIZE:]
            if not self.pending:
                self.event.clear()
            return batch
    
    async def run(self):
        while self.running:
            batch = await self.get_batch()
            if not batch:
                await asyncio.sleep(0.005)
                continue
            
            if check_memory_pressure():
                cleanup_old_sessions()
                clear_vram()
            
            tasks = [self.process_request(req) for req in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_request(self, request: PendingRequest):
        try:
            await self.generate_response(request)
        except Exception as e:
            await request.response_queue.put(("error", str(e)))
            await request.response_queue.put(None)
    
    async def generate_response(self, request: PendingRequest):
        messages = []
        
        # 將所有訊息的 content 轉換為 list 格式（多模態模型需要）
        for msg in request.messages:
            role = msg["role"]
            content = msg["content"]
            
            # 確保 content 是正確的格式
            if isinstance(content, str):
                # 字串轉換為 list of dict 格式
                formatted_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # 已經是 list 格式，保持不變
                formatted_content = content
            else:
                formatted_content = [{"type": "text", "text": str(content)}]
            
            messages.append({"role": role, "content": formatted_content})
        
        # 處理圖片：加入到最後一個 user 訊息中
        if request.images and messages and messages[-1]["role"] == "user":
            image_parts = [{"type": "image", "image": img} for img in request.images]
            # 圖片放在文字前面
            messages[-1]["content"] = image_parts + messages[-1]["content"]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(model.device, dtype=torch.bfloat16)
        
        seq_len = input_ids.shape[1]
        past_key_values = DynamicCache()
        
        position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)
        
        for start in range(0, seq_len, BATCH_CHUNK_SIZE):
            end = min(start + BATCH_CHUNK_SIZE, seq_len)
            chunk_ids = input_ids[:, start:end]
            chunk_position_ids = position_ids[:, start:end]
            
            with torch.no_grad():
                if start == 0 and pixel_values is not None:
                    outputs = model(
                        input_ids=chunk_ids,
                        attention_mask=attention_mask[:, :end],
                        position_ids=chunk_position_ids,
                        pixel_values=pixel_values,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    outputs = model(
                        input_ids=chunk_ids,
                        attention_mask=attention_mask[:, :end],
                        position_ids=chunk_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values
        
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_pos = seq_len
        
        eos_token_ids = [processor.tokenizer.eos_token_id]
        if hasattr(processor.tokenizer, "convert_tokens_to_ids"):
            for token in ["<end_of_turn>", "<eos>", "</s>"]:
                tid = processor.tokenizer.convert_tokens_to_ids(token)
                if tid is not None and tid != processor.tokenizer.unk_token_id:
                    eos_token_ids.append(tid)
        
        generated_text = ""
        
        for step in range(MAX_NEW_TOKENS):
            token_id = next_token_id.item()
            
            if token_id in eos_token_ids:
                break
            
            token_text = processor.tokenizer.decode([token_id], skip_special_tokens=True)
            generated_text += token_text
            await request.response_queue.put(("token", token_text))
            
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=model.device, dtype=attention_mask.dtype)
            ], dim=1)
            
            current_pos += 1
            new_position_ids = torch.tensor([[current_pos - 1]], device=model.device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=next_token_id,
                    attention_mask=attention_mask,
                    position_ids=new_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
            
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            if step % 50 == 0:
                await asyncio.sleep(0)
        
        await request.response_queue.put(("done", generated_text))
        await request.response_queue.put(None)

async def stream_with_tools(uid: str, prompt: str, images: Optional[List[str]] = None):
    session = get_or_create_session(uid)
    
    async with session.lock:
        if not session.messages:
            session.messages.append({"role": "user", "content": SYSTEM_PROMPT})
            session.messages.append({"role": "assistant", "content": "I understand. I have access to these tools and will use them when needed to help you. How can I assist you?"})
        
        session.messages.append({"role": "user", "content": prompt})
        
        parsed_images = None
        if images:
            parsed_images = [decode_image(img) for img in images]
        
        max_tool_iterations = 15
        
        for iteration in range(max_tool_iterations):
            response_queue = asyncio.Queue()
            
            request = PendingRequest(
                uid=uid,
                messages=session.messages.copy(),
                images=parsed_images if iteration == 0 else None,
                response_queue=response_queue,
                session=session,
            )
            
            await batch_processor.add_request(request)
            
            full_response = ""
            while True:
                item = await response_queue.get()
                if item is None:
                    break
                msg_type, content = item
                if msg_type == "error":
                    yield f"\n[Error: {content}]\n"
                    return
                elif msg_type == "token":
                    full_response += content
                    yield content
                elif msg_type == "done":
                    full_response = content
            
            session.messages.append({"role": "assistant", "content": full_response})
            
            tool_calls = parse_tool_calls(full_response)
            
            if not tool_calls:
                break
            
            tool_results = []
            for tc in tool_calls:
                result = execute_tool(tc)
                tool_results.append(
                    f"<tool_result>\n<name>{tc['name']}</name>\n<output>\n{result}\n</output>\n</tool_result>"
                )
            
            tool_result_message = "\n".join(tool_results)
            session.messages.append({"role": "user", "content": tool_result_message})
            
            yield "\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global batch_processor
    load_model()
    batch_processor = ContinuousBatchProcessor()
    batch_task = asyncio.create_task(batch_processor.run())
    yield
    batch_processor.running = False
    batch_task.cancel()
    try:
        await batch_task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(request: ChatRequest):
    async def generate():
        async for token in stream_with_tools(request.uid, request.prompt, request.images):
            yield token.encode("utf-8")
    
    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.delete("/session/{uid}")
async def delete_session(uid: str):
    if uid in sessions:
        del sessions[uid]
        clear_vram()
        return {"status": "deleted"}
    return {"status": "not_found"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vram_gb": round(get_vram_usage(), 2),
        "vram_ratio": round(get_vram_ratio(), 2),
        "sessions": len(sessions),
    }


@app.post("/clear_memory")
async def clear_memory():
    cleanup_old_sessions()
    clear_vram()
    return {
        "status": "cleared",
        "vram_gb": round(get_vram_usage(), 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)