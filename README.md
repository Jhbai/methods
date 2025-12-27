--- 
# Description

It's a demo test in windows(for desktop/laptop testing in most of users' enviroment, so instead of linux, there are some more problems to fix). 

First of all, pip install the "compiled version" of llama_cpp_python for windows(since assembly/machine code is within the same platform)

Second, install the cuda toolkits and move them to the venv where you run your code

Third, modify the code in the llama_cpp_python python code

# Example of SOP

(1) Run "pip install llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cu123 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --prefer-binary --no-cache-dir --force-reinstall"

(2) Go to "https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows" and install the toolkits

(3) Move cublas64_12.dll, cublasLt64_12.dll, cudart64_12.dll to "{path}\{venv}\Library\bin"

(4) Modify the code in "{path}\{venv}\Lib\site-packages\llama_cpp", where the code is in this repo (just copy paste)

# Reason for doing this

Because when  trying to update the compiled llama_cpp_python package, it appears the latest version available is only 0.3.4. However, there are latent bugs within the coroutine and CUDA library implementations.

# Service Command

You can start your 127.0.0.1:8000 service by 
```bash
python -m llama_cpp.server --model "D:\LLM\coder\qwen2.5-coder-7b-instruct-q4_0.gguf" --n_gpu_layers -1 --host 0.0.0.0 --port 8000 --chat_format qwen-agent --n_ctx 10000
```

# LangChain

The LangChain Code you can use for testing:

```python
import os
import subprocess
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, BaseMessage

os.environ["OPENAI_API_KEY"] = "sk-no-key-required"
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"

llm = ChatOpenAI(
    model="qwen-agent", 
    temperature=0.1
)

@tool
def execute_shell_command(command: str):
    print(f"\n[Tool: Shell] 執行指令: {command}")
    try:
        # 為了安全，這裡僅演示，真實環境中請注意安全
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        output = result.stdout if result.stdout else result.stderr
        return f"Command Output:\n{output}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool
def read_file(file_path: str):
    print(f"\n[Tool: File] 讀取檔案: {file_path}")
    if not os.path.exists(file_path):
        return "Error: File not found."
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_to_file(file_path: str, content: str):
    print(f"\n[Tool: File] 寫入檔案: {file_path}")
    try:
        if not content or len(content.strip()) == 0:
            print("   -> [警告] AI 試圖寫入空檔案，拒絕執行。")
            return "Error: content argument cannot be empty! You must provide the actual code/text you want to write in the 'content' argument. Do not just create an empty file."
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool
def get_current_stock_price(symbol: str):
    print(f"\n[Tool: Stock] 查詢股價: {symbol}")
    if "TSLA" in symbol.upper():
        return "TSLA: 215.5 USD"
    elif "2330" in symbol:
        return "2330: 980 TWD"
    else:
        return f"{symbol}: 100 USD (Mock Data)"

@tool
def task_completed(result: str):
    return "TASK_COMPLETED_SIGNAL"

tools = [execute_shell_command, read_file, write_to_file, get_current_stock_price, task_completed]
llm_with_tools = llm.bind_tools(tools)

CLINE_SYSTEM_PROMPT = """你是一個頂尖的 AI 軟體工程師，擁有執行 Shell 指令、讀寫檔案以及使用各種工具的能力。
你的名字是 ClineClone。你的目標是**實際執行**使用者的需求，而不僅僅是給出建議。
你的工作不是「教」使用者怎麼寫程式，而是「直接」幫使用者寫好檔案並執行。

### 核心原則 (CRITICAL RULES):
1. **禁止只說不做 (No Talk, All Action)**:
   - **絕對禁止**在對話中直接貼出你「打算」寫的程式碼。
   - 如果你要寫程式碼，**必須**使用 `write_to_file` 工具將其寫入硬碟。
   - 只有在檔案寫入成功後，你才能告訴使用者你完成了什麼。

2. **驗證與執行 (Verify & Execute)**:
   - 寫完程式碼後，不要假設它是對的。使用 `execute_shell_command` 來執行它 (例如 `python main.py`)。
   - 如果執行報錯，請閱讀錯誤訊息，然後再次使用 `write_to_file` 修正程式碼。

3. **工具使用規範**:
   - 想要列出目錄? 用 `execute_shell_command` 執行 `ls -F`。
   - 想要寫 Python 檔? 用 `write_to_file`。
   - 想要安裝套件? 用 `execute_shell_command` 執行 `pip install ...`。
   - 任務完成了? 用 `task_completed`。

4. **一次做一件事**:
   - 不要試圖解釋代碼，直接寫。
   - 寫完後，務必使用 `execute_shell_command` 進行測試 (例如 `python main.py`)。

5. **遇到錯誤要修正**:
   - 如果執行失敗，閱讀錯誤訊息，然後再次呼叫 `write_to_file` 修正代碼。

6. **禁止在對話框輸出程式碼**:
   - 所有的程式碼都必須放在 `write_to_file` 的參數裡。
   - 不要在回應中用 markdown (```python ...```) 貼出代碼，那樣無法執行。直接寫進檔案！  

## 你的運作模式
1. **分析任務**：仔細理解使用者的需求。
2. **逐步執行**：不要試圖一次做完所有事。先使用工具獲取資訊（例如列出檔案、讀取代碼），根據回傳結果再決定下一步。
3. **工具使用**：
   - 如果需要獲取數據，請使用 `get_current_stock_price`。
   - 如果需要操作系統，請使用 `execute_shell_command` (例如 `ls -F`, `python script.py`)。
   - 如果需要寫代碼，請使用 `write_to_file`。
4. **結束任務**：當你確認任務已經完成，**必須** 呼叫 `task_completed` 工具來提交最終結果，否則我會認為你還在工作中。

## 回應格式
在呼叫工具之前，請先簡短描述你的思考過程 (Thought)，告訴使用者你為什麼要這樣做。
"""

def run_agent_loop(user_input: str):
    print(f"User: {user_input}")
    messages = [
        SystemMessage(content=CLINE_SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]

    iteration = 0
    max_iterations = 10  # 防止無限迴圈的安全機制

    print("\n=== Agent Loop Started ===")

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Step {iteration} ---")
        
        try:
            ai_msg = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f"LLM Error: {e}")
            break

        if ai_msg.content:
            print(f"AI Thought: {ai_msg.content}")
        messages.append(ai_msg)

        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                print(f"tool_name: {tool_name},\ntool_args: {tool_args},\ntool_id: {tool_id}")

                if tool_name == "task_completed":
                    print(f"\n[SUCCESS] 任務完成: {tool_args.get('result')}")
                    return
                selected_tool = {t.name: t for t in tools}.get(tool_name)
                
                if selected_tool:
                    tool_output = selected_tool.invoke(tool_args)
                    messages.append(ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_id
                    ))
                else:
                    print(f"Error: Tool {tool_name} not found.")
        else:
            break
            
    if iteration >= max_iterations:
        print("\n[Warning] 達到最大迭代次數，強制停止。")

if __name__ == "__main__":
    query = "幫我創建一個main.py，並在main.py裡面編輯寫一個python service，然後可以輸入command做linux指令的執行。這個service的框架是要用FastAPI寫的。最終在這個main.py內，要有一個uvicorn的server啟動程式碼。"
    run_agent_loop(query)
```
