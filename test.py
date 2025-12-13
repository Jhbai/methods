import requests
import uuid
from typing import Optional, Generator


def chat(prompt: str, uid: str, base_url: str = "http://localhost:8000") -> Generator[str, None, None]:
    payload = {
        "uid": uid,
        "prompt": prompt,
        "images": None
    }
    
    response = requests.post(
        f"{base_url}/chat",
        json=payload,
        stream=True,
        headers={"Content-Type": "application/json"}
    )
    
    response.raise_for_status()
    
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            yield chunk


def chat_complete(prompt: str, uid: str, base_url: str = "http://localhost:8000") -> str:
    full_response = ""
    for token in chat(prompt, uid, base_url):
        print(token, end="", flush=True)
        full_response += token
    print()
    return full_response


def new_session() -> str:
    return str(uuid.uuid4())


def clear_session(uid: str, base_url: str = "http://localhost:8000") -> dict:
    response = requests.delete(f"{base_url}/session/{uid}")
    return response.json()


def health_check(base_url: str = "http://localhost:8000") -> dict:
    response = requests.get(f"{base_url}/health")
    return response.json()


if __name__ == "__main__":
    uid = new_session()
    print(f"Session: {uid}\n")
    
    print("=" * 50)
    print("Test 1: Basic conversation")
    print("=" * 50)
    chat_complete("Hello! What can you do?", uid)
    
    print("\n" + "=" * 50)
    print("Test 2: List files")
    print("=" * 50)
    chat_complete("List all Python files in the current directory", uid)
    
    print("\n" + "=" * 50)
    print("Test 3: Execute command")
    print("=" * 50)
    chat_complete("Run 'echo Hello World' command", uid)
    
    print("\n" + "=" * 50)
    print("Test 4: Create and read file")
    print("=" * 50)
    chat_complete("Create a file called test_output.txt with the content 'This is a test file created by AI'", uid)
    
    print("\n" + "=" * 50)
    print("Test 5: Read the created file")
    print("=" * 50)
    chat_complete("Read the file test_output.txt", uid)
    
    print("\n" + "=" * 50)
    print("Test 6: Multi-turn memory test")
    print("=" * 50)
    chat_complete("What was the content of the file I asked you to create earlier?", uid)
    
    print("\n" + "=" * 50)
    print("Health check:")
    print("=" * 50)
    print(health_check())
    
    print("\n" + "=" * 50)
    print("Cleanup:")
    print("=" * 50)
    print(clear_session(uid))