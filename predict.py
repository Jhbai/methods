import requests
import base64

def predict(params):
    """
    透過 HTTP 請求將資料發送到 main.py 的 /chat API，並將串流回應轉換為生成器。

    Args:
        params (dict): 包含 'prompt' (str), 'uid' (str),
                       和可選的 'image' (str, base64 編碼字串 或 base64 字串列表) 的字典。

    Yields:
        str: 從 API 接收到的每個文字片段。
    """
    api_url = "http://localhost:8000/chat"  # 假設 main.py 運行在這個地址

    prompt = params.get("prompt")
    uid = params.get("uid")
    image_data = params.get("image")

    if not prompt or not uid:
        raise ValueError("Params must contain 'prompt' and 'uid'.")

    request_payload = {
        "uid": uid,
        "prompt": prompt,
        "images": None
    }

    if image_data:
        # main.py 期望圖片是 base64 編碼的字串列表，這裡假設輸入的 'image' 已經是 base64 字串
        if isinstance(image_data, str):
            request_payload["images"] = [image_data]
        elif isinstance(image_data, list):
            request_payload["images"] = image_data
        else:
            raise ValueError("Unsupported image format. 'image' should be a base64 string or a list of base64 strings.")

    try:
        with requests.post(api_url, json=request_payload, stream=True, timeout=None) as response:
            response.raise_for_status()  # 檢查 HTTP 錯誤
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk.decode("utf-8")
    except requests.exceptions.RequestException as e:
        print(f"API 請求失敗: {e}")
        yield f"[錯誤: API 請求失敗 - {e}]"
    except Exception as e:
        print(f"發生未知錯誤: {e}")
        yield f"[錯誤: 發生未知錯誤 - {e}]"


# 範例使用 (如果需要在 predict.py 內執行測試)
def main():
    example_params = {
        "uid": "test_user_456",
        "prompt": "寫一個關於未來科技的短故事。",
        # "image": "base64_encoded_image_string_here" # 可選：如果需要發送圖片
    }
    
    print("開始串流回應...")
    try:
        for part in predict(example_params):
            print(part, end="")
    except ValueError as e:
        print(f"參數錯誤: {e}")
    print("\n串流結束。")

if __name__ == "__main__":
    main()
