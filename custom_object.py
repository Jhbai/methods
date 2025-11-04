import requests
import httpx # 推薦用於非同步
from typing import Any, List, Mapping, Optional, Dict

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

class CustomFastAPI_LLM(LLM):
    """
    一個自定義的 LangChain LLM 類，它透過 HTTP 請求呼叫我們自架的 FastAPI 服務。
    """
    
    # 您的 FastAPI 服務的 URL
    api_url: str = "http://127.0.0.1:8000/v1/completion/chat"
    
    @property
    def _llm_type(self) -> str:
        """回傳 LLM 的類型名稱。"""
        return "my_llm_service"

    def _call(self, prompt: str, uid: str) -> str:
        """
        實現同步的 _call 方法。
        """
        
        # 組合請求的 JSON payload
        payload = {"prompt": prompt, "uid": uid}

        # 發送 POST 請求
        with requests.post(self.api_url, json=payload, stream=True) as r:
            try:
                r.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error occurs when calling myLLM service: {e}") from e
            for chunk in r.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")

    async def _acall(self, prompt: str, uid: str) -> str:
        """
        實現非同步的 _acall 方法。
        """
        # 1. 組合 payload
        payload = {"prompt": prompt, "uid": uid}
        
        # 2. 使用 httpx 發送非同步請求
        with httpx.stream("POST", self.api_url, json=payload) as r:
            r.raise_for_status()
            for chunk in r.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")
                

    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        提供在初始化時設定的預設參數。
        """
        return {"temperature": 1, "max_tokens": 30768}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        幫助 LangChain 識別和快取此 LLM 實例的參數。
        """
        return {"api_url": self.api_url, **self._default_params}
