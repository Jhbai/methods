import requests
import httpx # 推薦用於非同步
from typing import Any, List, Mapping, Optional, Dict

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

class CustomFastAPI_LLM(LLM):
    """
    一個自定義的 LangChain LLM 類，
    它透過 HTTP 請求呼叫我們自架的 FastAPI 服務。
    """
    
    # 您的 FastAPI 服務的 URL
    api_url: str = "http://127.0.0.1:8000/generate"
    
    # 可以在初始化時傳入其他參數
    temperature: float = 0.7
    max_tokens: int = 256
    
    @property
    def _llm_type(self) -> str:
        """回傳 LLM 的類型名稱。"""
        return "custom_fastapi_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        實現同步的 _call 方法。
        """
        
        # 1. 組合請求的 JSON payload
        # 合併預設參數和運行時傳入的 kwargs
        merged_kwargs = {**self._default_params, **kwargs}
        
        payload = {
            "prompt": prompt,
            "stop": stop,
            **merged_kwargs
        }

        # 2. 發送 POST 請求
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status() # 如果狀態碼不是 2xx，則拋出異常
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling FastAPI service: {e}") from e

        # 3. 解析回應
        result = response.json()
        
        # 這裡我們只回傳 'text'，符合 LLM 類的標準
        # 您也可以在 run_manager 中處理 'tokens_used' 等 metadata
        
        return result['text']

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        (推薦實作) 實現非同步的 _acall 方法。
        """
        
        # 1. 組合 payload
        merged_kwargs = {**self._default_params, **kwargs}
        payload = {
            "prompt": prompt,
            "stop": stop,
            **merged_kwargs
        }
        
        # 2. 使用 httpx 發送非同步請求
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.api_url, json=payload, timeout=60.0)
                response.raise_for_status()
            except httpx.RequestError as e:
                raise RuntimeError(f"Async error calling FastAPI service: {e}") from e

        # 3. 解析回應
        result = response.json()
        return result['text']

    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        提供在初始化時設定的預設參數。
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        幫助 LangChain 識別和快取此 LLM 實例的參數。
        """
        return {"api_url": self.api_url, **self._default_params}
