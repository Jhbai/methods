from __future__ import annotations

import os
import json
import typing
import contextlib

# 【修正 1】正確引入 Semaphore
from anyio import Lock, Semaphore
from functools import partial
from typing import Iterator, List, Optional, Union, Dict, Any

import llama_cpp

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi import Depends, FastAPI, APIRouter, Request, HTTPException, status, Body
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from sse_starlette.sse import EventSourceResponse
from starlette_context.plugins import RequestIdPlugin  # type: ignore
from starlette_context.middleware import RawContextMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from llama_cpp.server.model import (
    LlamaProxy,
)
from llama_cpp.server.settings import (
    ConfigFileSettings,
    Settings,
    ModelSettings,
    ServerSettings,
)
from llama_cpp.server.types import (
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    # CreateChatCompletionRequest, # 註解掉，改用 Dict 接收
    ModelList,
    TokenizeInputRequest,
    TokenizeInputResponse,
    TokenizeInputCountResponse,
    DetokenizeInputRequest,
    DetokenizeInputResponse,
)
from llama_cpp.server.errors import RouteErrorHandler


router = APIRouter(route_class=RouteErrorHandler)

_server_settings: Optional[ServerSettings] = None


def set_server_settings(server_settings: ServerSettings):
    global _server_settings
    _server_settings = server_settings


def get_server_settings():
    yield _server_settings


_llama_proxy: Optional[LlamaProxy] = None

# 【修正 2】使用 Semaphore(1) 取代 Lock
# 這是避免 "The current task is not holding this lock" 的關鍵
llama_outer_lock = Semaphore(1)
llama_inner_lock = Semaphore(1)


def set_llama_proxy(model_settings: List[ModelSettings]):
    global _llama_proxy
    _llama_proxy = LlamaProxy(models=model_settings)


async def get_llama_proxy():
    # 這個函式負責獲取模型代理並上鎖
    # Semaphore 允許不同 Task 進行 acquire/release，確保即使發生錯誤也能釋放
    await llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        await llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            yield _llama_proxy
        finally:
            # 無論是否發生 400/500 錯誤，這裡都會執行
            # Semaphore.release() 不會檢查 Task ID，所以不會報 RuntimeError
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()


_ping_message_factory: typing.Optional[typing.Callable[[], bytes]] = None


def set_ping_message_factory(factory: typing.Callable[[], bytes]):
    global _ping_message_factory
    _ping_message_factory = factory


def create_app(
    settings: Settings | None = None,
    server_settings: ServerSettings | None = None,
    model_settings: List[ModelSettings] | None = None,
):
    config_file = os.environ.get("CONFIG_FILE", None)
    if config_file is not None:
        if not os.path.exists(config_file):
            raise ValueError(f"Config file {config_file} not found!")
        with open(config_file, "rb") as f:
            # Check if yaml file
            if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                import yaml

                config_file_settings = ConfigFileSettings.model_validate_json(
                    json.dumps(yaml.safe_load(f))
                )
            else:
                config_file_settings = ConfigFileSettings.model_validate_json(f.read())
            server_settings = ServerSettings.model_validate(config_file_settings)
            model_settings = config_file_settings.models

    if server_settings is None and model_settings is None:
        if settings is None:
            settings = Settings()
        server_settings = ServerSettings.model_validate(settings)
        model_settings = [ModelSettings.model_validate(settings)]

    assert (
        server_settings is not None and model_settings is not None
    ), "server_settings and model_settings must be provided together"

    set_server_settings(server_settings)
    middleware = [Middleware(RawContextMiddleware, plugins=(RequestIdPlugin(),))]
    app = FastAPI(
        middleware=middleware,
        title="🦙 llama.cpp Python API",
        version=llama_cpp.__version__,
        root_path=server_settings.root_path,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 加入錯誤處理，攔截驗證錯誤
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        print(f"\n🔥 [400 Bad Request] Validation Error: {exc.errors()}")
        return JSONResponse(
            status_code=400,
            content={"detail": exc.errors(), "body": str(exc.body)},
        )

    app.include_router(router)

    assert model_settings is not None
    set_llama_proxy(model_settings=model_settings)

    if server_settings.disable_ping_events:
        set_ping_message_factory(lambda: bytes())

    return app


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream[typing.Any],
    iterator: Iterator[typing.Any],
    on_complete: typing.Optional[typing.Callable[[], typing.Awaitable[None]]] = None,
):
    server_settings = next(get_server_settings())
    interrupt_requests = (
        server_settings.interrupt_requests if server_settings else False
    )
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                await inner_send_chan.send(dict(data=json.dumps(chunk)))
                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()
                
                # 【修正 3】解決 AttributeError: 'SemaphoreAdapter' object has no attribute 'locked'
                # Semaphore 沒有 locked() 方法，我們檢查它的 value 是否為 0 (0 代表已被佔用/鎖定)
                if interrupt_requests and llama_outer_lock.value == 0:
                    await inner_send_chan.send(dict(data="[DONE]"))
                    raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(dict(data="[DONE]"))
        except anyio.get_cancelled_exc_class() as e:
            print("disconnected")
            with anyio.move_on_after(1, shield=True):
                print(f"Disconnected from client (via refresh/close) {request.client}")
                raise e
        finally:
            if on_complete:
                await on_complete()


def _logit_bias_tokens_to_input_ids(
    llama: llama_cpp.Llama,
    logit_bias: Dict[str, float],
) -> Dict[str, float]:
    to_bias: Dict[str, float] = {}
    for token, score in logit_bias.items():
        token = token.encode("utf-8")
        for input_id in llama.tokenize(token, add_bos=False, special=True):
            to_bias[str(input_id)] = score
    return to_bias


# Setup Bearer authentication scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def authenticate(
    settings: Settings = Depends(get_server_settings),
    authorization: Optional[str] = Depends(bearer_scheme),
):
    # Skip API key check if it's not set in settings
    if settings.api_key is None:
        return True

    # check bearer credentials against the api_key
    if authorization and authorization.credentials == settings.api_key:
        # api key is valid
        return authorization.credentials

    # raise http error 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )


openai_v1_tag = "OpenAI V1"


@router.post(
    "/v1/completions",
    summary="Completion",
    dependencies=[Depends(authenticate)],
    response_model=Union[
        llama_cpp.CreateCompletionResponse,
        str,
    ],
    responses={
        "200": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {"$ref": "#/components/schemas/CreateCompletionResponse"}
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True. "
                        + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
                        "example": """data: {... see CreateCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
                    }
                },
            },
        }
    },
    tags=[openai_v1_tag],
)
@router.post(
    "/v1/engines/copilot-codex/completions",
    include_in_schema=False,
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
) -> llama_cpp.Completion:
    exit_stack = contextlib.AsyncExitStack()
    llama_proxy = await exit_stack.enter_async_context(contextlib.asynccontextmanager(get_llama_proxy)())
    if llama_proxy is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not available",
        )
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    llama = llama_proxy(
        body.model
        if request.url.path != "/v1/engines/copilot-codex/completions"
        else "copilot-codex"
    )

    exclude = {
        "n",
        "best_of",
        "logit_bias_type",
        "user",
        "min_tokens",
    }
    kwargs = body.model_dump(exclude=exclude)

    if body.logit_bias is not None:
        kwargs["logit_bias"] = (
            _logit_bias_tokens_to_input_ids(llama, body.logit_bias)
            if body.logit_bias_type == "tokens"
            else body.logit_bias
        )

    if body.grammar is not None:
        kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

    if body.min_tokens > 0:
        _min_tokens_logits_processor = llama_cpp.LogitsProcessorList(
            [llama_cpp.MinTokensLogitsProcessor(body.min_tokens, llama.token_eos())]
        )
        if "logits_processor" not in kwargs:
            kwargs["logits_processor"] = _min_tokens_logits_processor
        else:
            kwargs["logits_processor"].extend(_min_tokens_logits_processor)

    try:
        iterator_or_completion: Union[
            llama_cpp.CreateCompletionResponse,
            Iterator[llama_cpp.CreateCompletionStreamResponse],
        ] = await run_in_threadpool(llama, **kwargs)
    except Exception as err:
        await exit_stack.aclose()
        raise err

    if isinstance(iterator_or_completion, Iterator):
        first_response = await run_in_threadpool(next, iterator_or_completion)
        def iterator() -> Iterator[llama_cpp.CreateCompletionStreamResponse]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
                on_complete=exit_stack.aclose,
            ),
            sep="\n",
            ping_message_factory=_ping_message_factory,
        )
    else:
        await exit_stack.aclose()
        return iterator_or_completion


@router.post(
    "/v1/embeddings",
    summary="Embedding",
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def create_embedding(
    request: CreateEmbeddingRequest,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
):
    return await run_in_threadpool(
        llama_proxy(request.model).create_embedding,
        **request.model_dump(exclude={"user"}),
    )


# 【修正 4】使用 Dict 接收 Body，徹底繞過 Pydantic 驗證，解決 400 Bad Request
@router.post(
    "/v1/chat/completions",
    summary="Chat",
    dependencies=[Depends(authenticate)],
    response_model=Union[llama_cpp.ChatCompletion, str],
    responses={
        "200": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {
                                "$ref": "#/components/schemas/CreateChatCompletionResponse"
                            }
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True"
                        + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
                        "example": """data: {... see CreateChatCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
                    }
                },
            },
        }
    },
    tags=[openai_v1_tag],
)
async def create_chat_completion(
    request: Request,
    body: Dict[str, Any] = Body(
        openapi_examples={
            "normal": {
                "summary": "Chat Completion",
                "value": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                },
            },
        }
    ),
) -> llama_cpp.ChatCompletion:
    exit_stack = contextlib.AsyncExitStack()
    llama_proxy = await exit_stack.enter_async_context(contextlib.asynccontextmanager(get_llama_proxy)())
    if llama_proxy is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not available",
        )
    
    # 這裡直接用 body (dict) 來處理，不再使用 model_dump
    exclude_keys = {
        "n",
        "logit_bias_type",
        "user",
        "min_tokens",
        "stream_options",
        "parallel_tool_calls"
    }
    
    # 複製 body 內容並排除不支援的 key
    kwargs = {k: v for k, v in body.items() if k not in exclude_keys}
    
    llama = llama_proxy(kwargs.get("model"))
    
    # 處理 Logit Bias
    logit_bias = kwargs.get("logit_bias")
    logit_bias_type = kwargs.get("logit_bias_type")
    if logit_bias is not None:
        kwargs["logit_bias"] = (
            _logit_bias_tokens_to_input_ids(llama, logit_bias)
            if logit_bias_type == "tokens"
            else logit_bias
        )

    # 處理 Grammar
    grammar_str = kwargs.get("grammar")
    if grammar_str is not None:
        kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(grammar_str)

    # 處理 Min Tokens
    min_tokens = kwargs.get("min_tokens", 0)
    if min_tokens > 0:
        _min_tokens_logits_processor = llama_cpp.LogitsProcessorList(
            [llama_cpp.MinTokensLogitsProcessor(min_tokens, llama.token_eos())]
        )
        if "logits_processor" not in kwargs:
            kwargs["logits_processor"] = _min_tokens_logits_processor
        else:
            kwargs["logits_processor"].extend(_min_tokens_logits_processor)

    try:
        iterator_or_completion: Union[
            llama_cpp.ChatCompletion, Iterator[llama_cpp.ChatCompletionChunk]
        ] = await run_in_threadpool(llama.create_chat_completion, **kwargs)
    except Exception as err:
        await exit_stack.aclose()
        print(f"🔥 Llama execution error: {err}")
        raise err

    if isinstance(iterator_or_completion, Iterator):
        first_response = await run_in_threadpool(next, iterator_or_completion)
        def iterator() -> Iterator[llama_cpp.ChatCompletionChunk]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
                on_complete=exit_stack.aclose,
            ),
            sep="\n",
            ping_message_factory=_ping_message_factory,
        )
    else:
        await exit_stack.aclose()
        return iterator_or_completion


@router.get(
    "/v1/models",
    summary="Models",
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def get_models(
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> ModelList:
    return {
        "object": "list",
        "data": [
            {
                "id": model_alias,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
            for model_alias in llama_proxy
        ],
    }


extras_tag = "Extras"


@router.post(
    "/extras/tokenize",
    summary="Tokenize",
    dependencies=[Depends(authenticate)],
    tags=[extras_tag],
)
async def tokenize(
    body: TokenizeInputRequest,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> TokenizeInputResponse:
    tokens = llama_proxy(body.model).tokenize(body.input.encode("utf-8"), special=True)

    return TokenizeInputResponse(tokens=tokens)


@router.post(
    "/extras/tokenize/count",
    summary="Tokenize Count",
    dependencies=[Depends(authenticate)],
    tags=[extras_tag],
)
async def count_query_tokens(
    body: TokenizeInputRequest,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> TokenizeInputCountResponse:
    tokens = llama_proxy(body.model).tokenize(body.input.encode("utf-8"), special=True)

    return TokenizeInputCountResponse(count=len(tokens))


@router.post(
    "/extras/detokenize",
    summary="Detokenize",
    dependencies=[Depends(authenticate)],
    tags=[extras_tag],
)
async def detokenize(
    body: DetokenizeInputRequest,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> DetokenizeInputResponse:
    text = llama_proxy(body.model).detokenize(body.tokens).decode("utf-8")

    return DetokenizeInputResponse(text=text)
