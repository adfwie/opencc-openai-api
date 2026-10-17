import time
import uuid
import json
import unicodedata
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from opencc import OpenCC

# 初始化服务
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 预加载所有常用转换方向
converters = {
    "s2t": OpenCC("s2t"),
    "t2s": OpenCC("t2s"),
    "s2tw": OpenCC("s2tw"),
    "tw2s": OpenCC("tw2s"),
    "s2hk": OpenCC("s2hk"),
    "hk2s": OpenCC("hk2s"),
}

# 默认转换方向（工具不指定模型时生效，可自行修改）
DEFAULT_CONVERTER = "s2t"

# OpenAI 请求格式定义
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None


# 核心接口：完全兼容 OpenAI /v1/chat/completions
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 提取最后一条用户消息
    user_text = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_text = msg.content
            break

    # 根据模型名自动匹配转换方向
    converter_key = DEFAULT_CONVERTER
    for key in converters.keys():
        if key in request.model.lower():
            converter_key = key
            break

    # 执行纯规则繁简转换 + Unicode NFC标准化
    raw_convert = converters[converter_key].convert(user_text)
    converted = unicodedata.normalize("NFC", raw_convert)

    # 非流式返回（标准 OpenAI 格式）
    if not request.stream:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": converted},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_text),
                "completion_tokens": len(converted),
                "total_tokens": len(user_text) + len(converted)
            }
        }

    # ==========【修复重点】流式只输出单次完整内容，不再分片！==========
    def stream_generator():
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": converted}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# 模型列表接口，部分工具会调用校验
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": k, "object": "model", "owned_by": "local"} for k in converters.keys()]
    }


if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" 允许局域网访问；本机使用写127.0.0.1
    uvicorn.run(app, host="0.0.0.0", port=8000)
