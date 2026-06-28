# app/server.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import time
import re
import json

from rag.retrieval import retrieve
from rag.prompt import build_messages
from rag.config import MODEL_MAP, OLLAMA_CHAT_MODEL

app = FastAPI()

# ==========================================================
# State（簡易セッション）
# ==========================================================
pending_actions = {}

class ChatRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: list
    session_id: str = "default"
    stream: bool = False

print("LOADED server.py")

# ==========================================================
# Models
# ==========================================================
@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model"}
            for m in MODEL_MAP
        ],
    }

# ==========================================================
# Heuristic router
# ==========================================================
def needs_file_action(query: str) -> bool:
    keywords = [
        "作って", "作成", "生成", "ファイル",
        "スクリプト", "コード", "インプット",
        "input file", "write a",
    ]
    return any(k in query for k in keywords)

# ==========================================================
# Thinking tag remover
# ==========================================================
def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ==========================================================
# Ollama呼び出し（非同期）
# ==========================================================
async def call_ollama(model: str, messages: list) -> str:
    print(f"=== CALLING OLLAMA: {model} ===", flush=True)
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
            },
        )
        print("=== OLLAMA RESPONDED ===", flush=True)
        print(f"=== STATUS: {response.status_code} ===", flush=True)
        data = response.json()
        print(f"=== DATA KEYS: {list(data.keys())} ===", flush=True)
        content = data["message"]["content"]
        print(f"=== CONTENT LENGTH: {len(content)} ===", flush=True)
        stripped = strip_thinking(content)
        print(f"=== STRIPPED LENGTH: {len(stripped)} ===", flush=True)
        return stripped

# ==========================================================
# Streaming response generator
# ==========================================================
async def stream_response(model: str, content: str):
    """OpenAI互換のSSEストリームを生成"""
    chunk = {
        "id": "chatcmpl-local",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # 終了チャンク
    final = {
        "id": "chatcmpl-local",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"

# ==========================================================
# Response helper
# ==========================================================
def _make_response(model: str, content: str, stream: bool):
    if stream:
        return StreamingResponse(
            stream_response(model, content),
            media_type="text/event-stream",
        )
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

# ==========================================================
# Chat endpoint
# ==========================================================
@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):

    print("=== REQUEST RECEIVED ===", flush=True)
    print(f"=== STREAM: {req.stream} ===", flush=True)

    query = next(
        m["content"]
        for m in reversed(req.messages)
        if m["role"] == "user"
    )
    print(f"=== QUERY: {query} ===", flush=True)

    # --------------------------------------------------
    # STEP 1: confirmation flow
    # --------------------------------------------------
    if req.session_id in pending_actions:
        if "yes" in query.lower() or "y" == query.lower().strip():
            action = pending_actions.pop(req.session_id)
            messages = build_messages(req.messages, action["context"])
            llm_model = MODEL_MAP.get(req.model, OLLAMA_CHAT_MODEL)
            content = await call_ollama(llm_model, messages)
        else:
            pending_actions.pop(req.session_id, None)
            content = "OK, cancelled."
        return _make_response(req.model, content, req.stream)

    # --------------------------------------------------
    # STEP 2: detect file-related intent
    # --------------------------------------------------
    if needs_file_action(query):
        print("=== FILE ACTION DETECTED ===", flush=True)
        docs = retrieve(query)
        pending_actions[req.session_id] = {
            "query": query,
            "context": docs,
            "messages": req.messages,
        }
        return _make_response(
            req.model,
            "File generation requested. Do you want to proceed? (yes/no)",
            req.stream,
        )

    # --------------------------------------------------
    # STEP 3: normal QA
    # --------------------------------------------------
    print("=== STEP 3: NORMAL QA ===", flush=True)
    docs = retrieve(query)
    print("=== DOCS RETRIEVED ===", flush=True)
    messages = build_messages(req.messages, docs)
    print("=== MESSAGES BUILT ===", flush=True)

    llm_model = MODEL_MAP.get(req.model, OLLAMA_CHAT_MODEL)
    content = await call_ollama(llm_model, messages)
    print(f"=== CONTENT RECEIVED IN CHAT ===", flush=True) 
    return _make_response(req.model, content, req.stream)
