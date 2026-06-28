# rag/retrieval.py

import time
import httpx
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from .config import DB_PATH, RERANK_MODEL, TOP_K, PAGE_CONTENT_LIMIT

# =========================
# DB load
# =========================
t0 = time.time()

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embeddings
)

reranker = CrossEncoder(RERANK_MODEL)

print(f"DB LOADED: {time.time() - t0:.3f}s", flush=True)


# =========================
# Rerank
# =========================
def rerank(query: str, docs: list) -> list:
    if not docs:
        return []

    pairs = []
    for doc in docs:
        title = doc.metadata.get("title", "")
        text = doc.page_content[:PAGE_CONTENT_LIMIT]
        pairs.append((query, f"{title}\n{text}"))

    scores = reranker.predict(pairs)

    ranked = []
    for score, doc in zip(scores, docs):
        title = doc.metadata.get("title", "").lower()
        if query.lower() in title:
            score += 0.8
        if any(cmd in title for cmd in ("fix", "compute", "dump", "region", "variable")):
            score += 0.2
        ranked.append((score, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:TOP_K]]


# =========================
# retrieval
# =========================
def retrieve(query: str):
    # ChromaDBから多めに取得してrerankerで絞る
    docs = db.similarity_search(query, k=8)

    docs = rerank(query, docs)

    for d in docs:
        if hasattr(d, "page_content"):
            d.page_content = d.page_content[:500]

    return docs
