# rag/config.py
from pathlib import Path

# ==========================================================
# Project
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Directories
# ==========================================================
DB_PATH = PROJECT_ROOT / "rag_db"
DOCUMENTS_PATH = PROJECT_ROOT / "documents"

# ==========================================================
# Ollama Models
# ==========================================================
OLLAMA_CHAT_MODEL = "qwen3.5:9b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
RERANK_MODEL = "BAAI/bge-reranker-base"

# Model name aliases accepted by the OpenAI-compatible API
MODEL_MAP = {
    "gpt-4o-mini": OLLAMA_CHAT_MODEL,
    "qwen3.5:9b": OLLAMA_CHAT_MODEL,
}

# ==========================================================
# Retrieval
# ==========================================================
# Number of chunks retrieved from ChromaDB
SEARCH_K = 8
# Number of chunks kept after reranking
TOP_K = 3
# Max characters per chunk sent to the reranker
PAGE_CONTENT_LIMIT = 1500

# ==========================================================
# Chunking
# ==========================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==========================================================
# Prompt
# ==========================================================
# Max characters of context passed to the LLM
CONTEXT_MAX_CHARS = 1200

# ==========================================================
# Server
# ==========================================================
HOST = "127.0.0.1"
PORT = 8000

# ==========================================================
# Debug
# ==========================================================
DEBUG = True
PRINT_REQUEST = True
PRINT_RAG = True
PRINT_CONTEXT = False
PRINT_OLLAMA_MESSAGES = False
