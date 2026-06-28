# rag/prompt.py

from .config import CONTEXT_MAX_CHARS

# ==========================================================
# Context builder
# ==========================================================
def build_context(docs: list) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        if isinstance(doc, str):
            text = doc
            title = "unknown"
            source = "unknown"
        else:
            text = getattr(doc, "page_content", "")
            metadata = getattr(doc, "metadata", {}) or {}
            title = metadata.get("title", "unknown")
            source = metadata.get("source", "unknown")
        text = text[:300]
        block = f"""
[DOC {i}]
TITLE: {title}
SOURCE: {source}
{text}
"""
        blocks.append(block)
    return "\n\n".join(blocks)[:1200]

# ==========================================================
# Prompt builder
# ==========================================================
def build_messages(messages: list, context: str):
    system_messages = []
    conversation = []
    for m in messages:
        if m["role"] == "system":
            system_messages.append(m["content"])
        else:
            conversation.append(m)

    merged_system = "\n\n".join(system_messages)

    ollama_messages = []
    if merged_system:
        ollama_messages.append({
            "role": "system",
            "content": merged_system
        })

    # Last user detection
    last_user_content = None
    for m in reversed(conversation):
        if m["role"] == "user":
            last_user_content = m["content"]
            break

    last_user_added = False
    for m in conversation:
        role = m["role"]
        content = m["content"]
        if role == "user" and content == last_user_content and not last_user_added:
            content = f"""Use the following reference material if it is relevant.
=====================
{context}
=====================

IMPORTANT: You MUST reply in the same language as the question below.
If the question is in Japanese, reply entirely in Japanese.
If the question is in English, reply entirely in English.

Question: {content}"""
            last_user_added = True
        ollama_messages.append({
            "role": role,
            "content": content
        })

    return ollama_messages
