#!/usr/bin/env python3
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.loader import load_file
from rag.config import (
    DB_PATH,
    DOCUMENTS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def main():
    start_time = time.time()
    print("\n===== BUILD START =====\n")

    # ==========================================================
    # Load documents
    # ==========================================================
    docs = []
    files = list(DOCUMENTS_PATH.rglob("*"))
    print(f"[INFO] Found {len(files)} files")
    for i, file in enumerate(files):
        if not file.is_file():
            continue
        if i % 50 == 0:
            print(f"[LOAD] {i}/{len(files)} files processed")
        try:
            loaded = load_file(file)
            docs.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load {file}: {e}")
    print(f"\n[INFO] Loaded {len(docs)} document(s)\n")

    # ==========================================================
    # Chunking
    # ==========================================================
    print("[INFO] Starting chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Generated {len(chunks)} chunks\n")

    # ==========================================================
    # Embedding（HuggingFace、ollamaを使わない）
    # ==========================================================
    print("[INFO] Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ==========================================================
    # 既存DBを削除して再構築
    # ==========================================================
    import shutil
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)
        print(f"[INFO] Deleted existing DB at {DB_PATH}")

    # ==========================================================
    # Build Chroma DB
    # ==========================================================
    print("[INFO] Starting ChromaDB build...")
    t0 = time.time()

    BATCH_SIZE = 500
    db = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"[EMBED] Processing {i} → {i + len(batch)} / {len(chunks)}")
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(DB_PATH),
            )
        else:
            db.add_documents(batch)

    print(f"[INFO] ChromaDB build finished in {time.time() - t0:.2f}s")

    elapsed = time.time() - start_time
    print("\n===== BUILD COMPLETE =====")
    print(f"[TOTAL] {elapsed:.2f} sec")
    print(f"[TOTAL CHUNKS] {len(chunks)}")

if __name__ == "__main__":
    main()
