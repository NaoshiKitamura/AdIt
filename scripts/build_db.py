#!/usr/bin/env python3

import time
import shutil
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild DB from scratch")
    args = parser.parse_args()

    start_time = time.time()
    print("\n===== BUILD START =====\n")

    # ==========================================================
    # Embedding model
    # ==========================================================
    print("[INFO] Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ==========================================================
    # Rebuild or incremental
    # ==========================================================
    if args.rebuild and DB_PATH.exists():
        shutil.rmtree(DB_PATH)
        print(f"[INFO] Deleted existing DB at {DB_PATH}")

    # Load or create DB
    db = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )

    # Get already registered sources
    existing_sources = set()
    if not args.rebuild:
        try:
            existing = db.get()
            for metadata in existing["metadatas"]:
                if metadata and "source" in metadata:
                    existing_sources.add(metadata["source"])
            print(f"[INFO] Found {len(existing_sources)} already registered sources")
        except Exception as e:
            print(f"[INFO] No existing DB found: {e}")

    # ==========================================================
    # Load documents
    # ==========================================================
    docs = []
    files = list(DOCUMENTS_PATH.rglob("*"))
    print(f"[INFO] Found {len(files)} files in documents/")

    skipped = 0
    for file in files:
        if not file.is_file():
            continue
        if str(file) in existing_sources:
            skipped += 1
            continue
        try:
            loaded = load_file(file)
            docs.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load {file}: {e}")

    print(f"[INFO] Skipped {skipped} already registered files")
    print(f"[INFO] Loaded {len(docs)} new document(s)\n")

    if not docs:
        print("[INFO] No new documents to add. DB is up to date.")
        return

    # ==========================================================
    # Chunking
    # ==========================================================
    print("[INFO] Starting chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Generated {len(chunks)} new chunks\n")

    # ==========================================================
    # Add to DB
    # ==========================================================
    print("[INFO] Adding to ChromaDB...")
    t0 = time.time()

    BATCH_SIZE = 500
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"[EMBED] Processing {i} → {i + len(batch)} / {len(chunks)}")
        db.add_documents(batch)

    print(f"[INFO] ChromaDB update finished in {time.time() - t0:.2f}s")

    # ==========================================================
    # Finish
    # ==========================================================
    elapsed = time.time() - start_time
    print("\n===== BUILD COMPLETE =====")
    print(f"[TOTAL] {elapsed:.2f} sec")
    print(f"[TOTAL NEW CHUNKS] {len(chunks)}")


if __name__ == "__main__":
    main()
