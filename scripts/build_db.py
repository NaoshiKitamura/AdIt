#!/usr/bin/env python3

import json
import time
import shutil
from pathlib import Path
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

# Path to store chunk config snapshot
CHUNK_CONFIG_PATH = DB_PATH / ".chunk_config.json"


def load_chunk_config() -> dict:
    """Load the chunk config used when the DB was last built."""
    if CHUNK_CONFIG_PATH.exists():
        with open(CHUNK_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_chunk_config():
    """Save the current chunk config to disk."""
    CHUNK_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNK_CONFIG_PATH, "w") as f:
        json.dump({"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}, f)


def chunk_config_changed(saved: dict) -> bool:
    """Return True if CHUNK_SIZE or CHUNK_OVERLAP changed since last build."""
    return (
        saved.get("chunk_size") != CHUNK_SIZE
        or saved.get("chunk_overlap") != CHUNK_OVERLAP
    )


def get_sources_by_config(db: Chroma) -> tuple[set, set]:
    """
    Return two sets of source paths from the existing DB:
      - sources with the current chunk config  (can be skipped)
      - sources with a different chunk config  (need re-embedding)
    """
    current = {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}
    up_to_date = set()
    outdated = set()

    try:
        existing = db.get()
        for metadata in existing["metadatas"]:
            if not metadata or "source" not in metadata:
                continue
            source = metadata["source"]
            meta_config = {
                "chunk_size": metadata.get("chunk_size"),
                "chunk_overlap": metadata.get("chunk_overlap"),
            }
            if meta_config == current:
                up_to_date.add(source)
            else:
                outdated.add(source)
    except Exception as e:
        print(f"[INFO] No existing DB found: {e}")

    return up_to_date, outdated


def delete_chunks_by_source(db: Chroma, sources: set):
    """Delete all chunks belonging to the given source paths."""
    if not sources:
        return
    existing = db.get()
    ids_to_delete = [
        id_
        for id_, meta in zip(existing["ids"], existing["metadatas"])
        if meta and meta.get("source") in sources
    ]
    if ids_to_delete:
        db.delete(ids=ids_to_delete)
        print(f"[INFO] Deleted {len(ids_to_delete)} outdated chunks from {len(sources)} source(s)")


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

    db = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )

    # ==========================================================
    # Determine which sources need (re-)embedding
    # ==========================================================
    if args.rebuild:
        sources_to_skip = set()
        sources_to_reembed = set()
    else:
        sources_to_skip, sources_to_reembed = get_sources_by_config(db)

        if sources_to_reembed:
            print(f"[INFO] Chunk config changed for {len(sources_to_reembed)} source(s) — will re-embed")
            delete_chunks_by_source(db, sources_to_reembed)
        else:
            print(f"[INFO] Found {len(sources_to_skip)} up-to-date source(s)")

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
        # Skip only if up-to-date with current chunk config
        if str(file) in sources_to_skip:
            skipped += 1
            continue
        try:
            loaded = load_file(file)
            docs.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load {file}: {e}")

    print(f"[INFO] Skipped {skipped} already up-to-date file(s)")
    print(f"[INFO] Loaded {len(docs)} document(s) to process\n")

    if not docs:
        print("[INFO] No new or changed documents to add. DB is up to date.")
        save_chunk_config()
        return

    # ==========================================================
    # Chunking — embed chunk config into each chunk's metadata
    # ==========================================================
    print("[INFO] Starting chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # Attach chunk config to metadata for future comparison
    for chunk in chunks:
        chunk.metadata["chunk_size"] = CHUNK_SIZE
        chunk.metadata["chunk_overlap"] = CHUNK_OVERLAP

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
    # Save chunk config snapshot
    # ==========================================================
    save_chunk_config()

    # ==========================================================
    # Finish
    # ==========================================================
    elapsed = time.time() - start_time
    print("\n===== BUILD COMPLETE =====")
    print(f"[TOTAL] {elapsed:.2f} sec")
    print(f"[TOTAL NEW CHUNKS] {len(chunks)}")


if __name__ == "__main__":
    main()
