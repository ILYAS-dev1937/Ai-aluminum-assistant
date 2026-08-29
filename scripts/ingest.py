"""ingest.py — Load chunks.json into ChromaDB + BM25 index."""
import json
import pickle
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
import re

from config import CHUNKS_FILE, CHROMA_DIR, BM25_INDEX_FILE, LOCAL_EMBEDDING_MODEL


def tokenize(text: str) -> list:
    """Simple tokenizer for BM25."""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def main():
    print("Loading chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # ── ChromaDB ──────────────────────────────────────────────────
    print(f"Initializing ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Use local embeddings (bge-m3 is best for multilingual technical text)
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=LOCAL_EMBEDDING_MODEL,
        device="cpu",  # Change to "cuda" if you have GPU
    )

    collection = client.get_or_create_collection(
        name="maval_rubis",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare batch upload
    ids = [c["id"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [
        {
            "title": c.get("title", ""),
            "category": c.get("metadata", {}).get("category", "general"),
            "series": c.get("metadata", {}).get("series", "Both"),
            "profile_ref": c.get("metadata", {}).get("profile_ref", ""),
            "source_page": c.get("metadata", {}).get("source_page", ""),
            "keywords": ", ".join(c.get("metadata", {}).get("keywords", [])),
        }
        for c in chunks
    ]

    # Upsert in batches (Chroma handles duplicates by ID)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = i + batch_size
        print(f"  Upserting batch {i}-{end}...")
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"✅ ChromaDB ready: {collection.count()} chunks")

    # ── BM25 Index ────────────────────────────────────────────────
    print("Building BM25 index...")
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    BM25_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "documents": documents}, f)

    print(f"✅ BM25 index saved to {BM25_INDEX_FILE}")
    print("Next: run 'python query.py' or 'streamlit run app.py'")


if __name__ == "__main__":
    main()
