"""Configuration for MAVAL AI Aluminum Assistant."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================
# ollama Configuration
# =========================

# Ollama
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.1:latest"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1"
)


# =========================
# Data paths
# =========================

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

BM25_INDEX_FILE = PROJECT_ROOT / "data" / "processed" / "bm25_index.pkl"


# =========================
# Embeddings
# =========================

EMBEDDING_MODE = os.getenv(
    "EMBEDDING_MODE",
    "local"
)

LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "BAAI/bge-m3"
)


# =========================
# Retrieval
# =========================

TOP_K_VECTOR = 10

TOP_K_BM25 = 10

TOP_K_FINAL = 3

RRF_K = 60


# =========================
# Safety
# =========================

ENABLE_SAFETY_CHECK = True