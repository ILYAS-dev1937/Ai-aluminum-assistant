"""retriever.py -- Hybrid search: BM25 + Vector + Metadata filters + RRF."""
import json
import pickle
import re
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import (
    CHROMA_DIR, BM25_INDEX_FILE, LOCAL_EMBEDDING_MODEL,
    TOP_K_VECTOR, TOP_K_BM25, TOP_K_FINAL, RRF_K
)


class MavalRetriever:
    def __init__(self):
        print("Loading retriever...")
        # ChromaDB
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=LOCAL_EMBEDDING_MODEL,
            device="cpu",
        )
        self.collection = self.client.get_collection(
            name="maval_rubis",
            embedding_function=self.embed_fn,
        )

        # BM25
        with open(BM25_INDEX_FILE, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.bm25_ids = data["ids"]
        self.bm25_docs = data["documents"]

        # Reference extraction regex
        self.ref_pattern = re.compile(r"\b\d{3,4}[A-Z]?\b")

        print("  ChromaDB: {} chunks".format(self.collection.count()))
        print("  BM25: ready")

    def _extract_refs(self, query: str) -> List[str]:
        """Extract profile/accessory reference numbers from query."""
        return list(set(self.ref_pattern.findall(query)))

    def _bm25_search(self, query: str, k: int = TOP_K_BM25) -> List[tuple]:
        """Return list of (chunk_id, score)."""
        tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.bm25_ids[i], scores[i]) for i in top_indices]

    def _vector_search(
        self,
        query: str,
        series_filter: str = None,
        k: int = TOP_K_VECTOR,
    ) -> List[tuple]:
        """Return list of (chunk_id, distance). Lower distance = better."""
        where_filter = None
        if series_filter and series_filter != "Both":
            where_filter = {"series": {"$eq": series_filter}}

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        ids = results["ids"][0]
        distances = results["distances"][0]
        # Convert distance to score (lower distance = higher score)
        # Cosine distance is 0-2, so we invert: score = 2 - distance
        return [(cid, 2.0 - dist) for cid, dist in zip(ids, distances)]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[tuple],
        vector_results: List[tuple],
        ref_boosts: Dict[str, float] = None,
    ) -> List[str]:
        """RRF: fuse ranked lists. Returns ordered chunk IDs."""
        scores = {}

        # BM25 ranks
        for rank, (cid, _) in enumerate(bm25_results):
            scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

        # Vector ranks
        for rank, (cid, _) in enumerate(vector_results):
            scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

        # Boost exact reference matches
        if ref_boosts:
            for cid, boost in ref_boosts.items():
                if cid in scores:
                    scores[cid] += boost

        # Sort by fused score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked]

    def search(
        self,
        query: str,
        series_filter: str = None,
        top_k: int = TOP_K_FINAL,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search pipeline:
        1. Extract reference numbers from query
        2. BM25 keyword search
        3. Vector semantic search (with optional series metadata filter)
        4. Reciprocal Rank Fusion
        5. Return full chunk objects
        """
        refs = self._extract_refs(query)
        if refs:
            print("  Detected refs in query: {}".format(refs))

        # 1. BM25
        bm25_results = self._bm25_search(query)

        # 2. Vector
        vector_results = self._vector_search(query, series_filter=series_filter)

        # 3. Build ref boost map
        ref_boosts = {}
        if refs:
            for cid, meta in zip(self.bm25_ids, self.collection.get(ids=self.bm25_ids)["metadatas"]):
                profile_ref = meta.get("profile_ref", "") if meta else ""
                if any(r in profile_ref for r in refs):
                    ref_boosts[cid] = 0.5  # Strong boost for exact ref match

        # 4. Fuse
        fused_ids = self._reciprocal_rank_fusion(bm25_results, vector_results, ref_boosts)

        # 5. Fetch full chunks
        final_ids = fused_ids[:top_k]
        if not final_ids:
            return []

        chroma_results = self.collection.get(ids=final_ids)

        # Reassemble into dicts
        output = []
        for i, cid in enumerate(final_ids):
            try:
                idx = chroma_results["ids"].index(cid)
                output.append({
                    "id": cid,
                    "content": chroma_results["documents"][idx],
                    "metadata": chroma_results["metadatas"][idx],
                })
            except ValueError:
                continue

        return output


# CLI test
if __name__ == "__main__":
    retriever = MavalRetriever()

    test_queries = [
        "What is the inertia of profile 998?",
        "Can I use 24mm glass in RUBIS 85?",
        "Which roller for a heavy 120kg door?",
        "What accessories for 2-panel RUBIS 95 with dormant 951?",
    ]

    for q in test_queries:
        print("\n" + "=" * 60)
        print("Query: {}".format(q))
        results = retriever.search(q)
        for r in results:
            cat = r["metadata"].get("category", "?")
            page = r["metadata"].get("source_page", "?")
            print("  -> {} | {} | page {}".format(r["id"], cat, page))