"""Query-time retrieval helpers for the RAG pipeline."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np

from rag.embed import get_embeddings


def retrieve(query: str, index: faiss.Index, chunks: list[str], k: int = 3) -> list[dict[str, Any]]:
    """Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User query string.
        index: FAISS index containing chunk embeddings.
        chunks: Ordered chunk list aligned with the index vectors.
        k: Number of results to return.

    Returns:
        A list of dictionaries containing chunk text, chunk index, and distance.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty.")
    if not chunks:
        raise ValueError("Chunks list is empty.")
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    query_embedding = get_embeddings([query])
    limit = min(k, len(chunks))

    distances, indices = index.search(np.asarray(query_embedding, dtype=np.float32), limit)

    results: list[dict[str, Any]] = []
    for chunk_idx, distance in zip(indices[0], distances[0]):
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            continue
        results.append(
            {
                "chunk": chunks[chunk_idx],
                "index": int(chunk_idx),
                "distance": float(distance),
            }
        )

    return results

