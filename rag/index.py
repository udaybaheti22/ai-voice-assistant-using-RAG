"""FAISS index creation helpers."""

from __future__ import annotations

import faiss
import numpy as np


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create and populate a FAISS L2 index from embeddings."""
    if embeddings.size == 0:
        raise ValueError("Embeddings array is empty.")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    vectors = np.asarray(embeddings, dtype=np.float32)
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index
