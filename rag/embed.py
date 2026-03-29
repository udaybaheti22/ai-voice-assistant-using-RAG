"""Embedding utilities built on sentence-transformers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it across calls."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embeddings(text_chunks: Iterable[str]) -> np.ndarray:
    """Generate embeddings for a sequence of text chunks.

    Args:
        text_chunks: Iterable of text segments to embed.

    Returns:
        A float32 NumPy array shaped as ``(n_chunks, embedding_dim)``.

    Raises:
        ValueError: If no valid text chunks are provided.
    """
    chunks = [chunk.strip() for chunk in text_chunks if chunk and chunk.strip()]
    if not chunks:
        raise ValueError("No text chunks were provided for embedding.")

    model = _get_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)
