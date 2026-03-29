"""Utilities for splitting raw text into overlapping chunks."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: Raw input text to split.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of non-empty text chunks.

    Raises:
        ValueError: If ``chunk_size`` is invalid or ``overlap`` is not smaller
            than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks
