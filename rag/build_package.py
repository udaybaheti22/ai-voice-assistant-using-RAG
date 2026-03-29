"""Build and load knowledge packages for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path
import pickle

import faiss

from rag.chunk import chunk_text
from rag.embed import get_embeddings
from rag.index import create_faiss_index

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def _load_text_files(raw_docs_dir: Path) -> list[str]:
    """Load non-empty text documents from a package directory."""
    if not raw_docs_dir.exists() or not raw_docs_dir.is_dir():
        raise FileNotFoundError(f"Raw documents directory not found: {raw_docs_dir}")

    text_files = sorted(raw_docs_dir.glob("*.txt"))
    if not text_files:
        raise FileNotFoundError(f"No .txt files found in: {raw_docs_dir}")

    documents: list[str] = []
    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(text)

    if not documents:
        raise ValueError(f"All documents in {raw_docs_dir} are empty.")

    return documents


def build_package(package_name: str, base_dir: str = "packages") -> Path:
    """Build a knowledge package and save the FAISS index plus chunks as pickle.

    Args:
        package_name: Name of the package folder under ``base_dir``.
        base_dir: Root directory containing package folders.

    Returns:
        Path to the generated ``index.pkl`` file.
    """
    if not package_name or not package_name.strip():
        raise ValueError("package_name must not be empty.")

    package_dir = Path(base_dir) / package_name
    raw_docs_dir = package_dir / "raw_docs"
    output_path = package_dir / "index.pkl"

    documents = _load_text_files(raw_docs_dir)

    all_chunks: list[str] = []
    for document in documents:
        all_chunks.extend(chunk_text(document))

    if not all_chunks:
        raise ValueError(f"No chunks were produced for package: {package_name}")

    embeddings = get_embeddings(all_chunks)
    index = create_faiss_index(embeddings)

    package_data = {
        "chunks": all_chunks,
        "index": faiss.serialize_index(index),
    }

    package_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(package_data, file_obj)

    return output_path


def load_package(package_name: str, base_dir: str = "packages") -> tuple[faiss.Index, list[str]]:
    """Load a previously built knowledge package from pickle."""
    package_path = Path(base_dir) / package_name / "index.pkl"
    if not package_path.exists():
        raise FileNotFoundError(f"Package index file not found: {package_path}")

    with package_path.open("rb") as file_obj:
        package_data = pickle.load(file_obj)

    if "index" not in package_data or "chunks" not in package_data:
        raise ValueError(f"Invalid package data in: {package_path}")

    index = faiss.deserialize_index(package_data["index"])
    chunks = package_data["chunks"]
    return index, chunks

build_package("finance")