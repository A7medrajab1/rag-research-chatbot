# src/vector_store.py
import faiss
import numpy as np
import pickle
import os
from src.embeddings import embed_texts
from src.config import FAISS_INDEX_PATH, CHUNKS_PKL_PATH


def build_and_save_index(chunks: list[dict]) -> tuple:
    """Build FAISS index from chunks and save to disk."""
    if not chunks:
        raise ValueError("No chunks provided")

    texts = [chunk["content"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    dimension = embeddings.shape[1]

    # Use IndexFlatIP (Inner Product) with normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Ensure directory exists
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index saved with {index.ntotal} vectors (dim={dimension})")
    return index, chunks


def load_index() -> tuple:
    """Load FAISS index and chunks from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"Index not found at {FAISS_INDEX_PATH}. Run the pipeline first."
        )
    if not os.path.exists(CHUNKS_PKL_PATH):
        raise FileNotFoundError(
            f"Chunks not found at {CHUNKS_PKL_PATH}. Run the pipeline first."
        )

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PKL_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded index with {index.ntotal} vectors and {len(chunks)} chunks")
    return index, chunks


def index_exists() -> bool:
    """Check if index files exist on disk."""
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PKL_PATH)
