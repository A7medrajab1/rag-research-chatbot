# src/embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def warmup():
    """Pre-load the model so first query is fast."""
    get_model()
    print("Embedding model ready.")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    model = get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,  # disable progress bar for speed
        normalize_embeddings=True,
        batch_size=64,
    )
    return np.array(embeddings, dtype=np.float32)