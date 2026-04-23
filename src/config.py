# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DOCUMENTS_DIR = "documents"
RAW_TEXT_PATH = os.path.join(DOCUMENTS_DIR, "uctp.txt")
CLEANED_TEXT_PATH = os.path.join(DOCUMENTS_DIR, "cleaned_text.txt")
CHUNKS_JSON_PATH = os.path.join(DOCUMENTS_DIR, "chunks.json")
FAISS_INDEX_PATH = os.path.join(DOCUMENTS_DIR, "faiss.index")
CHUNKS_PKL_PATH = os.path.join(DOCUMENTS_DIR, "chunks.pkl")

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300  # increased from 200

# LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "z-ai/glm-4.5-air:free")

# Retrieval
TOP_K = 8  # increased from 5
SIMILARITY_THRESHOLD = 0.15