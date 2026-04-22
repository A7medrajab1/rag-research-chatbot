# src/pipeline.py
"""
Pipeline to preprocess, chunk, embed, and index documents.
Run this ONCE before using the chatbot.
"""
from src.config import (
    RAW_TEXT_PATH, CLEANED_TEXT_PATH, CHUNKS_JSON_PATH
)
from src.preprocessing import preprocess_file
from src.chunking import split_text_into_chunks, save_chunks, load_chunks
from src.vector_store import build_and_save_index, index_exists


def run_pipeline(force_rebuild: bool = False):
    """Run the full indexing pipeline."""

    if index_exists() and not force_rebuild:
        print("Index already exists. Use force_rebuild=True to rebuild.")
        return

    print("=" * 50)
    print("Step 1: Preprocessing")
    print("=" * 50)
    cleaned_text = preprocess_file(RAW_TEXT_PATH, CLEANED_TEXT_PATH)

    print("\n" + "=" * 50)
    print("Step 2: Chunking")
    print("=" * 50)
    chunks = split_text_into_chunks(cleaned_text, source="uctp.txt")
    save_chunks(chunks, CHUNKS_JSON_PATH)

    print("\n" + "=" * 50)
    print("Step 3: Building Index")
    print("=" * 50)
    build_and_save_index(chunks)

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build RAG index pipeline")
    parser.add_argument("--force", action="store_true", help="Force rebuild index")
    args = parser.parse_args()
    run_pipeline(force_rebuild=args.force)