# src/chunking.py 
import json
import re

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text_into_chunks(text: str, source: str = "unknown",
                           chunk_size: int = CHUNK_SIZE,
                           chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split text into chunks with metadata."""
    if not text or not text.strip():
        raise ValueError("Input text is empty")

    documents = [
        Document(page_content=text, metadata={"source": source})
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",      # triple newline (section breaks)
            "\n\n",         # paragraph breaks
            "\n",           # line breaks
            ". ",           # sentences
            " ",            # words
            "",
        ],
        length_function=len,
    )

    raw_chunks = splitter.split_documents(documents)

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        content = chunk.page_content.strip()
        if len(content) < 30:
            continue

        # Try to detect section/heading for metadata
        heading = detect_heading(content)

        chunks.append({
            "id": i,
            "content": content,
            "metadata": {**chunk.metadata, "heading": heading},
            "length": len(content),
        })

    print(f"Created {len(chunks)} chunks (skipped {len(raw_chunks) - len(chunks)} short chunks)")
    print(f"Avg chunk size: {sum(c['length'] for c in chunks) // len(chunks)} chars")
    return chunks


def detect_heading(text: str) -> str:
    """Try to extract section heading from chunk."""
    first_line = text.split("\n")[0].strip()
    # If first line is short and looks like a heading
    if len(first_line) < 100 and not first_line.endswith("."):
        return first_line
    return ""


def save_chunks(chunks: list[dict], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {file_path}")


def load_chunks(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)