import re

def clean_text(text: str) -> str:
    """Clean raw text by removing references, citations, URLs, and fixing formatting."""
    # Remove bracketed references like [1], [2,3]
    text = re.sub(r"$$\s*\d+(?:\s*,\s*\d+)*\s*$$", "", text)

    # Remove inline citations like (Author 2024)
    text = re.sub(r"\([A-Za-z][A-Za-z\s,&\.]*\d{4}\)", "", text)

    # Fix hyphenated line breaks
    text = re.sub(r"-\s*\n\s*", "", text)

    # Normalize paragraph spacing
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Normalize whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove space before punctuation
    text = re.sub(r"\s+([.,;:])", r"\1", text)

    return text.strip()


def load_txt(file_path: str) -> str:
    """Load text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {e}")


def preprocess_file(input_path: str, output_path: str) -> str:
    """Load, clean, and save preprocessed text."""
    raw_text = load_txt(input_path)
    cleaned = clean_text(raw_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Preprocessed text saved to {output_path} ({len(cleaned)} chars)")
    return cleaned