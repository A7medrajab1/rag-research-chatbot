import re
import numpy as np
from src.embeddings import embed_texts
from src.config import TOP_K


def search(query: str, index, chunks: list[dict],
           k: int = TOP_K, score_threshold: float = None) -> list[dict]:
    """Search for relevant chunks using the FAISS index."""
    query_vector = embed_texts([query])
    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        score = float(distances[0][i])
        if score_threshold is not None and score < score_threshold:
            continue
        results.append({
            "content": chunks[idx]["content"],
            "score": score,
            "chunk_id": chunks[idx].get("id", idx),
        })
    return results


def expand_query(query: str) -> list[str]:
    """Generate sub-queries using rules only (no LLM needed)."""
    sub_queries = []
    query_lower = query.lower().strip()

    # 1. Comparison: "difference between X and Y"
    comparison_patterns = [
        r"(?:difference|compare|comparison|vs|versus)\s+(?:between\s+)?(.+?)\s+(?:and|vs|versus)\s+(.+)",
        r"(.+?)\s+(?:vs|versus|compared to|compared with)\s+(.+)",
    ]
    for pattern in comparison_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            topic_a = match.group(1).strip()
            topic_b = match.group(2).strip()
            sub_queries.extend([
                topic_a,
                topic_b,
                f"{topic_a} characteristics features",
                f"{topic_b} characteristics features",
            ])
            break

    # 2. "X in Y" pattern
    in_match = re.search(r"(.+?)\s+in\s+(.+)", query, re.IGNORECASE)
    if in_match:
        topic = in_match.group(1).strip()
        context = in_match.group(2).strip()
        sub_queries.extend([topic, f"{topic} definition examples", context])

    # 3. "what is/are X" pattern
    what_match = re.search(r"what\s+(?:is|are)\s+(?:the\s+)?(.+)", query, re.IGNORECASE)
    if what_match:
        topic = what_match.group(1).strip()
        clean_topic = re.sub(r"\s+in\s+.+$", "", topic, flags=re.IGNORECASE)
        if clean_topic != topic:
            sub_queries.extend([clean_topic, f"{clean_topic} definition"])

    # 4. Keyword extraction: remove stop words and search key terms
    stop_words = {"what", "is", "are", "the", "a", "an", "in", "of", "for",
                  "and", "or", "how", "does", "do", "can", "could", "would",
                  "about", "tell", "me", "talk", "explain", "describe", "between"}
    words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
    if len(words) >= 2:
        sub_queries.append(" ".join(words))

    return sub_queries


def multi_query_search(query: str, index, chunks: list[dict],
                       client=None, model_name: str = None,
                       k: int = TOP_K, score_threshold: float = None) -> list[dict]:
    """Expand query and combine results for better retrieval."""

    # Rule-based expansion only (no LLM needed)
    sub_queries = expand_query(query)

    # Deduplicate
    all_queries = [query] + sub_queries
    seen = set()
    unique_queries = []
    for q in all_queries:
        q_lower = q.lower().strip()
        if q_lower not in seen and len(q_lower) > 2:
            seen.add(q_lower)
            unique_queries.append(q)

    print(f"   Searching with {len(unique_queries)} queries:")
    for i, q in enumerate(unique_queries):
        print(f"     {i+1}. {q}")

    # Search with all queries and combine
    seen_chunk_ids = set()
    all_results = []

    for q in unique_queries:
        results = search(q, index, chunks, k=k, score_threshold=score_threshold)
        for r in results:
            if r["chunk_id"] not in seen_chunk_ids:
                seen_chunk_ids.add(r["chunk_id"])
                all_results.append(r)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:k * 2]