# src\chatbot.py
from openai import OpenAI
from src.vector_store import load_index
from src.retrieval import search, multi_query_search
from src.embeddings import warmup
from src.config import OPENROUTER_API_KEY, MODEL_NAME, TOP_K


def create_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def format_context(results: list[dict]) -> str:
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}] (relevance: {r['score']:.3f})\n{r['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question based ONLY on the provided context.

Rules:
1. If the answer is found in the context, provide a clear and detailed answer.
2. If the question asks for a comparison, look for information about EACH item separately in the context and synthesize a comparison.
3. If the answer is partially found, provide what you can and mention what's missing.
4. If the answer is NOT found in the context, say "I couldn't find this information in the available documents."
5. Always be specific and cite which parts of the context support your answer.
6. Do not make up information."""


def generate_answer(client: OpenAI, query: str, results: list[dict]) -> str:
    if not results:
        return "No relevant documents found for your query."

    context = format_context(results)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}",
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def main():
    print("Initializing...")
    warmup()  # load embedding model once at startup

    print("Loading index...")
    index, chunks = load_index()

    print("Connecting to LLM...")
    client = create_client()

    print("\n✅ RAG Chatbot Ready!")
    print("Commands: 'exit' to quit | 'mode' to toggle search mode\n")
    print("=" * 50)

    use_multi_query = True

    while True:
        query = input("\n📝 Ask: ").strip()

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "mode":
            use_multi_query = not use_multi_query
            mode_name = "Multi-Query" if use_multi_query else "Simple"
            print(f"Switched to {mode_name} search mode")
            continue

        if use_multi_query:
            results = multi_query_search(
                query, index, chunks, client, MODEL_NAME,
                k=TOP_K, score_threshold=0.15,
            )
        else:
            results = search(
                query, index, chunks, k=TOP_K, score_threshold=0.15,
            )

        print(f"\n🔍 Found {len(results)} relevant chunks")
        if results:
            print(f"   Best score: {results[0]['score']:.3f}")
            print(f"   Worst score: {results[-1]['score']:.3f}")

        answer = generate_answer(client, query, results)

        print(f"\n💡 Answer:\n{answer}")
        print("=" * 50)


if __name__ == "__main__":
    main()