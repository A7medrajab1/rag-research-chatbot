from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from openai import OpenAI
from src.vector_store import load_index, index_exists
from src.retrieval import search, multi_query_search
from src.embeddings import warmup
from src.config import OPENROUTER_API_KEY, MODEL_NAME, TOP_K


# ============================================
# Global State
# ============================================
state = {
    "index": None,
    "chunks": None,
    "client": None,
}


# ============================================
# Startup / Shutdown
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, index, and client on startup."""
    print("🔄 Starting up...")

    warmup()

    if not index_exists():
        raise RuntimeError("Index not found. Run: python -m src.pipeline")
    state["index"], state["chunks"] = load_index()

    # OpenRouter instead of OpenAI direct
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")
    state["client"] = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    print("✅ Ready!")
    yield

    print("👋 Shutting down...")
    state["index"] = None
    state["chunks"] = None
    state["client"] = None

# ============================================
# App
# ============================================
app = FastAPI(
    title="RAG Research Chatbot API",
    description="Ask questions about research documents using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request / Response Models
# ============================================
class QueryRequest(BaseModel):
    question: str
    use_multi_query: bool = True
    top_k: int = TOP_K
    score_threshold: float = 0.15

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are hard constraints in university timetabling?",
                "use_multi_query": True,
                "top_k": 8,
                "score_threshold": 0.15,
            }
        }


class Source(BaseModel):
    content: str
    score: float
    chunk_id: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    num_sources: int
    query_used: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_chunks: int
    model: str


# ============================================
# System Prompt
# ============================================
SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question based ONLY on the provided context.

Rules:
1. If the answer is found in the context, provide a clear and detailed answer.
2. If the question asks for a comparison, look for information about EACH item separately in the context and synthesize a comparison.
3. If the answer is partially found, provide what you can and mention what's missing.
4. If the answer is NOT found in the context, say "I couldn't find this information in the available documents."
5. Always be specific and cite which parts of the context support your answer.
6. Do not make up information."""


# ============================================
# Helper Functions
# ============================================
def format_context(results: list[dict]) -> str:
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}] (relevance: {r['score']:.3f})\n{r['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, results: list[dict]) -> str:
    if not results:
        return "No relevant documents found for your query."

    context = format_context(results)

    try:
        response = state["client"].chat.completions.create(
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

        content = response.choices[0].message.content
        return content if content else "The model returned an empty response. Please try again."

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return "⚠️ Rate limited. Please wait a moment and try again."
        return f"Error generating answer: {error_msg}"


# ============================================
# Endpoints
# ============================================
@app.get("/", tags=["General"])
async def root():
    return {"message": "RAG Research Chatbot API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Check if the API is running and index is loaded."""
    return HealthResponse(
        status="healthy" if state["index"] else "not ready",
        index_loaded=state["index"] is not None,
        num_chunks=len(state["chunks"]) if state["chunks"] else 0,
        model=MODEL_NAME,
    )


@app.post("/ask", response_model=QueryResponse, tags=["Chat"])
async def ask(request: QueryRequest):
    """Ask a question and get an answer based on the research documents."""
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not state["index"]:
        raise HTTPException(status_code=503, detail="Index not loaded")

    try:
        if request.use_multi_query:
            results = multi_query_search(
                query=question,
                index=state["index"],
                chunks=state["chunks"],
                client=state["client"],
                model_name=MODEL_NAME,
                k=request.top_k,
                score_threshold=request.score_threshold,
            )
        else:
            results = search(
                query=question,
                index=state["index"],
                chunks=state["chunks"],
                k=request.top_k,
                score_threshold=request.score_threshold,
            )

        answer = generate_answer(question, results)

        sources = [
            Source(
                content=r["content"][:500],
                score=r["score"],
                chunk_id=r["chunk_id"],
            )
            for r in results
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            num_sources=len(sources),
            query_used=question,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", tags=["Search"])
async def search_only(request: QueryRequest):
    """Search for relevant chunks without generating an answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not state["index"]:
        raise HTTPException(status_code=503, detail="Index not loaded")

    results = search(
        query=request.question,
        index=state["index"],
        chunks=state["chunks"],
        k=request.top_k,
        score_threshold=request.score_threshold,
    )

    return {
        "query": request.question,
        "num_results": len(results),
        "results": [
            {
                "content": r["content"][:500],
                "score": r["score"],
                "chunk_id": r["chunk_id"],
            }
            for r in results
        ],
    }