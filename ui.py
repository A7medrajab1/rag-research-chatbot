import streamlit as st
import requests
import time

# ============================================
# Configuration
# ============================================
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="RAG Research Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #1a1f2e;
        border-left: 4px solid #4a9eff;
        padding: 15px 20px;
        border-radius: 0 12px 12px 0;
        margin: 10px 0;
        color: #e0e0e0;
    }
    
    .bot-message {
        background-color: #1a2a1a;
        border-left: 4px solid #4aff7f;
        padding: 15px 20px;
        border-radius: 0 12px 12px 0;
        margin: 10px 0;
        color: #e0e0e0;
    }
    
    /* Sources styling */
    .source-box {
        background-color: #1a1a2e;
        border: 1px solid #333;
        padding: 12px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 0.85em;
        color: #b0b0b0;
    }
    
    .score-badge {
        background-color: #4a9eff;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    
    /* Title */
    .title {
        text-align: center;
        color: #4a9eff;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    
    /* Sidebar */
    .sidebar-info {
        background-color: #1a1f2e;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1a1f2e;
        color: white;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Helper Functions
# ============================================
def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        return None


def ask_question(question: str, use_multi_query: bool = True,
                 top_k: int = 8, score_threshold: float = 0.15) -> dict:
    """Send a question to the API and get the answer."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "use_multi_query": use_multi_query,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code} - {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the server is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Try again."}


# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    use_multi_query = st.toggle("Multi-Query Search", value=True,
                                help="Expand your query into multiple sub-queries for better results")

    top_k = st.slider("Number of Sources", min_value=1, max_value=15,
                       value=8, help="How many chunks to retrieve")

    score_threshold = st.slider("Score Threshold", min_value=0.0,
                                max_value=1.0, value=0.15, step=0.05,
                                help="Minimum similarity score (higher = stricter)")

    show_sources = st.toggle("Show Sources", value=True,
                             help="Display retrieved document chunks")

    st.markdown("---")

    # API Health Check
    st.markdown("## 📡 API Status")
    health = check_api_health()
    if health:
        st.success(f"✅ Connected")
        st.markdown(f"""
        <div class='sidebar-info'>
            <b>Chunks:</b> {health.get('num_chunks', 'N/A')}<br>
            <b>Model:</b> {health.get('model', 'N/A')}<br>
            <b>Status:</b> {health.get('status', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ API not running")
        st.markdown("Run: `uvicorn app:app --port 8000`")

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    ### 📖 About
    RAG Chatbot for research papers using 
    semantic search and LLMs.
    
    """)


# ============================================
# Main Chat Area
# ============================================
st.markdown('<p class="title">🤖 RAG Research Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about research documents</p>',
            unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])

            # Show sources if available
            if show_sources and "sources" in message:
                with st.expander(f"📚 Sources ({message['num_sources']} chunks)"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"""
                        <div class='source-box'>
                            <span class='score-badge'>Score: {source['score']:.3f}</span>
                            &nbsp; Chunk #{source['chunk_id']}<br><br>
                            {source['content'][:300]}...
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about the research paper..."):

    # Check API first
    if not check_api_health():
        st.error("❌ API is not running. Start it with: `uvicorn app:app --port 8000`")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching and generating answer..."):
                start_time = time.time()
                result = ask_question(
                    prompt,
                    use_multi_query=use_multi_query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                elapsed = time.time() - start_time

            if "error" in result:
                st.error(result["error"])
                bot_message = {"role": "assistant", "content": result["error"]}
            else:
                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                num_sources = result.get("num_sources", 0)

                st.markdown(answer)
                st.caption(f"⏱️ {elapsed:.1f}s | 📚 {num_sources} sources")

                # Show sources
                if show_sources and sources:
                    with st.expander(f"📚 Sources ({num_sources} chunks)"):
                        for i, source in enumerate(sources):
                            st.markdown(f"""
                            <div class='source-box'>
                                <span class='score-badge'>Score: {source['score']:.3f}</span>
                                &nbsp; Chunk #{source['chunk_id']}<br><br>
                                {source['content'][:300]}...
                            </div>
                            """, unsafe_allow_html=True)

                bot_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "num_sources": num_sources,
                }

            st.session_state.messages.append(bot_message)