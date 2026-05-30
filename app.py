# app.py — AI Chat with RAG backend
from __future__ import annotations

import logging
import os
import time

import streamlit as st

from rag import answer, build_vector_store, is_vector_store_ready, load_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chat",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "vector_store" not in st.session_state and is_vector_store_ready():
    st.session_state.vector_store = load_vector_store()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word for English text."""
    return max(1, int(len(text.split()) * 1.3))


def _stream_text(text: str):
    """Yield words with a small delay to simulate streaming."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.025)


# ── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global font (preserve Material Symbols) ── */
    *:not([class*="material"]):not([data-testid="stSidebarCollapseButton"] span):not([data-testid="stSidebarCollapsedControl"] span) {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    [class*="material-symbols"],
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapsedControl"] span {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* ── Hide chrome ── */
    #MainMenu, footer, .stDeployButton { display: none !important; }
    .stApp > header {
        background: transparent !important;
        border: none !important;
    }

    /* ── Animations ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes skeletonPulse {
        0%, 100% { opacity: 0.04; }
        50%      { opacity: 0.09; }
    }
    @keyframes dotPulse {
        0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
        40%           { opacity: 1;   transform: scale(1); }
    }

    /* ── Backgrounds ── */
    .stApp, .main, .main .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stChatMessage"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background-color: #09090b !important;
    }
    .main .block-container {
        max-width: 60rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding: 0.5rem 1.5rem 6rem !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #1a1a1f; border-radius: 2px; }

    /* ═══════ SIDEBAR ═══════ */
    section[data-testid="stSidebar"] {
        background-color: #111114 !important;
        border-right: 1px solid #1c1c22 !important;
    }
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] .block-container,
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
    section[data-testid="stSidebar"] [data-testid="stFileUploader"],
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #111114 !important;
    }

    /* Collapse button */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        opacity: 0.4 !important;
        visibility: visible !important;
        display: flex !important;
        transition: opacity 0.2s ease !important;
        z-index: 999 !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="stSidebarCollapseButton"]:hover { opacity: 0.9 !important; }
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button {
        background: transparent !important;
        border: none !important;
        color: #444 !important;
    }
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:hover { color: #aaa !important; }

    /* Sidebar file uploader */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"],
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] div {
        background-color: #111114 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background-color: #111114 !important;
        border: 1px solid #1c1c22 !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover {
        border-color: #2a2a38 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        background-color: #111114 !important;
        color: #444 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: #191920 !important;
        border: 1px solid #1c1c22 !important;
        color: #2dd4bf !important;
        border-radius: 6px !important;
        font-size: 0.76rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        background-color: #1e1e28 !important;
        border-color: #2dd4bf44 !important;
    }

    /* Sidebar primary button */
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background: #2dd4bf !important;
        border: none !important;
        color: #09090b !important;
        font-weight: 700 !important;
        padding: 0.55rem 1rem !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover,
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {
        background: #5eead4 !important;
        box-shadow: 0 2px 12px rgba(45,212,191,0.2) !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"]:active,
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:active {
        background: #14b8a6 !important;
    }

    /* Sidebar secondary button */
    section[data-testid="stSidebar"] .stButton button[kind="secondary"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
        background: transparent !important;
        border: 1px solid #1e1e28 !important;
        color: #666 !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
        border-color: #2a2a38 !important;
        color: #999 !important;
    }

    /* Sidebar alerts */
    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(45,212,191,0.05) !important;
        border: 1px solid #1e2e2c !important;
        border-radius: 6px !important;
        font-size: 0.76rem !important;
    }

    /* ── Sidebar custom HTML ── */
    .sb-brand {
        display: flex; align-items: center; gap: 0.6rem;
        padding-bottom: 1.2rem; margin-bottom: 1rem;
        border-bottom: 1px solid #15151d;
    }
    .sb-brand-icon {
        width: 30px; height: 30px; border-radius: 8px;
        background: #2dd4bf;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.72rem; font-weight: 800; color: #09090b;
        letter-spacing: -0.03em;
    }
    .sb-brand-name { font-size: 0.88rem; font-weight: 700; color: #d4d4d8; letter-spacing: -0.02em; }
    .sb-brand-sub  { font-size: 0.64rem; color: #3f3f46; font-weight: 400; margin-top: 1px; }
    .sb-section    { font-size: 0.64rem; color: #3f3f46; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin: 1rem 0 0.5rem; }
    .sb-divider    { height: 1px; background: #15151d; margin: 1rem 0; }
    .sb-status     {
        display: inline-flex; align-items: center; gap: 0.35rem;
        padding: 0.25rem 0.7rem; border-radius: 999px;
        font-size: 0.68rem; font-weight: 600;
    }
    .sb-status.ready   { background: rgba(45,212,191,0.06); color: #2dd4bf; border: 1px solid #1e2e2c; }
    .sb-status.waiting { background: rgba(250,204,21,0.04); color: #a18413; border: 1px solid #2a2517; }
    .sb-dot {
        width: 5px; height: 5px; border-radius: 50%;
    }
    .sb-dot.active   { background: #2dd4bf; box-shadow: 0 0 6px rgba(45,212,191,0.5); }
    .sb-dot.inactive { background: #a18413; opacity: 0.6; }

    /* ═══════ CHAT HEADER ═══════ */
    .chat-header {
        text-align: center; padding: 2rem 0 1rem;
        animation: fadeIn 0.4s ease;
    }
    .chat-header h1 {
        font-size: 1.5rem; font-weight: 700; color: #d4d4d8;
        letter-spacing: -0.03em; margin: 0 0 0.3rem;
    }
    .chat-header p {
        color: #3f3f46; font-size: 0.78rem; margin: 0;
    }

    /* ═══════ CHAT MESSAGES (st.chat_message) ═══════ */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.6rem 0 !important;
        animation: fadeIn 0.3s ease;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.84rem !important;
        line-height: 1.7 !important;
        color: #a1a1aa !important;
    }
    /* User messages — slightly brighter */
    [data-testid="stChatMessage"][data-testid-suffix="user"] [data-testid="stMarkdownContainer"] p,
    .stChatMessage:has(img[alt="user"]) [data-testid="stMarkdownContainer"] p {
        color: #d4d4d8 !important;
    }

    /* Chat avatars */
    [data-testid="stChatMessage"] img {
        border-radius: 6px !important;
        width: 26px !important;
        height: 26px !important;
    }

    /* ═══════ SKELETON LOADER ═══════ */
    .skeleton-wrap {
        padding: 0.8rem 0; animation: fadeIn 0.3s ease;
    }
    .skeleton-line {
        height: 10px; border-radius: 4px;
        background: #ffffff; margin-bottom: 0.55rem;
        animation: skeletonPulse 1.4s ease-in-out infinite;
    }
    .skeleton-line:nth-child(1) { width: 85%; animation-delay: 0s; }
    .skeleton-line:nth-child(2) { width: 70%; animation-delay: 0.15s; }
    .skeleton-line:nth-child(3) { width: 55%; animation-delay: 0.3s; }

    .thinking-dots {
        display: flex; gap: 4px; padding: 0.4rem 0;
    }
    .thinking-dots span {
        width: 5px; height: 5px; border-radius: 50%;
        background: #2dd4bf; display: inline-block;
    }
    .thinking-dots span:nth-child(1) { animation: dotPulse 1.2s ease-in-out infinite; }
    .thinking-dots span:nth-child(2) { animation: dotPulse 1.2s ease-in-out 0.2s infinite; }
    .thinking-dots span:nth-child(3) { animation: dotPulse 1.2s ease-in-out 0.4s infinite; }

    /* ═══════ TOKEN BADGE ═══════ */
    .token-badge {
        font-size: 0.64rem; color: #3f3f46; font-weight: 500;
        margin-top: 0.15rem; letter-spacing: 0.01em;
    }

    /* ═══════ BOTTOM CHAT INPUT ═══════ */
    [data-testid="stBottom"] {
        background: linear-gradient(to top, #09090b 70%, transparent) !important;
        padding-top: 1.5rem !important;
    }
    [data-testid="stChatInput"] {
        background: #09090b !important;
    }
    [data-testid="stChatInput"] > div {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid #1e1e28 !important;
        border-radius: 12px !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #2dd4bf44 !important;
        box-shadow: 0 0 0 2px rgba(45,212,191,0.04) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #d4d4d8 !important;
        font-size: 0.84rem !important;
        caret-color: #2dd4bf !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #2a2a32 !important; }
    [data-testid="stChatInput"] button {
        background: #2dd4bf !important;
        border: none !important;
        color: #09090b !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: #5eead4 !important;
    }
    [data-testid="stChatInput"] button svg { color: #09090b !important; fill: #09090b !important; }

    /* ═══════ SUGGESTION BUTTONS ═══════ */
    .suggestions {
        display: flex; flex-wrap: wrap; gap: 0.4rem;
        justify-content: center; margin-top: 0.5rem;
    }
    div[data-testid="stMainBlockContainer"] > div > div > div > .stButton button {
        border-radius: 999px !important;
        border: 1px solid #1e1e28 !important;
        background: transparent !important;
        color: #52525b !important;
        font-size: 0.74rem !important;
        font-weight: 500 !important;
        padding: 0.3rem 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stMainBlockContainer"] > div > div > div > .stButton button:hover {
        border-color: #2dd4bf33 !important;
        color: #a1a1aa !important;
        background: rgba(45,212,191,0.03) !important;
    }

    /* ═══════ EMPTY STATE ═══════ */
    .empty-state {
        text-align: center; padding: 3rem 1rem;
        animation: fadeIn 0.4s ease 0.15s both;
    }
    .empty-state h3 {
        color: #3f3f46; font-size: 0.85rem; font-weight: 500; margin: 0 0 0.25rem;
    }
    .empty-state p {
        color: #27272a; font-size: 0.74rem; margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>What can I help you find?</h1>
    <p>Ask questions about your uploaded documents</p>
</div>
""", unsafe_allow_html=True)

# ── Render chat history ─────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "tokens" in msg:
            st.markdown(
                f'<div class="token-badge">{msg["tokens"]} tokens</div>',
                unsafe_allow_html=True,
            )

# ── Empty state ─────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h3>No messages yet</h3>
        <p>Upload documents in the sidebar, then start asking questions</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    suggestions = ["Summarize", "Key findings", "Extract data", "Explain"]
    for i, text in enumerate(suggestions):
        with cols[i]:
            if st.button(text, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = text
                st.rerun()

# ── Chat input (pinned to bottom) ──────────────────────────────────
prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.pending_query = prompt

# ── Process query ───────────────────────────────────────────────────
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Show skeleton loader, then stream the response
    with st.chat_message("assistant"):
        # Skeleton placeholder
        skeleton = st.empty()
        skeleton.markdown("""
        <div class="skeleton-wrap">
            <div class="thinking-dots"><span></span><span></span><span></span></div>
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
        </div>
        """, unsafe_allow_html=True)

        # Get answer
        ai_text = ""
        t_start = time.perf_counter()
        try:
            if "vector_store" in st.session_state:
                result = answer(query, st.session_state.vector_store)
                ai_text = result["answer"]
            else:
                ai_text = "No documents loaded. Upload PDFs in the sidebar and build the vector store first."
        except Exception as e:
            logger.error("Answer failed: %s", e, exc_info=True)
            err = str(e)
            if "429" in err:
                ai_text = "Rate-limited. Please wait a moment and try again."
            else:
                ai_text = f"Error: {err[:200]}"
        t_elapsed = time.perf_counter() - t_start

        # Clear skeleton, stream the answer
        skeleton.empty()
        st.write_stream(_stream_text(ai_text))

        # Token stats
        prompt_tokens = _estimate_tokens(query)
        completion_tokens = _estimate_tokens(ai_text)
        total_tokens = prompt_tokens + completion_tokens
        st.markdown(
            f'<div class="token-badge">{total_tokens} tokens &middot; {t_elapsed:.1f}s</div>',
            unsafe_allow_html=True,
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_text,
        "tokens": total_tokens,
    })

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">AI</div>
        <div>
            <div class="sb-brand-name">AI Chat</div>
            <div class="sb-brand-sub">RAG Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Documents
    st.markdown('<div class="sb-section">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploaded_files:
        os.makedirs("Artifacts", exist_ok=True)
        for f in uploaded_files:
            path = os.path.join("Artifacts", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s)")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Vector store
    st.markdown('<div class="sb-section">Vector Store</div>', unsafe_allow_html=True)
    if st.button("Build vector store", use_container_width=True, type="primary"):
        logger.info("Ingesting...")
        with st.spinner("Building..."):
            try:
                st.session_state.vector_store = build_vector_store()
                st.success("Vector store ready")
            except Exception as e:
                logger.error("Ingest failed: %s", e)
                st.error(str(e))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if is_vector_store_ready():
        st.markdown(
            '<div class="sb-status ready"><div class="sb-dot active"></div>Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-status waiting"><div class="sb-dot inactive"></div>No data</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Clear chat
    if st.session_state.messages:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()
