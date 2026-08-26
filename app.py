# app.py — AI Chat with RAG backend (Enterprise Edition)
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime

import streamlit as st

from rag import answer, build_vector_store, is_vector_store_ready, load_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Database helpers ────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")


def _get_db():
    """Return a SQLite connection with WAL mode for performance."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    return conn


def _create_session(title: str = "New chat") -> str:
    """Create a new chat session and return its ID."""
    sid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    conn = _get_db()
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (sid, title, now, now),
    )
    conn.commit()
    conn.close()
    return sid


def _list_sessions() -> list[dict]:
    """List all sessions ordered by most recent."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def _load_messages(session_id: str) -> list[dict]:
    """Load all messages for a session."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT role, content, tokens FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "tokens": r[2]} for r in rows]


def _save_message(session_id: str, role: str, content: str, tokens: int = 0):
    """Save a message and update session timestamp."""
    now = datetime.now().isoformat()
    conn = _get_db()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, tokens, now),
    )
    conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
    )
    conn.commit()
    conn.close()


def _update_session_title(session_id: str, title: str):
    """Update session title (auto-named from first message)."""
    conn = _get_db()
    conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
    conn.commit()
    conn.close()


def _delete_session(session_id: str):
    """Delete a session and its messages."""
    conn = _get_db()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def _delete_database_and_storage():
    """Clear the hybrid index and raw PDFs from disk."""
    import shutil
    from rag.config import settings

    # 1. Delete vector store files and directory
    if os.path.exists(settings.VECTOR_STORE_PATH):
        try:
            shutil.rmtree(settings.VECTOR_STORE_PATH)
            logger.info("Deleted vector store: %s", settings.VECTOR_STORE_PATH)
        except Exception as e:
            logger.error("Failed to delete vector store path: %s", e)

    # 2. Delete uploaded PDFs in Artifacts directory
    if os.path.exists(settings.ARTIFACTS_DIR):
        try:
            shutil.rmtree(settings.ARTIFACTS_DIR)
            logger.info("Deleted artifacts directory: %s", settings.ARTIFACTS_DIR)
        except Exception as e:
            logger.error("Failed to delete artifacts directory: %s", e)

    # 3. Clear st.session_state vector store
    if "vector_store" in st.session_state:
        del st.session_state.vector_store


# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind | Grounded RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ───────────────────────────────────────────────────
if "active_view" not in st.session_state:
    st.session_state.active_view = "home"
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "vector_store" not in st.session_state and is_vector_store_ready():
    st.session_state.vector_store = load_vector_store()

# Session management
if "current_session" not in st.session_state:
    sessions = _list_sessions()
    if sessions:
        st.session_state.current_session = sessions[0]["id"]
    else:
        st.session_state.current_session = _create_session()

# Load messages from DB
if "messages" not in st.session_state:
    st.session_state.messages = _load_messages(st.session_state.current_session)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word for English text."""
    return max(1, int(len(text.split()) * 1.3))


def _switch_session(session_id: str):
    """Switch to a different chat session."""
    st.session_state.current_session = session_id
    st.session_state.messages = _load_messages(session_id)
    st.session_state.pending_query = None


def _new_chat():
    """Start a new chat session."""
    sid = _create_session()
    st.session_state.current_session = sid
    st.session_state.messages = []
    st.session_state.pending_query = None


# ══════════════════════════════════════════════════════════════════════
# ENTERPRISE CSS — Complete Design System
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp, p, div, h1, h2, h3, h4, h5, h6, textarea, input, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    #MainMenu, footer, .stDeployButton { display: none !important; }
    .stApp > header {
        background: transparent !important;
        border: none !important;
    }

    /* Base App background */
    .stApp, .main, .main .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stChatMessage"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background-color: #0b0d11 !important;
    }

    /* Container constraints */
    .main .block-container {
        max-width: 52rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding: 1.5rem 1.5rem 8rem !important;
        padding-bottom: 10rem !important;
    }
    [data-testid="stBottomBlockContainer"] {
        max-width: 54rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }

    /* Main scroll area — prevent content overflow into chat input */
    [data-testid="stMain"] {
        overflow-y: auto !important;
    }
    [data-testid="stMain"] .block-container {
        position: relative;
        z-index: 1;
    }

    /* Hide default chat avatars */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatar"],
    [data-testid="stChatMessage"] img,
    [data-testid="stChatMessage"] svg {
        display: none !important;
    }

    /* ══════════════════════════════════════════════════════════════
       ANIMATION CLASSES
       ══════════════════════════════════════════════════════════════ */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .anim-fade-up {
        animation: fadeUp 0.5s ease-out forwards;
    }

    /* ══════════════════════════════════════════════════════════════
       ENTERPRISE SIDEBAR STYLING
       ══════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background-color: #12151c !important;
        border-right: 1px solid #252a36 !important;
        width: 280px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 1.5rem !important;
        background-color: #12151c !important;
    }

    /* Brand Header */
    .sb-brand-card {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        border-radius: 12px;
        background: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.15);
        margin-bottom: 1.25rem;
    }
    .sb-brand-icon {
        width: 34px; height: 34px; border-radius: 8px;
        background: #6366f1;
        display: flex; align-items: center; justify-content: center;
        color: #ffffff; font-weight: 800; font-size: 0.8rem;
        flex-shrink: 0;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
    }
    .sb-brand-name { font-size: 0.95rem; font-weight: 700; color: #f1f3f7; letter-spacing: -0.02em; }
    .sb-brand-sub { font-size: 0.7rem; color: #a1a9b8; font-weight: 500; }

    .sb-section-label {
        font-size: 0.65rem; color: #a1a9b8; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 1.25rem 0 0.5rem;
    }

    .sb-divider { height: 1px; background: #252a36; margin: 1rem 0; }

    /* Custom Status Badges */
    .sb-status {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.35rem 0.75rem; border-radius: 999px;
        font-size: 0.7rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;
    }
    .sb-status.ready { background: rgba(52, 211, 153, 0.1); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.25); }
    .sb-status.waiting { background: rgba(248, 113, 113, 0.1); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.25); }
    .sb-dot { width: 6px; height: 6px; border-radius: 50%; }
    .sb-dot.active { background: #34d399; box-shadow: 0 0 8px #34d399; }
    .sb-dot.inactive { background: #f87171; }

    /* Sidebar Button Overrides */
    section[data-testid="stSidebar"] .stButton button {
        background: #181c25 !important;
        border: 1px solid #252a36 !important;
        color: #f1f3f7 !important;
        border-radius: 8px !important;
        font-size: 0.825rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #1f2430 !important;
        border-color: #353f54 !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background: #6366f1 !important;
        border-color: #818cf8 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.25) !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background: #4f46e5 !important;
    }

    /* Sidebar Column Buttons (Session List) */
    section[data-testid="stSidebar"] [data-testid="column"] button {
        padding: 6px 10px !important;
        font-size: 0.8rem !important;
        border-radius: 6px !important;
    }

    /* Column 1: Session Title Button */
    section[data-testid="stSidebar"] [data-testid="column"]:first-child button {
        text-align: left !important;
        justify-content: flex-start !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }

    /* Column 2: Delete Button (renders as "x") */
    section[data-testid="stSidebar"] [data-testid="column"]:last-child button {
        background: transparent !important;
        border: none !important;
        color: #5c6370 !important;
        font-weight: bold !important;
        text-align: center !important;
        justify-content: center !important;
    }
    section[data-testid="stSidebar"] [data-testid="column"]:last-child button:hover {
        background: rgba(239, 68, 68, 0.1) !important;
        color: #f87171 !important;
    }

    /* ══════════════════════════════════════════════════════════════
       WELCOME HERO & PROMPT STARTERS
       ══════════════════════════════════════════════════════════════ */
    .hero-card {
        padding: 2.25rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #12151c 0%, #181c25 100%);
        border: 1px solid #252a36;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.3rem 0.75rem; border-radius: 999px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #818cf8; font-size: 0.725rem; font-weight: 600;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 1.85rem; font-weight: 800; color: #f1f3f7;
        letter-spacing: -0.025em; line-height: 1.25; margin-bottom: 0.6rem;
    }
    .hero-desc {
        font-size: 0.875rem; color: #a1a9b8; line-height: 1.65;
        margin-bottom: 1.5rem; max-width: 38rem;
    }

    .prompt-starter-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }
    .prompt-card {
        padding: 1rem;
        border-radius: 12px;
        background: #12151c;
        border: 1px solid #252a36;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .prompt-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: #181c25;
        transform: translateY(-2px);
    }
    .prompt-card-title {
        font-size: 0.825rem; font-weight: 600; color: #f1f3f7; margin-bottom: 0.25rem;
        display: flex; align-items: center;
    }
    .prompt-card-desc {
        font-size: 0.725rem; color: #a1a9b8; line-height: 1.4;
    }

    /* ══════════════════════════════════════════════════════════════
       CHAT WELCOME EMPTY STATE
       ══════════════════════════════════════════════════════════════ */
    .chat-welcome {
        text-align: center;
        padding: 3rem 1.5rem;
        max-width: 600px;
        margin: 2rem auto;
    }
    .chat-welcome h2 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f3f7;
        margin-bottom: 0.5rem;
    }
    .chat-welcome p {
        font-size: 0.875rem;
        color: #a1a9b8;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    .welcome-icon {
        width: 56px; height: 56px;
        border-radius: 16px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #818cf8;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1.25rem;
    }

    /* ══════════════════════════════════════════════════════════════
       CHAT MESSAGES & BUBBLES
       ══════════════════════════════════════════════════════════════ */
    .user-msg {
        display: flex;
        justify-content: flex-end;
        margin: 1.25rem 0;
        position: relative;
        z-index: 1;
    }
    .user-msg-bubble {
        background: #6366f1;
        color: #ffffff;
        border-radius: 18px 18px 2px 18px;
        padding: 0.85rem 1.25rem;
        max-width: 78%;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25);
        word-break: break-word;
    }
    .user-msg-bubble p {
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #ffffff !important;
        font-weight: 500;
    }

    .assistant-msg {
        margin: 1.25rem 0;
        background: #12151c;
        border: 1px solid #252a36;
        border-radius: 18px 18px 18px 2px;
        padding: 1.15rem 1.35rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        position: relative;
        z-index: 1;
        max-width: 100%;
        word-break: break-word;
        overflow: hidden;
    }
    .assistant-msg p {
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        line-height: 1.7;
        color: #e6edf3 !important;
    }

    /* Typing Indicator */
    .typing-container {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.75rem 1.15rem; border-radius: 12px;
        background: #12151c; border: 1px solid #252a36;
        width: fit-content; margin: 1rem 0;
        position: relative; z-index: 1;
    }
    .typing-pulse {
        width: 8px; height: 8px; border-radius: 50%;
        background: #6366f1; animation: pulse 1.2s infinite ease-in-out;
    }
    .typing-text { font-size: 0.8rem; color: #a1a9b8; font-weight: 500; }

    @keyframes pulse {
        0%, 100% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.1); }
    }

    /* ══════════════════════════════════════════════════════════════
       CHAT INPUT PINNED BAR
       ══════════════════════════════════════════════════════════════ */
    [data-testid="stBottom"] {
        background: linear-gradient(to top, #0b0d11 80%, transparent) !important;
        padding-top: 1.5rem !important;
        padding-bottom: 1.75rem !important;
        border-top: none !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        z-index: 50 !important;
        position: relative !important;
    }
    [data-testid="stChatInput"] > div {
        background: #12151c !important;
        border: 1px solid #252a36 !important;
        border-radius: 16px !important;
        padding: 8px 12px 8px 18px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.15) !important;
    }
    textarea[data-testid="stChatInputTextArea"] {
        color: #f1f3f7 !important;
        font-size: 0.925rem !important;
        caret-color: #6366f1 !important;
    }
    textarea[data-testid="stChatInputTextArea"]::placeholder {
        color: #5c6370 !important;
    }
    [data-testid="stChatInput"] button {
        background: #6366f1 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: #4f46e5 !important;
    }

    /* ══════════════════════════════════════════════════════════════
       FILE UPLOADER (Main Content Area)
       ══════════════════════════════════════════════════════════════ */
    [data-testid="stFileUploader"] section {
        background-color: #181c25 !important;
        border: 1px dashed #252a36 !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
    }
    [data-testid="stFileUploader"] label {
        color: #a1a9b8 !important;
    }
    [data-testid="stFileUploader"] small {
        color: #5c6370 !important;
    }

    /* ══════════════════════════════════════════════════════════════
       MAIN CONTENT AREA BUTTON OVERRIDES
       ══════════════════════════════════════════════════════════════ */
    .main .block-container .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .main .block-container .stButton button[kind="primary"],
    .main .block-container button[data-testid="baseButton-primary"] {
        background: #6366f1 !important;
        border-color: #818cf8 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.25) !important;
    }
    .main .block-container .stButton button[kind="primary"]:hover {
        background: #4f46e5 !important;
    }
    .main .block-container .stButton button[kind="secondary"],
    .main .block-container button[data-testid="baseButton-secondary"] {
        background: #181c25 !important;
        border: 1px solid #252a36 !important;
        color: #f1f3f7 !important;
    }
    .main .block-container .stButton button[kind="secondary"]:hover {
        background: #1f2430 !important;
        border-color: #353f54 !important;
    }

    /* Spinner & success/error overrides */
    .stSpinner > div { color: #818cf8 !important; }
    .stSuccess { background: rgba(52, 211, 153, 0.08) !important; border-color: rgba(52, 211, 153, 0.2) !important; color: #34d399 !important; border-radius: 10px !important; }
    .stAlert { border-radius: 10px !important; }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR (rendered first so navigation state is set before main panel)
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand-card">
        <div class="sb-brand-icon">DM</div>
        <div>
            <div class="sb-brand-name">DocMind</div>
            <div class="sb-brand-sub">Enterprise RAG</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ──
    st.markdown('<div class="sb-section-label">Navigation</div>', unsafe_allow_html=True)

    if st.button("◈  Overview", use_container_width=True, key="nav_home",
                 type="primary" if st.session_state.active_view == "home" else "secondary"):
        st.session_state.active_view = "home"
        st.rerun()

    if st.button("◉  Inquiry Assistant", use_container_width=True, key="nav_chat",
                 type="primary" if st.session_state.active_view == "chat" else "secondary"):
        st.session_state.active_view = "chat"
        st.rerun()

    if st.button("◎  Knowledge Vault", use_container_width=True, key="nav_docs",
                 type="primary" if st.session_state.active_view == "documents" else "secondary"):
        st.session_state.active_view = "documents"
        st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── New Inquiry ──
    if st.button("＋ New Inquiry", use_container_width=True, type="primary", key="new_chat_btn"):
        _new_chat()
        st.session_state.active_view = "chat"
        st.rerun()

    # ── Inquiry History ──
    st.markdown('<div class="sb-section-label">Inquiry History</div>', unsafe_allow_html=True)
    sessions = _list_sessions()
    for sess in sessions:
        is_active = sess["id"] == st.session_state.current_session

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                sess["title"],
                key=f"sess_{sess['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                _switch_session(sess["id"])
                st.session_state.active_view = "chat"
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{sess['id']}"):
                _delete_session(sess["id"])
                if sess["id"] == st.session_state.current_session:
                    remaining = _list_sessions()
                    if remaining:
                        _switch_session(remaining[0]["id"])
                    else:
                        _new_chat()
                st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Status Footer ──
    if is_vector_store_ready():
        st.markdown(
            '<div class="sb-status ready"><div class="sb-dot active"></div>Hybrid Store Online</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-status waiting"><div class="sb-dot inactive"></div>Store Offline</div>',
            unsafe_allow_html=True,
        )

    # Clear current chat
    if st.session_state.messages:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Clear Conversation", use_container_width=True, key="clear_chat"):
            _delete_session(st.session_state.current_session)
            _new_chat()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA — Conditional View Rendering
# ══════════════════════════════════════════════════════════════════════

if st.session_state.active_view == "home":
    # ══════════════════════════════════════════════════════════════
    # OVERVIEW VIEW
    # ══════════════════════════════════════════════════════════════
    vs_status = "Ready" if is_vector_store_ready() else "Offline"

    st.markdown(f"""
    <div class="anim-fade-up">
        <div class="hero-card" style="position: relative; overflow: hidden;">
            <div style="position: relative; z-index: 1;">
                <div class="hero-badge">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 9.7a1 1 0 0 1-.68 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 .76-.97l8-2a1 1 0 0 1 .48 0l8 2A1 1 0 0 1 20 6z"/><path d="m9 12 2 2 4-4"/></svg>
                    <span style="margin-left:4px">Enterprise Grade Context Verification</span>
                </div>
                <h1 class="hero-title">Grounded Document Intelligence</h1>
                <p class="hero-desc">
                    Upload internal PDFs, synthesize dense embeddings with hybrid sparse retrieval (BM25 + RRF), and query your knowledge base with verifiable citations and confidence scoring.
                </p>
            </div>
            <div style="position: absolute; right: -30px; bottom: -30px; opacity: 0.03; pointer-events: none; color: #ffffff;">
                <svg xmlns="http://www.w3.org/2000/svg" width="240" height="240" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 15h3"/><path d="M1 9h3"/><path d="M1 15h3"/></svg>
            </div>
        </div>

        <!-- Pipeline Features Grid -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px;">
            <div style="padding: 24px; border-radius: 14px; border: 1px solid #252a36; background: #12151c;">
                <div style="width: 40px; height: 40px; border-radius: 10px; background: rgba(59, 130, 246, 0.08); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.15); display: flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>
                </div>
                <h3 style="font-size: 13px; font-weight: 700; margin-bottom: 6px; color: #f1f3f7;">1. Multi-Stage Ingestion</h3>
                <p style="font-size: 12px; line-height: 1.6; color: #a1a9b8;">PDF text parsing, semantic chunking, and metadata attribution with localized offline storage.</p>
            </div>

            <div style="padding: 24px; border-radius: 14px; border: 1px solid #252a36; background: #12151c;">
                <div style="width: 40px; height: 40px; border-radius: 10px; background: rgba(139, 92, 246, 0.08); color: #8b5cf6; border: 1px solid rgba(139, 92, 246, 0.15); display: flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="14" y="14" width="4" height="6" rx="2"/><rect x="6" y="14" width="4" height="6" rx="2"/><rect x="6" y="4" width="4" height="6" rx="2"/><path d="M14 4h4v6h-4"/></svg>
                </div>
                <h3 style="font-size: 13px; font-weight: 700; margin-bottom: 6px; color: #f1f3f7;">2. Hybrid Retrieval</h3>
                <p style="font-size: 12px; line-height: 1.6; color: #a1a9b8;">Combines dense vector similarity with sparse BM25 keyword matching using Reciprocal Rank Fusion.</p>
            </div>

            <div style="padding: 24px; border-radius: 14px; border: 1px solid #252a36; background: #12151c;">
                <div style="width: 40px; height: 40px; border-radius: 10px; background: rgba(52, 211, 153, 0.08); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.15); display: flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m2 12 5.25 5L18 6"/><path d="m16 16 2.25 2L23 8"/></svg>
                </div>
                <h3 style="font-size: 13px; font-weight: 700; margin-bottom: 6px; color: #f1f3f7;">3. Grounded Synthesis</h3>
                <p style="font-size: 12px; line-height: 1.6; color: #a1a9b8;">High-precision answers with granular source citations and mathematical confidence metrics.</p>
            </div>
        </div>

        <!-- Telemetry Card -->
        <div style="padding: 24px; border-radius: 14px; border: 1px solid #252a36; background: #12151c;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                <h3 style="font-size: 13px; font-weight: 700; display: flex; align-items: center; gap: 8px; color: #f1f3f7;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #34d399;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    <span>RAG Subsystem Telemetry</span>
                </h3>
                <span style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #5c6370;">Synchronized</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
                <div style="padding: 14px; border-radius: 10px; border: 1px solid #252a36; background: #181c25;">
                    <label style="font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #5c6370; display: block; margin-bottom: 6px;">Engine State</label>
                    <div style="font-size: 13px; font-weight: 700; color: #34d399;">Active</div>
                </div>
                <div style="padding: 14px; border-radius: 10px; border: 1px solid #252a36; background: #181c25;">
                    <label style="font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #5c6370; display: block; margin-bottom: 6px;">Index Status</label>
                    <div style="font-size: 13px; font-weight: 700; color: #f1f3f7;">{vs_status}</div>
                </div>
                <div style="padding: 14px; border-radius: 10px; border: 1px solid #252a36; background: #181c25;">
                    <label style="font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #5c6370; display: block; margin-bottom: 6px;">Persistence</label>
                    <div style="font-size: 13px; font-weight: 700; color: #f1f3f7;">SQLite WAL</div>
                </div>
                <div style="padding: 14px; border-radius: 10px; border: 1px solid #252a36; background: #181c25;">
                    <label style="font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #5c6370; display: block; margin-bottom: 6px;">Privacy Mode</label>
                    <div style="font-size: 13px; font-weight: 700; color: #818cf8;">Offline Local</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


elif st.session_state.active_view == "chat":
    # ══════════════════════════════════════════════════════════════
    # INQUIRY ASSISTANT VIEW
    # ══════════════════════════════════════════════════════════════
    if not st.session_state.messages:
        st.markdown("""
        <div class="chat-welcome anim-fade-up">
            <div class="welcome-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
            </div>
            <h2>Inquiry Assistant</h2>
            <p>Ask queries against your indexed PDF repository. Answers are strictly grounded with citations and confidence metrics.</p>

            <div class="prompt-starter-grid">
                <div class="prompt-card">
                    <div class="prompt-card-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;color:#818cf8;"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="m9 15 2 2 4-4"/></svg>
                        Executive Summary
                    </div>
                    <div class="prompt-card-desc">Synthesize key objectives and conclusions.</div>
                </div>
                <div class="prompt-card">
                    <div class="prompt-card-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;color:#818cf8;"><path d="M20 13c0 5-3.5 7.5-7.66 9.7a1 1 0 0 1-.68 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 .76-.97l8-2a1 1 0 0 1 .48 0l8 2A1 1 0 0 1 20 6z"/><path d="M12 22V12"/><path d="M12 8h.01"/></svg>
                        Compliance &amp; Guidelines
                    </div>
                    <div class="prompt-card-desc">Extract all guidelines, constraints, and policies.</div>
                </div>
                <div class="prompt-card">
                    <div class="prompt-card-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;color:#818cf8;"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 15h3"/><path d="M1 9h3"/><path d="M1 15h3"/></svg>
                        Technical Architecture
                    </div>
                    <div class="prompt-card-desc">Outline system design and methodology.</div>
                </div>
                <div class="prompt-card">
                    <div class="prompt-card-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;color:#818cf8;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>
                        Risk Assessment
                    </div>
                    <div class="prompt-card-desc">Identify critical bottlenecks and caveats.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Render chat history with custom bubbles
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-msg"><div class="user-msg-bubble"><p>{msg["content"]}</p></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-msg"><p>{msg["content"]}</p></div>',
                    unsafe_allow_html=True,
                )

    # ── Chat Input (only shown on chat view) ──
    prompt = st.chat_input("Inquire about indexed documents...")

    if prompt:
        st.session_state.pending_query = prompt

    # ── Process Query ──
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None

        # Save & show user message
        _save_message(st.session_state.current_session, "user", query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Auto-title session from first message
        if len(st.session_state.messages) == 1:
            title = query[:50] + ("..." if len(query) > 50 else "")
            _update_session_title(st.session_state.current_session, title)

        # Render user message
        st.markdown(
            f'<div class="user-msg"><div class="user-msg-bubble"><p>{query}</p></div></div>',
            unsafe_allow_html=True,
        )

        # Animated Typing Indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            '''<div class="typing-container">
                <div class="typing-pulse"></div>
                <span class="typing-text">Synthesizing grounded context...</span>
            </div>''',
            unsafe_allow_html=True,
        )

        ai_text = ""
        try:
            if "vector_store" in st.session_state:
                result = answer(query, st.session_state.vector_store)
                ai_text = result.get("answer", "No response generated.")
            else:
                ai_text = "No documents loaded. Please upload PDF files in the Knowledge Vault and consolidate the index first."
        except Exception as e:
            logger.error("Answer failed: %s", e, exc_info=True)
            err = str(e)
            if "429" in err:
                ai_text = "Rate-limited. Please wait a moment and try again."
            else:
                ai_text = f"Error: {err[:200]}"

        # Clear indicator and render the answer
        typing_placeholder.empty()
        st.markdown(
            f'<div class="assistant-msg"><p>{ai_text}</p></div>',
            unsafe_allow_html=True,
        )

        # Token count & Save to DB
        prompt_tokens = _estimate_tokens(query)
        completion_tokens = _estimate_tokens(ai_text)
        total_tokens = prompt_tokens + completion_tokens

        _save_message(st.session_state.current_session, "assistant", ai_text, total_tokens)
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_text,
            "tokens": total_tokens,
        })
        st.rerun()


elif st.session_state.active_view == "documents":
    # ══════════════════════════════════════════════════════════════
    # KNOWLEDGE VAULT VIEW
    # ══════════════════════════════════════════════════════════════
    idx_status = "Online (Hybrid Indexed)" if is_vector_store_ready() else "Offline (Deposit PDFs & Consolidate)"
    dot_color = "#34d399" if is_vector_store_ready() else "#f87171"

    st.markdown("""
    <div class="anim-fade-up">
        <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 20px; border-bottom: 1px solid #252a36; margin-bottom: 24px;">
            <div>
                <h2 style="font-size: 1.5rem; font-weight: 700; color: #f1f3f7; margin-bottom: 4px;">Knowledge Vault</h2>
                <p style="font-size: 0.8rem; color: #a1a9b8;">Manage and deposit PDF documents for hybrid semantic indexing.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploaded_files:
        os.makedirs("Artifacts", exist_ok=True)
        for f in uploaded_files:
            path = os.path.join("Artifacts", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Deposited {len(uploaded_files)} PDF(s) into vault")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Consolidate Index", use_container_width=True, type="primary", key="vault_build"):
            with st.spinner("Rebuilding Hybrid Vector Store..."):
                try:
                    st.session_state.vector_store = build_vector_store()
                    st.success("Vector Store Synchronized!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    with col2:
        if st.button("Delete Database & Storage", use_container_width=True, key="vault_delete"):
            with st.spinner("Deleting database and storage..."):
                _delete_database_and_storage()
                st.success("Vector index and raw files cleared.")
                time.sleep(1.0)
                st.rerun()

    # Info Cards
    st.markdown(f"""
    <div style="margin-top: 24px; display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px;">
        <!-- Ingestion Workflow -->
        <div style="padding: 24px; border-radius: 14px; border: 1px solid #252a36; background: #12151c;">
            <h3 style="font-size: 13px; font-weight: 700; display: flex; align-items: center; gap: 8px; margin-bottom: 16px; color: #f1f3f7;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:#818cf8;"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                <span>Ingestion Workflow</span>
            </h3>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 12px; font-size: 12px; color: #a1a9b8; line-height: 1.5;">
                    <span style="width: 20px; height: 20px; border-radius: 6px; background: #181c25; border: 1px solid #252a36; display: flex; align-items: center; justify-content: center; font-size: 10px; font-family: monospace; font-weight: 700; color: #818cf8; flex-shrink: 0;">1</span>
                    <span>Upload your PDF documents to the local vault storage.</span>
                </li>
                <li style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 12px; font-size: 12px; color: #a1a9b8; line-height: 1.5;">
                    <span style="width: 20px; height: 20px; border-radius: 6px; background: #181c25; border: 1px solid #252a36; display: flex; align-items: center; justify-content: center; font-size: 10px; font-family: monospace; font-weight: 700; color: #818cf8; flex-shrink: 0;">2</span>
                    <span>Click <strong style="color:#f1f3f7;">Consolidate Index</strong> to parse text, compute embeddings, and build BM25 indices.</span>
                </li>
                <li style="display: flex; align-items: flex-start; gap: 10px; font-size: 12px; color: #a1a9b8; line-height: 1.5;">
                    <span style="width: 20px; height: 20px; border-radius: 6px; background: #181c25; border: 1px solid #252a36; display: flex; align-items: center; justify-content: center; font-size: 10px; font-family: monospace; font-weight: 700; color: #818cf8; flex-shrink: 0;">3</span>
                    <span>Query the <strong style="color:#f1f3f7;">Inquiry Assistant</strong> with guaranteed source grounding.</span>
                </li>
            </ul>
        </div>

        <!-- Index Health Card -->
        <div style="padding: 24px; border-radius: 14px; border: 1px solid #252a36; background: #12151c; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #5c6370; margin-bottom: 10px;">Index Health</div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: {dot_color}; display: inline-block;"></span>
                    <span style="font-size: 14px; font-weight: 700; color: #f1f3f7;">{idx_status}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
