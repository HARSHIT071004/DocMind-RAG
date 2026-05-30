# app.py — AI Chat with RAG backend
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


# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chat",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ───────────────────────────────────────────────────
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


def _stream_text(text: str):
    """Yield words with a small delay to simulate streaming."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.025)


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





# ── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp, p, div, h1, h2, h3, h4, h5, h6, textarea, input, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .material-symbols-rounded, .material-icons, [class*="material-symbols"], [class*="icon"], svg, span.material-symbols-rounded {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    #MainMenu, footer, .stDeployButton { display: none !important; }
    .stApp > header {
        background: transparent !important;
        border: none !important;
    }

    /* ══════════════════════════════════════════════════════════════
       HIDE CHAT AVATARS / ICONS
       ══════════════════════════════════════════════════════════════ */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatar"],
    [data-testid="stChatMessage"] [data-testid="stImage"],
    [data-testid="stChatMessage"] .stImage,
    [data-testid="stChatMessage"] img,
    [data-testid="stChatMessage"] svg,
    .stChatMessage img,
    .stChatMessage svg,
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        max-width: 0 !important;
        max-height: 0 !important;
        visibility: hidden !important;
        overflow: hidden !important;
        position: absolute !important;
    }
    [data-testid="stChatMessage"] > [data-testid*="avatar"],
    [data-testid="stChatMessage"] > div > [data-testid*="avatar"],
    [data-testid="stChatMessage"] > div:first-child > div:first-child:has(img),
    [data-testid="stChatMessage"] > div:first-child > div:first-child:has(svg) {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stChatMessage"] > div:first-child {
        gap: 0 !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .stApp, .main, .main .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stChatMessage"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background-color: #0a0a0c !important;
    }

    .main .block-container {
        max-width: 48rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding: 0 1.5rem 7rem !important;
    }
    [data-testid="stBottomBlockContainer"] {
        max-width: 54rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #1a1a1f; border-radius: 2px; }

    /* ══════════════════════════════════════════════════════════════
       SIDEBAR
       ══════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: none !important;
        width: 260px !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }

    /* Push first sidebar element to the top */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0 !important;
        padding-bottom: 1rem !important;
        background-color: #171717 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child,
    section[data-testid="stSidebar"] .element-container:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    section[data-testid="stSidebar"] *::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        background: transparent !important;
    }
    section[data-testid="stSidebar"] * {
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }

    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
    section[data-testid="stSidebar"] [data-testid="stFileUploader"],
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #171717 !important;
    }

    /* ── Sidebar collapse button — clean default look ── */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        background: transparent !important;
        border: none !important;
        color: #a1a1aa !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="collapsedControl"]:hover {
        background: rgba(255,255,255,0.06) !important;
        color: #ececec !important;
    }

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

    /* ══════════════════════════════════════════════════════════════
       SIDEBAR BUTTONS
       ══════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] .stButton button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #ececec !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 400 !important;
        padding: 8px 12px !important;
        min-height: 40px !important;
        height: auto !important;
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
        transition: background 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .stButton button p {
        font-size: 0.88rem !important;
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #2f2f2f !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background: #212121 !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        align-items: center !important;
        margin-bottom: 2px !important;
        border-radius: 8px !important;
        transition: background 0.2s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover {
        background: #212121 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton button {
        background: transparent !important;
        border-radius: 8px 0 0 8px !important;
        padding-right: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton button p {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        width: 100% !important;
        display: block !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton button {
        background: transparent !important;
        border-radius: 0 8px 8px 0 !important;
        color: #8e8e8e !important;
        justify-content: center !important;
        padding: 8px !important;
        opacity: 0;
        transition: opacity 0.2s ease, color 0.2s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover [data-testid="column"]:last-child .stButton button {
        opacity: 1;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton button:hover {
        color: #ef4444 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(45,212,191,0.05) !important;
        border: 1px solid #1e2e2c !important;
        border-radius: 6px !important;
        font-size: 0.76rem !important;
    }

    .sb-brand {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0 !important; margin: 0 !important;
        border-bottom: none !important;
    }
    .sb-brand-icon {
        width: 28px; height: 28px; border-radius: 7px;
        background: linear-gradient(135deg, #2dd4bf, #14b8a6);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.65rem; font-weight: 800; color: #09090b;
        letter-spacing: -0.03em; flex-shrink: 0;
    }
    .sb-brand-name { font-size: 0.85rem; font-weight: 700; color: #d4d4d8; letter-spacing: -0.02em; }
    .sb-brand-sub  { font-size: 0.62rem; color: #3f3f46; font-weight: 400; margin-top: 1px; }
    .sb-section    { font-size: 0.62rem; color: #3f3f46; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin: 1rem 0 0.4rem; }
    .sb-divider    { height: 1px; background: #1a1a1f; margin: 0.8rem 0; }
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

    .chat-history-item {
        padding: 0.45rem 0.65rem;
        border-radius: 6px;
        cursor: pointer;
        transition: background 0.15s ease;
        margin-bottom: 2px;
    }
    .chat-history-item:hover {
        background: rgba(255,255,255,0.04);
    }
    .chat-history-item.active {
        background: rgba(45,212,191,0.06);
    }
    .chat-history-title {
        font-size: 0.78rem;
        color: #a1a1aa;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.4;
    }
    .chat-history-item.active .chat-history-title {
        color: #d4d4d8;
    }
    .chat-history-date {
        font-size: 0.6rem;
        color: #3f3f46;
        margin-top: 1px;
    }

    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 15vh;
        margin-bottom: 2.5rem;
        animation: fadeIn 0.5s ease;
    }
    .welcome-icon {
        width: 56px; height: 56px; border-radius: 50%;
        background: #ffffff;
        display: flex; align-items: center; justify-content: center;
        color: #0a0a0c;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    }
    .welcome-title {
        font-size: 1.85rem;
        font-weight: 500;
        color: #ececec;
        letter-spacing: -0.02em;
        margin: 0;
        text-align: center;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        animation: slideIn 0.25s ease;
        max-width: 100% !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.88rem !important;
        line-height: 1.75 !important;
        color: #d4d4d8 !important;
    }

    .user-msg {
        display: flex;
        justify-content: flex-end;
        margin: 0.4rem 0;
    }
    .user-msg-bubble {
        background: #1e1e28;
        border-radius: 16px 16px 4px 16px;
        padding: 0.7rem 1rem;
        max-width: 75%;
        animation: slideIn 0.2s ease;
    }
    .user-msg-bubble p {
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.65;
        color: #e4e4e7;
    }
    .assistant-msg {
        margin: 0.4rem 0;
        animation: slideIn 0.25s ease;
    }
    .assistant-msg p {
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.75;
        color: #a1a1aa;
    }

    .typing-indicator {
        display: flex; align-items: center; gap: 5px;
        padding: 0.6rem 0;
    }
    .typing-indicator span {
        width: 7px; height: 7px; border-radius: 50%;
        background: #3f3f46; display: inline-block;
        animation: typingDot 1.4s ease-in-out infinite;
    }
    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typingDot {
        0%, 60%, 100% { opacity: 0.3; transform: scale(0.8); }
        30% { opacity: 1; transform: scale(1); }
    }

    /* ══════════════════════════════════════════════════════════════
       CHAT INPUT — PERPLEXITY STYLE
       ══════════════════════════════════════════════════════════════ */
    [data-testid="stBottom"] {
        background: linear-gradient(to top, #0a0a0c 70%, transparent) !important;
        padding-top: 1.5rem !important;
        padding-bottom: 1.75rem !important;
        border-top: none !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
    }
    [data-testid="stBottomBlockContainer"] {
        padding-bottom: 0 !important;
    }
    [data-testid="stChatInput"] {
        background: transparent !important;
        padding: 0 !important;
    }
    [data-testid="stChatInput"] > div {
        background: #1c1c1f !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 20px !important;
        padding: 6px 10px 6px 20px !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35), 0 1px 4px rgba(0,0,0,0.2) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        max-width: 100% !important;
        align-items: flex-end !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(255,255,255,0.18) !important;
        box-shadow: 0 4px 32px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.06) !important;
    }
    div[data-testid="stChatInput"] div[data-baseweb="textarea"],
    div[data-testid="stChatInput"] div[data-baseweb="textarea"] > div {
        min-height: 60px !important;
        background: transparent !important;
    }
    textarea[data-testid="stChatInputTextArea"] {
        background: transparent !important;
        color: #ececec !important;
        font-size: 0.97rem !important;
        padding: 16px 0 !important;
        /* height is set inline by Streamlit — overridden via JS below */
        min-height: 60px !important;
        max-height: 200px !important;
        caret-color: #ececec !important;
        line-height: 1.6 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        resize: none !important;
        overflow-y: auto !important;
    }
    textarea[data-testid="stChatInputTextArea"]:focus,
    textarea[data-testid="stChatInputTextArea"]:focus-visible {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    textarea[data-testid="stChatInputTextArea"]::placeholder {
        color: #52525b !important;
        font-size: 0.97rem !important;
    }
    [data-testid="stChatInput"] button {
        background: #e4e4e7 !important;
        border-radius: 12px !important;
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 0 6px 8px !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
        border: none !important;
        flex-shrink: 0 !important;
        align-self: flex-end !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: #ffffff !important;
        transform: scale(1.04) !important;
    }
    [data-testid="stChatInput"] button:active {
        transform: scale(0.96) !important;
    }
    [data-testid="stChatInput"] button svg {
        color: #09090b !important;
        fill: #09090b !important;
        width: 17px !important;
        height: 17px !important;
    }

    .msg-separator {
        height: 1px;
        background: #141418;
        margin: 0.2rem 0;
    }

    .stApp > header {
        height: 0 !important;
        min-height: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Force chat input height via JS (Streamlit sets inline style that overrides CSS) ──
st.markdown("""
<script>
(function() {
    function fixTextarea() {
        const ta = document.querySelector('textarea[data-testid="stChatInputTextArea"]');
        if (!ta) return;
        // Remove Streamlit's inline height so our CSS min-height takes over
        ta.style.removeProperty('height');
        ta.style.minHeight = '60px';
        ta.style.maxHeight = '200px';
        ta.style.overflowY = 'auto';
        ta.addEventListener('input', function() {
            ta.style.removeProperty('height');
        }, { once: false });
    }
    const observer = new MutationObserver(fixTextarea);
    observer.observe(document.body, { childList: true, subtree: true });
    fixTextarea();
})();
</script>
""", unsafe_allow_html=True)

# ── Render chat UI ──────────────────────────────────────────────────

if not st.session_state.messages:
    # Welcome / empty state
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
        </div>
        <div class="welcome-title">How can I help you today?</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render chat history with custom HTML for alignment
    for i, msg in enumerate(st.session_state.messages):
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
        # Subtle separator
        if i < len(st.session_state.messages) - 1:
            st.markdown('<div class="msg-separator"></div>', unsafe_allow_html=True)


# ── Chat input (pinned to bottom) ──────────────────────────────────
prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.pending_query = prompt

# ── Process query ───────────────────────────────────────────────────
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

    # Typing indicator, then answer
    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        '<div class="typing-indicator"><span></span><span></span><span></span></div>',
        unsafe_allow_html=True,
    )

    ai_text = ""
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

    # Clear typing indicator, stream the answer
    typing_placeholder.empty()
    st.write_stream(_stream_text(ai_text))

    # Token count for DB
    prompt_tokens = _estimate_tokens(query)
    completion_tokens = _estimate_tokens(ai_text)
    total_tokens = prompt_tokens + completion_tokens

    # Save to DB
    _save_message(st.session_state.current_session, "assistant", ai_text, total_tokens)
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_text,
        "tokens": total_tokens,
    })


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
        </div>
        <div>
            <div class="sb-brand-name">DocMind</div>
            <div class="sb-brand-sub">RAG Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New chat button
    if st.button("New chat", use_container_width=True, type="primary"):
        _new_chat()
        st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

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
    if st.button("Build vector store", use_container_width=True, type="primary", key="build_vs"):
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

    # Chat history
    st.markdown('<div class="sb-section">Recent chats</div>', unsafe_allow_html=True)
    sessions = _list_sessions()
    for sess in sessions:  # Show all sessions
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
                st.rerun()
        with col2:
            if st.button("x", key=f"del_{sess['id']}"):
                _delete_session(sess["id"])
                if sess["id"] == st.session_state.current_session:
                    remaining = _list_sessions()
                    if remaining:
                        _switch_session(remaining[0]["id"])
                    else:
                        _new_chat()
                st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Clear current chat
    if st.session_state.messages:
        if st.button("Clear conversation", use_container_width=True, key="clear_chat"):
            _delete_session(st.session_state.current_session)
            _new_chat()
            st.rerun()



