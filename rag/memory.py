import sqlite3
from datetime import datetime

from rag.config import settings


DB_PATH = "chat_history.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, title TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id))""")
    conn.commit()
    return conn


def create_session(title: str = "New chat") -> str:
    import uuid
    sid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (sid, title, now, now),
    )
    conn.commit()
    conn.close()
    return sid


def save_message(session_id: str, role: str, content: str) -> None:
    now = datetime.now().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now),
    )
    conn.execute(
        "UPDATE sessions SET updated_at=? WHERE id=?",
        (now, session_id),
    )
    conn.commit()
    conn.close()


def get_recent_history(session_id: str, max_turns: int = 4) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT role, content FROM messages
           WHERE session_id=? ORDER BY id DESC LIMIT ?""",
        (session_id, max_turns * 2),
    ).fetchall()
    conn.close()
    messages = [{"role": r["role"], "content": r["content"]} for r in rows]
    messages.reverse()
    return messages


def format_history_for_prompt(history: list[dict], max_tokens: int = 1000) -> str:
    lines = []
    token_estimate = 0
    for msg in reversed(history):
        label = "User" if msg["role"] == "user" else "Assistant"
        line = f"{label}: {msg['content']}"
        tokens = len(line) // 4
        if token_estimate + tokens > max_tokens:
            break
        lines.insert(0, line)
        token_estimate += tokens
    return "\n".join(lines)
