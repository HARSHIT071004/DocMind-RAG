from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from rag import answer, build_vector_store, is_vector_store_ready, load_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevent sentence-transformers from making network calls on startup
os.environ.setdefault("HF_HUB_OFFLINE", "1")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")

# ── In-memory vector store cache ────────────────────────────────────
_vector_store = None
if is_vector_store_ready():
    try:
        _vector_store = load_vector_store()
    except Exception as e:
        logger.warning("Could not load vector store on startup: %s", e)


# ── DB helpers ───────────────────────────────────────────────────────
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, title TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        role TEXT NOT NULL, content TEXT NOT NULL, tokens INTEGER DEFAULT 0,
        created_at TEXT NOT NULL, FOREIGN KEY (session_id) REFERENCES sessions(id))""")
    conn.commit()
    return conn


def _create_session(title="New chat"):
    sid, now = str(uuid.uuid4()), datetime.now().isoformat()
    conn = _db()
    conn.execute("INSERT INTO sessions VALUES (?,?,?,?)", (sid, title, now, now))
    conn.commit(); conn.close()
    return sid


def _list_sessions():
    conn = _db()
    rows = conn.execute(
        "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def _load_messages(sid):
    conn = _db()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id", (sid,)
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def _save_message(sid, role, content, tokens=0):
    now = datetime.now().isoformat()
    conn = _db()
    conn.execute(
        "INSERT INTO messages (session_id,role,content,tokens,created_at) VALUES (?,?,?,?,?)",
        (sid, role, content, tokens, now))
    conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, sid))
    conn.commit(); conn.close()


def _update_title(sid, title):
    conn = _db()
    conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, sid))
    conn.commit(); conn.close()


def _delete_session(sid):
    conn = _db()
    conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
    conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
    conn.commit(); conn.close()


# ── Routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    return jsonify(_list_sessions())


@app.route("/api/sessions", methods=["POST"])
def create_session():
    title = request.json.get("title", "New chat") if request.json else "New chat"
    sid = _create_session(title)
    return jsonify({"id": sid, "title": title})


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def delete_session(sid):
    _delete_session(sid)
    return jsonify({"ok": True})


@app.route("/api/sessions/<sid>/messages", methods=["GET"])
def get_messages(sid):
    return jsonify(_load_messages(sid))


@app.route("/api/chat", methods=["POST"])
def chat():
    global _vector_store
    data = request.json or {}
    sid = data.get("session_id")
    query = data.get("message", "").strip()
    if not sid or not query:
        return jsonify({"error": "session_id and message required"}), 400

    _save_message(sid, "user", query)

    # Auto-title on first message
    msgs = _load_messages(sid)
    if len(msgs) == 1:
        _update_title(sid, query[:50] + ("..." if len(query) > 50 else ""))

    def generate():
        global _vector_store
        try:
            if _vector_store is None:
                text = "No documents loaded. Upload PDFs and build the vector store first."
            else:
                result = answer(query, _vector_store)
                text = result["answer"]
        except Exception as e:
            logger.error("Answer failed: %s", e, exc_info=True)
            err = str(e)
            text = "Rate-limited. Please wait and try again." if "429" in err else f"Error: {err[:200]}"

        # Stream word by word via SSE
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'token': chunk})}\n\n"

        # Save full answer
        tokens = max(1, int(len(text.split()) * 1.3))
        _save_message(sid, "assistant", text, tokens)
        yield f"data: {json.dumps({'done': True, 'full': text})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files"}), 400
    os.makedirs("Artifacts", exist_ok=True)
    saved = []
    for f in files:
        if f.filename.endswith(".pdf"):
            path = os.path.join("Artifacts", f.filename)
            f.save(path)
            saved.append(f.filename)
    return jsonify({"saved": saved})


@app.route("/api/vector-store/build", methods=["POST"])
def build_vs():
    global _vector_store
    try:
        _vector_store = build_vector_store()
        return jsonify({"ok": True})
    except Exception as e:
        logger.error("Build failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/vector-store/status", methods=["GET"])
def vs_status():
    return jsonify({"ready": is_vector_store_ready()})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
