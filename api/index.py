from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logging.basicConfig(level=logging.INFO, format="%(message)s")

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["20 per minute"],
    storage_uri="memory://",
)

_vector_store = None
_db_path = "/tmp/chat_history.db"


def _get_db():
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, title TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL, role TEXT NOT NULL,
        content TEXT NOT NULL, created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id))""")
    conn.commit()
    return conn


def _get_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    try:
        from rag import index_exists, load_index
        if index_exists():
            _vector_store = load_index()
    except Exception as e:
        logging.warning("vector_store_load_failed: %s", e)
    return _vector_store


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    conn = _get_db()
    rows = conn.execute(
        "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return jsonify([{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows])


@app.route("/api/sessions", methods=["POST"])
def create_session_api():
    title = request.json.get("title", "New chat") if request.json else "New chat"
    sid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    conn = _get_db()
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (sid, title, now, now),
    )
    conn.commit()
    conn.close()
    return jsonify({"id": sid, "title": title})


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def delete_session_api(sid):
    conn = _get_db()
    conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
    conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/sessions/<sid>/messages", methods=["GET"])
def get_messages_api(sid):
    conn = _get_db()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
        (sid,),
    ).fetchall()
    conn.close()
    return jsonify([{"role": r[0], "content": r[1]} for r in rows])


@app.route("/api/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    data = request.json or {}
    sid = data.get("session_id")
    query = data.get("message", "").strip()

    if not sid or not query:
        return jsonify({"error": "session_id and message required"}), 400

    now = datetime.now().isoformat()
    conn = _get_db()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (sid, "user", query, now),
    )
    conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, sid))
    msgs = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", (sid,)
    ).fetchone()[0]
    if msgs <= 2:
        conn.execute(
            "UPDATE sessions SET title=? WHERE id=?",
            (query[:50] + ("..." if len(query) > 50 else ""), sid),
        )
    conn.commit()
    conn.close()

    def generate():
        vs = _get_vector_store()
        try:
            if vs is None:
                text = "No documents loaded. Upload PDFs and build the vector store first."
                confidence = 0.0
            else:
                from rag import answer
                result = answer(query, vs, session_id=sid)
                text = result.get("answer", "No answer generated.")
                confidence = result.get("confidence", 0.0)
        except Exception as e:
            logging.error("answer_failed: %s", e)
            text = f"Error: {str(e)[:200]}"
            confidence = 0.0

        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True, 'full': text, 'confidence': confidence})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files"}), 400
    os.makedirs("/tmp/Artifacts", exist_ok=True)
    saved = []
    for f in files:
        if f.filename and f.filename.endswith(".pdf"):
            path = os.path.join("/tmp/Artifacts", f.filename)
            f.save(path)
            saved.append(f.filename)
    return jsonify({"saved": saved})


@app.route("/api/vector-store/status", methods=["GET"])
def vs_status():
    try:
        from rag import index_exists
        ready = index_exists()
    except Exception:
        ready = False
    return jsonify({"ready": ready, "building": False})


@app.route("/api/vector-store/build", methods=["POST"])
def build_vs():
    global _vector_store
    try:
        from rag import build_index
        _vector_store = build_index()
        return jsonify({"ok": True, "message": "Build complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vector-store/delete", methods=["POST"])
def delete_vs():
    global _vector_store
    import shutil
    for path in ["/tmp/vector_store", "/tmp/Artifacts"]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception:
                pass
    _vector_store = None
    return jsonify({"ok": True})


handler = app
