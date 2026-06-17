from __future__ import annotations

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import structlog
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from rag import answer, build_index, index_exists, load_index
from rag.memory import create_session, get_recent_history, save_message

logging.basicConfig(level=logging.INFO, format="%(message)s")
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)

os.environ.setdefault("HF_HUB_OFFLINE", "1")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["20 per minute"],
    storage_uri="memory://",
)

_index_lock = ThreadPoolExecutor(max_workers=1)
_index_future = None
_vector_store = None

if index_exists():
    try:
        _vector_store = load_index()
        logger.info("vector_store_loaded")
    except Exception as e:
        logger.warning("vector_store_load_failed", error=str(e))


def _request_id() -> str:
    return request.headers.get("X-Request-Id") or str(uuid.uuid4())


@app.before_request
def _before_request():
    request.rid = _request_id()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request.rid)


@app.after_request
def _after_request(response):
    response.headers["X-Request-Id"] = request.rid
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "vector_store_ready": index_exists(),
        "vector_store_loaded": _vector_store is not None,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    conn = _db()
    rows = conn.execute(
        "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return jsonify([{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows])


@app.route("/api/sessions", methods=["POST"])
def create_session_api():
    title = request.json.get("title", "New chat") if request.json else "New chat"
    sid = create_session(title)
    return jsonify({"id": sid, "title": title})


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def delete_session_api(sid):
    conn = _db()
    conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
    conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/sessions/<sid>/messages", methods=["GET"])
def get_messages_api(sid):
    msgs = get_recent_history(sid, max_turns=100)
    return jsonify(msgs)


@app.route("/api/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    global _vector_store
    data = request.json or {}
    sid = data.get("session_id")
    query = data.get("message", "").strip()

    if not sid or not query:
        return jsonify({"error": "session_id and message required"}), 400

    save_message(sid, "user", query)

    msgs = get_recent_history(sid, max_turns=1)
    if len(msgs) <= 2:
        conn = _db()
        conn.execute("UPDATE sessions SET title=? WHERE id=?", (query[:50] + ("..." if len(query) > 50 else ""), sid))
        conn.commit()
        conn.close()

    def generate():
        try:
            if _vector_store is None:
                text = "No documents loaded. Upload PDFs and build the vector store first."
                confidence = 0.0
            else:
                result = answer(query, _vector_store, session_id=sid)
                text = result.get("answer", "No answer generated.")
                confidence = result.get("confidence", 0.0)

        except Exception as e:
            logger.error("answer_failed", error=str(e))
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
    os.makedirs("Artifacts", exist_ok=True)
    saved = []
    for f in files:
        if f.filename and f.filename.endswith(".pdf"):
            path = os.path.join("Artifacts", f.filename)
            f.save(path)
            saved.append(f.filename)
    return jsonify({"saved": saved})


@app.route("/api/vector-store/status", methods=["GET"])
def vs_status():
    global _index_future
    return jsonify({
        "ready": index_exists(),
        "building": _index_future is not None and not _index_future.done(),
    })


@app.route("/api/vector-store/build", methods=["POST"])
def build_vs():
    global _vector_store, _index_future

    if _index_future and not _index_future.done():
        return jsonify({"error": "Build already in progress"}), 409

    def _rebuild():
        global _vector_store
        logger.info("rebuild_started")
        _vector_store = build_index()
        logger.info("rebuild_complete", chunks=len(_vector_store.chunks))
        return _vector_store

    _index_future = _index_lock.submit(_rebuild)
    return jsonify({"ok": True, "message": "Build started in background"})


@app.route("/api/build-status", methods=["GET"])
def build_status():
    global _index_future
    if _index_future is None:
        return jsonify({"status": "idle"})
    if _index_future.done():
        exc = _index_future.exception()
        if exc:
            return jsonify({"status": "failed", "error": str(exc)})
        return jsonify({"status": "completed"})
    return jsonify({"status": "building"})


def _db():
    import sqlite3
    conn = sqlite3.connect("chat_history.db")
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


if __name__ == "__main__":
    import waitress
    logger.info("starting_server", port=5000)
    waitress.serve(app, host="0.0.0.0", port=5000, threads=8)
