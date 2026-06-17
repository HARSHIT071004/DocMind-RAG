<div align="center">
  <h1>DocMind-RAG</h1>
  <p><strong>Production-Grade Hybrid RAG System for Document Question Answering</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11"/>
    <img src="https://img.shields.io/badge/flask-3.1-green" alt="Flask"/>
    <img src="https://img.shields.io/badge/FAISS+BM25-Hybrid-orange" alt="Hybrid Search"/>
    <img src="https://img.shields.io/badge/docker-ready-2496ED" alt="Docker Ready"/>
    <img src="https://img.shields.io/badge/license-GPLv3-red" alt="License"/>
  </p>
</div>

---

## Architecture

DocMind-RAG is a full-stack Retrieval-Augmented Generation system combining dense and sparse retrieval with advanced reranking, confidence scoring, and corrective RAG for accurate document-based question answering.

```
User Query → Query Cache → Hybrid Retrieval → MMR Diversify
→ Cross-Encoder Rerank → CRAG (Corrective RAG) → LLM Synthesis
→ Confidence Scoring → SSE Streamed Response
```

### Components

| Layer | Technology |
|---|---|
| **Ingestion** | PyMuPDF (pdf→text), Small-to-Big chunking with parent-child mapping |
| **Dense Retrieval** | FAISS (all-MiniLM-L6-v2 embeddings) |
| **Sparse Retrieval** | BM25Okapi (keyword-based) |
| **Fusion** | Weighted RRF (Reciprocal Rank Fusion) |
| **Diversification** | MMR (Maximum Marginal Relevance) |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM-L-2-v2) |
| **LLM** | OpenRouter (multi-model with fallbacks) |
| **Corrective RAG** | Confidence threshold → re-retrieve on low confidence |
| **Caching** | LRU + TTL-based query cache |
| **Memory** | SQLite conversation history (per-session) |
| **API** | Flask + Waitress with SSE streaming, rate limiting |
| **Logging** | Structured logging (structlog) |
| **Container** | Docker (python:3.11-slim) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Local Setup

```bash
git clone https://github.com/HARSHIT071004/DocMind-RAG.git
cd DocMind-RAG

python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/mac: source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Run

```bash
# Build the vector index (ingest PDFs from Artifacts/)
python -c "from rag import build_index; build_index()"

# Start the API server
python server.py
```

The server starts on **http://localhost:5000**.

### Docker

```bash
docker build -t docmind-rag .
docker run -p 5000:5000 -e OPENROUTER_API_KEY=sk-or-v1-... docmind-rag
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/chat` | POST | Ask a question (returns SSE stream) |
| `/history/<session_id>` | GET | Get conversation history |

### `/chat` Request

```json
{
  "question": "What technical skills does the candidate have?",
  "session_id": "user-abc-123"
}
```

### SSE Response

```
data: {"type": "token", "content": "The candidate..."}
data: {"type": "confidence", "value": 0.87}
data: {"type": "done"}
```

---

## Evaluation

Run RAGAS benchmarks to assess retrieval and generation quality:

```bash
python -m rag.evaluation
```

Output: `evaluation_results.json` with metrics:

| Metric | Description |
|---|---|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevancy | How relevant is the answer to the question? |
| Context Precision | Are all retrieved chunks relevant? |
| Context Recall | Were all necessary chunks retrieved? |

---

## Project Structure

```
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── server.py                    # Flask API + SSE
├── templates/
│   └── index.html               # Web UI
├── rag/
│   ├── __init__.py
│   ├── config.py                # Pydantic settings
│   ├── ingestion.py             # PDF chunking
│   ├── hybrid_index.py          # FAISS + BM25 build/load
│   ├── retriever.py             # Hybrid retrieval + MMR + reranker
│   ├── pipeline.py              # answer() orchestrator
│   ├── cache.py                 # Query cache with TTL
│   ├── memory.py                # SQLite conversation memory
│   └── evaluation.py            # RAGAS evaluation pipeline
└── Artifacts/                   # Place PDFs here before indexing
```

---

## License

Distributed under the GNU General Public License v3.0. See `LICENSE` for details.
