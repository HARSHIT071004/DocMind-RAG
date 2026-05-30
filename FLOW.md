# Question-Answering System using RAG — Complete Flow Document

## 1. What Is This Project?

A **Retrieval-Augmented Generation (RAG)** system that:
- Ingests a PDF document (`Artifacts/Articles.pdf`)
- Creates vector embeddings from its content
- Lets users ask natural-language questions
- Retrieves the most relevant chunks from the PDF
- Passes them as context to an LLM (Llama3 via Groq) to generate a precise answer

Built with **Streamlit** (UI) + **LangChain** (orchestration).

---

## 2. Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend / UI | Streamlit |
| Orchestration | LangChain |
| LLM Inference | Groq (Llama3-8b-8192) |
| Embeddings | Google Generative AI (`models/embedding-001`) |
| Vector Store | FAISS (local, CPU) |
| PDF Loading | PyPDFDirectoryLoader (via pypdf/PyPDF2) |
| Text Splitting | RecursiveCharacterTextSplitter |
| Environment | python-dotenv |

---

## 3. Project Structure

```
Question-Answering-System-using-RAG/
├── app.py                     # THE ONLY source file — full application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── LICENSE                    # GNU GPL v3
├── .gitignore                 # Git ignore rules
├── Artifacts/
│   └── Articles.pdf           # Source PDF document
└── FLOW.md                    # This file
```

---

## 4. Complete Execution Flow (Step by Step)

### PHASE 0 — Startup & Initialization

1. **Environment Loading** (`dotenv`):
   - Reads `.env` file from the project root
   - Extracts `API_KEY` → set as `groq_api_key`
   - Extracts `GOOGLE_API_KEY` → set as `OS env var "GOOGLE_API_KEY"`

2. **Streamlit UI Setup**:
   - Page title: *"Document Question Answering System"*
   - Layout: wide
   - Shows caption: *"Initially Ingest the Data into Vector Store and then ask questions."*

3. **LLM Initialization**:
   ```python
   llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
   ```
   - Connects to Groq cloud API
   - Uses the `Llama3-8b-8192` model (8 billion parameters, 8192 context window)

4. **Prompt Template**:
   ```python
   prompt_template = """
   Answer the questions based on the provided context only.
   Please provide the most accurate response based on the question
   <context>
   {context}
   <context>
   Questions:{input}
   """
   ```
   - Instructs the LLM to answer **only** from the given context
   - `{context}` placeholder → retrieved document chunks
   - `{input}` placeholder → user's question

---

### PHASE 1 — Document Ingestion (triggered by button click)

User clicks **"Ingest the Data into Vector Store"** button → calls `vector_embedding()`.

```
vector_embedding()
│
├── Check: Already ingested? (st.session_state.vectors exists)
│   └── If yes → skip (idempotent)
│
├── Step 1: Create Embedding Model
│   └── GoogleGenerativeAIEmbeddings(model="models/embedding-001")
│   └── Connects to Google's embedding API
│
├── Step 2: Load PDF Documents
│   └── PyPDFDirectoryLoader("./Artifacts")
│   └── Scans ./Artifacts/ directory for all PDFs
│   └── Returns list of Document objects (page_content, metadata)
│   └── Loads Articles.pdf
│
├── Step 3: Split Documents into Chunks
│   └── RecursiveCharacterTextSplitter(
│   │       chunk_size=1000,     # 1000 characters per chunk
│   │       chunk_overlap=200    # 200 character overlap between chunks
│   │   )
│   └── Takes docs[:20] (first 20 pages only — LIMITATION)
│   └── Returns smaller Document chunks
│
└── Step 4: Create & Store Vector Embeddings
    └── FAISS.from_documents(final_documents, st.session_state.embeddings)
    └── For each chunk:
    │       ├── Generate embedding vector via Google API
    │       └── Store in FAISS index (in-memory)
    └── Saved to: st.session_state.vectors
    └── UI shows: "Data is Ingested in vector store database..."
```

**Result**: A FAISS vector index containing embeddings of up to 20 PDF pages, stored in Streamlit's session state.

---

### PHASE 2 — Question Answering (triggered by user typing a question)

User types a question in the text input → `prompt1` becomes non-empty.

```
User enters question
│
├── Check: st.session_state.vectors exists?
│   └── NO  → Show error: "Please Ingest Data First..."
│   └── YES → Proceed
│
├── Step 1: Create Document Chain
│   └── create_stuff_documents_chain(llm, prompt)
│   └── "Stuff" method: stuffs all retrieved docs into the prompt's {context}
│   └── Combines LLM + prompt template into a chain
│
├── Step 2: Create Retriever
│   └── st.session_state.vectors.as_retriever()
│   └── Returns a retriever that does similarity search on FAISS
│
├── Step 3: Create Retrieval Chain
│   └── create_retrieval_chain(retriever, document_chain)
│   └── Links retriever → document_chain
│
├── Step 4: Start Timer
│   └── time.process_time() → records CPU time
│
├── Step 5: Invoke Chain
│   └── retrieval_chain.invoke({'input': prompt1})
│   └── Internally:
│   │       ├── Embeds user question via Google Embeddings
│   │       ├── FAISS similarity search → top-k relevant chunks
│   │       ├── Formats chunks into {context} in prompt
│   │       ├── Sends prompt + context to Groq LLM
│   │       └── LLM returns answer based solely on context
│   └── Returns: {"input": ..., "context": ..., "answer": "..."}
│
├── Step 6: Display Results
│   ├── Response time: time.process_time() - start
│   └── Answer: response['answer']
│
└── Error handling: Any exception caught → shows "Please Ingest Data First"
```

---

## 5. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT APP (app.py)                      │
│                                                                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  User Clicks  │───▶│ vector_embedding │───▶│  FAISS Vector │  │
│  │ "Ingest Data" │    │   () function    │    │    Store      │  │
│  └──────────────┘    │                  │    │ (in-memory)   │  │
│                      │ 1. GoogleEmbed   │    └───────┬───────┘  │
│                      │ 2. Load PDF      │            │           │
│                      │ 3. Split chunks  │            │           │
│                      │ 4. FAISS.from_   │            │           │
│                      │    documents()   │            │           │
│                      └──────────────────┘            │           │
│                                                      │           │
│  ┌──────────────┐    ┌──────────────────┐            │           │
│  │ User Types   │───▶│ retrieval_chain  │◀───────────┘           │
│  │ a Question   │    │ .invoke(...)     │                        │
│  └──────────────┘    │                  │                        │
│                      │ ┌──────────────┐ │                        │
│                      │ │  Retriever   │ │  FAISS similarity      │
│                      │ │ (as_retriever)│─┼── search on question  │
│                      │ └──────┬───────┘ │                        │
│                      │        │         │                        │
│                      │ ┌──────▼───────┐ │                        │
│                      │ │ Document     │ │  Stuff docs into       │
│                      │ │ Chain (Stuff)│─┼── prompt template      │
│                      │ └──────┬───────┘ │                        │
│                      │        │         │                        │
│                      │ ┌──────▼───────┐ │                        │
│                      │ │   Groq LLM   │ │  Generate answer       │
│                      │ │ (Llama3-8b)  │─┼── from context         │
│                      │ └──────────────┘ │                        │
│                      └──────────────────┘                        │
│                           │                                       │
│                           ▼                                       │
│                     Display Answer                                │
│                     + Response Time                               │
└─────────────────────────────────────────────────────────────────┘

                   EXTERNAL SERVICES
┌─────────────────────┐    ┌─────────────────────┐
│   Google Generative │    │   Groq Cloud API    │
│   AI (Embeddings)   │    │   (Llama3-8b-8192)  │
│   models/embedding- │    │                     │
│   001               │    │                     │
└─────────────────────┘    └─────────────────────┘
```

---

## 6. Data Flow

```
PDF File (Articles.pdf)
        │
        ▼
PyPDFDirectoryLoader ────► List[Document] (raw text with metadata)
        │
        ▼
RecursiveCharacterTextSplitter ────► List[Document] (chunks of 1000 chars)
        │                              (only first 20 pages)
        ▼
GoogleGenerativeAIEmbeddings ────► Vector Embeddings (768-dim)
        │
        ▼
FAISS.from_documents() ────► FAISS Index (in-memory)
        │
        ▼  (when user asks question)

User Question
        │
        ▼
FAISS similarity_search ────► Top-k relevant chunks (as context)
        │
        ▼
Prompt Template ────► {context} + {input} filled in
        │
        ▼
Groq LLM (Llama3-8b-8192) ────► Generated Answer
        │
        ▼
Streamlit UI ────► Display answer + response time
```

---

## 7. Key Limitations (Documented)

1. **Only first 20 pages**: `docs[:20]` — only the first 20 pages of the PDF are processed.
2. **In-memory vector store**: FAISS is not persisted to disk — data is lost when the app restarts.
3. **Single PDF**: Only `Artifacts/Articles.pdf` is loaded (no multi-doc support).
4. **No source citation**: The answer is shown without citing which chunk/page it came from.
5. **No streaming**: The LLM response is returned all at once, not token-by-token.

---

## 8. How to Run

```bash
# 1. Create & activate environment (conda recommended)
conda create -n rag python=3.10 -y
conda activate rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up .env file with:
#    API_KEY=<your_groq_api_key>
#    GOOGLE_API_KEY=<your_google_api_key>

# 4. Run the app
streamlit run app.py
```

---

## 9. Dependencies (requirements.txt)

| Package | Role |
|---------|------|
| `faiss-cpu` | Vector similarity search |
| `groq` | Groq API client |
| `langchain-groq` | LangChain Groq integration |
| `PyPDF2` | PDF parsing |
| `langchain_google_genai` | Google AI embeddings |
| `langchain` | Core RAG orchestration |
| `streamlit` | Web UI |
| `langchain_community` | FAISS, PDF loader, etc. |
| `python-dotenv` | .env loading |
| `pypdf` | PDF parsing (used by LangChain) |
| `google-cloud-aiplatform>=1.38` | Google Cloud AI SDK |

---

## 10. Environment Variables (`.env`)

```
API_KEY=<Groq API Key>          # Used for LLM inference (Llama3-8b-8192)
GOOGLE_API_KEY=<Google API Key> # Used for embeddings (models/embedding-001)
```
