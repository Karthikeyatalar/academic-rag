# Academic RAG Assistant

A Multi-Document Academic Retrieval-Augmented Generation (RAG) assistant implemented with FastAPI, ChromaDB, Sentence-Transformers, and Ollama, featuring page-level citation, a two-tier anti-hallucination guardrail, and a modern, responsive web application.

---

## 🏛️ System Architecture

```
                    ┌─────────────────────────┐
                    │    Web UI (HTML/CSS/JS) │
                    └───────────┬─────────────┘
                                │ Query / Upload
                                ▼
                    ┌─────────────────────────┐
                    │     FastAPI Backend     │
                    │      (app/main.py)      │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Parser Engine │       │ Vector Store  │       │ LLM Generator │
│ (PyMuPDF,     │       │ (ChromaDB +   │       │ (Ollama       │
│  python-docx) │       │  MiniLM-L6-v2)│       │  Llama3 JSON) │
└───────────────┘       └───────┬───────┘       └───────┬───────┘
                                │                       │
            ┌───────────────────┴───────────────────────┴───┐
            ▼                                               ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│       Stage-1 Guardrail      │        │       Stage-2 Guardrail      │
│     Relevance Check (Cosine) │        │  Verbatim Citation Check     │
│   Refuse if distance > 0.60  │        │ Exact Substring & Length >=10│
└──────────────────────────────┘        └──────────────────────────────┘
```

### Two-Tier Anti-Hallucination Guardrail

1. **Stage-1 (Pre-Retrieval Distance Filter)**:
   - Measures cosine distance in embedding space between the user prompt and indexed document chunks.
   - If the best chunk cosine distance exceeds `DISTANCE_THRESHOLD` (0.60) or no documents match, the system rejects the query immediately before querying the LLM, eliminating compute costs and out-of-scope hallucinations.

2. **Stage-2 (Post-Generation Verbatim Quote Verification)**:
   - Inspects the structured JSON output returned by the LLM.
   - Normalizes whitespace and casing, and checks whether `verbatim_quote` (minimum 10 characters) is an exact substring within the retrieved context chunks.
   - If the LLM generates a quote that cannot be located in the source text, the response is flipped to `status: refused`.

---

## 📂 Project Structure

```
academic-rag/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration parameters and thresholds
│   ├── parser.py           # Multi-format document parser (PDF, DOCX, TXT)
│   ├── vector_store.py     # ChromaDB session-isolated vector collections
│   ├── guardrails.py       # Stage-1 and Stage-2 anti-hallucination guardrails
│   ├── generator.py        # Zero-shot JSON prompt engineering & Ollama client
│   ├── main.py             # FastAPI REST API routes and static asset serving
│   └── static/             # Modern Web Frontend
│       ├── index.html      # Responsive Academic UI Shell
│       ├── style.css       # Academic Design System & CSS Variables
│       └── app.js          # Client-side controller (Fetch API, Drag & Drop)
├── data/
│   └── chromadb/           # ChromaDB persistent storage directory
├── tests/
│   ├── __init__.py
│   ├── sample_docs/        # Synthetic documents directory
│   └── test_guardrails.py  # Guardrail test suite (True Positive, Adversarial, Hallucination)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🎨 Interactive Web Application Features

- **Document Dropzone**: Drag-and-drop or browse multiple `.pdf`, `.docx`, and `.txt` files with format badges and file size previews.
- **Session Isolation**: Generate fresh session IDs, view and copy active session badges, or drop Chroma collections with one click.
- **Knowledge Base Telemetry**: Live counter of indexed chunks and grounded documents.
- **Two-Tier Guardrail Inspector**: Visual telemetry badges displaying Stage-1 relevance verification ($\le 0.60$) and Stage-2 verbatim citation verification.
- **Verbatim Quote & Page Citation**: Formatted blockquotes with document name and exact page number.
- **Quick Prompts**: One-click sample questions including factual queries and adversarial out-of-scope tests.
- **Inquiry History**: Chronological log of previous questions with instant review.

---

## ⚙️ Configuration (`app/config.py`)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `CHUNK_SIZE` | `500` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `60` | Overlap size between adjacent chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer dense model |
| `CHROMA_PATH` | `./data/chromadb` | Directory for persistent vector store |
| `LLM_API_URL` | `http://localhost:11434/api/chat` | Ollama chat endpoint |
| `LLM_MODEL` | `llama3:latest` | Ollama model identifier |
| `DISTANCE_THRESHOLD` | `0.60` | Cosine distance refusal cutoff |
| `REFUSAL_TEXT` | `"The uploaded documents do not contain information to answer this question."` | Standard refusal message |

---

## 🚀 Setup & Launching the Website

### 1. Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.ai/) with `llama3:latest` pulled. *(If Ollama is offline, the backend automatically utilizes deterministic extractive fallback from the grounded chunks so the web app remains fully testable out-of-the-box!)*

### 2. Activate Virtual Environment & Install Dependencies

```bash
cd "d:\projects\Machine learning\academic-rag"

# On Windows:
.\.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Start the Web Server

**Option A (Simplest — Windows Batch):**
Double-click `run.bat` or run:
```cmd
run.bat
```

**Option B (Direct execution via virtual environment in PowerShell):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Option C (If activating virtual environment in PowerShell):**
If PowerShell blocks running scripts (`PSSecurityException`), enable script execution for your session first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to:
- 🌐 **Web Application**: `http://127.0.0.1:8000/`
- 📑 **Swagger API Docs**: `http://127.0.0.1:8000/docs`
- 🩺 **System Status**: `http://127.0.0.1:8000/api/status`

---

## 🧪 Running the Tests

Execute the comprehensive test suite verifying both guardrail stages, document parsing, and session isolation:

```powershell
# Via batch:
test.bat

# Or direct Python in PowerShell:
.\.venv\Scripts\python.exe -m pytest tests/test_guardrails.py -v
```

The test suite covers:
- **Test Case 1 (True Positive)**: Valid question answered with page-2 citation and exact quote verification.
- **Test Case 2 (Out-of-Scope Pre-Retrieval Refusal)**: Adversarial question flagged by Stage-1 cosine distance filter.
- **Test Case 3 (Post-Guardrail Hallucination Check)**: LLM quote invention detected and flipped to refused by Stage-2 guardrail.
- **Unit Tests**: Cosine threshold calculations, quote normalization, and unsupported file rejection.

---

## 📡 API Endpoints

### 1. Web Interface
`GET /`
- Serves the interactive HTML5/CSS3/JavaScript web frontend.

### 2. Runtime Status
`GET /api/status`
- Reports ChromaDB configuration, active embedding model, and Ollama connectivity.

### 3. Create Session
`POST /session/create`
- Generates an isolated session ID for document indexing and retrieval.

### 4. Upload Documents
`POST /upload`
- Multipart form upload with `session_id` and `files` (`.pdf`, `.docx`, `.txt`).
- Attaches page numbers to every chunk for precise citation.

### 5. Query Assistant
`POST /query`
- Takes `session_id` and `question`.
- Runs dense retrieval, Stage-1 relevance check, LLM generation, and Stage-2 citation verification.

### 6. Delete Session
`DELETE /session/{session_id}`
- Cleans up and deletes the ChromaDB collection associated with the session.
#   a c a d e m i c - r a g  
 