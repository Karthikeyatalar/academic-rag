import logging
import os
import secrets
import string
from typing import List, Optional
import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.generator import generate_answer
from app.guardrails import check_relevance, verify_citation
from app.parser import extract_document_chunks
from app.vector_store import delete_collection, index_chunks, retrieve_top_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Academic RAG Assistant",
    description="Multi-Document Academic RAG Assistant with strict citation and two-tier anti-hallucination guardrail.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class SessionCreateResponse(BaseModel):
    session_id: str = Field(..., description="Unique 8-character session identifier")


class UploadResponse(BaseModel):
    session_id: str
    chunks_indexed: int
    files_processed: int


class QueryRequest(BaseModel):
    session_id: str
    question: str


class QueryResponse(BaseModel):
    status: str = Field(..., description="'answered' or 'refused'")
    answer: str = Field(..., description="Concise factual answer or refusal text")
    verbatim_quote: str = Field(..., description="Exact sentence copied verbatim from the context")
    cited_page: int = Field(..., description="Page number of the source document")
    source_document: str = Field(..., description="Filename of the source document")


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool


def _generate_session_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_index():
    """Serves the academic web frontend."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Academic RAG Assistant API is online. /docs for OpenAPI specifications."}


@app.get("/api/status")
def get_system_status():
    """Returns runtime health and configuration metadata."""
    ollama_online = False
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        ollama_online = (r.status_code == 200)
    except Exception:
        ollama_online = False

    return {
        "status": "online",
        "database": "SQLite 3",
        "db_path": config.SQLITE_DB_PATH,
        "ollama_connected": ollama_online,
        "embedding_model": config.EMBEDDING_MODEL,
        "distance_threshold": config.DISTANCE_THRESHOLD,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


@app.post("/session/create", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    """Generates and returns a random 8-character session_id."""
    session_id = _generate_session_id(8)
    return SessionCreateResponse(session_id=session_id)


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    session_id_query: Optional[str] = Query(None, alias="session_id"),
) -> UploadResponse:
    """
    Ingests, parses, and indexes uploaded documents into ChromaDB.
    Returns the total count of chunks indexed.
    """
    effective_session_id = session_id or session_id_query
    if not effective_session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required (either as a form field or query parameter).",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    total_chunks_indexed = 0

    for file in files:
        filename = file.filename or "unknown.txt"
        try:
            content = await file.read()
            chunks = extract_document_chunks(content, filename)
            if chunks:
                count = index_chunks(effective_session_id, chunks)
                total_chunks_indexed += count
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error("Failed to process file '%s': %s", filename, e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process file '{filename}': {str(e)}",
            )

    return UploadResponse(
        session_id=effective_session_id,
        chunks_indexed=total_chunks_indexed,
        files_processed=len(files),
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Executes dense retrieval, performs Stage-1 relevance check, generates an
    answer via LLM, and verifies citation through Stage-2 quote verification.
    """
    refusal_response = QueryResponse(
        status="refused",
        answer=config.REFUSAL_TEXT,
        verbatim_quote="",
        cited_page=0,
        source_document="",
    )

    # 1. Dense retrieval
    chunks = retrieve_top_k(request.session_id, request.question, top_k=4)

    # 2. Stage-1 Guardrail: Relevance check
    if not check_relevance(chunks, config.DISTANCE_THRESHOLD):
        logger.info("Stage-1 Guardrail failed: Chunks outside relevance threshold.")
        return refusal_response

    # 3. LLM generation
    llm_output = generate_answer(request.question, chunks)

    # If LLM explicitly refused, return refusal
    if llm_output.get("status") != "answered":
        logger.info("LLM returned status: %s", llm_output.get("status"))
        return refusal_response

    # 4. Stage-2 Guardrail: Verbatim citation verification
    quote = llm_output.get("verbatim_quote", "")
    if not verify_citation(quote, chunks):
        logger.info("Stage-2 Guardrail failed: Verbatim quote not found in retrieved chunks.")
        return refusal_response

    # Check that cited_page is positive
    cited_page = llm_output.get("cited_page", 0)
    source_document = llm_output.get("source_document", "")

    return QueryResponse(
        status="answered",
        answer=llm_output.get("answer", ""),
        verbatim_quote=quote,
        cited_page=cited_page,
        source_document=source_document,
    )


@app.delete("/session/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str) -> DeleteSessionResponse:
    """Drops the session collection from ChromaDB."""
    deleted = delete_collection(session_id)
    return DeleteSessionResponse(session_id=session_id, deleted=deleted)
