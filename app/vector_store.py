import os
import sqlite3
import threading
from typing import Any, Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from app import config

_model: SentenceTransformer | None = None
_db_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Returns or lazily creates the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def get_db_connection() -> sqlite3.Connection:
    """Returns an active SQLite database connection with tables initialized."""
    os.makedirs(os.path.dirname(os.path.abspath(config.SQLITE_DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with _db_lock:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_id ON document_chunks(session_id)
                """
            )
    return conn


def get_collection(session_id: str):
    """
    Backwards-compatible helper returning session metadata and count.
    """
    conn = get_db_connection()
    with conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE session_id = ?",
            (session_id,),
        )
        count = cursor.fetchone()[0]
    return {"session_id": session_id, "count": count}


def index_chunks(session_id: str, chunks: List[Dict[str, Any]]) -> int:
    """
    Generates embeddings for document chunks and persists them into SQLite.

    Args:
        session_id: The session identifier.
        chunks: List of chunk dictionaries containing 'chunk_id', 'text', and 'metadata'.

    Returns:
        Number of chunks indexed.
    """
    if not chunks:
        return 0

    model = get_embedding_model()
    texts = [c["text"] for c in chunks]

    # Generate unit-normalized embeddings for fast cosine distance computation
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings_np = np.asarray(embeddings, dtype=np.float32)

    conn = get_db_connection()
    records = []
    for i, c in enumerate(chunks):
        cid = c["chunk_id"]
        txt = c["text"]
        meta = c.get("metadata", {})
        source = meta.get("source", "document")
        page = int(meta.get("page", 1))
        emb_blob = embeddings_np[i].tobytes()
        records.append((cid, session_id, txt, source, page, emb_blob))

    with _db_lock:
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO document_chunks 
                (chunk_id, session_id, text, source, page, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    return len(chunks)


def retrieve_top_k(session_id: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Retrieves the top_k most similar chunks for the query from SQLite
    using cosine distance over unit-normalized embeddings.

    Args:
        session_id: The session identifier.
        query: Query string.
        top_k: Number of chunks to retrieve.

    Returns:
        List of chunks with text, metadata, and cosine_distance.
    """
    conn = get_db_connection()
    with _db_lock:
        with conn:
            cursor = conn.execute(
                """
                SELECT chunk_id, text, source, page, embedding 
                FROM document_chunks 
                WHERE session_id = ?
                """,
                (session_id,),
            )
            rows = cursor.fetchall()

    if not rows:
        return []

    # Encode query with unit normalization
    model = get_embedding_model()
    query_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    query_vec = np.asarray(query_emb[0], dtype=np.float32)

    # Reconstruct matrix of stored embeddings
    emb_list = [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
    emb_matrix = np.vstack(emb_list)

    # Cosine distance = 1.0 - (emb_matrix · query_vec) for unit-normalized vectors
    sims = np.dot(emb_matrix, query_vec)
    distances = 1.0 - sims

    # Sort indices by distance ascending
    sorted_indices = np.argsort(distances)
    top_indices = sorted_indices[:top_k]

    retrieved: List[Dict[str, Any]] = []
    for idx in top_indices:
        row = rows[idx]
        dist = float(distances[idx])
        retrieved.append({
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "metadata": {
                "source": row["source"],
                "page": row["page"],
                "chunk_id": row["chunk_id"],
            },
            "cosine_distance": dist,
        })

    return retrieved


def delete_collection(session_id: str) -> bool:
    """
    Deletes all document chunks belonging to session_id from SQLite.

    Args:
        session_id: The session identifier.

    Returns:
        True if operation completed successfully.
    """
    conn = get_db_connection()
    with _db_lock:
        with conn:
            conn.execute(
                "DELETE FROM document_chunks WHERE session_id = ?",
                (session_id,),
            )
    return True
