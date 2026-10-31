import os

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "60"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./data/academic_rag.db")
CHROMA_PATH: str = SQLITE_DB_PATH  # Backward-compatibility alias
LLM_API_URL: str = os.getenv("LLM_API_URL", "http://localhost:11434/api/chat")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3:latest")
DISTANCE_THRESHOLD: float = float(os.getenv("DISTANCE_THRESHOLD", "0.60"))
REFUSAL_TEXT: str = os.getenv(
    "REFUSAL_TEXT",
    "The uploaded documents do not contain information to answer this question."
)
