import io
import os
from typing import Any, Dict, List
import docx
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import config


def extract_document_chunks(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts text and chunks from an uploaded file (PDF, DOCX, TXT) with page-level metadata.

    Args:
        file_bytes: Raw bytes of the document.
        filename: Name of the document including extension.

    Returns:
        List of dictionaries containing 'chunk_id', 'text', and 'metadata'.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext not in ["pdf", "docx", "txt"]:
        raise ValueError(
            f"Unsupported file extension: .{ext} for file '{filename}'. "
            "Supported formats are: .pdf, .docx, .txt"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks: List[Dict[str, Any]] = []

    if ext == "pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            if not page_text or not page_text.strip():
                continue
            splits = text_splitter.split_text(page_text)
            for idx, split_text in enumerate(splits):
                chunk_id = f"{filename}_p{page_num + 1}_c{idx}"
                metadata = {
                    "source": filename,
                    "page": page_num + 1,
                    "chunk_id": chunk_id,
                }
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": split_text,
                    "metadata": metadata,
                })
        doc.close()

    elif ext == "docx":
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        if full_text.strip():
            splits = text_splitter.split_text(full_text)
            for idx, split_text in enumerate(splits):
                chunk_id = f"{filename}_p1_c{idx}"
                metadata = {
                    "source": filename,
                    "page": 1,
                    "chunk_id": chunk_id,
                }
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": split_text,
                    "metadata": metadata,
                })

    elif ext == "txt":
        full_text = file_bytes.decode("utf-8", errors="replace")
        if full_text.strip():
            splits = text_splitter.split_text(full_text)
            for idx, split_text in enumerate(splits):
                chunk_id = f"{filename}_p1_c{idx}"
                metadata = {
                    "source": filename,
                    "page": 1,
                    "chunk_id": chunk_id,
                }
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": split_text,
                    "metadata": metadata,
                })

    return chunks

