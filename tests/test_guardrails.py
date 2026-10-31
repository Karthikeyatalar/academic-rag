import os
from unittest.mock import patch
import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from app import config
from app.guardrails import check_relevance, verify_citation
from app.main import app
from app.parser import extract_document_chunks

SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")


def create_synthetic_pdf(filepath: str) -> None:
    """
    Programmatically creates a synthetic 2-page PDF containing specific academic facts.
    - Page 1: Abstract and methodology context.
    - Page 2: Specific convergence and parameter facts.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Abstract and Methodology\n\n"
        "Machine learning models often rely on iterative gradient descent methods. "
        "The experiments were initialized using Adam optimizer with standard momentum "
        "parameters on benchmark datasets. Initial loss was measured at 2.45 before any training steps.",
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Experimental Results and Convergence Analysis\n\n"
        "During fine-tuning across 100 iterations, the model showed steady decrease in loss. "
        "The algorithm convergence was reached at epoch 42 with a learning rate of 0.005 on page 2. "
        "The final validation accuracy stabilized at 98.4%.",
    )

    doc.save(filepath)
    doc.close()


@pytest.fixture(scope="module")
def client():
    """TestClient fixture for FastAPI."""
    return TestClient(app)


@pytest.fixture(scope="module")
def setup_document(client):
    """
    Creates a synthetic PDF, creates a session, uploads the PDF,
    yields the session_id and filepath, and drops the session upon teardown.
    """
    pdf_path = os.path.join(SAMPLE_DOCS_DIR, "synthetic_paper.pdf")
    create_synthetic_pdf(pdf_path)

    # 1. Create session
    create_resp = client.post("/session/create")
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    # 2. Upload document
    with open(pdf_path, "rb") as f:
        upload_resp = client.post(
            "/upload",
            data={"session_id": session_id},
            files={"files": ("synthetic_paper.pdf", f, "application/pdf")},
        )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["chunks_indexed"] > 0

    yield session_id, pdf_path

    # Teardown
    client.delete(f"/session/{session_id}")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


# =====================================================================
# Specification Test Cases
# =====================================================================


def test_case_1_true_positive(client, setup_document):
    """
    Test Case 1 (True Positive):
    Query a fact present in the PDF. Assert response contains status == 'answered',
    cited_page == 2, and verbatim_quote is present in the context.
    """
    session_id, _ = setup_document
    question = "At what epoch and learning rate was algorithm convergence reached?"

    # Mock the LLM generation to return accurate factual answer grounded in Page 2
    mocked_llm_response = {
        "status": "answered",
        "answer": "The algorithm convergence was reached at epoch 42 with a learning rate of 0.005.",
        "verbatim_quote": "The algorithm convergence was reached at epoch 42 with a learning rate of 0.005 on page 2.",
        "cited_page": 2,
        "source_document": "synthetic_paper.pdf",
    }

    with patch("app.main.generate_answer", return_value=mocked_llm_response):
        response = client.post(
            "/query",
            json={"session_id": session_id, "question": question},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "answered"
    assert data["cited_page"] == 2
    assert data["source_document"] == "synthetic_paper.pdf"
    assert "epoch 42" in data["answer"]
    assert data["verbatim_quote"] == "The algorithm convergence was reached at epoch 42 with a learning rate of 0.005 on page 2."


def test_case_2_out_of_scope_pre_retrieval_refusal(client, setup_document):
    """
    Test Case 2 (Out-of-Scope Pre-Retrieval Refusal):
    Ask an adversarial question (e.g. 'Who was the first emperor of Rome?').
    Assert Stage-1 guardrail flags high distance and returns REFUSAL_TEXT.
    """
    session_id, _ = setup_document
    adversarial_question = "Who was the first emperor of Rome and when did he rule?"

    # We do NOT mock generate_answer; Stage-1 should abort before generating
    response = client.post(
        "/query",
        json={"session_id": session_id, "question": adversarial_question},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "refused"
    assert data["answer"] == config.REFUSAL_TEXT
    assert data["verbatim_quote"] == ""
    assert data["cited_page"] == 0
    assert data["source_document"] == ""


def test_case_3_post_guardrail_hallucination_check(client, setup_document):
    """
    Test Case 3 (Post-Guardrail Hallucination Check):
    Mock an LLM response where the answer invents a quote not in the document.
    Assert Stage-2 guardrail catches the discrepancy and flips response to status == 'refused'.
    """
    session_id, _ = setup_document
    # A query relevant to the document so Stage-1 passes
    question = "What optimization method was used in the experiments?"

    # LLM hallucinates a quote that does not exist anywhere in the context
    hallucinated_llm_response = {
        "status": "answered",
        "answer": "The model used quantum simulated annealing with 500 spin states.",
        "verbatim_quote": "Quantum annealing with 500 spin states was used as the primary optimizer.",
        "cited_page": 2,
        "source_document": "synthetic_paper.pdf",
    }

    with patch("app.main.generate_answer", return_value=hallucinated_llm_response):
        response = client.post(
            "/query",
            json={"session_id": session_id, "question": question},
        )

    assert response.status_code == 200
    data = response.json()

    # Discrepancy caught by Stage-2 guardrail: must flip to refused
    assert data["status"] == "refused"
    assert data["answer"] == config.REFUSAL_TEXT
    assert data["verbatim_quote"] == ""
    assert data["cited_page"] == 0
    assert data["source_document"] == ""


# =====================================================================
# Unit Tests for Guardrails, Parser, and Edge Cases
# =====================================================================


def test_guardrails_unit():
    """Unit tests for check_relevance and verify_citation functions."""
    # check_relevance tests
    assert not check_relevance([], 0.60)
    assert check_relevance([{"cosine_distance": 0.45}], 0.60)
    assert not check_relevance([{"cosine_distance": 0.75}], 0.60)
    assert check_relevance([{"cosine_distance": 0.75}, {"cosine_distance": 0.35}], 0.60)

    # verify_citation tests
    chunks = [
        {"text": "The algorithm convergence was reached at epoch 42 with a learning rate of 0.005."},
        {"text": "Another paragraph with different information."},
    ]

    # Exact match
    assert verify_citation(
        "The algorithm convergence was reached at epoch 42 with a learning rate of 0.005.",
        chunks,
    )

    # Case & whitespace variance
    assert verify_citation(
        "  the algorithm   convergence was   reached at epoch 42  ",
        chunks,
    )

    # Substring shorter than 10 characters should be rejected
    assert not verify_citation("algorithm", chunks)
    assert not verify_citation("short", chunks)

    # Empty or None quote
    assert not verify_citation("", chunks)
    assert not verify_citation(None, chunks)

    # Hallucinated quote
    assert not verify_citation(
        "This sentence does not exist anywhere in the chunks.",
        chunks,
    )


def test_parser_unsupported_extension():
    """Tests that extract_document_chunks raises explicit ValueError on unsupported formats."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        extract_document_chunks(b"some content", "file.xyz")


def test_parser_txt_and_docx():
    """Tests that extract_document_chunks handles TXT and DOCX files correctly."""
    import io
    import docx

    # 1. Test TXT
    txt_content = b"This is an academic research document on deep learning algorithms."
    txt_chunks = extract_document_chunks(txt_content, "notes.txt")
    assert len(txt_chunks) > 0
    assert txt_chunks[0]["metadata"]["source"] == "notes.txt"
    assert txt_chunks[0]["metadata"]["page"] == 1
    assert "deep learning" in txt_chunks[0]["text"]

    # 2. Test DOCX
    doc = docx.Document()
    doc.add_paragraph("First paragraph discussing dataset statistics.")
    doc.add_paragraph("Second paragraph detailing ablation experiments.")
    buffer = io.BytesIO()
    doc.save(buffer)
    docx_bytes = buffer.getvalue()

    docx_chunks = extract_document_chunks(docx_bytes, "paper.docx")
    assert len(docx_chunks) > 0
    assert docx_chunks[0]["metadata"]["source"] == "paper.docx"
    assert docx_chunks[0]["metadata"]["page"] == 1
    assert "dataset statistics" in docx_chunks[0]["text"]

