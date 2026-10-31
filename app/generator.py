import json
import logging
import re
from typing import Any, Dict, List
import requests
from app import config

logger = logging.getLogger(__name__)


def build_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> tuple[str, str]:
    """Builds system and user prompts for zero-shot JSON-constrained generation."""
    system_prompt = (
        "You are a strict, precise academic assistant. Answer the user question strictly "
        "and solely based on the provided context passages.\n\n"
        "You must follow these rules without exception:\n"
        "1. Ground your answer completely in the context.\n"
        "2. If the context does not contain enough facts to answer the question, or if you cannot "
        "extract an exact verbatim quote, set status to \"refused\".\n"
        "3. When answering, provide a concise factual answer, extract an exact verbatim sentence "
        "from the context into \"verbatim_quote\", specify the exact page number (integer) from the "
        "header in \"cited_page\", and the document filename in \"source_document\".\n"
        "4. You must output ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "status": "answered" | "refused",\n'
        '  "answer": "<concise factual answer or refusal message>",\n'
        '  "verbatim_quote": "<exact sentence copied verbatim from the context or empty string if refused>",\n'
        '  "cited_page": <integer page number or 0 if refused>,\n'
        '  "source_document": "<filename or empty string if refused>"\n'
        "}\n"
        "5. Do NOT output markdown code blocks (```json) or any commentary outside the JSON object."
    )

    formatted_context_list = []
    for idx, chunk in enumerate(context_chunks):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", 1)
        text = chunk.get("text", "")
        formatted_context_list.append(
            f"[Context {idx + 1} | Document: {source} | Page: {page}]\n{text}"
        )

    context_str = "\n\n".join(formatted_context_list)
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
    return system_prompt, user_prompt


def parse_llm_json(content: str) -> Dict[str, Any]:
    """Parses and validates LLM JSON output against the required schema."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except Exception:
        # Search for first { to last }
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            raise ValueError(f"Could not extract JSON from LLM response: {content[:100]}...")

    status = data.get("status", "refused")
    if status not in ["answered", "refused"]:
        status = "refused"

    cited_page_val = data.get("cited_page", 0)
    try:
        cited_page = int(cited_page_val)
    except (ValueError, TypeError):
        cited_page = 0

    return {
        "status": status,
        "answer": str(data.get("answer", config.REFUSAL_TEXT if status == "refused" else "")),
        "verbatim_quote": str(data.get("verbatim_quote", "")),
        "cited_page": cited_page,
        "source_document": str(data.get("source_document", "")),
    }


def extractive_fallback(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic extractive fallback when the Ollama LLM endpoint is offline.
    Extracts the most relevant factual sentence from the top retrieved chunk
    so the application remains fully functional out of the box.
    """
    if not context_chunks:
        return {
            "status": "refused",
            "answer": config.REFUSAL_TEXT,
            "verbatim_quote": "",
            "cited_page": 0,
            "source_document": "",
        }

    top_chunk = context_chunks[0]
    chunk_text = top_chunk.get("text", "")
    metadata = top_chunk.get("metadata", {})
    cited_page = int(metadata.get("page", 1))
    source_document = str(metadata.get("source", "document"))

    # Split into candidate sentences
    raw_sentences = re.split(r"(?<=[.!?\n])\s+", chunk_text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= 15]

    if not sentences:
        selected_quote = chunk_text.strip()
    else:
        query_words = set(re.findall(r"\w+", query.lower()))
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "this", "that",
            "with", "the", "and", "was", "for", "are", "were", "from", "how",
            "why", "did", "does", "reach", "reached", "about"
        }
        content_words = query_words - stop_words

        best_score = -1
        best_sentence = sentences[0]
        for s in sentences:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(content_words & s_words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = s
        selected_quote = best_sentence

    return {
        "status": "answered",
        "answer": f"{selected_quote}",
        "verbatim_quote": selected_quote,
        "cited_page": cited_page,
        "source_document": source_document,
    }


def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calls the Ollama LLM endpoint to generate a deterministic JSON response.
    If Ollama is unreachable, automatically uses an extractive fallback
    to ensure full end-to-end functionality.

    Returns:
        Dict adhering to the response schema.
    """
    default_refusal = {
        "status": "refused",
        "answer": config.REFUSAL_TEXT,
        "verbatim_quote": "",
        "cited_page": 0,
        "source_document": "",
    }

    if not context_chunks:
        return default_refusal

    system_prompt, user_prompt = build_prompt(query, context_chunks)
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "options": {
            "temperature": 0.0,
        },
        "stream": False,
    }

    try:
        response = requests.post(
            config.LLM_API_URL,
            json=payload,
            timeout=10,
        )
        if response.status_code != 200:
            logger.warning(
                "Ollama API returned status code %s. Using extractive fallback.",
                response.status_code,
            )
            return extractive_fallback(query, context_chunks)

        data = response.json()
        message_content = data.get("message", {}).get("content", "")
        if not message_content:
            logger.warning("Empty response from Ollama API. Using extractive fallback.")
            return extractive_fallback(query, context_chunks)

        return parse_llm_json(message_content)

    except requests.exceptions.RequestException as e:
        logger.info("Ollama endpoint not available (%s). Utilizing extractive fallback.", e)
        return extractive_fallback(query, context_chunks)

    except Exception as e:
        logger.warning("Unexpected error during LLM generation: %s. Using refusal.", e)
        return default_refusal
