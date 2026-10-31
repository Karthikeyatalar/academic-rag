from typing import Any, Dict, List


def check_relevance(retrieved_chunks: List[Dict[str, Any]], threshold: float) -> bool:
    """
    Pre-retrieval guardrail: Checks if the retrieved chunks meet the relevance threshold.

    If best_distance > threshold or retrieved list is empty, returns False (trigger refusal).

    Args:
        retrieved_chunks: List of retrieved chunks with 'cosine_distance'.
        threshold: Cosine distance cutoff threshold (e.g. 0.60).

    Returns:
        bool: True if relevant (best_distance <= threshold), False otherwise.
    """
    if not retrieved_chunks:
        return False

    best_distance = min(chunk.get("cosine_distance", 1.0) for chunk in retrieved_chunks)
    if best_distance > threshold:
        return False

    return True


def verify_citation(verbatim_quote: str, context_chunks: List[Dict[str, Any]]) -> bool:
    """
    Post-generation guardrail: Verifies that the cited verbatim_quote exists
    as an exact substring inside any of the retrieved chunks after normalizing
    whitespace and case.

    Returns False if missing or shorter than 10 characters.

    Args:
        verbatim_quote: The cited quote from the model generation.
        context_chunks: List of retrieved context chunks.

    Returns:
        bool: True if valid exact quote found in context, False otherwise.
    """
    if not verbatim_quote or not isinstance(verbatim_quote, str):
        return False

    # Normalize whitespace (collapse multiple spaces, tabs, newlines) and lowercase
    normalized_quote = " ".join(verbatim_quote.strip().split()).lower()

    # Reject if shorter than 10 characters
    if len(normalized_quote) < 10:
        return False

    if not context_chunks:
        return False

    for chunk in context_chunks:
        chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        normalized_chunk = " ".join(chunk_text.strip().split()).lower()
        if normalized_quote in normalized_chunk:
            return True

    return False

