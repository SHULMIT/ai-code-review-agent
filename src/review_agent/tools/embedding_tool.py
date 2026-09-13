"""Client for the configured embedding service - the same Ollama-compatible
/api/embed endpoint the remote Model Service instance already exposes.

Kept as its own tool, separate from model_service_tool.py, since the
embedding endpoint is configured independently
(Settings.embedding_service_url / resolved_embedding_service_url()) and may
move to a different server and have different latency requirements than text
generation later.
"""

import httpx
from pydantic import ValidationError

from review_agent.config import Settings
from review_agent.models.embedding_output import EmbeddingResponse

_EMBEDDING_MODEL_NAME = "embeddinggemma:latest"


class EmbeddingServiceError(Exception):
    """Raised when the embedding service cannot be called or its response cannot be parsed."""


async def get_embeddings(
    texts: list[str], settings: Settings, client: httpx.AsyncClient | None = None
) -> list[list[float]]:
    """Return one embedding vector per text from the configured embedding service.

    Every failure - timeout, connection error, a non-200 response, or an
    unparseable body - is wrapped in EmbeddingServiceError; no raw httpx
    exception ever escapes this function. A single batched request lets the
    deduplication service reuse each finding's embedding across all of its
    pairwise comparisons in one review run.

    client lets a caller pass in one pooled httpx.AsyncClient shared across
    an entire review run instead of opening a fresh TCP connection per call
    - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5. When omitted, this
    function falls back to its original self-contained per-call client, so
    every existing caller keeps working unchanged.
    """
    if not texts:
        return []

    url = f"{settings.resolved_embedding_service_url()}/api/embed"
    payload = {"model": _EMBEDDING_MODEL_NAME, "input": texts}

    try:
        if client is not None:
            response = await client.post(
                url, json=payload, timeout=settings.embedding_service_timeout_seconds
            )
            response.raise_for_status()
        else:
            async with httpx.AsyncClient(
                timeout=settings.embedding_service_timeout_seconds
            ) as owned_client:
                response = await owned_client.post(url, json=payload)
                response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise EmbeddingServiceError(
            f"Embedding service returned {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise EmbeddingServiceError("Embedding service request failed.") from exc

    try:
        parsed = EmbeddingResponse.model_validate(response.json())
        if len(parsed.embeddings) != len(texts):
            raise ValueError("embedding response count does not match input count")
        return parsed.embeddings
    except (ValueError, ValidationError) as exc:
        raise EmbeddingServiceError("Embedding service returned an unparseable response.") from exc


async def get_embedding(
    text: str, settings: Settings, client: httpx.AsyncClient | None = None
) -> list[float]:
    """Return one embedding while preserving the single-text tool interface."""

    return (await get_embeddings([text], settings, client=client))[0]
