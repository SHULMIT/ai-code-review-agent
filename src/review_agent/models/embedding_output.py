"""The embedding service's validated response shape for one /api/embed call."""

import math

from pydantic import BaseModel
from pydantic import field_validator


class EmbeddingResponse(BaseModel):
    """The embedding service's raw response - one vector per requested input."""

    embeddings: list[list[float]]

    @field_validator("embeddings")
    @classmethod
    def require_non_empty_finite_vectors(
        cls, vectors: list[list[float]]
    ) -> list[list[float]]:
        """Reject empty or non-finite vectors before they reach cosine similarity."""

        if not vectors:
            raise ValueError("embeddings must contain at least one vector")
        if any(not vector for vector in vectors):
            raise ValueError("embedding vectors must not be empty")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("embedding vectors must contain only finite values")
        return vectors
