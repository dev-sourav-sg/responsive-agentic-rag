from collections.abc import Sequence
from typing import Protocol

from responsive_agentic_rag.models.retrieval import RetrievalCandidate


class Reranker(Protocol):
    """Contract for deterministic or neural candidate rerankers."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        limit: int = 10,
    ) -> list[RetrievalCandidate]:
        """Return candidates ordered by reranker relevance."""
        ...