from typing import Protocol, Sequence, runtime_checkable

from responsive_agentic_rag.models.knowledge import ChunkRecord
from responsive_agentic_rag.models.retrieval import RetrievalCandidate

@runtime_checkable
class VectorStore(Protocol):
    """Abstraction for persistent vector storage."""

    def upsert(self, chunks: Sequence[ChunkRecord]) -> None:
        """Insert or update embedded chunks."""
        ...

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 10,
    ) -> list[RetrievalCandidate]:
        """Search for the most relevant chunks."""
        ...