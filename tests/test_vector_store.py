from collections.abc import Sequence

from responsive_agentic_rag.models.knowledge import ChunkRecord
from responsive_agentic_rag.models.retrieval import RetrievalCandidate
from responsive_agentic_rag.retrieval.vector_store import VectorStore


class FakeVectorStore:
    """Minimal in-memory implementation for contract testing."""

    def __init__(self) -> None:
        self.chunks: dict[str, ChunkRecord] = {}

    def upsert(self, chunks: Sequence[ChunkRecord]) -> None:
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 10,
    ) -> list[RetrievalCandidate]:
        return []


def test_fake_vector_store_implements_vector_store():
    store = FakeVectorStore()

    assert isinstance(store, VectorStore)


def test_vector_store_upsert_stores_chunks():
    store = FakeVectorStore()

    chunk = ChunkRecord(
        chunk_id="chunk-001",
        source_id="source-001",
        record_id="record-001",
        content="Payment settlement requires reconciliation.",
        chunk_index=0,
        token_count=5,
        embedding=[0.1, 0.2, 0.3],
        authority_score=0.9,
    )

    store.upsert([chunk])

    assert "chunk-001" in store.chunks
    assert store.chunks["chunk-001"].content == (
        "Payment settlement requires reconciliation."
    )