from types import SimpleNamespace

import pytest

from responsive_agentic_rag.retrieval.qdrant_store import QdrantVectorStore


class FakeQdrantClient:
    """Minimal fake Qdrant client for unit testing search behavior."""

    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_limit = None
        self.last_collection_name = None
        self.last_with_payload = None

    def query_points(
        self,
        collection_name,
        query,
        limit,
        with_payload,
    ):
        self.last_collection_name = collection_name
        self.last_query = query
        self.last_limit = limit
        self.last_with_payload = with_payload

        return SimpleNamespace(points=self.results)


def make_result(
    chunk_id="chunk-1",
    score=0.91,
    authority_score=0.8,
):
    """Create a fake Qdrant search result."""

    return SimpleNamespace(
        score=score,
        payload={
            "chunk_id": chunk_id,
            "source_id": "source-1",
            "source_type": "document",
            "source_location": "data/documents/payment_policy.pdf",
            "content": "Settlement processing takes four business days.",
            "metadata": {
                "owner": "Payments Team",
            },
            "authority_score": authority_score,
        },
    )


def test_search_returns_semantic_candidates():
    """Qdrant results should be mapped to RetrievalCandidate objects."""

    client = FakeQdrantClient(
        [
            make_result("chunk-1", 0.91, 0.8),
            make_result("chunk-2", 0.84, 0.7),
        ]
    )

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_dimension=3,
    )

    results = store.search(
        query_vector=[0.1, 0.2, 0.3],
        limit=2,
    )

    assert len(results) == 2

    first = results[0]
    assert first.chunk_id == "chunk-1"
    assert first.source_id == "source-1"
    assert first.source_type == "document"
    assert first.source_location == (
        "data/documents/payment_policy.pdf"
    )
    assert first.content == (
        "Settlement processing takes four business days."
    )
    assert first.metadata == {
        "owner": "Payments Team",
    }

    assert first.semantic_score == 0.91
    assert first.lexical_score == 0.0
    assert first.authority_score == 0.8

    # For T017-A semantic-only retrieval,
    # combined_score is currently the semantic score.
    assert first.combined_score == 0.91
    assert first.rank == 1

    second = results[1]
    assert second.chunk_id == "chunk-2"
    assert second.semantic_score == 0.84
    assert second.lexical_score == 0.0
    assert second.authority_score == 0.7
    assert second.combined_score == 0.84
    assert second.rank == 2


def test_search_passes_query_and_limit_to_qdrant():
    """The query vector and result limit should reach Qdrant."""

    client = FakeQdrantClient(
        [
            make_result("chunk-1"),
        ]
    )

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_dimension=3,
    )

    query_vector = [0.1, 0.2, 0.3]

    store.search(
        query_vector=query_vector,
        limit=7,
    )

    assert client.last_query == query_vector
    assert client.last_limit == 7
    assert client.last_collection_name == "test_collection"
    assert client.last_with_payload is True


def test_search_assigns_rank_starting_at_one():
    """Returned candidates should have deterministic one-based ranks."""

    client = FakeQdrantClient(
        [
            make_result("chunk-1", 0.95),
            make_result("chunk-2", 0.90),
            make_result("chunk-3", 0.85),
        ]
    )

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_dimension=3,
    )

    results = store.search(
        query_vector=[0.1, 0.2, 0.3],
        limit=3,
    )

    assert [result.rank for result in results] == [1, 2, 3]


def test_search_rejects_empty_query_vector():
    """An empty query vector should be rejected before calling Qdrant."""

    client = FakeQdrantClient([])

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_dimension=3,
    )

    with pytest.raises(
        ValueError,
        match="query_vector cannot be empty",
    ):
        store.search([])

    assert client.last_query is None


@pytest.mark.parametrize("limit", [0, -1])
def test_search_rejects_invalid_limit(limit):
    """Search should reject zero or negative result limits."""

    client = FakeQdrantClient([])

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_dimension=3,
    )

    with pytest.raises(
        ValueError,
        match="limit must be greater than 0",
    ):
        store.search(
            query_vector=[0.1, 0.2, 0.3],
            limit=limit,
        )

    assert client.last_query is None


def test_search_rejects_dimension_mismatch():
    """Query vector dimensions must match the Qdrant collection."""

    client = FakeQdrantClient([])

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_dimension=3,
    )

    with pytest.raises(
        ValueError,
        match="does not match configured dimension",
    ):
        store.search(
            query_vector=[0.1, 0.2],
            limit=10,
        )

    assert client.last_query is None


def test_search_uses_safe_defaults_for_optional_payload():
    """Missing optional payload fields should receive safe defaults."""

    result = SimpleNamespace(
        score=0.75,
        payload={
            "chunk_id": "chunk-1",
            "source_id": "source-1",
            "source_type": "document",
            "source_location": "test.pdf",
            "content": "Example content",
        },
    )

    client = FakeQdrantClient([result])

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_dimension=3,
    )

    results = store.search(
        query_vector=[0.1, 0.2, 0.3],
    )

    assert len(results) == 1
    assert results[0].authority_score == 0.0
    assert results[0].metadata == {}
    assert results[0].semantic_score == 0.75
    assert results[0].lexical_score == 0.0
    assert results[0].combined_score == 0.75
    assert results[0].rank == 1


def test_search_returns_empty_list_when_qdrant_has_no_matches():
    """No Qdrant matches should produce an empty candidate list."""

    client = FakeQdrantClient([])

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_dimension=3,
    )

    results = store.search(
        query_vector=[0.1, 0.2, 0.3],
        limit=10,
    )

    assert results == []