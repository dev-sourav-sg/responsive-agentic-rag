from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient

from responsive_agentic_rag.models.knowledge import ChunkRecord
from responsive_agentic_rag.retrieval.qdrant_store import QdrantVectorStore


def build_chunk(
    *,
    chunk_id: str = "chunk-001",
    embedding: list[float] | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source_id="source-001",
        record_id="record-001",
        content="Payment settlement requires reconciliation.",
        chunk_index=0,
        token_count=4,
        embedding=embedding or [0.1, 0.2, 0.3],
        section_path=["Settlement"],
        metadata={
            "source_type": "document",
            "source_location": "data/documents/payment_policy.pdf",
        },
        authority_score=0.9,
    )


def test_qdrant_store_initializes_with_configuration():
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=3,
    )

    assert store.client is client
    assert store.collection_name == "test_chunks"
    assert store.vector_dimension == 3


def test_ensure_collection_creates_collection():
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=3,
    )

    store.ensure_collection()

    collections = client.get_collections()

    assert "test_chunks" in {
        collection.name
        for collection in collections.collections
    }


def test_ensure_collection_is_idempotent():
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=3,
    )

    store.ensure_collection()
    store.ensure_collection()

    collections = client.get_collections()

    matching_collections = [
        collection.name
        for collection in collections.collections
        if collection.name == "test_chunks"
    ]

    assert matching_collections == ["test_chunks"]


def test_upsert_requires_embeddings():
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=3,
    )

    store.ensure_collection()

    chunk = build_chunk()
    chunk.embedding = None

    with pytest.raises(
        ValueError,
        match="does not have an embedding",
    ):
        store.upsert([chunk])


def test_upsert_empty_chunks_does_nothing():
    client = MagicMock(spec=QdrantClient)

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=3,
    )

    store.upsert([])

    client.upsert.assert_not_called()


def test_build_point_id_is_deterministic():
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=3,
    )

    first_id = store._build_point_id("chunk-001")
    second_id = store._build_point_id("chunk-001")

    assert first_id == second_id


def test_build_point_id_differs_for_different_chunks():
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=3,
    )

    first_id = store._build_point_id("chunk-001")
    second_id = store._build_point_id("chunk-002")

    assert first_id != second_id


def test_upsert_writes_chunk_to_qdrant():
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=3,
    )

    store.ensure_collection()

    chunk = build_chunk(
        chunk_id="chunk-001",
        embedding=[0.1, 0.2, 0.3],
    )

    store.upsert([chunk])

    points, next_offset = client.scroll(
        collection_name="test_chunks",
        limit=10,
        with_payload=True,
        with_vectors=True,
    )

    assert next_offset is None
    assert len(points) == 1

    point = points[0]

    assert point.id == store._build_point_id("chunk-001")

    expected = [0.1, 0.2, 0.3]
    expected_norm = sum(
        value * value
        for value in expected
    ) ** 0.5
    expected_normalized = [
        value / expected_norm
        for value in expected
    ]

    assert point.vector == pytest.approx(
        expected_normalized,
        abs=1e-6,
    )

    assert point.payload is not None
    assert point.payload["chunk_id"] == "chunk-001"
    assert point.payload["source_id"] == "source-001"
    assert point.payload["record_id"] == "record-001"
    assert point.payload["source_type"] == "document"
    assert (
        point.payload["source_location"]
        == "data/documents/payment_policy.pdf"
    )
    assert (
        point.payload["content"]
        == "Payment settlement requires reconciliation."
    )
    assert point.payload["chunk_index"] == 0
    assert point.payload["section_path"] == ["Settlement"]
    assert point.payload["token_count"] == 4
    assert point.payload["authority_score"] == 0.9


def test_upsert_updates_existing_chunk():
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=3,
    )

    store.ensure_collection()

    original_chunk = build_chunk(
        chunk_id="chunk-001",
        embedding=[0.1, 0.2, 0.3],
    )

    updated_chunk = build_chunk(
        chunk_id="chunk-001",
        embedding=[0.3, 0.2, 0.1],
    )
    updated_chunk.content = "Updated settlement content."

    store.upsert([original_chunk])
    store.upsert([updated_chunk])

    points, _ = client.scroll(
        collection_name="test_chunks",
        limit=10,
        with_payload=True,
        with_vectors=True,
    )

    assert len(points) == 1

    point = points[0]

    assert point.id == store._build_point_id("chunk-001")

    expected = [0.3, 0.2, 0.1]
    expected_norm = sum(
        value * value
        for value in expected
    ) ** 0.5
    expected_normalized = [
        value / expected_norm
        for value in expected
    ]

    assert point.vector == pytest.approx(
        expected_normalized,
        abs=1e-6,
    )

    assert point.payload is not None
    assert point.payload["chunk_id"] == "chunk-001"
    assert point.payload["content"] == "Updated settlement content."