from pathlib import Path

from responsive_agentic_rag.connectors.document_connector import (
    DocumentConnector,
)
from responsive_agentic_rag.ingestion.chunker import TextChunker
from responsive_agentic_rag.ingestion.embedding_service import (
    EmbeddingService,
)
from responsive_agentic_rag.ingestion.normalizer import IngestionNormalizer
from responsive_agentic_rag.models.knowledge import KnowledgeSource
from responsive_agentic_rag.retrieval.qdrant_client import (
    create_qdrant_client,
)
from responsive_agentic_rag.retrieval.qdrant_store import (
    QdrantVectorStore,
)


COLLECTION_NAME = "t011_real_persistence_test"


def test_t011_persists_real_embedded_chunk_to_qdrant():
    # 1. Connect to the real Docker Qdrant instance.
    client = create_qdrant_client()

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        vector_dimension=384,
    )

    store.ensure_collection()

    # 2. Read a real document through the existing connector.
    connector = DocumentConnector(
        Path("data/documents")
    )

    source_config = connector.list_sources()[0]
    raw_content = connector.fetch_content(source_config)

    # 3. Normalize the real document.
    source = KnowledgeSource(
        source_id="t011-real-source",
        source_type="document",
        title="T011 Real Persistence Test",
        source_location=source_config.source_location,
        metadata={
            **source_config.metadata,
            "source_type": "document",
            "source_location": source_config.source_location,
        },
    )

    normalizer = IngestionNormalizer()

    records = normalizer.normalize(
        source,
        raw_content,
    )

    assert records
    assert records[0].body_text

    # 4. Chunk the normalized content.
    chunker = TextChunker(
        max_chunk_characters=500,
        overlap_characters=50,
    )

    chunks = chunker.chunk(records[0])

    assert chunks

    # 5. Generate real embeddings.
    embedding_service = EmbeddingService()

    embedded_chunks = embedding_service.embed_chunks(
        chunks[:1]
    )

    assert len(embedded_chunks) == 1

    chunk = embedded_chunks[0]

    assert chunk.embedding is not None
    assert len(chunk.embedding) == 384

    # 6. Persist the real embedded chunk in Docker Qdrant.
    store.upsert([chunk])

    # 7. Read the point back from Qdrant.
    point_id = store._build_point_id(
        chunk.chunk_id
    )

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )

    assert len(points) == 1

    point = points[0]

    # 8. Validate the persisted vector.
    assert point.id == point_id
    assert point.vector is not None
    assert len(point.vector) == 384

    # Qdrant uses cosine distance and stores normalized vectors.
    vector_norm = sum(
        value * value
        for value in point.vector
    ) ** 0.5

    assert abs(vector_norm - 1.0) < 1e-5

    # 9. Validate the persisted provenance and content.
    assert point.payload is not None

    assert point.payload["chunk_id"] == chunk.chunk_id
    assert point.payload["source_id"] == chunk.source_id
    assert point.payload["record_id"] == chunk.record_id
    assert point.payload["source_type"] == "document"
    assert (
        point.payload["source_location"]
        == source_config.source_location
    )
    assert point.payload["content"] == chunk.content
    assert point.payload["chunk_index"] == chunk.chunk_index
    assert point.payload["token_count"] == chunk.token_count
    assert point.payload["authority_score"] == chunk.authority_score