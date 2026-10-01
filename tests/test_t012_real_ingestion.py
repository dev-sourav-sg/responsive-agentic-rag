from pathlib import Path

from responsive_agentic_rag.connectors.document_connector import (
    DocumentConnector,
)
from responsive_agentic_rag.ingestion.chunker import TextChunker
from responsive_agentic_rag.ingestion.embedding_service import (
    EmbeddingService,
)
from responsive_agentic_rag.ingestion.normalizer import IngestionNormalizer
from responsive_agentic_rag.ingestion.orchestrator import (
    IngestionOrchestrator,
)
from responsive_agentic_rag.retrieval.qdrant_client import (
    create_qdrant_client,
)
from responsive_agentic_rag.retrieval.qdrant_store import (
    QdrantVectorStore,
)


def test_t012_real_document_ingestion_to_qdrant():
    document_directory = Path("data/documents")

    connector = DocumentConnector(document_directory)

    normalizer = IngestionNormalizer()
    chunker = TextChunker(
        max_chunk_characters=500,
        overlap_characters=50,
    )
    embedding_service = EmbeddingService()

    client = create_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="t012_real_ingestion_test",
        vector_dimension=embedding_service.dimension,
    )

    vector_store.ensure_collection()

    orchestrator = IngestionOrchestrator(
        connector=connector,
        normalizer=normalizer,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = orchestrator.ingest_all()

    assert len(results) == 4

    assert all(
        result.records_created > 0
        for result in results
    )

    assert all(
        result.chunks_created > 0
        for result in results
    )

    collection_info = client.get_collection(
        "t012_real_ingestion_test"
    )

    assert collection_info.points_count > 0