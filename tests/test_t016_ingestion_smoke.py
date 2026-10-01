from pathlib import Path
from unittest.mock import patch

from responsive_agentic_rag.connectors.document_connector import (
    DocumentConnector,
)
from responsive_agentic_rag.connectors.website_connector import (
    WebsiteConnector,
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


COLLECTION_NAME = "t016_ingestion_smoke_test"


def test_t016_ingests_documents_and_website_into_shared_corpus():
    document_connector = DocumentConnector(
        Path("data/documents")
    )

    website_connector = WebsiteConnector(
        ["https://example.com"]
    )

    normalizer = IngestionNormalizer()

    chunker = TextChunker(
        max_chunk_characters=500,
        overlap_characters=50,
    )

    embedding_service = EmbeddingService()

    client = create_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        vector_dimension=embedding_service.dimension,
    )

    vector_store.ensure_collection()

    document_orchestrator = IngestionOrchestrator(
        connector=document_connector,
        normalizer=normalizer,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    website_orchestrator = IngestionOrchestrator(
        connector=website_connector,
        normalizer=normalizer,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    document_results = document_orchestrator.ingest_all()

    assert len(document_results) == 4

    assert all(
        result.records_created > 0
        for result in document_results
    )

    assert all(
        result.chunks_created > 0
        for result in document_results
    )

    html_content = b"""
    <html>
        <head>
            <title>Responsive Test Website</title>
        </head>
        <body>
            <h1>Settlement Processing</h1>
            <p>
                Merchant settlement processing requires
                reconciliation before funds are released.
            </p>
        </body>
    </html>
    """

    with patch(
        "responsive_agentic_rag.connectors.website_connector.urlopen"
    ) as mock_urlopen:
        mock_response = mock_urlopen.return_value.__enter__.return_value
        mock_response.read.return_value = html_content

        website_results = website_orchestrator.ingest_all()

    assert len(website_results) == 1

    assert website_results[0].records_created == 1
    assert website_results[0].chunks_created > 0

    collection = client.get_collection(
        COLLECTION_NAME
    )

    assert collection.points_count > 0

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    source_types = {
        point.payload["source_type"]
        for point in points
    }

    assert "document" in source_types
    assert "website" in source_types