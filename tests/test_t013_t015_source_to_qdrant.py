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
from responsive_agentic_rag.models.knowledge import KnowledgeSource
from responsive_agentic_rag.retrieval.qdrant_client import (
    create_qdrant_client,
)
from responsive_agentic_rag.retrieval.qdrant_store import (
    QdrantVectorStore,
)


COLLECTION_NAME = "t013_t015_source_validation"


def _create_orchestrator(connector):
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

    return IngestionOrchestrator(
        connector=connector,
        normalizer=normalizer,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    ), client


def test_document_and_website_sources_share_ingestion_and_storage_pipeline():
    document_connector = DocumentConnector(
        Path("data/documents")
    )

    orchestrator, client = _create_orchestrator(
        document_connector
    )

    document_results = orchestrator.ingest_all()

    assert len(document_results) == 4

    assert all(
        result.records_created > 0
        for result in document_results
    )

    assert all(
        result.chunks_created > 0
        for result in document_results
    )

    website_connector = WebsiteConnector(
        ["https://example.com"]
    )

    website_orchestrator, _ = _create_orchestrator(
        website_connector
    )

    html_content = b"""
    <html>
        <head>
            <title>Test Website</title>
        </head>
        <body>
            <h1>Payment Settlement</h1>
            <p>
                Settlement processing requires reconciliation
                before merchant funds are released.
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