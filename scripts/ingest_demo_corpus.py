from pathlib import Path

from responsive_agentic_rag.config.settings import settings
from responsive_agentic_rag.connectors.document_connector import DocumentConnector
from responsive_agentic_rag.connectors.website_connector import WebsiteConnector
from responsive_agentic_rag.ingestion.chunker import TextChunker
from responsive_agentic_rag.ingestion.embedding_service import EmbeddingService
from responsive_agentic_rag.ingestion.normalizer import IngestionNormalizer
from responsive_agentic_rag.ingestion.orchestrator import IngestionOrchestrator
from responsive_agentic_rag.retrieval.qdrant_client import create_qdrant_client
from responsive_agentic_rag.retrieval.qdrant_store import QdrantVectorStore


ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_DIRECTORY = (
    ROOT / "data" / "demo_corpus" / "documents"
)

WEBSITE_DIRECTORY = (
    ROOT / "data" / "demo_corpus" / "websites"
)

DEMO_COLLECTION = "demo_knowledge_chunks"

WEBSITE_BASE_URL = "http://localhost:8000"


def create_vector_store() -> QdrantVectorStore:
    """Create and initialize the Qdrant store used by the demo corpus."""
    client = create_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=DEMO_COLLECTION,
        vector_dimension=settings.embedding_dimension,
    )

    vector_store.ensure_collection()

    return vector_store

def create_pipeline(
    connector,
) -> IngestionOrchestrator:
    """Build the shared ingestion pipeline."""
    return IngestionOrchestrator(
        connector=connector,
        normalizer=IngestionNormalizer(),
        chunker=TextChunker(),
        embedding_service=EmbeddingService(
            model_name=settings.embedding_model,
        ),
        vector_store=create_vector_store(),
    )


def ingest_documents() -> list:
    """Ingest the generated PDF corpus."""
    connector = DocumentConnector(
        document_directory=DOCUMENT_DIRECTORY,
    )

    pipeline = create_pipeline(connector)

    return pipeline.ingest_all()


def ingest_websites() -> list:
    """Ingest generated HTML pages through WebsiteConnector."""
    website_urls = [
        f"{WEBSITE_BASE_URL}/{path.name}"
        for path in sorted(
            WEBSITE_DIRECTORY.glob("*.html")
        )
    ]

    connector = WebsiteConnector(
        urls=website_urls,
    )

    pipeline = create_pipeline(connector)

    return pipeline.ingest_all()


def print_results(
    source_type: str,
    results: list,
) -> None:
    """Print ingestion statistics."""
    source_count = len(results)

    record_count = sum(
        result.records_created
        for result in results
    )

    chunk_count = sum(
        result.chunks_created
        for result in results
    )

    print()
    print("=" * 60)
    print(f"{source_type.upper()} INGESTION")
    print("=" * 60)

    print(f"Sources : {source_count}")
    print(f"Records : {record_count}")
    print(f"Chunks  : {chunk_count}")

    print()
    print("Sample sources:")

    for result in results[:5]:
        print(
            f"  {result.source_location}"
            f" | records={result.records_created}"
            f" | chunks={result.chunks_created}"
        )


def main() -> None:
    print()
    print("=" * 60)
    print("RESPONSIVE AGENTIC RAG")
    print("DEMO CORPUS INGESTION")
    print("=" * 60)

    if not DOCUMENT_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Demo document directory not found: "
            f"{DOCUMENT_DIRECTORY}"
        )

    if not WEBSITE_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Demo website directory not found: "
            f"{WEBSITE_DIRECTORY}"
        )

    document_count = len(
        list(
            DOCUMENT_DIRECTORY.glob("*.pdf")
        )
    )

    website_count = len(
        list(
            WEBSITE_DIRECTORY.glob("*.html")
        )
    )

    print()
    print(f"PDF documents : {document_count}")
    print(f"Website pages : {website_count}")
    print(f"Qdrant collection: {DEMO_COLLECTION}")

    if document_count != 50:
        raise RuntimeError(
            f"Expected 50 PDF documents, "
            f"found {document_count}"
        )

    if website_count != 30:
        raise RuntimeError(
            f"Expected 30 website pages, "
            f"found {website_count}"
        )

    # ---------------------------------------------------------
    # Documents
    # ---------------------------------------------------------

    document_results = ingest_documents()

    print_results(
        "document",
        document_results,
    )

    # ---------------------------------------------------------
    # Websites
    # ---------------------------------------------------------

    website_results = ingest_websites()

    print_results(
        "website",
        website_results,
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    all_results = (
        document_results
        + website_results
    )

    total_sources = len(all_results)

    total_records = sum(
        result.records_created
        for result in all_results
    )

    total_chunks = sum(
        result.chunks_created
        for result in all_results
    )

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(f"Total sources : {total_sources}")
    print(f"Total records : {total_records}")
    print(f"Total chunks  : {total_chunks}")
    print(f"Qdrant        : {DEMO_COLLECTION}")

    print()
    print(
        "The complete demo corpus is now indexed."
    )


if __name__ == "__main__":
    main()