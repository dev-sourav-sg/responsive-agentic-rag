from dataclasses import dataclass

from responsive_agentic_rag.connectors.base import (
    KnowledgeSourceConfig,
    SourceConnector,
)
from responsive_agentic_rag.ingestion.chunker import TextChunker
from responsive_agentic_rag.ingestion.embedding_service import EmbeddingService
from responsive_agentic_rag.ingestion.normalizer import IngestionNormalizer
from responsive_agentic_rag.models.knowledge import KnowledgeSource
from responsive_agentic_rag.retrieval.vector_store import VectorStore


@dataclass(frozen=True)
class IngestionResult:
    source_id: str
    source_location: str
    records_created: int
    chunks_created: int


class IngestionOrchestrator:
    """Coordinate source acquisition and the shared ingestion pipeline."""

    def __init__(
        self,
        connector: SourceConnector,
        normalizer: IngestionNormalizer,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.connector = connector
        self.normalizer = normalizer
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest_source(
        self,
        source_config: KnowledgeSourceConfig,
    ) -> IngestionResult:
        raw_content = self.connector.fetch_content(source_config)

        source = self._build_knowledge_source(source_config)

        records = self.normalizer.normalize(
            source,
            raw_content,
        )

        chunks = []

        for record in records:
            chunks.extend(
                self.chunker.chunk(record)
            )

        embedded_chunks = self.embedding_service.embed_chunks(
            chunks
        )

        self.vector_store.upsert(
            embedded_chunks
        )

        return IngestionResult(
            source_id=source.source_id,
            source_location=source.source_location,
            records_created=len(records),
            chunks_created=len(embedded_chunks),
        )

    def ingest_all(self) -> list[IngestionResult]:
        """Ingest every source exposed by the connector."""
        results = []

        for source_config in self.connector.list_sources():
            results.append(
                self.ingest_source(source_config)
            )

        return results

    def _build_knowledge_source(
        self,
        source_config: KnowledgeSourceConfig,
    ) -> KnowledgeSource:
        metadata = dict(source_config.metadata)

        source_id = metadata.get("source_id")
        if not source_id:
            source_id = (
                f"{source_config.source_type}:"
                f"{source_config.source_location}"
            )

        title = metadata.get(
            "title",
            source_config.source_location,
        )

        return KnowledgeSource(
            source_id=source_id,
            source_type=source_config.source_type,
            title=title,
            source_location=source_config.source_location,
            version=metadata.get("version"),
            owner=metadata.get("owner"),
            publication_date=metadata.get("publication_date"),
            modification_date=metadata.get("modification_date"),
            authority=metadata.get("authority", 0.5),
            approval_status=metadata.get("approval_status"),
            metadata=metadata,
        )