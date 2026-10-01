from responsive_agentic_rag.connectors.base import KnowledgeSourceConfig
from responsive_agentic_rag.ingestion.orchestrator import (
    IngestionOrchestrator,
)
from responsive_agentic_rag.models.knowledge import (
    ChunkRecord,
    NormalizedSourceRecord,
)


class FakeConnector:
    source_type = "document"

    def __init__(self):
        self.sources = [
            KnowledgeSourceConfig(
                source_type="document",
                source_location="test-document.pdf",
                metadata={
                    "source_id": "doc-001",
                    "title": "Test Document",
                    "authority": 0.8,
                },
            )
        ]

    def list_sources(self):
        return self.sources

    def fetch_content(self, source):
        return b"raw document content"


class FakeNormalizer:
    def normalize(self, source, raw_content):
        assert raw_content == b"raw document content"

        return [
            NormalizedSourceRecord(
                record_id="record-001",
                source_id=source.source_id,
                source_type=source.source_type,
                title=source.title,
                body_text="Test normalized content.",
                metadata=source.metadata,
            )
        ]


class FakeChunker:
    def chunk(self, record):
        return [
            ChunkRecord(
                chunk_id="chunk-001",
                source_id=record.source_id,
                record_id=record.record_id,
                content=record.body_text,
                chunk_index=0,
                token_count=3,
                metadata=record.metadata,
                authority_score=0.0
            )
        ]


class FakeEmbeddingService:
    def embed_chunks(self, chunks):
        for chunk in chunks:
            chunk.embedding = [0.1, 0.2, 0.3]

        return chunks


class FakeVectorStore:
    def __init__(self):
        self.upserted_chunks = []

    def upsert(self, chunks):
        self.upserted_chunks.extend(chunks)


def test_ingest_source_coordinates_pipeline():
    vector_store = FakeVectorStore()

    orchestrator = IngestionOrchestrator(
        connector=FakeConnector(),
        normalizer=FakeNormalizer(),
        chunker=FakeChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    source = FakeConnector().list_sources()[0]

    result = orchestrator.ingest_source(source)

    assert result.source_id == "doc-001"
    assert result.source_location == "test-document.pdf"
    assert result.records_created == 1
    assert result.chunks_created == 1

    assert len(vector_store.upserted_chunks) == 1
    assert vector_store.upserted_chunks[0].embedding == [0.1, 0.2, 0.3]


def test_ingest_all_processes_all_sources():
    connector = FakeConnector()
    vector_store = FakeVectorStore()

    orchestrator = IngestionOrchestrator(
        connector=connector,
        normalizer=FakeNormalizer(),
        chunker=FakeChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    results = orchestrator.ingest_all()

    assert len(results) == 1
    assert results[0].source_id == "doc-001"
    assert len(vector_store.upserted_chunks) == 1