from __future__ import annotations

import hashlib
from pathlib import Path

from responsive_agentic_rag.ingestion.chunker import TextChunker
from responsive_agentic_rag.ingestion.embedding_service import EmbeddingService
from responsive_agentic_rag.ingestion.normalizer import IngestionNormalizer
from responsive_agentic_rag.models.knowledge import ChunkRecord, KnowledgeSource
from responsive_agentic_rag.retrieval.vector_store import VectorStore


class RuntimeIngestionService:
    """
    Ingest user-uploaded PDF/HTML content through the shared ingestion pipeline.

    Runtime flow:

        Uploaded file
            ↓
        KnowledgeSource
            ↓
        Normalization
            ↓
        Structure-aware chunking
            ↓
        Embedding
            ↓
        Vector store
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": "document",
        ".html": "website",
        ".htm": "website",
    }

    def __init__(
        self,
        normalizer: IngestionNormalizer,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.normalizer = normalizer
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest_uploaded_file(
        self,
        file_name: str,
        content: bytes,
        title: str | None = None,
        document_id: str | None = None,
        version: str | None = None,
        owner: str | None = None,
        approval_status: str = "approved",
        authority: float = 1.0,
    ) -> list[ChunkRecord]:
        """
        Ingest one uploaded PDF or HTML file.

        The uploaded file is already the source-acquisition boundary,
        so we do not need to persist it to disk or invoke a connector.
        Everything after acquisition uses the same shared ingestion
        components as the batch ingestion pipeline.
        """

        if not file_name:
            raise ValueError("file_name cannot be empty")

        if not content:
            raise ValueError(
                f"Uploaded file '{file_name}' is empty"
            )

        extension = Path(file_name).suffix.lower()

        source_type = self.SUPPORTED_EXTENSIONS.get(extension)

        if source_type is None:
            supported = ", ".join(
                sorted(self.SUPPORTED_EXTENSIONS.keys())
            )
            raise ValueError(
                f"Unsupported file type '{extension}'. "
                f"Supported types: {supported}"
            )

        if not 0.0 <= authority <= 1.0:
            raise ValueError(
                "authority must be between 0.0 and 1.0"
            )

        source_id = (
            f"upload:{document_id.strip()}"
            if document_id and document_id.strip()
            else self._build_source_id(
                file_name=file_name,
                content=content,
            )
        )

        source_title = (
            title.strip()
            if title and title.strip()
            else Path(file_name).stem
        )

        metadata = {
            "file_name": file_name,
            "source_type": source_type,
            "source_location": file_name,
            "document_id": document_id,
            "version": version,
            "owner": owner,
            "approval_status": approval_status,
            "authority": authority,
            "ingestion_mode": "runtime_upload",
        }

        source = KnowledgeSource(
            source_id=source_id,
            source_type=source_type,
            title=source_title,
            source_location=file_name,
            version=version,
            owner=owner,
            publication_date=None,
            modification_date=None,
            authority=authority,
            approval_status=approval_status,
            metadata=metadata,
        )

        # ---------------------------------------------------------------
        # 1. Normalize
        # ---------------------------------------------------------------

        records = self.normalizer.normalize(
            source=source,
            raw_content=content,
        )

        if not records:
            raise ValueError(
                f"No readable content was extracted from '{file_name}'"
            )

        # ---------------------------------------------------------------
        # 2. Chunk
        # ---------------------------------------------------------------

        chunks: list[ChunkRecord] = []

        for record in records:
            chunks.extend(
                self.chunker.chunk(record)
            )

        if not chunks:
            raise ValueError(
                f"No chunks were generated from '{file_name}'"
            )

        # ---------------------------------------------------------------
        # 3. Embed
        # ---------------------------------------------------------------

        embedded_chunks = self.embedding_service.embed_chunks(
            chunks
        )

        # ---------------------------------------------------------------
        # 4. Persist in Qdrant
        # ---------------------------------------------------------------

        self.vector_store.upsert(
            embedded_chunks
        )

        return embedded_chunks

    @staticmethod
    def _build_source_id(
        file_name: str,
        content: bytes,
    ) -> str:
        """
        Build a deterministic source ID when the user does not provide
        a document ID.
        """

        content_hash = hashlib.sha256(
            content
        ).hexdigest()[:16]

        return (
            f"upload:{Path(file_name).stem}:"
            f"{content_hash}"
        )