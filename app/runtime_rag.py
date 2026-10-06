from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient

from responsive_agentic_rag.ingestion.embedding_service import EmbeddingService
from responsive_agentic_rag.models.knowledge import ChunkRecord
from responsive_agentic_rag.models.retrieval import RetrievalCandidate
from responsive_agentic_rag.retrieval.bm25_index import BM25Index
from responsive_agentic_rag.retrieval.hybrid_search import HybridRetriever
from responsive_agentic_rag.retrieval.qdrant_store import QdrantVectorStore


class RuntimeRetrievalService:
    """Application-level retrieval service for the live Streamlit demo.

    Qdrant remains the persistent source of truth for vectors/chunks.
    BM25 is rebuilt from Qdrant payloads so runtime uploads become
    immediately searchable by both retrieval lanes.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_index = BM25Index()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
        )
        self.refresh_bm25()

    def refresh_bm25(self) -> int:
        chunks = self._load_chunks_from_qdrant()
        self.bm25_index.build(chunks)
        return len(chunks)

    def add_uploaded_chunks(self, chunks: Sequence[ChunkRecord]) -> int:
        """Rebuild BM25 from Qdrant after a successful upload.

        Re-reading Qdrant keeps BM25 aligned with the persistent vector
        store and avoids maintaining two independent document inventories.
        """
        return self.refresh_bm25()

    def retrieve(
        self,
        question: str,
        semantic_limit: int = 10,
        lexical_limit: int = 10,
        final_limit: int = 10,
    ) -> dict[str, list[RetrievalCandidate]]:
        if not question.strip():
            raise ValueError("question cannot be empty")

        query_vector = self.embedding_service.embed(question)

        semantic = self.vector_store.search(
            query_vector=query_vector,
            limit=semantic_limit,
        )

        lexical_results = self.bm25_index.search(
            query=question,
            limit=lexical_limit,
        )
        lexical_chunks = self.vector_store.get_chunks(
            [chunk_id for chunk_id, _ in lexical_results]
        )
        lexical_by_id = {chunk.chunk_id: chunk for chunk in lexical_chunks}

        lexical: list[RetrievalCandidate] = []
        for chunk_id, score in lexical_results:
            chunk = lexical_by_id.get(chunk_id)
            if chunk is None:
                continue

            metadata = dict(chunk.metadata)
            metadata.setdefault("source_type", metadata.get("source_type", "unknown"))
            metadata.setdefault(
                "source_location",
                metadata.get("source_location", "unknown"),
            )

            lexical.append(
                RetrievalCandidate(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_type=metadata.get("source_type", "unknown"),
                    source_location=metadata.get("source_location", "unknown"),
                    content=chunk.content,
                    metadata=metadata,
                    semantic_score=0.0,
                    lexical_score=float(score),
                    authority_score=chunk.authority_score,
                    combined_score=0.0,
                    rank=len(lexical) + 1,
                )
            )

        final = self.hybrid_retriever.search(
            query=question,
            query_vector=query_vector,
            limit=final_limit,
            semantic_limit=max(semantic_limit, 30),
            lexical_limit=max(lexical_limit, 30),
        )

        return {
            "semantic": list(semantic),
            "lexical": lexical,
            "final": list(final),
        }

    def _load_chunks_from_qdrant(self) -> list[ChunkRecord]:
        """Reconstruct ChunkRecord objects from Qdrant payloads for BM25."""
        chunks: list[ChunkRecord] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                metadata = dict(payload.get("metadata") or {})

                # Preserve top-level Qdrant fields as metadata too.
                metadata.setdefault("source_type", payload.get("source_type", "unknown"))
                metadata.setdefault(
                    "source_location",
                    payload.get("source_location", "unknown"),
                )

                chunks.append(
                    ChunkRecord(
                        chunk_id=str(payload["chunk_id"]),
                        source_id=str(payload["source_id"]),
                        record_id=str(payload["record_id"]),
                        content=str(payload["content"]),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        start_offset=None,
                        end_offset=None,
                        section_path=list(payload.get("section_path") or []),
                        token_count=int(payload.get("token_count", 0)),
                        embedding=None,
                        metadata=metadata,
                        authority_score=float(payload.get("authority_score", 0.0)),
                    )
                )

            if offset is None:
                break

        return chunks
