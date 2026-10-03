import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from responsive_agentic_rag.config.settings import settings
from responsive_agentic_rag.models.knowledge import ChunkRecord
from responsive_agentic_rag.models.retrieval import RetrievalCandidate
from responsive_agentic_rag.retrieval.vector_store import VectorStore


class QdrantVectorStore:
    """Qdrant-backed implementation of the VectorStore contract."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = settings.qdrant_collection,
        vector_dimension: int = settings.embedding_dimension,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = self.client.get_collections()

        existing_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name in existing_names:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dimension,
                distance=Distance.COSINE,
            ),
        )

    def upsert(self, chunks: Sequence[ChunkRecord]) -> None:
        """Insert or update embedded chunks."""
        if not chunks:
            return

        points: list[PointStruct] = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} does not have an embedding"
                )

            points.append(
                PointStruct(
                    id=self._build_point_id(chunk.chunk_id),
                    vector=chunk.embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "source_id": chunk.source_id,
                        "record_id": chunk.record_id,
                        "source_type": chunk.metadata.get(
                            "source_type",
                            "unknown",
                        ),
                        "source_location": chunk.metadata.get(
                            "source_location",
                            "unknown",
                        ),
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "section_path": chunk.section_path,
                        "token_count": chunk.token_count,
                        "authority_score": chunk.authority_score,
                        "metadata": chunk.metadata,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self, query_vector: Sequence[float],limit: int = 10,) -> list[RetrievalCandidate]:
        """Search for semantically similar chunks."""
        if not query_vector:
            raise ValueError("query_vector cannot be empty")

        if limit < 1:
            raise ValueError("limit must be greater than 0")

        if len(query_vector) != self.vector_dimension:
            raise ValueError(
                f"query_vector dimension {len(query_vector)} "
                f"does not match configured dimension "
                f"{self.vector_dimension}"
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            limit=limit,
            with_payload=True,
        ).points

        candidates: list[RetrievalCandidate] = []

        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}

            candidates.append(
                RetrievalCandidate(
                    chunk_id=str(payload["chunk_id"]),
                    source_id=str(payload["source_id"]),
                    source_type=payload["source_type"],
                    source_location=str(payload["source_location"]),
                    content=str(payload["content"]),
                    metadata=dict(payload.get("metadata", {})),
                    semantic_score=float(result.score),
                    lexical_score=0.0,
                    authority_score=float(
                        payload.get("authority_score", 0.0)
                    ),
                    combined_score=float(result.score),
                    rank=rank,
                )
            )

        return candidates

    def _build_point_id(self, chunk_id: str) -> str:
        """Create a deterministic UUID for a Qdrant point."""
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"responsive-agentic-rag:{chunk_id}",
            )
        )


assert isinstance(
    QdrantVectorStore(
        client=QdrantClient(":memory:")
    ),
    VectorStore,
)