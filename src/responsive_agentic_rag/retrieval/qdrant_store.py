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

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 10,
    ) -> list[RetrievalCandidate]:
        """Search for similar chunks."""
        raise NotImplementedError

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