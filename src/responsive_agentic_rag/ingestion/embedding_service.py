from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from responsive_agentic_rag.config.settings import settings
from responsive_agentic_rag.models.knowledge import ChunkRecord


class EmbeddingService:
    """Generate embeddings using the configured sentence-transformer model."""

    def __init__(
        self,
        model_name: str = settings.embedding_model,
    ) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding model."""
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        if not text.strip():
            raise ValueError("text cannot be empty")

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_many(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError(
                "texts cannot contain empty values"
            )

        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_chunks(
        self,
        chunks: list[ChunkRecord],
    ) -> list[ChunkRecord]:
        """Generate embeddings for chunk records."""

        if not chunks:
            return []

        embeddings = self.embed_many(
            [chunk.content for chunk in chunks]
        )

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks            
                    
        
            
        
            
    