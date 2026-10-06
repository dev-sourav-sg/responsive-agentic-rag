from qdrant_client import QdrantClient

from responsive_agentic_rag.config.settings import settings


def create_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from application configuration."""
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )