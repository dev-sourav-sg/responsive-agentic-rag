from responsive_agentic_rag.config.settings import settings
from responsive_agentic_rag.retrieval.qdrant_client import (
    create_qdrant_client,
)
from responsive_agentic_rag.retrieval.qdrant_store import (
    QdrantVectorStore,
)


def test_qdrant_store_connects_to_real_instance():
    client = create_qdrant_client()

    store = QdrantVectorStore(
        client=client,
        collection_name="t011_integration_test",
        vector_dimension=settings.embedding_dimension,
    )

    collections = client.get_collections()

    assert collections is not None

    store.ensure_collection()

    collection_names = {
        collection.name
        for collection in client.get_collections().collections
    }

    assert "t011_integration_test" in collection_names