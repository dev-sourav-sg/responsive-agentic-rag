from unittest.mock import patch

from responsive_agentic_rag.retrieval.qdrant_client import (
    create_qdrant_client,
)


def test_create_qdrant_client_uses_application_settings():
    with patch(
        "responsive_agentic_rag.retrieval.qdrant_client.QdrantClient"
    ) as mock_client:
        create_qdrant_client()

        mock_client.assert_called_once_with(
            host="localhost",
            port=6333,
            api_key=None,
        )