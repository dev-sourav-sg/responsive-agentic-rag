from responsive_agentic_rag.config.settings import settings
from pathlib import Path

def test_default_settings():
    assert settings.app_environment == "local"
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.qdrant_collection == "knowledge_chunks"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_dimension == 384
    assert settings.document_directory == Path("data/documents")
    assert settings.website_urls == []
