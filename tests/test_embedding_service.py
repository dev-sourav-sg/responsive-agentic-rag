from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from responsive_agentic_rag.connectors.document_connector import (
    DocumentConnector,
)
from responsive_agentic_rag.ingestion.embedding_service import (
    EmbeddingService,
)
from responsive_agentic_rag.ingestion.normalizer import (
    IngestionNormalizer,
)
from responsive_agentic_rag.ingestion.chunker import (
    TextChunker,
)
from responsive_agentic_rag.models.knowledge import KnowledgeSource


@patch(
    "responsive_agentic_rag.ingestion.embedding_service.SentenceTransformer"
)
def test_embedding_service_uses_configured_model(mock_model_class):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = [0.1, 0.2, 0.3]

    mock_model_class.return_value = mock_model

    service = EmbeddingService(
        model_name="test-model"
    )

    mock_model_class.assert_called_once_with("test-model")
    assert service.model_name == "test-model"


@patch(
    "responsive_agentic_rag.ingestion.embedding_service.SentenceTransformer"
)
def test_embedding_service_returns_single_embedding(
    mock_model_class,
):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = MagicMock(
        tolist=lambda: [0.1, 0.2, 0.3]
    )

    mock_model_class.return_value = mock_model

    service = EmbeddingService(
        model_name="test-model"
    )

    result = service.embed("payment settlement")

    assert result == [0.1, 0.2, 0.3]

    mock_model.encode.assert_called_once_with(
        "payment settlement",
        normalize_embeddings=True,
    )


@patch(
    "responsive_agentic_rag.ingestion.embedding_service.SentenceTransformer"
)
def test_embedding_service_returns_multiple_embeddings(
    mock_model_class,
):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = MagicMock(
        tolist=lambda: [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    mock_model_class.return_value = mock_model

    service = EmbeddingService(
        model_name="test-model"
    )

    result = service.embed_many(
        [
            "payment",
            "settlement",
        ]
    )

    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    mock_model.encode.assert_called_once_with(
        [
            "payment",
            "settlement",
        ],
        normalize_embeddings=True,
    )


@patch(
    "responsive_agentic_rag.ingestion.embedding_service.SentenceTransformer"
)
def test_embedding_service_returns_dimension(
    mock_model_class,
):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384

    mock_model_class.return_value = mock_model

    service = EmbeddingService(
        model_name="test-model"
    )

    assert service.dimension == 384


@patch(
    "responsive_agentic_rag.ingestion.embedding_service.SentenceTransformer"
)
def test_embedding_service_rejects_empty_text(
    mock_model_class,
):
    mock_model_class.return_value = MagicMock()

    service = EmbeddingService(
        model_name="test-model"
    )

    with pytest.raises(ValueError, match="text cannot be empty"):
        service.embed("")


@patch(
    "responsive_agentic_rag.ingestion.embedding_service.SentenceTransformer"
)
def test_embedding_service_rejects_empty_text_in_batch(
    mock_model_class,
):
    mock_model_class.return_value = MagicMock()

    service = EmbeddingService(
        model_name="test-model"
    )

    with pytest.raises(
        ValueError,
        match="texts cannot contain empty values",
    ):
        service.embed_many(
            [
                "payment",
                "",
                "settlement",
            ]
        )


@patch(
    "responsive_agentic_rag.ingestion.embedding_service.SentenceTransformer"
)
def test_embedding_service_handles_empty_batch(
    mock_model_class,
):
    mock_model_class.return_value = MagicMock()

    service = EmbeddingService(
        model_name="test-model"
    )

    assert service.embed_many([]) == []


def test_t010_end_to_end_document_to_embedded_chunks():
    # 1. Acquire document
    connector = DocumentConnector(
        Path("data/documents")
    )

    source_config = connector.list_sources()[0]
    raw_content = connector.fetch_content(source_config)

    # 2. Build canonical source
    source = KnowledgeSource(
        source_id="t010-doc-001",
        source_type="document",
        title="T010 Integration Document",
        source_location=source_config.source_location,
        metadata=source_config.metadata,
    )

    # 3. Normalize
    normalizer = IngestionNormalizer()

    records = normalizer.normalize(
        source,
        raw_content,
    )

    assert records
    assert records[0].body_text

    # 4. Chunk
    chunker = TextChunker(
        max_chunk_characters=500,
        overlap_characters=50,
    )

    chunks = chunker.chunk(records[0])

    assert chunks
    assert all(chunk.content for chunk in chunks)

    # 5. Generate embeddings
    embedding_service = EmbeddingService()

    embedded_chunks = embedding_service.embed_chunks(
        chunks
    )

    # 6. Validate final T010 output
    assert len(embedded_chunks) == len(chunks)

    for chunk in embedded_chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384
        assert all(
            isinstance(value, float)
            for value in chunk.embedding
        )

        norm = sum(
            value * value
            for value in chunk.embedding
        ) ** 0.5

        assert abs(norm - 1.0) < 1e-5