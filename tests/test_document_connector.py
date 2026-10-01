from pathlib import Path

from responsive_agentic_rag.connectors.base import KnowledgeSourceConfig
from responsive_agentic_rag.connectors.document_connector import DocumentConnector


DOCUMENT_DIRECTORY = Path("data/documents")


def test_document_connector_discovers_pdf_documents():
    connector = DocumentConnector(DOCUMENT_DIRECTORY)

    sources = connector.list_sources()

    assert len(sources) == 4
    assert all(source.source_type == "document" for source in sources)
    assert all(source.source_location.endswith(".pdf") for source in sources)


def test_document_connector_fetches_pdf_bytes():
    connector = DocumentConnector(DOCUMENT_DIRECTORY)

    sources = connector.list_sources()
    source = sources[0]

    content = connector.fetch_content(source)

    assert isinstance(content, bytes)
    assert len(content) > 0
    assert content.startswith(b"%PDF")


def test_document_connector_rejects_missing_document():
    connector = DocumentConnector(DOCUMENT_DIRECTORY)

    source = KnowledgeSourceConfig(
        source_type="document",
        source_location="data/documents/does_not_exist.pdf",
    )

    try:
        connector.fetch_content(source)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
