from responsive_agentic_rag.connectors.base import (
    KnowledgeSourceConfig,
    SourceConnector,
)
from responsive_agentic_rag.connectors.registry import ConnectorRegistry


class FakeDocumentConnector:
    source_type = "document"

    def list_sources(self):
        return [
            KnowledgeSourceConfig(
                source_type="document",
                source_location="data/documents/example.pdf",
            )
        ]

    def fetch_content(self, source):
        return {"content": "example document"}


class FakeWebsiteConnector:
    source_type = "website"

    def list_sources(self):
        return [
            KnowledgeSourceConfig(
                source_type="website",
                source_location="https://example.com",
            )
        ]

    def fetch_content(self, source):
        return {"content": "example website"}


def test_source_connector_contract():
    connector = FakeDocumentConnector()

    assert isinstance(connector, SourceConnector)
    assert connector.source_type == "document"

    sources = connector.list_sources()

    assert len(sources) == 1
    assert sources[0].source_type == "document"

    content = connector.fetch_content(sources[0])

    assert content["content"] == "example document"


def test_connector_registry():
    registry = ConnectorRegistry()

    document_connector = FakeDocumentConnector()
    website_connector = FakeWebsiteConnector()

    registry.register(document_connector)
    registry.register(website_connector)

    assert registry.get("document") is document_connector
    assert registry.get("website") is website_connector
    assert registry.source_types() == ["document", "website"]


def test_connector_registry_rejects_unknown_source_type():
    registry = ConnectorRegistry()

    try:
        registry.get("sharepoint")
        assert False, "Expected ValueError for unknown connector"
    except ValueError as exc:
        assert "sharepoint" in str(exc)
