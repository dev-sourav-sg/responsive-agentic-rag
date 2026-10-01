from unittest.mock import patch

from responsive_agentic_rag.connectors.base import KnowledgeSourceConfig
from responsive_agentic_rag.connectors.website_connector import WebsiteConnector


WEBSITE_URLS = [
    "https://example.com/policy",
    "https://example.com/settlement",
]


def test_website_connector_lists_configured_urls():
    connector = WebsiteConnector(WEBSITE_URLS)

    sources = connector.list_sources()

    assert len(sources) == 2
    assert all(source.source_type == "website" for source in sources)
    assert sources[0].source_location == WEBSITE_URLS[0]
    assert sources[1].source_location == WEBSITE_URLS[1]


@patch("responsive_agentic_rag.connectors.website_connector.urlopen")
def test_website_connector_fetches_raw_content(mock_urlopen):
    mock_response = mock_urlopen.return_value.__enter__.return_value
    mock_response.read.return_value = b"<html><body>Test content</body></html>"

    connector = WebsiteConnector(WEBSITE_URLS)

    source = connector.list_sources()[0]
    content = connector.fetch_content(source)

    assert isinstance(content, bytes)
    assert content == b"<html><body>Test content</body></html>"
    mock_urlopen.assert_called_once()


def test_website_connector_preserves_url_metadata():
    connector = WebsiteConnector(WEBSITE_URLS)

    source = connector.list_sources()[0]

    assert source.metadata["url"] == WEBSITE_URLS[0]