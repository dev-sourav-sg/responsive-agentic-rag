from typing import Sequence
from urllib.request import Request, urlopen

from responsive_agentic_rag.connectors.base import (
    KnowledgeSourceConfig,
    SourceConnector,
)


class WebsiteConnector:
    """Connector for discovering and fetching configured website URLs."""

    source_type = "website"

    def __init__(self, urls: Sequence[str]) -> None:
        self.urls = list(urls)

    def list_sources(self) -> Sequence[KnowledgeSourceConfig]:
        """Return configured website URLs as source configurations."""
        return [
            KnowledgeSourceConfig(
                source_type="website",
                source_location=url,
                metadata={
                    "url": url,
                },
            )
            for url in self.urls
        ]

    def fetch_content(self, source: KnowledgeSourceConfig) -> object:
        """Fetch raw website content for a configured URL."""
        request = Request(
            source.source_location,
            headers={
                "User-Agent": "responsive-agentic-rag/0.1",
            },
        )

        with urlopen(request, timeout=15) as response:
            return response.read()


assert isinstance(WebsiteConnector([]), SourceConnector)