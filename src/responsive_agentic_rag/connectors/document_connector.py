from pathlib import Path
from typing import Sequence

from responsive_agentic_rag.connectors.base import (
    KnowledgeSourceConfig,
    SourceConnector,
)


class DocumentConnector:
    """Connector for discovering and reading local document files."""

    source_type = "document"

    def __init__(self, document_directory: Path) -> None:
        self.document_directory = document_directory

    def list_sources(self) -> Sequence[KnowledgeSourceConfig]:
        """Return the documents available in the configured directory."""
        sources = []

        for path in sorted(self.document_directory.glob("*.pdf")):
            sources.append(
                KnowledgeSourceConfig(
                    source_type="document",
                    source_location=str(path),
                    metadata={
                        "file_name": path.name,
                        "file_size": path.stat().st_size,
                    },
                )
            )

        return sources

    def fetch_content(self, source: KnowledgeSourceConfig) -> object:
        """Read the raw bytes of a configured document."""
        path = Path(source.source_location)

        if not path.is_file():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        return path.read_bytes()


assert isinstance(DocumentConnector(Path("data/documents")), SourceConnector)