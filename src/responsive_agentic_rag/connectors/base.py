from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from responsive_agentic_rag.models.knowledge import SourceType


class KnowledgeSourceConfig(BaseModel):
    """Configuration required for a connector to acquire a source."""

    source_type: SourceType
    source_location: str
    metadata: dict[str, Any] = Field(default_factory=dict)

@runtime_checkable
class SourceConnector(Protocol):
    """Contract implemented by every source-specific connector."""

    source_type: str

    def list_sources(self) -> Sequence[KnowledgeSourceConfig]:
        """Return the sources available through this connector."""
        ...

    def fetch_content(self, source: KnowledgeSourceConfig) -> object:
        """Fetch raw content for a configured source."""
        ...
    