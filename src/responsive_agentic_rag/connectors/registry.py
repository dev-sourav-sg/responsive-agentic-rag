from responsive_agentic_rag.connectors.base import SourceConnector


class ConnectorRegistry:
    """Registry for source-specific connectors."""

    def __init__(self) -> None:
        self._connectors: dict[str, SourceConnector] = {}

    def register(self, connector: SourceConnector) -> None:
        self._connectors[connector.source_type] = connector

    def get(self, source_type: str) -> SourceConnector:
        try:
            return self._connectors[source_type]
        except KeyError as exc:
            raise ValueError(
                f"No connector registered for source type: {source_type}"
            ) from exc

    def source_types(self) -> list[str]:
        return list(self._connectors.keys())
