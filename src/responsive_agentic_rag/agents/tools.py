from typing import Protocol, runtime_checkable

from responsive_agentic_rag.ingestion.embedding_service import EmbeddingService
from responsive_agentic_rag.models.retrieval import EvidenceSet, RetrievalQuery
from responsive_agentic_rag.retrieval.evidence_sufficiency import (
    EvidenceSufficiency,
)
from responsive_agentic_rag.retrieval.hybrid_search import HybridRetriever


@runtime_checkable
class RetrievalToolContract(Protocol):
    """Contract exposed to agent orchestration for deterministic retrieval."""

    def retrieve(self, query: RetrievalQuery) -> EvidenceSet:
        """Retrieve grounded evidence for a query."""
        ...


class RetrievalTool:
    """Agent-facing boundary for deterministic evidence retrieval."""

    def __init__(
        self,
        retriever: HybridRetriever,
        embedding_service: EmbeddingService,
        evidence_sufficiency: EvidenceSufficiency,
    ) -> None:
        self._retriever = retriever
        self._embedding_service = embedding_service
        self._evidence_sufficiency = evidence_sufficiency

    def retrieve(self, query: RetrievalQuery) -> EvidenceSet:
        """Embed the query, retrieve evidence, and assess sufficiency."""
        query_vector = self._embedding_service.embed(query.query_text)

        candidates = self._retriever.search(
            query=query.query_text,
            query_vector=query_vector,
            limit=query.max_results,
        )

        sufficiency_assessment = self._evidence_sufficiency.assess(
            candidates
        )

        supporting_sources = list(
            dict.fromkeys(candidate.source_id for candidate in candidates)
        )

        return EvidenceSet(
            query_id=query.query_text,
            candidates=candidates,
            sufficiency_assessment=sufficiency_assessment,
            supporting_sources=supporting_sources,
        )