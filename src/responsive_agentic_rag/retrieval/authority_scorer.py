from typing import Any

from responsive_agentic_rag.models.retrieval import RetrievalCandidate


class AuthorityScorer:
    """Deterministically calculates source authority for retrieval candidates."""

    DEFAULT_APPROVAL_MULTIPLIERS = {
        "approved": 1.0,
        "pending": 0.75,
        "review": 0.75,
        "draft": 0.50,
        "deprecated": 0.25,
    }

    DEFAULT_SOURCE_TYPE_MULTIPLIERS = {
        "document": 1.0,
        "website": 0.90,
    }

    def __init__(
        self,
        approval_multipliers: dict[str, float] | None = None,
        source_type_multipliers: dict[str, float] | None = None,
        unknown_approval_multiplier: float = 0.50,
        unknown_source_type_multiplier: float = 1.0,
    ) -> None:
        self._approval_multipliers = (
            approval_multipliers
            or self.DEFAULT_APPROVAL_MULTIPLIERS.copy()
        )
        self._source_type_multipliers = (
            source_type_multipliers
            or self.DEFAULT_SOURCE_TYPE_MULTIPLIERS.copy()
        )
        self._unknown_approval_multiplier = unknown_approval_multiplier
        self._unknown_source_type_multiplier = unknown_source_type_multiplier

        self._validate_multipliers()

    def score(
        self,
        base_authority: float,
        metadata: dict[str, Any],
    ) -> float:
        if not 0.0 <= base_authority <= 1.0:
            raise ValueError(
                "base_authority must be between 0.0 and 1.0"
            )

        approval_status = str(
            metadata.get("approval_status", "")
        ).strip().lower()

        source_type = str(
            metadata.get("source_type", "")
        ).strip().lower()

        approval_multiplier = self._approval_multipliers.get(
            approval_status,
            self._unknown_approval_multiplier,
        )

        source_type_multiplier = self._source_type_multipliers.get(
            source_type,
            self._unknown_source_type_multiplier,
        )

        score = (
            base_authority
            * approval_multiplier
            * source_type_multiplier
        )

        return max(0.0, min(1.0, score))

    def enrich_candidate(
        self,
        candidate: RetrievalCandidate,
    ) -> RetrievalCandidate:
        effective_authority = self.score(
            base_authority=candidate.authority_score,
            metadata=candidate.metadata,
        )

        return candidate.model_copy(
            update={
                "authority_score": effective_authority,
            }
        )

    def _validate_multipliers(self) -> None:
        multipliers = [
            *self._approval_multipliers.values(),
            *self._source_type_multipliers.values(),
            self._unknown_approval_multiplier,
            self._unknown_source_type_multiplier,
        ]

        if any(
            not 0.0 <= value <= 1.0
            for value in multipliers
        ):
            raise ValueError(
                "All authority multipliers must be between 0.0 and 1.0"
            )