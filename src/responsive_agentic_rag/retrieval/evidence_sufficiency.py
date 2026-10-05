from collections.abc import Sequence

from responsive_agentic_rag.models.retrieval import RetrievalCandidate


class EvidenceSufficiency:
    """Deterministic gate that decides whether retrieved evidence is sufficient."""

    def __init__(
        self,
        minimum_candidates: int = 1,
        minimum_score: float = 0.0,
    ) -> None:
        if minimum_candidates < 1:
            raise ValueError("minimum_candidates must be at least 1")

        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0.0 and 1.0")

        self._minimum_candidates = minimum_candidates
        self._minimum_score = minimum_score

    def assess(
        self,
        candidates: Sequence[RetrievalCandidate],
    ) -> bool:
        """Return True when retrieved evidence meets the minimum threshold."""

        if len(candidates) < self._minimum_candidates:
            return False

        return any(
            candidate.combined_score >= self._minimum_score
            for candidate in candidates
        )