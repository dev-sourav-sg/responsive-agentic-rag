from collections.abc import Sequence

from responsive_agentic_rag.models.retrieval import RetrievalCandidate


class DeterministicReranker:
    """Deterministically rerank retrieved candidates using relevance and authority."""

    def __init__(
        self,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.3,
        authority_weight: float = 0.2,
    ) -> None:
        total_weight = (
            semantic_weight
            + lexical_weight
            + authority_weight
        )

        if total_weight <= 0:
            raise ValueError("at least one reranker weight must be positive")

        if any(
            weight < 0
            for weight in (
                semantic_weight,
                lexical_weight,
                authority_weight,
            )
        ):
            raise ValueError("reranker weights cannot be negative")

        self._semantic_weight = semantic_weight / total_weight
        self._lexical_weight = lexical_weight / total_weight
        self._authority_weight = authority_weight / total_weight

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        limit: int = 10,
    ) -> list[RetrievalCandidate]:
        if not query.strip():
            raise ValueError("query cannot be empty")

        if limit < 1:
            raise ValueError("limit must be greater than 0")

        if not candidates:
            return []

        semantic_scores = [
            candidate.semantic_score
            for candidate in candidates
        ]
        lexical_scores = [
            candidate.lexical_score
            for candidate in candidates
        ]

        normalized_semantic = self._normalize_scores(
            semantic_scores
        )
        normalized_lexical = self._normalize_scores(
            lexical_scores
        )

        scored_candidates: list[RetrievalCandidate] = []

        for index, candidate in enumerate(candidates):
            rerank_score = (
                self._semantic_weight
                * normalized_semantic[index]
                + self._lexical_weight
                * normalized_lexical[index]
                + self._authority_weight
                * candidate.authority_score
            )

            scored_candidates.append(
                candidate.model_copy(
                    update={
                        "combined_score": rerank_score,
                        "metadata": {
                            **candidate.metadata,
                            "reranker_score": rerank_score,
                        },
                    }
                )
            )

        scored_candidates.sort(
            key=lambda candidate: (
                -candidate.combined_score,
                -candidate.authority_score,
                candidate.chunk_id,
            )
        )

        return [
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(
                scored_candidates[:limit],
                start=1,
            )
        ]

    @staticmethod
    def _normalize_scores(
        scores: Sequence[float],
    ) -> list[float]:
        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [0.0 for _ in scores]

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]