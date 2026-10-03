from collections.abc import Sequence

from responsive_agentic_rag.models.retrieval import RetrievalCandidate
from responsive_agentic_rag.retrieval.bm25_index import BM25Index
from responsive_agentic_rag.retrieval.vector_store import VectorStore


class HybridRetriever:
    """Combine semantic and lexical retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be greater than 0")

        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        limit: int = 10,
        semantic_limit: int = 30,
        lexical_limit: int = 30,
    ) -> list[RetrievalCandidate]:
        """Run semantic and lexical retrieval and fuse results using RRF."""

        if not query.strip():
            raise ValueError("query cannot be empty")

        if limit < 1:
            raise ValueError("limit must be greater than 0")

        if semantic_limit < 1:
            raise ValueError("semantic_limit must be greater than 0")

        if lexical_limit < 1:
            raise ValueError("lexical_limit must be greater than 0")

        semantic_results = self._vector_store.search(
            query_vector=query_vector,
            limit=semantic_limit,
        )

        lexical_results = self._bm25_index.search(
            query=query,
            limit=lexical_limit,
        )

        return self._fuse_results(
            semantic_results=semantic_results,
            lexical_results=lexical_results,
            limit=limit,
        )

    def _fuse_results(
        self,
        semantic_results: Sequence[RetrievalCandidate],
        lexical_results: Sequence[tuple[str, float]],
        limit: int,
    ) -> list[RetrievalCandidate]:
        """Merge semantic and lexical results using Reciprocal Rank Fusion."""

        candidates: dict[str, RetrievalCandidate] = {}

        semantic_ranks: dict[str, int] = {}
        lexical_ranks: dict[str, int] = {}

        # ------------------------------------------------------------------
        # Semantic retrieval results
        # ------------------------------------------------------------------
        for rank, candidate in enumerate(semantic_results, start=1):
            semantic_ranks[candidate.chunk_id] = rank

            candidates[candidate.chunk_id] = candidate.model_copy(
                update={
                    "semantic_score": candidate.semantic_score,
                    "lexical_score": 0.0,
                    "combined_score": 0.0,
                    "rank": 0,
                    "metadata": {
                        **candidate.metadata,
                        "semantic_rank": rank,
                    },
                }
            )

        # ------------------------------------------------------------------
        # Lexical retrieval results
        # ------------------------------------------------------------------
        for rank, (chunk_id, lexical_score) in enumerate(
            lexical_results,
            start=1,
        ):
            lexical_ranks[chunk_id] = rank

            if chunk_id in candidates:
                existing = candidates[chunk_id]

                candidates[chunk_id] = existing.model_copy(
                    update={
                        "lexical_score": lexical_score,
                        "metadata": {
                            **existing.metadata,
                            "lexical_rank": rank,
                        },
                    }
                )
            else:
                # BM25 currently returns only chunk_id + score.
                #
                # Therefore a lexical-only result cannot be converted into
                # a complete RetrievalCandidate unless the chunk metadata
                # is available from another source.
                #
                # We deliberately do not fabricate source/content metadata.
                continue

        # ------------------------------------------------------------------
        # Calculate RRF
        # ------------------------------------------------------------------
        fused_candidates: list[RetrievalCandidate] = []

        for chunk_id, candidate in candidates.items():
            semantic_rank = semantic_ranks.get(chunk_id)
            lexical_rank = lexical_ranks.get(chunk_id)

            rrf_score = 0.0

            if semantic_rank is not None:
                rrf_score += 1.0 / (
                    self._rrf_k + semantic_rank
                )

            if lexical_rank is not None:
                rrf_score += 1.0 / (
                    self._rrf_k + lexical_rank
                )

            fused_candidates.append(
                candidate.model_copy(
                    update={
                        "combined_score": rrf_score,
                    }
                )
            )

        # ------------------------------------------------------------------
        # Deterministic ordering
        # ------------------------------------------------------------------
        fused_candidates.sort(
            key=lambda candidate: (
                -candidate.combined_score,
                candidate.chunk_id,
            )
        )

        # Assign final rank after RRF ordering.
        ranked_candidates: list[RetrievalCandidate] = []

        for rank, candidate in enumerate(
            fused_candidates[:limit],
            start=1,
        ):
            ranked_candidates.append(
                candidate.model_copy(
                    update={
                        "rank": rank,
                    }
                )
            )

        return ranked_candidates