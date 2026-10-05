from collections.abc import Sequence

from responsive_agentic_rag.models.knowledge import ChunkRecord
from responsive_agentic_rag.models.retrieval import RetrievalCandidate
from responsive_agentic_rag.retrieval.authority_scorer import AuthorityScorer
from responsive_agentic_rag.retrieval.bm25_index import BM25Index
from responsive_agentic_rag.retrieval.reranker import Reranker
from responsive_agentic_rag.retrieval.vector_store import VectorStore


class HybridRetriever:
    """Combines semantic and lexical retrieval using reciprocal rank fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        reranker: Reranker | None = None,
        authority_scorer: AuthorityScorer | None = None,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")

        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._reranker = reranker
        self._authority_scorer = authority_scorer
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        limit: int = 10,
        semantic_limit: int | None = None,
        lexical_limit: int | None = None,
    ) -> list[RetrievalCandidate]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if not query_vector:
            raise ValueError("query_vector must not be empty")

        if limit < 1:
            raise ValueError("limit must be >= 1")

        semantic_limit = semantic_limit or limit
        lexical_limit = lexical_limit or limit

        if semantic_limit < 1:
            raise ValueError("semantic_limit must be >= 1")

        if lexical_limit < 1:
            raise ValueError("lexical_limit must be >= 1")

        semantic_candidates = self._vector_store.search(
            query_vector=query_vector,
            limit=semantic_limit,
        )

        lexical_results = self._bm25_index.search(
            query=query,
            limit=lexical_limit,
        )

        lexical_candidates = self._resolve_lexical_candidates(
            lexical_results=lexical_results,
            semantic_candidates=semantic_candidates,
        )

        fused_candidates = self._rrf_fuse(
            semantic_candidates=semantic_candidates,
            lexical_candidates=lexical_candidates,
        )

        if self._authority_scorer is not None:
            fused_candidates = [
                self._authority_scorer.enrich_candidate(candidate)
                for candidate in fused_candidates
            ]

        if self._reranker is not None:
            return self._reranker.rerank(
                query=query,
                candidates=fused_candidates,
                limit=limit,
            )

        return fused_candidates[:limit]

    def _resolve_lexical_candidates(
        self,
        lexical_results: Sequence[tuple[str, float]],
        semantic_candidates: Sequence[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """
        Resolve BM25 results into RetrievalCandidates.

        If a chunk already exists in semantic retrieval results, reuse that
        candidate and add its lexical score. Only BM25-only chunks require
        a VectorStore lookup.
        """
        if not lexical_results:
            return []

        semantic_by_id = {
            candidate.chunk_id: candidate
            for candidate in semantic_candidates
        }

        unresolved_ids = [
            chunk_id
            for chunk_id, _ in lexical_results
            if chunk_id not in semantic_by_id
        ]

        chunks_by_id: dict[str, ChunkRecord] = {}

        if unresolved_ids:
            chunks = self._vector_store.get_chunks(unresolved_ids)

            chunks_by_id = {
                chunk.chunk_id: chunk
                for chunk in chunks
            }

        candidates: list[RetrievalCandidate] = []

        for chunk_id, lexical_score in lexical_results:
            semantic_candidate = semantic_by_id.get(chunk_id)

            if semantic_candidate is not None:
                candidates.append(
                    semantic_candidate.model_copy(
                        update={
                            "lexical_score": lexical_score,
                        }
                    )
                )
                continue

            chunk = chunks_by_id.get(chunk_id)

            if chunk is None:
                continue

            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_type=chunk.metadata.get(
                        "source_type",
                        "document",
                    ),
                    source_location=chunk.metadata.get(
                        "source_location",
                        "",
                    ),
                    content=chunk.content,
                    metadata=chunk.metadata,
                    semantic_score=0.0,
                    lexical_score=lexical_score,
                    authority_score=chunk.authority_score,
                    combined_score=0.0,
                    rank=0,
                )
            )

        return candidates

    def _rrf_fuse(
        self,
        semantic_candidates: Sequence[RetrievalCandidate],
        lexical_candidates: Sequence[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """
        Fuse semantic and lexical rankings using Reciprocal Rank Fusion.

        RRF score:

            1 / (k + semantic_rank)
            +
            1 / (k + lexical_rank)

        Authority is deliberately NOT included here. Authority is applied
        after retrieval fusion so that source authority does not distort
        the underlying semantic/lexical retrieval ranks.
        """
        candidates_by_id: dict[str, RetrievalCandidate] = {}

        # Preserve semantic candidates first.
        for candidate in semantic_candidates:
            candidates_by_id[candidate.chunk_id] = candidate

        # Merge lexical candidates.
        #
        # If the chunk already exists in semantic results, preserve all
        # semantic/provenance fields and add the lexical score.
        for candidate in lexical_candidates:
            existing = candidates_by_id.get(candidate.chunk_id)

            if existing is None:
                candidates_by_id[candidate.chunk_id] = candidate
                continue

            candidates_by_id[candidate.chunk_id] = existing.model_copy(
                update={
                    "lexical_score": candidate.lexical_score,
                }
            )

        scores: dict[str, float] = {
            chunk_id: 0.0
            for chunk_id in candidates_by_id
        }

        # Semantic contribution.
        for rank, candidate in enumerate(
            semantic_candidates,
            start=1,
        ):
            scores[candidate.chunk_id] += 1.0 / (
                self._rrf_k + rank
            )

        # Lexical contribution.
        for rank, candidate in enumerate(
            lexical_candidates,
            start=1,
        ):
            scores[candidate.chunk_id] += 1.0 / (
                self._rrf_k + rank
            )

        fused_candidates: list[RetrievalCandidate] = []

        for candidate in candidates_by_id.values():
            fused_candidates.append(
                candidate.model_copy(
                    update={
                        "combined_score": scores[
                            candidate.chunk_id
                        ],
                    }
                )
            )

        # Deterministic ordering:
        # 1. Higher RRF score first
        # 2. Chunk ID as stable tie-breaker
        fused_candidates.sort(
            key=lambda candidate: (
                -candidate.combined_score,
                candidate.chunk_id,
            )
        )

        return [
            candidate.model_copy(
                update={
                    "rank": rank,
                }
            )
            for rank, candidate in enumerate(
                fused_candidates,
                start=1,
            )
        ]