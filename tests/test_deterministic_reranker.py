import pytest

from responsive_agentic_rag.models.retrieval import RetrievalCandidate
from responsive_agentic_rag.retrieval.deterministic_reranker import (
    DeterministicReranker,
)


def make_candidate(
    chunk_id: str,
    semantic_score: float,
    lexical_score: float,
    authority_score: float,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        source_type="document",
        source_location=f"/documents/{chunk_id}.pdf",
        content=f"Content for {chunk_id}",
        semantic_score=semantic_score,
        lexical_score=lexical_score,
        authority_score=authority_score,
    )


def test_reranks_candidates_using_weighted_signals():
    reranker = DeterministicReranker()

    candidates = [
        make_candidate("chunk-a", 0.9, 0.2, 0.4),
        make_candidate("chunk-b", 0.7, 0.8, 0.9),
        make_candidate("chunk-c", 0.2, 0.1, 0.5),
    ]

    results = reranker.rerank(
        query="What is the settlement policy?",
        candidates=candidates,
        limit=3,
    )

    assert [candidate.chunk_id for candidate in results] == [
        "chunk-b",
        "chunk-a",
        "chunk-c",
    ]


def test_assigns_final_ranks():
    reranker = DeterministicReranker()

    candidates = [
        make_candidate("chunk-a", 0.9, 0.9, 0.9),
        make_candidate("chunk-b", 0.5, 0.5, 0.5),
    ]

    results = reranker.rerank(
        query="test query",
        candidates=candidates,
        limit=2,
    )

    assert [candidate.rank for candidate in results] == [1, 2]


def test_preserves_candidate_metadata():
    reranker = DeterministicReranker()

    candidate = make_candidate("chunk-a", 0.8, 0.6, 0.9)
    candidate = candidate.model_copy(
        update={"metadata": {"semantic_rank": 2}}
    )

    results = reranker.rerank(
        query="test query",
        candidates=[candidate],
    )

    assert results[0].metadata["semantic_rank"] == 2
    assert "reranker_score" in results[0].metadata


def test_limit_is_respected():
    reranker = DeterministicReranker()

    candidates = [
        make_candidate("chunk-a", 0.9, 0.9, 0.9),
        make_candidate("chunk-b", 0.8, 0.8, 0.8),
        make_candidate("chunk-c", 0.7, 0.7, 0.7),
    ]

    results = reranker.rerank(
        query="test query",
        candidates=candidates,
        limit=2,
    )

    assert len(results) == 2


def test_empty_candidates_return_empty_result():
    reranker = DeterministicReranker()

    assert reranker.rerank(
        query="test query",
        candidates=[],
    ) == []


def test_rejects_empty_query():
    reranker = DeterministicReranker()

    with pytest.raises(ValueError, match="query cannot be empty"):
        reranker.rerank(
            query="   ",
            candidates=[],
        )


def test_rejects_invalid_limit():
    reranker = DeterministicReranker()

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        reranker.rerank(
            query="test query",
            candidates=[],
            limit=0,
        )


def test_rejects_negative_weights():
    with pytest.raises(
        ValueError,
        match="reranker weights cannot be negative",
    ):
        DeterministicReranker(
            semantic_weight=-0.1,
            lexical_weight=0.5,
            authority_weight=0.6,
        )


def test_rejects_zero_total_weight():
    with pytest.raises(
        ValueError,
        match="at least one reranker weight must be positive",
    ):
        DeterministicReranker(
            semantic_weight=0.0,
            lexical_weight=0.0,
            authority_weight=0.0,
        )


def test_weights_are_normalized():
    reranker = DeterministicReranker(
        semantic_weight=5.0,
        lexical_weight=3.0,
        authority_weight=2.0,
    )

    candidates = [
        make_candidate("chunk-a", 1.0, 0.0, 0.0),
        make_candidate("chunk-b", 0.0, 1.0, 0.0),
        make_candidate("chunk-c", 0.0, 0.0, 1.0),
    ]

    results = reranker.rerank(
        query="test query",
        candidates=candidates,
        limit=3,
    )

    assert [candidate.chunk_id for candidate in results] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]


def test_tie_breaking_is_deterministic():
    reranker = DeterministicReranker()

    candidates = [
        make_candidate("chunk-b", 0.5, 0.5, 0.5),
        make_candidate("chunk-a", 0.5, 0.5, 0.5),
    ]

    first = reranker.rerank(
        query="test query",
        candidates=candidates,
    )

    second = reranker.rerank(
        query="test query",
        candidates=candidates,
    )

    assert [candidate.chunk_id for candidate in first] == [
        "chunk-a",
        "chunk-b",
    ]

    assert [candidate.chunk_id for candidate in first] == [
        candidate.chunk_id for candidate in second
    ]