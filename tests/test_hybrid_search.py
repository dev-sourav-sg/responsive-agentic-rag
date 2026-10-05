from collections.abc import Sequence

import pytest

from responsive_agentic_rag.models.knowledge import ChunkRecord
from responsive_agentic_rag.models.retrieval import RetrievalCandidate
from responsive_agentic_rag.retrieval.bm25_index import BM25Index
from responsive_agentic_rag.retrieval.hybrid_search import HybridRetriever
from responsive_agentic_rag.retrieval.deterministic_reranker import (
    DeterministicReranker,
)


class FakeVectorStore:
    """Deterministic fake semantic store for hybrid retrieval tests."""

    def __init__(
        self,
        results: Sequence[RetrievalCandidate],
        chunks: Sequence[ChunkRecord] = (),
    ) -> None:
        self.results = list(results)
        self.chunks = list(chunks)
        self.last_query_vector = None
        self.last_limit = None
        self.last_chunk_ids = None

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 10,
    ) -> list[RetrievalCandidate]:
        self.last_query_vector = query_vector
        self.last_limit = limit
        return self.results[:limit]

    def get_chunks(
        self,
        chunk_ids: Sequence[str],
    ) -> list[ChunkRecord]:
        self.last_chunk_ids = list(chunk_ids)

        return [
            chunk
            for chunk in self.chunks
            if chunk.chunk_id in chunk_ids
        ]


class FakeBM25Index:
    """Deterministic fake BM25 index for hybrid retrieval tests."""

    def __init__(
        self,
        results: Sequence[tuple[str, float]],
    ) -> None:
        self.results = list(results)
        self.last_query = None
        self.last_limit = None

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        self.last_query = query
        self.last_limit = limit
        return self.results[:limit]


def make_candidate(
    chunk_id: str,
    semantic_score: float,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        source_type="document",
        source_location=f"document://{chunk_id}",
        content=f"Content for {chunk_id}",
        metadata={},
        semantic_score=semantic_score,
        lexical_score=0.0,
        combined_score=semantic_score,
        rank=1,
    )


def make_chunk(
    chunk_id: str,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        record_id=f"record-{chunk_id}",
        content=f"Content for {chunk_id}",
        chunk_index=0,
        token_count=3,
        metadata={
            "source_type": "document",
            "source_location": f"document://{chunk_id}",
        },
        authority_score=0.5,
    )


def test_semantic_only_result_is_retained() -> None:
    semantic_results = [
        make_candidate("chunk-1", 0.95),
    ]

    vector_store = FakeVectorStore(semantic_results)
    bm25_index = FakeBM25Index([])

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    results = retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].semantic_score == 0.95
    assert results[0].lexical_score == 0.0
    assert results[0].combined_score == pytest.approx(1 / 61)


def test_lexical_only_result_is_resolved_to_complete_candidate() -> None:
    semantic_results = []

    lexical_results = [
        ("chunk-2", 4.25),
    ]

    vector_store = FakeVectorStore(
        results=semantic_results,
        chunks=[
            make_chunk("chunk-2"),
        ],
    )

    bm25_index = FakeBM25Index(lexical_results)

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    results = retriever.search(
        query="chargeback",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert len(results) == 1

    result = results[0]

    assert result.chunk_id == "chunk-2"
    assert result.content == "Content for chunk-2"
    assert result.source_id == "source-chunk-2"
    assert result.source_type == "document"
    assert result.source_location == "document://chunk-2"
    assert result.semantic_score == 0.0
    assert result.lexical_score == 4.25
    assert result.combined_score == pytest.approx(1 / 61)
    assert result.rank == 1


def test_result_found_by_both_retrievers_is_merged() -> None:
    semantic_results = [
        make_candidate("chunk-1", 0.90),
    ]

    lexical_results = [
        ("chunk-1", 5.50),
    ]

    vector_store = FakeVectorStore(semantic_results)
    bm25_index = FakeBM25Index(lexical_results)

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    results = retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert len(results) == 1

    result = results[0]

    assert result.chunk_id == "chunk-1"
    assert result.semantic_score == 0.90
    assert result.lexical_score == 5.50

    expected_rrf = (1 / 61) + (1 / 61)

    assert result.combined_score == pytest.approx(expected_rrf)


def test_rrf_uses_original_ranks() -> None:
    semantic_results = [
        make_candidate("chunk-1", 0.95),
        make_candidate("chunk-2", 0.85),
    ]

    lexical_results = [
        ("chunk-2", 5.0),
        ("chunk-1", 4.0),
    ]

    vector_store = FakeVectorStore(semantic_results)
    bm25_index = FakeBM25Index(lexical_results)

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    results = retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert len(results) == 2

    chunk_1 = next(
        result
        for result in results
        if result.chunk_id == "chunk-1"
    )

    chunk_2 = next(
        result
        for result in results
        if result.chunk_id == "chunk-2"
    )

    expected_chunk_1 = (1 / 61) + (1 / 62)
    expected_chunk_2 = (1 / 62) + (1 / 61)

    assert chunk_1.combined_score == pytest.approx(
        expected_chunk_1
    )
    assert chunk_2.combined_score == pytest.approx(
        expected_chunk_2
    )


def test_limit_is_respected() -> None:
    semantic_results = [
        make_candidate("chunk-1", 0.95),
        make_candidate("chunk-2", 0.90),
        make_candidate("chunk-3", 0.85),
    ]

    vector_store = FakeVectorStore(semantic_results)
    bm25_index = FakeBM25Index([])

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    results = retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
        limit=2,
    )

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_semantic_and_lexical_limits_are_forwarded() -> None:
    vector_store = FakeVectorStore(
        [
            make_candidate("chunk-1", 0.95),
        ]
    )

    bm25_index = FakeBM25Index(
        [
            ("chunk-1", 5.0),
        ]
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
        limit=5,
        semantic_limit=30,
        lexical_limit=40,
    )

    assert vector_store.last_limit == 30
    assert bm25_index.last_limit == 40
    assert bm25_index.last_query == "settlement"


def test_rrf_k_is_configurable() -> None:
    semantic_results = [
        make_candidate("chunk-1", 0.95),
    ]

    vector_store = FakeVectorStore(semantic_results)
    bm25_index = FakeBM25Index([])

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        rrf_k=10,
    )

    results = retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert results[0].combined_score == pytest.approx(1 / 11)


def test_invalid_rrf_k_is_rejected() -> None:
    vector_store = FakeVectorStore([])
    bm25_index = FakeBM25Index([])

    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            rrf_k=0,
        )


@pytest.mark.parametrize(
    ("query", "limit"),
    [
        ("", 10),
        ("   ", 10),
        ("settlement", 0),
    ],
)
def test_invalid_search_arguments_are_rejected(
    query: str,
    limit: int,
) -> None:
    vector_store = FakeVectorStore([])
    bm25_index = FakeBM25Index([])

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    with pytest.raises(ValueError):
        retriever.search(
            query=query,
            query_vector=[0.1, 0.2, 0.3],
            limit=limit,
        )


def test_results_are_deterministically_ordered() -> None:
    semantic_results = [
        make_candidate("chunk-b", 0.90),
        make_candidate("chunk-a", 0.90),
    ]

    vector_store = FakeVectorStore(semantic_results)
    bm25_index = FakeBM25Index([])

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    first_results = retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
    )

    second_results = retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert [result.chunk_id for result in first_results] == [
        result.chunk_id for result in second_results
    ]


def test_original_metadata_is_preserved() -> None:
    candidate = make_candidate("chunk-1", 0.95)
    candidate.metadata = {
        "source_type": "document",
        "source_location": "document://policy.pdf",
        "authority": 0.9,
    }

    vector_store = FakeVectorStore([candidate])
    bm25_index = FakeBM25Index(
        [("chunk-1", 4.0)],
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    results = retriever.search(
        query="policy",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert results[0].metadata["source_type"] == "document"
    assert (
        results[0].metadata["source_location"]
        == "document://policy.pdf"
    )
    assert results[0].metadata["authority"] == 0.9


def test_final_ranks_are_assigned_after_rrf_sorting() -> None:
    semantic_results = [
        make_candidate("chunk-1", 0.95),
        make_candidate("chunk-2", 0.85),
    ]

    lexical_results = [
        ("chunk-2", 5.0),
        ("chunk-1", 4.0),
    ]

    vector_store = FakeVectorStore(semantic_results)
    bm25_index = FakeBM25Index(lexical_results)

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    results = retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert [result.rank for result in results] == [1, 2]
    assert all(result.rank > 0 for result in results)


def test_lexical_only_chunks_are_resolved_in_one_lookup() -> None:
    lexical_results = [
        ("chunk-1", 5.0),
        ("chunk-2", 4.0),
    ]

    vector_store = FakeVectorStore(
        results=[],
        chunks=[
            make_chunk("chunk-1"),
            make_chunk("chunk-2"),
        ],
    )

    bm25_index = FakeBM25Index(lexical_results)

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    retriever.search(
        query="settlement",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert vector_store.last_chunk_ids == [
        "chunk-1",
        "chunk-2",
    ]

def test_hybrid_retriever_applies_reranker():
    candidates = [
        make_candidate("chunk-a", 0.9).model_copy(
            update={"authority_score": 0.2}
        ),
        make_candidate("chunk-b", 0.7).model_copy(
            update={"authority_score": 0.9}
        ),
    ]

    vector_store = FakeVectorStore(results=candidates)
    bm25_index = FakeBM25Index(
        results=[
            ("chunk-a", 0.2),
            ("chunk-b", 0.8),
        ]
    )

    reranker = DeterministicReranker(
    semantic_weight=0.3,
    lexical_weight=0.3,
    authority_weight=0.4,
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=reranker,
    )

    results = retriever.search(
        query="settlement policy",
        query_vector=[0.1, 0.2, 0.3],
        limit=2,
    )

    assert results[0].chunk_id == "chunk-b"
    assert results[0].rank == 1
    assert "reranker_score" in results[0].metadata


def test_hybrid_retriever_without_reranker_preserves_t017_behavior():
    candidates = [
        make_candidate("chunk-a", 0.9),
        make_candidate("chunk-b", 0.7),
    ]

    vector_store = FakeVectorStore(results=candidates)
    bm25_index = FakeBM25Index(
        results=[
            ("chunk-a", 0.2),
            ("chunk-b", 0.8),
        ]
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    results = retriever.search(
        query="settlement policy",
        query_vector=[0.1, 0.2, 0.3],
        limit=2,
    )

    assert "reranker_score" not in results[0].metadata