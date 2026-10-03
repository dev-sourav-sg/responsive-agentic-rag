from responsive_agentic_rag.models.knowledge import ChunkRecord
from responsive_agentic_rag.retrieval.bm25_index import BM25Index


def make_chunk(
    chunk_id: str,
    content: str,
) -> ChunkRecord:
    """Create a minimal ChunkRecord for BM25 tests."""

    return ChunkRecord(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        record_id=f"record-{chunk_id}",
        content=content,
        chunk_index=0,
        token_count=len(content.split()),
        authority_score=0.0,
    )


def test_build_and_search_returns_relevant_chunks():
    """BM25 should return chunks matching the query terms."""

    chunks = [
        make_chunk(
            "chunk-1",
            "Settlement processing takes four business days.",
        ),
        make_chunk(
            "chunk-2",
            "Merchant onboarding requires identity verification.",
        ),
        make_chunk(
            "chunk-3",
            "Reconciliation compares processor transaction files.",
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    results = index.search("settlement processing")

    assert results
    assert results[0][0] == "chunk-1"
    assert results[0][1] > 0.0


def test_exact_domain_terms_are_retrieved():
    """BM25 should be effective for exact domain terminology."""

    chunks = [
        make_chunk(
            "chunk-1",
            "Fiserv reconciliation files are processed daily.",
        ),
        make_chunk(
            "chunk-2",
            "Payoneer handles merchant settlement processing.",
        ),
        make_chunk(
            "chunk-3",
            "Merchant support handles onboarding requests.",
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    results = index.search("Fiserv reconciliation")

    assert results
    assert results[0][0] == "chunk-1"
    assert results[0][1] > 0.0


def test_multiple_matching_chunks_receive_scores():
    """Multiple relevant chunks should be returned."""

    chunks = [
        make_chunk(
            "chunk-1",
            "Settlement processing takes four business days.",
        ),
        make_chunk(
            "chunk-2",
            "Settlement failures require operational review.",
        ),
        make_chunk(
            "chunk-3",
            "Merchant onboarding requires KYC verification.",
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    results = index.search("settlement")

    result_ids = {chunk_id for chunk_id, _ in results}

    assert "chunk-1" in result_ids
    assert "chunk-2" in result_ids
    assert "chunk-3" not in result_ids


def test_search_respects_limit():
    """Search should return no more than the requested number of results."""

    chunks = [
        make_chunk(
            "chunk-1",
            "Settlement processing takes four business days.",
        ),
        make_chunk(
            "chunk-2",
            "Settlement failures require operational review.",
        ),
        make_chunk(
            "chunk-3",
            "Settlement requests are sent to Payoneer.",
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    results = index.search(
        "settlement",
        limit=2,
    )

    assert len(results) == 2


def test_empty_index_returns_empty_results():
    """Searching an index that has not been built should return no results."""

    index = BM25Index()

    results = index.search("settlement")

    assert results == []


def test_build_with_empty_chunks_creates_empty_index():
    """Building with no chunks should produce an empty searchable index."""

    index = BM25Index()

    index.build([])

    results = index.search("settlement")

    assert results == []


def test_search_rejects_empty_query():
    """An empty or whitespace-only query should be rejected."""

    chunks = [
        make_chunk(
            "chunk-1",
            "Settlement processing takes four business days.",
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    import pytest

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        index.search("")

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        index.search("   ")


def test_search_rejects_invalid_limit():
    """Zero or negative limits should be rejected."""

    chunks = [
        make_chunk(
            "chunk-1",
            "Settlement processing takes four business days.",
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    import pytest

    with pytest.raises(
        ValueError,
        match="limit must be greater than 0",
    ):
        index.search("settlement", limit=0)

    with pytest.raises(
        ValueError,
        match="limit must be greater than 0",
    ):
        index.search("settlement", limit=-1)


def test_rebuilding_index_replaces_previous_contents():
    """Rebuilding should replace the old corpus rather than append to it."""

    initial_chunks = [
        make_chunk(
            "chunk-old",
            "Legacy reconciliation process.",
        ),
        make_chunk(
            "chunk-other",
            "Merchant onboarding requires verification.",
        ),
    ]

    new_chunks = [
        make_chunk(
            "chunk-new",
            "Settlement processing through Payoneer.",
        ),
        make_chunk(
            "chunk-new-other",
            "Merchant support handles operational requests.",
        ),
    ]

    index = BM25Index()

    index.build(initial_chunks)

    old_results = index.search("reconciliation")

    assert old_results
    assert old_results[0][0] == "chunk-old"

    index.build(new_chunks)

    new_results = index.search("settlement")

    assert new_results
    assert new_results[0][0] == "chunk-new"

    old_results_after_rebuild = index.search("reconciliation")

    assert old_results_after_rebuild == []


def test_search_results_are_deterministic():
    """Repeated searches over the same index should return the same ordering."""

    chunks = [
        make_chunk(
            "chunk-a",
            "Settlement policy applies to merchants.",
        ),
        make_chunk(
            "chunk-b",
            "Settlement policy applies to processors.",
        ),
        make_chunk(
            "chunk-c",
            "Settlement policy defines processing windows.",
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    first_results = index.search(
        "settlement policy",
        limit=3,
    )

    second_results = index.search(
        "settlement policy",
        limit=3,
    )

    assert first_results == second_results


def test_unmatched_query_returns_empty_results():
    """A query with no matching terms should return no results."""

    chunks = [
        make_chunk(
            "chunk-1",
            "Settlement processing takes four business days.",
        ),
        make_chunk(
            "chunk-2",
            "Merchant onboarding requires identity verification.",
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    results = index.search("cryptocurrency mining")

    assert results == []