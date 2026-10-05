import pytest

from responsive_agentic_rag.models.retrieval import RetrievalCandidate
from responsive_agentic_rag.retrieval.evidence_sufficiency import EvidenceSufficiency


def make_candidate(
    chunk_id: str,
    combined_score: float,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        source_id="source-1",
        source_type="document",
        source_location="test.pdf",
        content=f"content for {chunk_id}",
        combined_score=combined_score,
    )


def test_returns_true_when_minimum_evidence_is_present():
    gate = EvidenceSufficiency(
        minimum_candidates=1,
        minimum_score=0.5,
    )

    candidates = [
        make_candidate("chunk-1", 0.8),
    ]

    assert gate.assess(candidates) is True


def test_returns_false_when_no_candidates_are_present():
    gate = EvidenceSufficiency(
        minimum_candidates=1,
        minimum_score=0.5,
    )

    assert gate.assess([]) is False


def test_returns_false_when_candidates_do_not_meet_score_threshold():
    gate = EvidenceSufficiency(
        minimum_candidates=1,
        minimum_score=0.5,
    )

    candidates = [
        make_candidate("chunk-1", 0.3),
        make_candidate("chunk-2", 0.4),
    ]

    assert gate.assess(candidates) is False


def test_requires_configured_number_of_candidates():
    gate = EvidenceSufficiency(
        minimum_candidates=2,
        minimum_score=0.5,
    )

    candidates = [
        make_candidate("chunk-1", 0.8),
    ]

    assert gate.assess(candidates) is False


def test_any_candidate_can_satisfy_score_threshold():
    gate = EvidenceSufficiency(
        minimum_candidates=2,
        minimum_score=0.5,
    )

    candidates = [
        make_candidate("chunk-1", 0.3),
        make_candidate("chunk-2", 0.7),
    ]

    assert gate.assess(candidates) is True


def test_rejects_invalid_minimum_candidates():
    with pytest.raises(ValueError):
        EvidenceSufficiency(minimum_candidates=0)


def test_rejects_invalid_minimum_score():
    with pytest.raises(ValueError):
        EvidenceSufficiency(minimum_score=1.5)