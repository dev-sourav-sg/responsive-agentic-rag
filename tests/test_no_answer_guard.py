import pytest

from responsive_agentic_rag.models.retrieval import EvidenceSet
from responsive_agentic_rag.retrieval.no_answer_guard import (
    InsufficientEvidenceError,
    NoAnswerGuard,
)


def test_allows_answer_when_evidence_is_sufficient():
    evidence = EvidenceSet(
        query_id="query-1",
        sufficiency_assessment=True,
        supporting_sources=["source-1"],
        candidates=[],
    )

    NoAnswerGuard().enforce(evidence)


def test_blocks_answer_when_evidence_is_insufficient():
    evidence = EvidenceSet(
        query_id="query-1",
        sufficiency_assessment=False,
        supporting_sources=[],
        candidates=[],
    )

    with pytest.raises(InsufficientEvidenceError):
        NoAnswerGuard().enforce(evidence)


def test_blocks_answer_when_sufficiency_is_unknown():
    evidence = EvidenceSet(
        query_id="query-1",
        sufficiency_assessment=None,
        supporting_sources=[],
        candidates=[],
    )

    with pytest.raises(InsufficientEvidenceError):
        NoAnswerGuard().enforce(evidence)


def test_error_message_is_explicit():
    evidence = EvidenceSet(
        query_id="query-1",
        sufficiency_assessment=False,
        supporting_sources=[],
        candidates=[],
    )

    with pytest.raises(
        InsufficientEvidenceError,
        match="insufficient.*grounded answer",
    ):
        NoAnswerGuard().enforce(evidence)