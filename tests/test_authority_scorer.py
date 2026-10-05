import pytest

from responsive_agentic_rag.models.retrieval import RetrievalCandidate
from responsive_agentic_rag.retrieval.authority_scorer import AuthorityScorer


def test_approved_document_preserves_base_authority():
    scorer = AuthorityScorer()

    result = scorer.score(
        base_authority=0.9,
        metadata={
            "approval_status": "approved",
            "source_type": "document",
        },
    )

    assert result == pytest.approx(0.9)


def test_draft_document_reduces_authority():
    scorer = AuthorityScorer()

    result = scorer.score(
        base_authority=0.8,
        metadata={
            "approval_status": "draft",
            "source_type": "document",
        },
    )

    assert result == pytest.approx(0.4)


def test_deprecated_document_has_low_authority():
    scorer = AuthorityScorer()

    result = scorer.score(
        base_authority=0.8,
        metadata={
            "approval_status": "deprecated",
            "source_type": "document",
        },
    )

    assert result == pytest.approx(0.2)


def test_website_source_type_applies_multiplier():
    scorer = AuthorityScorer()

    result = scorer.score(
        base_authority=1.0,
        metadata={
            "approval_status": "approved",
            "source_type": "website",
        },
    )

    assert result == pytest.approx(0.9)


def test_unknown_approval_status_uses_default_multiplier():
    scorer = AuthorityScorer()

    result = scorer.score(
        base_authority=0.8,
        metadata={
            "approval_status": "something_unknown",
            "source_type": "document",
        },
    )

    assert result == pytest.approx(0.4)


def test_unknown_source_type_uses_default_multiplier():
    scorer = AuthorityScorer()

    result = scorer.score(
        base_authority=0.8,
        metadata={
            "approval_status": "approved",
            "source_type": "unknown",
        },
    )

    assert result == pytest.approx(0.8)


def test_score_is_clamped_to_valid_range():
    scorer = AuthorityScorer(
        approval_multipliers={"approved": 1.0},
        source_type_multipliers={"document": 1.0},
    )

    assert scorer.score(
        base_authority=1.0,
        metadata={
            "approval_status": "approved",
            "source_type": "document",
        },
    ) == pytest.approx(1.0)


def test_rejects_invalid_base_authority():
    scorer = AuthorityScorer()

    with pytest.raises(ValueError):
        scorer.score(
            base_authority=1.1,
            metadata={
                "approval_status": "approved",
                "source_type": "document",
            },
        )


def test_rejects_invalid_negative_base_authority():
    scorer = AuthorityScorer()

    with pytest.raises(ValueError):
        scorer.score(
            base_authority=-0.1,
            metadata={
                "approval_status": "approved",
                "source_type": "document",
            },
        )


def test_rejects_invalid_configured_multiplier():
    with pytest.raises(ValueError):
        AuthorityScorer(
            approval_multipliers={"approved": 1.5},
        )


def test_enrich_candidate_applies_effective_authority():
    scorer = AuthorityScorer()

    candidate = RetrievalCandidate(
        chunk_id="chunk-1",
        source_id="source-1",
        source_type="document",
        source_location="policy.pdf",
        content="Settlement policy",
        authority_score=0.8,
        metadata={
            "approval_status": "draft",
            "source_type": "document",
        },
    )

    enriched = scorer.enrich_candidate(candidate)

    assert enriched.authority_score == pytest.approx(0.4)
    assert enriched.chunk_id == "chunk-1"
    assert enriched.content == "Settlement policy"


def test_enrich_candidate_does_not_mutate_original():
    scorer = AuthorityScorer()

    candidate = RetrievalCandidate(
        chunk_id="chunk-1",
        source_id="source-1",
        source_type="document",
        source_location="policy.pdf",
        content="Settlement policy",
        authority_score=0.8,
        metadata={
            "approval_status": "draft",
            "source_type": "document",
        },
    )

    enriched = scorer.enrich_candidate(candidate)

    assert candidate.authority_score == pytest.approx(0.8)
    assert enriched.authority_score == pytest.approx(0.4)