from responsive_agentic_rag.agents.answer_synthesizer import (
    AnswerSynthesizerContract,
    DeterministicAnswerSynthesizer,
)
from responsive_agentic_rag.models.retrieval import (
    AnswerRecord,
    EvidenceSet,
    RetrievalCandidate,
)


def make_candidate(
    chunk_id: str,
    source_id: str,
    source_location: str,
    content: str,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        source_id=source_id,
        source_type="document",
        source_location=source_location,
        content=content,
        combined_score=0.85,
        authority_score=0.9,
    )


def test_synthesizer_implements_contract():
    synthesizer = DeterministicAnswerSynthesizer()

    assert isinstance(synthesizer, AnswerSynthesizerContract)


def test_synthesizer_returns_grounded_answer_when_evidence_is_sufficient():
    synthesizer = DeterministicAnswerSynthesizer()

    evidence = EvidenceSet(
        query_id="q1",
        sufficiency_assessment=True,
        candidates=[
            make_candidate(
                "chunk-1",
                "source-1",
                "payment_policy.pdf",
                "Payments must be settled within four business days.",
            ),
        ],
        supporting_sources=["source-1"],
    )

    result = synthesizer.synthesize(evidence)

    assert isinstance(result, AnswerRecord)
    assert result.grounding_status == "grounded"
    assert (
        "Payments must be settled within four business days."
        in result.answer_text
    )
    assert result.citations == ["payment_policy.pdf"]
    assert result.source_ids == ["source-1"]
    assert result.missing_evidence_note is None


def test_synthesizer_returns_no_answer_when_evidence_is_insufficient():
    synthesizer = DeterministicAnswerSynthesizer()

    evidence = EvidenceSet(
        query_id="q2",
        sufficiency_assessment=False,
        candidates=[],
        supporting_sources=[],
    )

    result = synthesizer.synthesize(evidence)

    assert isinstance(result, AnswerRecord)
    assert result.grounding_status == "insufficient_evidence"
    assert result.citations == []
    assert result.source_ids == []
    assert result.missing_evidence_note is not None
    assert "sufficient evidence" in result.answer_text


def test_synthesizer_deduplicates_citations_and_sources():
    synthesizer = DeterministicAnswerSynthesizer()

    evidence = EvidenceSet(
        query_id="q3",
        sufficiency_assessment=True,
        candidates=[
            make_candidate(
                "chunk-1",
                "source-1",
                "payment_policy.pdf",
                "Payment evidence one.",
            ),
            make_candidate(
                "chunk-2",
                "source-1",
                "payment_policy.pdf",
                "Payment evidence two.",
            ),
            make_candidate(
                "chunk-3",
                "source-2",
                "settlement_policy.pdf",
                "Settlement evidence.",
            ),
        ],
        supporting_sources=["source-1", "source-2"],
    )

    result = synthesizer.synthesize(evidence)

    assert result.citations == [
        "payment_policy.pdf",
        "settlement_policy.pdf",
    ]
    assert result.source_ids == ["source-1", "source-2"]