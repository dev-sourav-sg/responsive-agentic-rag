from responsive_agentic_rag.agents.answer_assembler import (
    AnswerAssembler,
    AnswerAssemblerContract,
)
from responsive_agentic_rag.models.retrieval import (
    AnswerRecord,
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


def test_answer_assembler_implements_contract():
    assembler = AnswerAssembler()

    assert isinstance(assembler, AnswerAssemblerContract)


def test_assembles_grounded_answer_with_evidence_derived_citations():
    assembler = AnswerAssembler()

    candidates = [
        make_candidate(
            "chunk-1",
            "source-1",
            "payment_policy.pdf",
            "Payments must be settled within four business days.",
        ),
        make_candidate(
            "chunk-2",
            "source-2",
            "settlement_policy.pdf",
            "Settlement processing occurs after reconciliation.",
        ),
    ]

    result = assembler.assemble(
        answer_text="Payments must be settled within four business days.",
        candidates=candidates,
    )

    assert isinstance(result, AnswerRecord)
    assert result.answer_text == (
        "Payments must be settled within four business days."
    )
    assert result.grounding_status == "grounded"
    assert result.citations == [
        "payment_policy.pdf",
        "settlement_policy.pdf",
    ]
    assert result.source_ids == ["source-1", "source-2"]


def test_deduplicates_citations_and_source_ids():
    assembler = AnswerAssembler()

    candidates = [
        make_candidate(
            "chunk-1",
            "source-1",
            "payment_policy.pdf",
            "Evidence one.",
        ),
        make_candidate(
            "chunk-2",
            "source-1",
            "payment_policy.pdf",
            "Evidence two.",
        ),
        make_candidate(
            "chunk-3",
            "source-2",
            "settlement_policy.pdf",
            "Evidence three.",
        ),
    ]

    result = assembler.assemble(
        answer_text="Combined answer.",
        candidates=candidates,
    )

    assert result.citations == [
        "payment_policy.pdf",
        "settlement_policy.pdf",
    ]
    assert result.source_ids == ["source-1", "source-2"]


def test_insufficient_evidence_removes_citations():
    assembler = AnswerAssembler()

    candidates = [
        make_candidate(
            "chunk-1",
            "source-1",
            "payment_policy.pdf",
            "Weak evidence.",
        ),
    ]

    result = assembler.assemble(
        answer_text="I don't have sufficient evidence to answer this.",
        candidates=candidates,
        grounding_status="insufficient_evidence",
    )

    assert result.grounding_status == "insufficient_evidence"
    assert result.citations == []
    assert result.source_ids == []
    assert result.missing_evidence_note is not None


def test_rejects_empty_answer():
    assembler = AnswerAssembler()

    candidates = [
        make_candidate(
            "chunk-1",
            "source-1",
            "payment_policy.pdf",
            "Evidence.",
        ),
    ]

    try:
        assembler.assemble(
            answer_text="   ",
            candidates=candidates,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "answer_text must not be empty"


def test_rejects_invalid_grounding_status():
    assembler = AnswerAssembler()

    candidates = [
        make_candidate(
            "chunk-1",
            "source-1",
            "payment_policy.pdf",
            "Evidence.",
        ),
    ]

    try:
        assembler.assemble(
            answer_text="Answer.",
            candidates=candidates,
            grounding_status="unknown",
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "grounding_status" in str(exc)