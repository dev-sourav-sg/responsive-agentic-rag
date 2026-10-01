
import pytest
from pydantic import ValidationError

from responsive_agentic_rag.models.knowledge import (
    ChunkRecord,
    KnowledgeSource,
    NormalizedSourceRecord,
)
from responsive_agentic_rag.models.retrieval import (
    AnswerRecord,
    EvaluationCase,
    EvidenceSet,
    RetrievalCandidate,
    RetrievalQuery,
)


def test_knowledge_source():
    source = KnowledgeSource(
        source_id="doc-001",
        source_type="document",
        title="Payment Policy",
        source_location="data/documents/payment-policy.pdf",
        authority=0.9,
    )

    assert source.source_id == "doc-001"
    assert source.source_type == "document"
    assert source.authority == 0.9


def test_normalized_source_record():
    record = NormalizedSourceRecord(
        record_id="record-001",
        source_id="doc-001",
        source_type="document",
        title="Payment Policy",
        section_path=["Payments", "Refunds"],
        body_text="Refunds are processed within five business days.",
    )

    assert record.section_path == ["Payments", "Refunds"]
    assert "five business days" in record.body_text


def test_chunk_record():
    chunk = ChunkRecord(
        chunk_id="chunk-001",
        source_id="doc-001",
        record_id="record-001",
        content="Refunds are processed within five business days.",
        chunk_index=0,
        token_count=8,
        authority_score=0.9,
    )

    assert chunk.chunk_index == 0
    assert chunk.authority_score == 0.9


def test_retrieval_query_defaults():
    query = RetrievalQuery(query_text="What is the refund policy?")

    assert query.max_results == 10
    assert query.hybrid_mode == "hybrid"
    assert query.source_filters == {}


def test_retrieval_candidate():
    candidate = RetrievalCandidate(
        chunk_id="chunk-001",
        source_id="doc-001",
        source_type="document",
        source_location="data/documents/payment-policy.pdf",
        content="Refunds are processed within five business days.",
        semantic_score=0.91,
        lexical_score=0.84,
        authority_score=0.9,
        combined_score=0.89,
        rank=1,
    )

    assert candidate.rank == 1
    assert candidate.combined_score == 0.89


def test_evidence_set():
    evidence = EvidenceSet(
        query_id="query-001",
        supporting_sources=["doc-001"],
    )

    assert evidence.query_id == "query-001"
    assert evidence.candidates == []


def test_answer_record():
    answer = AnswerRecord(
        answer_text="Refunds are processed within five business days.",
        citations=["doc-001"],
        source_ids=["doc-001"],
    )

    assert answer.grounding_status == "grounded"
    assert answer.citations == ["doc-001"]


def test_evaluation_case():
    case = EvaluationCase(
        case_id="eval-001",
        question="What is the refund policy?",
        expected_answer="Refunds are processed within five business days.",
        answerable=True,
        source_types=["document"],
        difficulty="easy",
        tags=["direct-answer"],
    )

    assert case.answerable is True
    assert case.source_types == ["document"]


def test_invalid_authority_is_rejected():
    with pytest.raises(ValidationError):
        KnowledgeSource(
            source_id="doc-001",
            source_type="document",
            title="Payment Policy",
            source_location="payment-policy.pdf",
            authority=1.5,
        )


def test_invalid_source_type_is_rejected():
    with pytest.raises(ValidationError):
        KnowledgeSource(
            source_id="doc-001",
            source_type="database",
            title="Payment Policy",
            source_location="payment-policy.pdf",
        )


def test_invalid_max_results_is_rejected():
    with pytest.raises(ValidationError):
        RetrievalQuery(
            query_text="What is the refund policy?",
            max_results=0,
        )
