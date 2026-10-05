from responsive_agentic_rag.agents.adk_orchestrator import (
    ADKRetrievalOrchestrator,
)
from responsive_agentic_rag.agents.answer_assembler import AnswerAssembler
from responsive_agentic_rag.agents.answer_synthesizer import (
    DeterministicAnswerSynthesizer,
)
from responsive_agentic_rag.models.retrieval import (
    EvidenceSet,
    RetrievalCandidate,
)
from responsive_agentic_rag.retrieval.no_answer_guard import NoAnswerGuard


class FakeRetrievalTool:
    def __init__(self, evidence: EvidenceSet) -> None:
        self.evidence = evidence
        self.received_query = None

    def retrieve(self, query):
        self.received_query = query
        return self.evidence


def test_t023_answerable_query_returns_grounded_answer_with_citations():
    evidence = EvidenceSet(
        query_id="What is the settlement period?",
        sufficiency_assessment=True,
        supporting_sources=["settlement-policy"],
        candidates=[
            RetrievalCandidate(
                chunk_id="chunk-1",
                source_id="settlement-policy",
                source_type="document",
                source_location="data/documents/settlement_policy.pdf",
                content="The standard settlement period is four business days.",
                semantic_score=0.92,
                lexical_score=0.88,
                authority_score=0.95,
                combined_score=0.91,
                rank=1,
            )
        ],
    )

    retrieval_tool = FakeRetrievalTool(evidence)

    synthesizer = DeterministicAnswerSynthesizer(
        assembler=AnswerAssembler()
    )

    orchestrator = ADKRetrievalOrchestrator(
        retrieval_tool=retrieval_tool,
        answer_synthesizer=synthesizer,
        model="test-model",
        no_answer_guard=NoAnswerGuard(),
        answer_assembler=AnswerAssembler(),
    )

    result = orchestrator.answer("What is the settlement period?")

    assert result.grounding_status == "grounded"
    assert result.answer_text != ""
    assert "four business days" in result.answer_text

    assert result.citations == [
        "data/documents/settlement_policy.pdf"
    ]

    assert result.source_ids == ["settlement-policy"]

    assert retrieval_tool.received_query is not None
    assert retrieval_tool.received_query.query_text == (
        "What is the settlement period?"
    )

def test_t023_insufficient_evidence_returns_no_answer_without_synthesis():
    evidence = EvidenceSet(
        query_id="What is the policy for an unsupported scenario?",
        sufficiency_assessment=False,
        supporting_sources=[],
        candidates=[],
    )

    retrieval_tool = FakeRetrievalTool(evidence)

    class FailingAnswerSynthesizer:
        def synthesize(self, evidence):
            raise AssertionError(
                "Synthesis must not run when evidence is insufficient."
            )

    orchestrator = ADKRetrievalOrchestrator(
        retrieval_tool=retrieval_tool,
        answer_synthesizer=FailingAnswerSynthesizer(),
        model="test-model",
        no_answer_guard=NoAnswerGuard(),
        answer_assembler=AnswerAssembler(),
    )

    result = orchestrator.answer(
        "What is the policy for an unsupported scenario?"
    )

    assert result.grounding_status == "insufficient_evidence"
    assert result.citations == []
    assert result.source_ids == []
    assert result.missing_evidence_note is not None
    assert result.answer_text == (
        "I don't have sufficient evidence in the available "
        "knowledge sources to answer this question."
    )