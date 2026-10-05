from responsive_agentic_rag.agents.adk_orchestrator import (
    ADKRetrievalOrchestrator,
)
from responsive_agentic_rag.models.retrieval import (
    AnswerRecord,
    EvidenceSet,
    RetrievalCandidate,
    RetrievalQuery,
)
from responsive_agentic_rag.agents.answer_assembler import AnswerAssembler


class FakeRetrievalTool:
    def __init__(self) -> None:
        self.received_query: RetrievalQuery | None = None
        self.call_count = 0

    def retrieve(self, query: RetrievalQuery) -> EvidenceSet:
        self.received_query = query
        self.call_count += 1

        return EvidenceSet(
            query_id=query.query_text,
            sufficiency_assessment=True,
            supporting_sources=["source-1"],
            candidates=[
                RetrievalCandidate(
                    chunk_id="chunk-1",
                    source_id="source-1",
                    source_type="document",
                    source_location="test.pdf",
                    content="The settlement period is four days.",
                    combined_score=0.91,
                    authority_score=0.9,
                )
            ],
        )


class FakeAnswerSynthesizer:
    def __init__(self) -> None:
        self.received_evidence: EvidenceSet | None = None

    def synthesize(self, evidence: EvidenceSet) -> AnswerRecord:
        self.received_evidence = evidence

        return AnswerRecord(
            answer_text="The settlement period is four days.",
            citations=["test.pdf"],
            source_ids=["source-1"],
            grounding_status="grounded",
        )


def create_orchestrator():
    retrieval_tool = FakeRetrievalTool()
    answer_synthesizer = FakeAnswerSynthesizer()

    orchestrator = ADKRetrievalOrchestrator(
        retrieval_tool=retrieval_tool,
        answer_synthesizer=answer_synthesizer,
        model="test-model",
    )

    return orchestrator, retrieval_tool, answer_synthesizer


def test_retrieve_evidence_delegates_to_retrieval_tool():
    orchestrator, retrieval_tool, _ = create_orchestrator()

    result = orchestrator.retrieve_evidence(
        "What is the settlement period?"
    )

    assert retrieval_tool.received_query is not None
    assert (
        retrieval_tool.received_query.query_text
        == "What is the settlement period?"
    )

    assert result["query"] == "What is the settlement period?"
    assert result["sufficient"] is True
    assert result["supporting_sources"] == ["source-1"]


def test_retrieve_evidence_returns_provenance_and_content():
    orchestrator, _, _ = create_orchestrator()

    result = orchestrator.retrieve_evidence(
        "What is the settlement period?"
    )

    candidate = result["candidates"][0]

    assert candidate["chunk_id"] == "chunk-1"
    assert candidate["source_id"] == "source-1"
    assert candidate["source_type"] == "document"
    assert candidate["source_location"] == "test.pdf"
    assert candidate["content"] == "The settlement period is four days."
    assert candidate["combined_score"] == 0.91
    assert candidate["authority_score"] == 0.9


def test_answer_delegates_evidence_to_answer_synthesizer():
    orchestrator, _, answer_synthesizer = create_orchestrator()

    result = orchestrator.answer(
        "What is the settlement period?"
    )

    assert answer_synthesizer.received_evidence is not None
    assert (
        answer_synthesizer.received_evidence.query_id
        == "What is the settlement period?"
    )

    assert isinstance(result, AnswerRecord)
    assert result.answer_text == "The settlement period is four days."
    assert result.grounding_status == "grounded"
    assert result.citations == ["test.pdf"]
    assert result.source_ids == ["source-1"]


def test_build_agent_creates_google_adk_agent():
    orchestrator, _, _ = create_orchestrator()

    agent = orchestrator.build_agent()

    assert agent.name == "responsive_rag_agent"
    assert agent.description
    assert agent.instruction
    assert len(agent.tools) == 1


def test_answer_returns_no_answer_when_evidence_is_insufficient():
    retrieval_tool = FakeRetrievalTool()

    retrieval_tool.retrieve = lambda query: EvidenceSet(
        query_id=query.query_text,
        sufficiency_assessment=False,
        supporting_sources=[],
        candidates=[],
    )

    from responsive_agentic_rag.agents.answer_synthesizer import (
        DeterministicAnswerSynthesizer,
    )

    orchestrator = ADKRetrievalOrchestrator(
        retrieval_tool=retrieval_tool,
        answer_synthesizer=DeterministicAnswerSynthesizer(
        assembler=AnswerAssembler()
        ),
        model="test-model",
    )

    result = orchestrator.answer(
        "What is the policy for an unsupported scenario?"
    )

    assert isinstance(result, AnswerRecord)
    assert result.grounding_status == "insufficient_evidence"
    assert result.citations == []
    assert result.source_ids == []
    assert result.missing_evidence_note is not None


def test_answer_retrieves_evidence_exactly_once():
    orchestrator, retrieval_tool, _ = create_orchestrator()

    orchestrator.answer(
        "What is the settlement period?"
    )

    assert retrieval_tool.call_count == 1

def test_answer_does_not_synthesize_when_evidence_is_insufficient():
    retrieval_tool = FakeRetrievalTool()

    retrieval_tool.retrieve = lambda query: EvidenceSet(
        query_id=query.query_text,
        sufficiency_assessment=False,
        supporting_sources=[],
        candidates=[],
    )

    class FailingAnswerSynthesizer:
        def synthesize(self, evidence):
            raise AssertionError(
                "Answer synthesis must not run when evidence is insufficient."
            )

    orchestrator = ADKRetrievalOrchestrator(
        retrieval_tool=retrieval_tool,
        answer_synthesizer=FailingAnswerSynthesizer(),
        model="test-model",
    )

    result = orchestrator.answer(
        "What is the policy for an unsupported scenario?"
    )

    assert isinstance(result, AnswerRecord)
    assert result.grounding_status == "insufficient_evidence"
    assert result.citations == []
    assert result.source_ids == []