from google.adk.agents import Agent

from responsive_agentic_rag.agents.answer_assembler import (
    AnswerAssembler,
    AnswerAssemblerContract,
)
from responsive_agentic_rag.agents.answer_synthesizer import (
    AnswerSynthesizerContract,
)
from responsive_agentic_rag.agents.tools import RetrievalToolContract
from responsive_agentic_rag.models.retrieval import (
    AnswerRecord,
    EvidenceSet,
    RetrievalQuery,
)
from responsive_agentic_rag.retrieval.no_answer_guard import (
    InsufficientEvidenceError,
    NoAnswerGuard,
)


class ADKRetrievalOrchestrator:
    def __init__(
        self,
        retrieval_tool: RetrievalToolContract,
        answer_synthesizer: AnswerSynthesizerContract,
        model: str,
        no_answer_guard: NoAnswerGuard | None = None,
        answer_assembler: AnswerAssemblerContract | None = None,
    ) -> None:
        self._retrieval_tool = retrieval_tool
        self._answer_synthesizer = answer_synthesizer
        self._model = model
        self._no_answer_guard = no_answer_guard or NoAnswerGuard()
        self._answer_assembler = answer_assembler or AnswerAssembler()

    def _retrieve(self, query: str) -> EvidenceSet:
        retrieval_query = RetrievalQuery(
            query_text=query,
            max_results=10,
        )
        return self._retrieval_tool.retrieve(retrieval_query)

    def retrieve_evidence(self, query: str) -> dict:
        evidence = self._retrieve(query)

        return {
            "query": query,
            "sufficient": evidence.sufficiency_assessment,
            "supporting_sources": evidence.supporting_sources,
            "candidates": [
                {
                    "chunk_id": candidate.chunk_id,
                    "source_id": candidate.source_id,
                    "source_type": candidate.source_type,
                    "source_location": candidate.source_location,
                    "content": candidate.content,
                    "combined_score": candidate.combined_score,
                    "authority_score": candidate.authority_score,
                }
                for candidate in evidence.candidates
            ],
        }

    def answer(self, query: str) -> AnswerRecord:
        evidence = self._retrieve(query)

        try:
            self._no_answer_guard.enforce(evidence)
        except InsufficientEvidenceError:
            return self._answer_assembler.assemble(
                answer_text=(
                    "I don't have sufficient evidence in the available "
                    "knowledge sources to answer this question."
                ),
                candidates=[],
                grounding_status="insufficient_evidence",
            )

        return self._answer_synthesizer.synthesize(evidence)

    def build_agent(self) -> Agent:
        return Agent(
            name="responsive_rag_agent",
            description=(
                "A grounded enterprise knowledge assistant that uses "
                "retrieved evidence to answer questions."
            ),
            model=self._model,
            instruction=(
                "You are a grounded enterprise knowledge assistant. "
                "Use the retrieve_evidence tool to obtain supporting "
                "evidence before answering factual questions. "
                "Do not rely on pretrained knowledge when the answer "
                "requires information from the knowledge corpus. "
                "If the retrieval result indicates insufficient evidence, "
                "do not invent an answer. State that the available evidence "
                "is insufficient. "
                "When evidence is sufficient, answer using only the "
                "retrieved evidence and identify the supporting sources."
            ),
            tools=[self.retrieve_evidence],
        )