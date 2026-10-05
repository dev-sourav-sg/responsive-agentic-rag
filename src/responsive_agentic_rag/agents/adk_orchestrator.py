from google.adk.agents import Agent

from responsive_agentic_rag.agents.answer_synthesizer import (
    AnswerSynthesizerContract,
)
from responsive_agentic_rag.agents.tools import RetrievalToolContract
from responsive_agentic_rag.models.retrieval import (
    AnswerRecord,
    EvidenceSet,
    RetrievalQuery,
)


class ADKRetrievalOrchestrator:
    """Google ADK orchestration boundary for grounded retrieval."""

    def __init__(
        self,
        retrieval_tool: RetrievalToolContract,
        answer_synthesizer: AnswerSynthesizerContract,
        model: str,
    ) -> None:
        self._retrieval_tool = retrieval_tool
        self._answer_synthesizer = answer_synthesizer
        self._model = model

    def _retrieve(self, query: str) -> EvidenceSet:
        """Execute deterministic retrieval and return the evidence set."""

        retrieval_query = RetrievalQuery(
            query_text=query,
            max_results=10,
        )

        return self._retrieval_tool.retrieve(retrieval_query)

    def retrieve_evidence(self, query: str) -> dict:
        """Expose deterministic retrieval to the ADK agent."""

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
        """Run deterministic retrieval followed by grounded synthesis."""

        evidence = self._retrieve(query)

        return self._answer_synthesizer.synthesize(evidence)

    def build_agent(self) -> Agent:
        """Build the Google ADK agent with retrieval as its tool."""

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