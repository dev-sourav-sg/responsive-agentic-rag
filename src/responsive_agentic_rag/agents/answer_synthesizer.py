from typing import Protocol, runtime_checkable

from responsive_agentic_rag.agents.answer_assembler import AnswerAssemblerContract
from responsive_agentic_rag.models.retrieval import AnswerRecord, EvidenceSet


@runtime_checkable
class AnswerSynthesizerContract(Protocol):
    def synthesize(self, evidence: EvidenceSet) -> AnswerRecord:
        ...


class DeterministicAnswerSynthesizer:
    def __init__(self, assembler: AnswerAssemblerContract) -> None:
        self._assembler = assembler

    def synthesize(self, evidence: EvidenceSet) -> AnswerRecord:
        if not evidence.sufficiency_assessment:
            return self._assembler.assemble(
                answer_text=(
                    "I don't have sufficient evidence in the available "
                    "knowledge sources to answer this question."
                ),
                candidates=evidence.candidates,
                grounding_status="insufficient_evidence",
            )

        evidence_text = "\n".join(
            f"- {candidate.content}" for candidate in evidence.candidates
        )

        return self._assembler.assemble(
            answer_text=evidence_text,
            candidates=evidence.candidates,
            grounding_status="grounded",
        )