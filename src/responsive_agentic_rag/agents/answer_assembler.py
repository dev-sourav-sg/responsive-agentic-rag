from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from responsive_agentic_rag.models.retrieval import (
    AnswerRecord,
    RetrievalCandidate,
)


@runtime_checkable
class AnswerAssemblerContract(Protocol):
    """Contract for assembling grounded answers and provenance."""

    def assemble(
        self,
        answer_text: str,
        candidates: Sequence[RetrievalCandidate],
        grounding_status: str = "grounded",
    ) -> AnswerRecord:
        """Assemble the final answer with evidence-derived citations."""
        ...


class AnswerAssembler:
    """Deterministically assembles final answers and citations."""

    def assemble(
        self,
        answer_text: str,
        candidates: Sequence[RetrievalCandidate],
        grounding_status: str = "grounded",
    ) -> AnswerRecord:
        """Build an AnswerRecord from answer text and retrieved evidence."""

        if not answer_text.strip():
            raise ValueError("answer_text must not be empty")

        if grounding_status not in {
            "grounded",
            "insufficient_evidence",
        }:
            raise ValueError(
                "grounding_status must be 'grounded' or "
                "'insufficient_evidence'"
            )

        source_ids = list(
            dict.fromkeys(
                candidate.source_id
                for candidate in candidates
            )
        )

        citations = list(
            dict.fromkeys(
                candidate.source_location
                for candidate in candidates
            )
        )

        if grounding_status == "insufficient_evidence":
            return AnswerRecord(
                answer_text=answer_text,
                citations=[],
                source_ids=[],
                grounding_status="insufficient_evidence",
                missing_evidence_note=(
                    "The available evidence was insufficient to "
                    "support a grounded answer."
                ),
            )

        return AnswerRecord(
            answer_text=answer_text,
            citations=citations,
            source_ids=source_ids,
            grounding_status="grounded",
        )