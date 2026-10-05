from typing import Protocol, runtime_checkable

from responsive_agentic_rag.models.retrieval import (
    AnswerRecord,
    EvidenceSet,
)


@runtime_checkable
class AnswerSynthesizerContract(Protocol):
    """Contract for converting retrieved evidence into a grounded answer."""

    def synthesize(self, evidence: EvidenceSet) -> AnswerRecord:
        """Synthesize an answer from retrieved evidence."""
        ...


class DeterministicAnswerSynthesizer:
    """Initial deterministic answer synthesis boundary.

    This component does not use an LLM. It establishes the contract
    between retrieved evidence and the final AnswerRecord.
    """

    def synthesize(self, evidence: EvidenceSet) -> AnswerRecord:
        """Create a grounded or insufficient-evidence answer."""

        if not evidence.sufficiency_assessment:
            return AnswerRecord(
                answer_text=(
                    "I don't have sufficient evidence in the available "
                    "knowledge sources to answer this question."
                ),
                citations=[],
                source_ids=[],
                grounding_status="insufficient_evidence",
                missing_evidence_note=(
                    "The retrieved evidence did not meet the configured "
                    "sufficiency threshold."
                ),
            )

        citations = [
            candidate.source_location
            for candidate in evidence.candidates
        ]

        source_ids = list(
            dict.fromkeys(
                candidate.source_id
                for candidate in evidence.candidates
            )
        )

        evidence_text = "\n".join(
            f"- {candidate.content}"
            for candidate in evidence.candidates
        )

        return AnswerRecord(
            answer_text=evidence_text,
            citations=list(dict.fromkeys(citations)),
            source_ids=source_ids,
            grounding_status="grounded",
        )