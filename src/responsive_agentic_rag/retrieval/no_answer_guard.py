from responsive_agentic_rag.models.retrieval import EvidenceSet


class InsufficientEvidenceError(RuntimeError):
    """Raised when retrieved evidence is not sufficient to support an answer."""


class NoAnswerGuard:
    """Enforces the policy that insufficient evidence cannot produce an answer."""

    def enforce(self, evidence: EvidenceSet) -> None:
        if not evidence.sufficiency_assessment:
            raise InsufficientEvidenceError(
                "The available evidence is insufficient to support a grounded answer."
            )