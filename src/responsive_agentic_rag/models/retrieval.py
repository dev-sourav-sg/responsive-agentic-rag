from typing import Any, Literal

from pydantic import BaseModel, Field

from responsive_agentic_rag.models.knowledge import SourceType


class RetrievalQuery(BaseModel):
    """Parameters used to retrieve evidence for a user query."""

    query_text: str
    user_context: dict[str, Any] = Field(default_factory=dict)

    source_filters: dict[str, Any] = Field(default_factory=dict)

    max_results: int = Field(default=10, ge=1)
    hybrid_mode: Literal["semantic", "lexical", "hybrid"] = "hybrid"

    min_authority: float | None = Field(default=None, ge=0.0, le=1.0)


class RetrievalCandidate(BaseModel):
    """A ranked chunk candidate returned by retrieval."""

    chunk_id: str
    source_id: str
    source_type: SourceType
    source_location: str

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    semantic_score: float = 0.0
    lexical_score: float = 0.0
    authority_score: float = Field(default=0.0, ge=0.0, le=1.0)

    combined_score: float = 0.0
    rank: int = Field(default=0, ge=0)


class EvidenceSet(BaseModel):
    """Evidence assembled for answer generation."""

    query_id: str
    candidates: list[RetrievalCandidate] = Field(default_factory=list)

    retrieval_summary: str | None = None
    sufficiency_assessment: bool | None = None

    supporting_sources: list[str] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    """Grounded answer produced from retrieved evidence."""

    answer_text: str

    citations: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    grounding_status: Literal["grounded", "insufficient_evidence"] = "grounded"

    missing_evidence_note: str | None = None


class EvaluationCase(BaseModel):
    """A repeatable evaluation case for retrieval and answer quality."""

    case_id: str
    question: str

    source_expectations: list[str] = Field(default_factory=list)

    expected_answer: str | None = None

    answerable: bool

    source_types: list[SourceType] = Field(default_factory=list)

    difficulty: str | None = None
    tags: list[str] = Field(default_factory=list)