from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


SourceType = Literal["document", "website"]


class KnowledgeSource(BaseModel):
    """Metadata describing an ingested knowledge source."""

    source_id: str
    source_type: SourceType
    title: str
    source_location: str

    version: str | None = None
    owner: str | None = None
    publication_date: datetime | None = None
    modification_date: datetime | None = None

    authority: float = Field(ge=0.0, le=1.0, default=0.5)
    approval_status: str | None = None

    created_at: datetime = Field(
    default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedSourceRecord(BaseModel):
    """Normalized content extracted from a knowledge source."""

    record_id: str
    source_id: str
    source_type: SourceType

    title: str
    section_path: list[str] = Field(default_factory=list)
    body_text: str

    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    """A retrievable chunk derived from a normalized source record."""

    chunk_id: str
    source_id: str
    record_id: str

    content: str
    chunk_index: int = Field(ge=0)

    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)

    section_path: list[str] = Field(default_factory=list)
    token_count: int = Field(ge=0)

    embedding: list[float] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    authority_score: float = Field(ge=0.0, le=1.0)