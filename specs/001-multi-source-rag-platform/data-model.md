# Data Model: Multi-Source Agentic RAG Platform

## Core Entities

### KnowledgeSource

Represents one source of truth in the corpus, regardless of whether it is a document or a website.

Fields:
- `source_id`: stable unique identifier
- `source_type`: `document` or `website`
- `title`: human-readable label for the source
- `source_location`: local file path, document URI, or website URL
- `version`: document revision or content version when available
- `owner`: responsible party or team
- `publication_date`: date of publication when known
- `modification_date`: last known update date
- `authority`: value representing trust or approval tier
- `approval_status`: approved, draft, external, or unknown
- `created_at`: ingestion timestamp
- `metadata`: additional source-specific fields

Relationships:
- One source may produce many `NormalizedSourceRecord` entries or many chunk records.

### NormalizedSourceRecord

The converged representation produced after source-specific extraction and cleanup.

Fields:
- `record_id`
- `source_id`
- `source_type`
- `title`
- `section_path`: logical document structure such as heading path
- `body_text`: normalized plain-text content
- `language`
- `metadata`

Relationships:
- Many records belong to a single source.
- Each record yields multiple chunks.

### ChunkRecord

A retrieval unit derived from a normalized source record.

Fields:
- `chunk_id`
- `source_id`
- `record_id`
- `content`
- `chunk_index`
- `start_offset` / `end_offset`
- `section_path`
- `token_count`
- `embedding`
- `metadata`
- `authority_score`

Relationships:
- Each chunk belongs to one normalized record and one source.
- Each chunk can appear in many retrieval results.

### RetrievalQuery

The structured input to the retrieval system.

Fields:
- `query_text`
- `user_context`
- `source_filters`
- `max_results`
- `hybrid_mode`
- `min_authority`

### RetrievalCandidate

A single ranked candidate returned by retrieval.

Fields:
- `chunk_id`
- `source_id`
- `source_type`
- `source_location`
- `content`
- `metadata`
- `semantic_score`
- `lexical_score`
- `authority_score`
- `combined_score`
- `rank`

Relationships:
- A candidate always references a chunk and a source.
- Multiple candidates can be combined into one evidence set.

### EvidenceSet

The collection of candidates used for a final answer.

Fields:
- `query_id`
- `candidates`
- `retrieval_summary`
- `sufficiency_assessment`
- `supporting_sources`

### AnswerRecord

The final output produced from retrieved evidence.

Fields:
- `answer_text`
- `citations`
- `source_ids`
- `grounding_status`
- `missing_evidence_note`

### EvaluationCase

A benchmark item used to assess quality and regressions.

Fields:
- `case_id`
- `question`
- `source_expectations`
- `expected_answer`
- `answerable`
- `source_types`
- `difficulty`
- `tags`

## Relationships

- A `KnowledgeSource` produces one or more `NormalizedSourceRecord` values.
- A `NormalizedSourceRecord` produces one or more `ChunkRecord` values.
- A `ChunkRecord` contributes to multiple `RetrievalCandidate` records across different queries.
- `RetrievalCandidate` values are assembled into an `EvidenceSet` for the agent.
- An `AnswerRecord` is grounded directly in the evidence set and cites the relevant sources.
- `EvaluationCase` items test retrieval, ranking, and answer faithfulness over time.

## Validation Rules

- Every chunk must be associated with exactly one source and one normalized record.
- Source metadata must be preserved for citation and authority-aware ranking.
- Retrieval candidates must include source identity, source location, and ranking metadata.
- Answer generation must be rejected or downgraded when no evidence meets the threshold for sufficiency.
- Source type and authority must be included in retrieval scoring and reporting.
