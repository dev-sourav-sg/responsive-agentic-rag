# Contracts: Source Connectors and Retrieval Tools

## Source Connector Contract

All source-specific connectors must implement the same functional contract before they reach the shared chunking and indexing pipeline.

```python
from typing import Protocol, Sequence

class SourceConnector(Protocol):
    source_type: str

    def list_sources(self) -> Sequence["KnowledgeSourceConfig"]: ...
    def fetch_content(self, source: "KnowledgeSourceConfig") -> object: ...

### Contract requirements
- Each connector must return source content in a form that can be processed by the shared normalization pipeline.
- Metadata required for authority, provenance, and citations must be preserved from the source.
- Connector-specific extraction failures must not silently discard the source; they must be logged and reported as ingestion errors.
- Source connectors must not perform shared normalization, chunking, embedding, or retrieval.
- Source connectors must not bypass the shared ingestion and retrieval pipelines.

## Retrieval Tool Contract

The retrieval layer must expose a deterministic interface to the agent.

```python
from typing import Protocol, Sequence

class RetrievalTool(Protocol):
    def retrieve(self, query: "RetrievalQuery") -> Sequence["RetrievalCandidate"]: ...
    def filter(self, query: "RetrievalQuery", candidates: Sequence["RetrievalCandidate"]) -> Sequence["RetrievalCandidate"]: ...
    def rerank(self, candidates: Sequence["RetrievalCandidate"]) -> Sequence["RetrievalCandidate"]: ...
```

### Contract requirements
- Retrieval results must contain chunk content, source identifiers, source type, location, metadata, and ranking information.
- Filtering and reranking must be explicit and independently testable.
- The tool contract must not rely on hidden LLM judgments for source selection.

## Agent Tool Contract

The ADK-backed agent can only access deterministic tools and must produce final answers from retrieved evidence.

```python
from typing import Protocol

class AgentTool(Protocol):
    def assess_sufficiency(self, evidence: "EvidenceSet") -> bool: ...
    def synthesize_answer(self, evidence: "EvidenceSet") -> "AnswerRecord": ...
```

### Contract requirements
- Agents must not bypass retrieval.
- The answer pipeline must require a grounded evidence set.
- Insufficient evidence must result in an explicit no-answer outcome.
