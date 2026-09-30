# Research: Multi-Source Agentic RAG Platform

## Decision Log

### Decision 1: Use a single Python package with explicit domain modules

**Decision**: Build the MVP as one Python project with isolated modules for connectors, normalization, chunking, retrieval, agent orchestration, and evaluation.

**Rationale**: The project already starts as a lightweight Python package, and the constitution emphasizes simplicity, modular architecture, and incremental delivery. This design gives clear boundaries without introducing service sprawl or distributed infrastructure.

**Alternatives considered**: A microservice split into separate document, web, and retrieval services was rejected because it would add operational complexity without strengthening the MVP’s user value.

### Decision 2: Normalize all sources into one common knowledge representation

**Decision**: Source-specific adapters will produce content that is converted into the common `KnowledgeSource` and `NormalizedSourceRecord` models before chunking and vector indexing.

**Rationale**: This satisfies the source-agnostic requirement and ensures that the downstream retrieval and agent layers never depend on source-specific logic.

**Alternatives considered**: Creating separate retrieval pipelines per source type was rejected because it would duplicate logic and fragment ranking behavior.

### Decision 3: Use deterministic retrieval services plus a bounded ADK orchestration layer

**Decision**: Retrieval, filtering, ranking, and reranking will be implemented in deterministic Python services; Google ADK will orchestrate these tools rather than owning the retrieval logic itself.

**Rationale**: The constitution requires bounded agentic reasoning, explicit tools, and grounded answers. Deterministic services produce inspectable evidence and maintainability, while the agent is responsible for orchestration and synthesis.

**Alternatives considered**: Letting the agent directly perform vector search or LLM-based ranking was rejected because it would reduce determinism and testability.

### Decision 4: Use a vector store and hybrid retrieval with authority-aware scoring

**Decision**: The product will combine semantic/vector retrieval with lexical overlap and explicit metadata weighting, then apply deterministic reranking using authority and source-type signals.

**Rationale**: The feature requirements explicitly mention semantic retrieval, metadata filtering, keyword retrieval, hybrid retrieval, and source authority. A weighted hybrid score allows these concerns to be tested and tuned without depending on undocumented LLM reasoning.

**Alternatives considered**: Pure semantic search alone was rejected because it is weak when authority or exact term matching matters; pure keyword search alone was rejected because it misses paraphrased questions.

### Decision 5: Use Qdrant as the vector store

**Decision**: Use Qdrant as the vector database for the MVP. Embeddings will be generated through a configurable embedding provider, while Qdrant will provide vector storage and similarity-search capabilities.

For local development, Qdrant may be deployed through the LocalStack Qdrant extension. The application will continue to access Qdrant through its standard client/API.

The application MUST expose an internal `VectorStore` abstraction so that the retrieval layer is not directly coupled to Qdrant-specific implementation details.

**Rationale**:
- Qdrant provides a dedicated vector database suitable for semantic retrieval.
- Qdrant has a straightforward local Docker-based deployment model.
- Qdrant can also be integrated into a LocalStack-based local development environment through the Qdrant extension.
- Using an internal `VectorStore` abstraction preserves the ability to change the vector-store implementation later without changing the core retrieval and agentic layers.
- Selecting one concrete vector database removes implementation ambiguity from the MVP and keeps the initial implementation focused.
- The embedding provider remains configurable so embedding generation is not unnecessarily coupled to the vector database.

**Alternatives considered**:
- **pgvector**: Not selected for the MVP because introducing PostgreSQL as the vector-storage dependency would add another persistence technology when the primary MVP requirement is vector retrieval.
- **Managed vector databases**: Not selected because the assignment requires a locally reproducible development environment and does not require a managed service.
- **Standalone Qdrant Docker deployment**: Supported as an alternative local deployment mechanism when LocalStack is not required. The application architecture remains unchanged because both deployment modes expose Qdrant through its standard client/API.

**Architectural constraint**:
The retrieval layer MUST depend on the application's `VectorStore` abstraction rather than directly embedding Qdrant-specific operations throughout the codebase.

### Decision 6: Apply clear prompt-injection and trust boundaries

**Decision**: All ingested content is treated as untrusted data; the system will not automatically execute or interpret retrieved text as instructions, and the agent will only use the retrieved evidence as context.

**Rationale**: The constitution explicitly calls out prompt injection risk and the need to treat retrieved content as untrusted.

**Alternatives considered**: Allowing free-form model interpretation of raw source content without sanitization was rejected as unsafe.

## Resolved Architecture Notes

- Documents and websites are both represented as `KnowledgeSource` entries with distinct source-specific adapters but one common downstream contract.
- Chunking happens after normalization to preserve section context and provenance.
- Embeddings are generated for chunks only, not entire source documents, to keep retrieval granular and support citation fidelity.
- Source authority metadata influences a deterministic ranking formula rather than ad hoc model judgment.
- The evaluation harness uses a curated dataset covering answerable, conflicting-source, mixed-source, and no-answer scenarios.

## Open questions resolved during planning

- There is no unresolved product ambiguity requiring a clarification cycle. The feature spec defines the required behavior, and the implementation plan turns the missing technical decisions into concrete implementation patterns.
