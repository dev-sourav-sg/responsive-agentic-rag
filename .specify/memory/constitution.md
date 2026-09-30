# Responsive Agentic RAG Constitution

## Core Principles

### I. Source-Agnostic Architecture

The platform MUST treat every knowledge source as an implementation of a common ingestion contract.

Documents, websites, and future sources such as SharePoint, Confluence, S3, Google Drive, or other repositories MUST converge into a normalized internal representation before chunking, embedding, and indexing.

Adding a new source MUST NOT require redesigning the retrieval or agent architecture.

### II. Grounded Retrieval and Answering

The system MUST prioritize answers grounded in retrieved source content.

Generated answers MUST be traceable to the retrieved evidence through source metadata and citations.

The system MUST NOT use an LLM-generated summary as a replacement for authoritative source content.

When sufficient evidence cannot be retrieved, the system SHOULD explicitly indicate that the available knowledge is insufficient rather than fabricate an answer.

### III. Deterministic Retrieval, Bounded Agentic Reasoning

Deterministic components SHOULD perform parsing, normalization, chunking, embedding, filtering, retrieval, ranking, and reranking.

Agents SHOULD be used primarily for reasoning, query understanding, retrieval orchestration, evidence assessment, and answer synthesis.

Agent autonomy MUST be bounded by explicit tools, contracts, and termination conditions.

An agent MUST NOT independently bypass retrieval controls or invent unavailable knowledge.

### IV. Evaluation-First Quality

Every meaningful retrieval or answering capability MUST have measurable evaluation criteria.

The evaluation suite SHOULD include:

- Direct factual questions
- Semantic/paraphrased questions
- Multi-document questions
- Conflicting or competing sources
- Source-authority scenarios
- Website versus document retrieval
- No-answer and insufficient-evidence scenarios

Retrieval quality SHOULD be measured using metrics such as Recall@K, Precision@K, and MRR where applicable.

Answer quality SHOULD consider correctness, groundedness, and citation accuracy.

### V. Extensible and Testable Components

Core capabilities MUST be exposed through clear interfaces and independently testable components.

Ingestion connectors, document normalization, chunking, embedding, storage, retrieval, reranking, and agent tools SHOULD have explicit contracts.

Dependencies on external infrastructure or model providers SHOULD be isolated behind adapters where practical.

New capabilities SHOULD be added through extension points rather than modifying unrelated components.

### VI. Simplicity and Incremental Delivery

The implementation MUST favor the simplest architecture that satisfies the current requirement.

Distributed infrastructure, asynchronous processing, Kafka, Kubernetes, advanced agent loops, or additional persistence layers MUST NOT be introduced solely for architectural sophistication.

Production-scale capabilities MAY be introduced when justified by measurable requirements.

The initial implementation MUST maintain a working end-to-end vertical slice.

## Technical Constraints

### Technology Stack

The initial implementation will use:

- Python 3.12+
- `uv` for Python environment and dependency management
- Google ADK for agent orchestration
- Vector search for semantic retrieval
- Local development infrastructure through LocalStack where applicable
- Docker for reproducible local execution

Infrastructure-specific implementations MUST remain replaceable where practical.

### Security

Secrets MUST NOT be committed to source control.

Credentials and API keys MUST be provided through environment variables or approved secret-management mechanisms.

The system MUST consider prompt injection and malicious content originating from retrieved documents or websites.

Retrieved content MUST be treated as untrusted data and MUST NOT automatically become executable instructions.

### Observability

Important pipeline stages MUST produce structured, actionable logs.

At minimum, the system SHOULD make it possible to trace:

- Source ingestion
- Document/chunk identifiers
- Retrieval queries
- Retrieved candidates
- Ranking/reranking decisions
- Agent execution
- Final evidence used for answering
- Evaluation results
- Errors and retries

Sensitive information MUST NOT be written to logs unnecessarily.

## Development Workflow

Development follows a specification-driven workflow:

1. Establish or update the project constitution.
2. Define the functional specification.
3. Clarify ambiguous requirements where necessary.
4. Create the technical implementation plan.
5. Generate actionable implementation tasks.
6. Implement incrementally.
7. Run unit and integration tests.
8. Evaluate retrieval and answer quality.
9. Review the implementation against the specification and constitution.

Spec Kit provides the primary specification lifecycle.

Lattice provides complementary engineering workflows for requirements refinement, design, implementation, and code review. The two systems MUST NOT create competing sources of truth.

The specification and approved architectural decisions remain authoritative.

## Quality Gates

Before considering a feature complete:

- The relevant specification MUST be satisfied.
- Automated tests MUST pass.
- New retrieval behavior MUST have evaluation coverage where applicable.
- Source citations MUST remain traceable.
- No secrets or credentials may be committed.
- Architectural deviations MUST be documented.
- Significant trade-offs SHOULD be recorded as Architecture Decision Records (ADRs).

## Governance

This constitution defines the engineering principles for the Responsive Agentic RAG project and takes precedence over convenience-based implementation decisions.

Changes to these principles MUST be intentional and documented.

Any amendment MUST:

1. Explain the reason for the change.
2. Identify affected specifications or implementation decisions.
3. Update the constitution version.
4. Record the amendment date.

The constitution SHOULD be reviewed whenever the project scope, architecture, or operational requirements materially change.

**Version**: 1.0.0 | **Ratified**: 2026-09-30 | **Last Amended**: 2026-09-30
