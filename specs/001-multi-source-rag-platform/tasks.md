# Tasks: Multi-Source Agentic RAG Platform

**Input**: Design documents from `/specs/001-multi-source-rag-platform/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create the base Python package structure and feature directories per plan in `src/responsive_agentic_rag/` and `tests/`
- [ ] T002 Initialize project configuration and dependency management in `pyproject.toml` for Python 3.12, `uv`, and local runtime tooling
- [ ] T003 Create environment configuration and settings schema in `src/responsive_agentic_rag/config/settings.py` for vector store, embedding model, document paths, and website sources
- [ ] T004 Create observability and structured logging scaffolding in `src/responsive_agentic_rag/observability/logging.py` for source processing, retrieval, and evaluation traceability

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the common data contracts and source abstraction that every story depends on

- [ ] T005 Define the canonical source and retrieval models in `src/responsive_agentic_rag/models/knowledge.py` and `src/responsive_agentic_rag/models/retrieval.py`, including `source_id`, `source_type`, `title`, `source_location`, `version`, `owner`, `publication_date`, `modification_date`, `authority`, `approval_status`, and the retrieval candidate fields
- [ ] T006 Implement the source connector protocol and registry in `src/responsive_agentic_rag/connectors/base.py` so document and website connectors converge on a single normalized knowledge contract
- [ ] T007 Implement the document connector adapter in
src/responsive_agentic_rag/connectors/document_connector.py
for discovering/fetching document content and preserving source metadata without performing shared normalization, chunking, embedding, or retrieval.
- [ ] T008 Implement the website connector adapter in
src/responsive_agentic_rag/connectors/website_connector.py
for fetching and extracting website content and preserving available source metadata without performing shared normalization, chunking, embedding, or retrieval.
- [ ] T009 Implement the ingestion normalizer and content contract in `src/responsive_agentic_rag/ingestion/normalizer.py` to standardize document and website data before chunking
- [ ] T010 Implement the chunking and embedding service in `src/responsive_agentic_rag/ingestion/chunker.py` and `src/responsive_agentic_rag/ingestion/embedding_service.py`, preserving document or section context while producing vector representations
- [ ] T011 Implement the vector store interface and index persistence in `src/responsive_agentic_rag/retrieval/vector_store.py` for chunk content, embeddings, and metadata

## Phase 3: User Story 1 - Ingest and index document and website knowledge (Priority: P1)

**Story goal**: Build the shared ingestion pipeline and make both source types queryable through one corpus.

**Independent test criteria**: A local ingestion run indexes a sample document set and website URLs and exposes the resulting chunk metadata for inspection.

- [ ] T012 [US1] Build the ingestion orchestration flow in `src/responsive_agentic_rag/app.py` and `src/responsive_agentic_rag/cli.py` to run document and website ingestion through the shared pipeline
- [ ] T013 [P] [US1] Add document connector validation tests covering document
identification, title, location, owner, version, dates, authority, and approval metadata preservation.
- [ ] T014 [P] [US1] Add website connector validation tests covering URL, title,
available publication dates, owner, and authority metadata preservation.
- [ ] T015 [US1] Add chunk storage validation in `src/responsive_agentic_rag/retrieval/vector_store.py` to ensure chunk content, embeddings, and metadata are retained for downstream retrieval
- [ ] T016 [US1] Add a smoke-test ingestion scenario in `tests/integration/` that ingests a sample set of documents and website URLs and asserts that both source types are represented in the indexed corpus

## Phase 4: User Story 2 - Ask natural-language questions and receive grounded answers with citations (Priority: P1)

**Story goal**: Deliver the end-to-end retrieval and answer pipeline with evidence-grounded citations.

**Independent test criteria**: A question can be answered from the corpus and the response includes retrieved evidence plus source citations; insufficient evidence triggers a no-answer result.

- [ ] T017 [US2] Implement deterministic semantic and lexical retrieval in `src/responsive_agentic_rag/retrieval/hybrid_search.py` with filtering and hybrid combination logic
- [ ] T018 [US2] Implement deterministic reranking in
src/responsive_agentic_rag/retrieval/ranker.py using relevance signals from semantic and lexical retrieval results.
- [ ] T019 [US2] Implement the deterministic RetrievalTool and agent-facing
contracts in src/responsive_agentic_rag/agents/tools.py, including evidence sufficiency assessment and answer synthesis boundaries, while ensuring the agent cannot bypass the RetrievalTool.
- [ ] T020 [US2] Implement the Google ADK orchestration flow in `src/responsive_agentic_rag/agents/adk_orchestrator.py` to understand the query, call retrieval tools, assess sufficiency, and synthesize grounded answers
- [ ] T021 [US2] Implement citation generation and final answer assembly in `src/responsive_agentic_rag/agents/tools.py` so responses explicitly tie back to retrieved source IDs and locations
- [ ] T022 [US2] Add a no-answer guard in `src/responsive_agentic_rag/agents/adk_orchestrator.py` that returns an explicit insufficient-evidence response when the corpus cannot support the question
- [ ] T023 [US2] Add an integration test in `tests/integration/` covering a direct answer case and an insufficient-evidence case for the agent workflow

## Phase 5: User Story 3 - Retrieve and rank evidence with authority-aware logic (Priority: P1)

**Story goal**: Make retrieval deterministic, observable, and authority-sensitive when sources conflict.

**Independent test criteria**: Retrieval results prioritize an approved enterprise source above a lower-trust website when the same fact is disputed.

- [ ] T024 [US3] Add authority and confidence scoring to the ranking pipeline
in src/responsive_agentic_rag/retrieval/ranker.py based on explicit metadata such as authority, approval_status, and source type.
- [ ] T025 [US3] Implement query-level metadata filters and retrieval configuration in `src/responsive_agentic_rag/models/retrieval.py` and `src/responsive_agentic_rag/retrieval/hybrid_search.py` for source type, location, owner, and authority constraints
- [ ] T026 [US3] Capture retrieval metadata in the evidence set so each result includes chunk content, source identifier, source type, source location, metadata, and ranking details
- [ ] T027 [US3] Add a conflict-resolution integration test in `tests/integration/` that verifies authority-aware ranking when approved and unapproved sources disagree

## Phase 6: User Story 4 - Evaluate quality and extend the system with additional connectors (Priority: P2)

**Story goal**: Ensure the platform is repeatably measurable and source-agnostic by design.

**Independent test criteria**: The evaluation dataset runs locally and reports retrieval and answer-quality metrics; a new connector can be added without changing the core retrieval and agent layers.

- [ ] T028 [US4] Create the evaluation dataset and domain scenarios in `src/responsive_agentic_rag/evaluation/dataset.py` covering direct factual, paraphrased, multi-source, conflicting-source, mixed-source, and no-answer cases
- [ ] T029 [US4] Implement the evaluation runner in `src/responsive_agentic_rag/evaluation/runner.py` to calculate recall, precision, MRR, groundedness, and citation correctness
- [ ] T030 [US4] Add a connector extension example in `src/responsive_agentic_rag/connectors/` showing how a future source type can be implemented without altering the core retrieval or agent architecture
- [ ] T031 [US4] Add a validation scenario in `tests/eval/` that executes the evaluation harness and asserts repeatable metric output for the MVP dataset

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Hardening, observability, safety, and local demonstration readiness

- [ ] T032 Add structured logging and traceability for ingestion, retrieval, ranking, and final answer synthesis in `src/responsive_agentic_rag/observability/logging.py`
- [ ] T033 Add prompt-injection and untrusted-content safeguards in `src/responsive_agentic_rag/ingestion/normalizer.py` and `src/responsive_agentic_rag/agents/tools.py` so retrieved content is treated as untrusted and never executed as instructions
- [ ] T034 Validate the source-agnostic architecture and connector contract against the plan in `src/responsive_agentic_rag/connectors/base.py` and `src/responsive_agentic_rag/connectors/` so future connectors remain modular
- [ ] T035 Finalize the local demo flow and documentation in `README.md` and `specs/001-multi-source-rag-platform/quickstart.md` to cover setup, ingestion, retrieval, answer generation, and evaluation
- [ ] T036 Run the local smoke validation for the end-to-end MVP flow: ingest sources, retrieve evidence, answer a question, and confirm no-answer behavior and citation reporting in the CLI or app

## Dependencies

- User Story 1 must complete before User Story 2 and User Story 3 because retrieval depends on the indexed corpus.
- User Story 2 depends on User Story 1 and the retrieval/ranking scaffolding.
- User Story 3 depends on User Story 1 and the foundational retrieval contracts because ranking operates on indexed chunks and retrieval candidates. It can be implemented independently of the ADK answer orchestration once the retrieval candidate model exists.
- User Story 4 depends on User Story 1 and User Story 2 because evaluation requires indexed sources and grounded answer outputs.
- Phase 7 depends on all story phases and acts as the final hardening and demo readiness gate.

## Parallel Execution Examples

- Parallel work for Phase 2: `T006`, `T007`, and `T008` can be developed in parallel because they are separate source connectors but share the same common contract.
- Parallel work for Phase 3: `T013` and `T014` can be developed in parallel because they validate source-specific metadata handling independently.
- Parallel work for Phase 4: `T017`, `T018`, and `T019` can be developed in parallel once the shared model contracts are in place.
- Parallel work for Phase 6: `T028` and `T029` can be developed alongside `T030` because the dataset and runner are independent of the connector extension example.

## Implementation Strategy

- Deliver the MVP by completing User Story 1 and User Story 2 first; these two stories provide the user-facing value of ingestion and grounded answers.
- Implement the authority-sensitive ranking and evaluation work in User Story 3 and User Story 4 as the second stage, after the core retrieval path is proven stable.
- Keep the architecture modular by using clear contracts and avoiding source-specific logic in the downstream agent and retrieval layers.
- Favor deterministic, inspectable retrieval and ranking logic over hidden model judgment, and treat evaluation as a mandatory quality gate for retrieval and answer changes.
