from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from responsive_agentic_rag.config.settings import settings
from responsive_agentic_rag.models.knowledge import ChunkRecord

from responsive_agentic_rag.agents.adk_orchestrator import (
    ADKRetrievalOrchestrator,
)
from responsive_agentic_rag.agents.answer_assembler import (
    AnswerAssembler,
)
from responsive_agentic_rag.agents.answer_synthesizer import (
    DeterministicAnswerSynthesizer,
)
from responsive_agentic_rag.agents.tools import (
    RetrievalTool,
    EvidenceSufficiency,
)

from responsive_agentic_rag.ingestion.embedding_service import (
    EmbeddingService,
)

from responsive_agentic_rag.retrieval.authority_scorer import (
    AuthorityScorer,
)
from responsive_agentic_rag.retrieval.bm25_index import (
    BM25Index,
)
from responsive_agentic_rag.retrieval.hybrid_search import (
    HybridRetriever,
)
from responsive_agentic_rag.retrieval.qdrant_store import (
    QdrantVectorStore,
)

# IMPORTANT:
# DeterministicReranker is in deterministic_reranker.py,
# not reranker.py.
from responsive_agentic_rag.retrieval.deterministic_reranker import (
    DeterministicReranker,
)


DEMO_COLLECTION = "demo_knowledge_chunks"


# =============================================================================
# LOAD DEMO CORPUS FROM QDRANT
# =============================================================================

def load_chunks() -> list[ChunkRecord]:
    """
    Load all chunks from the demo Qdrant collection.

    Qdrant is used as the persistent vector store.
    BM25 is rebuilt in memory from the same chunks for the demo.
    """

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    chunks: list[ChunkRecord] = []

    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=DEMO_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            chunks.append(
                ChunkRecord(
                    chunk_id=payload["chunk_id"],
                    source_id=payload["source_id"],
                    record_id=payload["record_id"],
                    content=payload["content"],
                    chunk_index=payload["chunk_index"],
                    section_path=payload.get(
                        "section_path",
                        [],
                    ),
                    token_count=payload["token_count"],
                    metadata=payload.get(
                        "metadata",
                        {},
                    ),
                    authority_score=payload.get(
                        "authority_score",
                        0.0,
                    ),
                )
            )

        if offset is None:
            break

    print(
        f"Loaded {len(chunks)} chunks from Qdrant"
    )

    return chunks


# =============================================================================
# BUILD RETRIEVAL STACK
# =============================================================================

def build_retrieval_stack(
    chunks: list[ChunkRecord],
) -> ADKRetrievalOrchestrator:
    """
    Build the complete deterministic retrieval stack.

    Architecture:

        Query
          |
          v
        EmbeddingService
          |
          +----------------------+
          |                      |
          v                      v
        Qdrant                  BM25
        Semantic               Lexical
        Retrieval              Retrieval
          |                      |
          +----------+-----------+
                     |
                     v
                    RRF
                     |
                     v
              Authority Scorer
                     |
                     v
             Deterministic
                Reranker
                     |
                     v
               EvidenceSet
                     |
                     v
            Evidence Sufficiency
                     |
                     v
               ADK Agent
    """

    print(
        f"Building retrieval stack from {len(chunks)} chunks..."
    )

    # -------------------------------------------------------------------------
    # Qdrant
    # -------------------------------------------------------------------------

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=DEMO_COLLECTION,
        vector_dimension=settings.embedding_dimension,
    )

    # -------------------------------------------------------------------------
    # BM25
    # -------------------------------------------------------------------------

    bm25_index = BM25Index()

    bm25_index.build(chunks)

    # -------------------------------------------------------------------------
    # Deterministic reranker
    # -------------------------------------------------------------------------

    reranker = DeterministicReranker()

    # -------------------------------------------------------------------------
    # Authority scorer
    # -------------------------------------------------------------------------

    authority_scorer = AuthorityScorer()

    # -------------------------------------------------------------------------
    # Hybrid retriever
    # -------------------------------------------------------------------------

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=reranker,
        authority_scorer=authority_scorer,
    )

    # -------------------------------------------------------------------------
    # Embedding service
    # -------------------------------------------------------------------------

    embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
    )

    # -------------------------------------------------------------------------
    # Evidence sufficiency
    # -------------------------------------------------------------------------

    evidence_sufficiency = EvidenceSufficiency()

    # -------------------------------------------------------------------------
    # Retrieval tool
    # -------------------------------------------------------------------------

    retrieval_tool = RetrievalTool(
        retriever=retriever,
        embedding_service=embedding_service,
        evidence_sufficiency=evidence_sufficiency,
    )

    # -------------------------------------------------------------------------
    # Answer assembler
    # -------------------------------------------------------------------------

    answer_assembler = AnswerAssembler()

    # -------------------------------------------------------------------------
    # Deterministic synthesizer
    # -------------------------------------------------------------------------

    answer_synthesizer = DeterministicAnswerSynthesizer(
        assembler=answer_assembler,
    )

    # -------------------------------------------------------------------------
    # ADK orchestrator
    # -------------------------------------------------------------------------

    orchestrator = ADKRetrievalOrchestrator(
        retrieval_tool=retrieval_tool,
        answer_synthesizer=answer_synthesizer,
        model="gemini-3.5-flash-lite",
        answer_assembler=answer_assembler,
    )

    return orchestrator


# =============================================================================
# PRINT DETERMINISTIC RESULT
# =============================================================================

def print_result(result) -> None:
    """
    Print a compact representation of an AnswerRecord.
    """

    print("\nANSWER:")
    print(result.answer_text)

    print("\nGROUNDING:")
    print(result.grounding_status)

    print("\nCITATIONS:")

    for citation in result.citations:
        print(f"  - {citation}")


# =============================================================================
# DETERMINISTIC RETRIEVAL DEMO
# =============================================================================

def run_deterministic_demo(
    orchestrator: ADKRetrievalOrchestrator,
) -> None:
    """
    Demonstrate the deterministic retrieval path.

    This proves that the retrieval layer works independently
    of Gemini.
    """

    questions = [
        "What is the standard settlement processing target?",

        "What settlement policy should take precedence "
        "when approved and draft policies conflict?",

        "What is the company's 2027 international expansion strategy?",
    ]

    for question in questions:

        print("\n" + "=" * 80)
        print(
            f"QUESTION: {question}"
        )
        print("=" * 80)

        try:

            result = orchestrator.answer(
                question
            )

            print_result(result)

        except Exception as exc:

            print("\nERROR:")
            print(
                f"{type(exc).__name__}: {exc}"
            )


# =============================================================================
# REAL GOOGLE ADK / GEMINI DEMO
# =============================================================================

def run_real_adk_demo(
    orchestrator: ADKRetrievalOrchestrator,
) -> None:
    """
    Invoke the actual Google ADK Runner.

    Runtime flow:

        User
          |
          v
        ADK Runner
          |
          v
        Gemini 2.5 Flash
          |
          v
        retrieve_evidence()
          |
          v
        Deterministic Retrieval
          |
          +----------------------+
          |                      |
          v                      v
        Qdrant                  BM25
          |                      |
          +----------+-----------+
                     |
                     v
                    RRF
                     |
                     v
             Authority Reranker
                     |
                     v
                EvidenceSet
                     |
                     v
                  Gemini
                     |
                     v
               Final Answer
    """

    print("\n" + "=" * 80)
    print("REAL GOOGLE ADK / GEMINI DEMO")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Build ADK agent
    # -------------------------------------------------------------------------

    agent = orchestrator.build_agent()

    print("\nADK AGENT CREATED")
    print("=" * 80)
    print(
        f"Agent name: {agent.name}"
    )
    print(
        "Model: gemini-3.5-flash-lite"
    )

    # -------------------------------------------------------------------------
    # Session service
    # -------------------------------------------------------------------------

    session_service = InMemorySessionService()

    session = session_service.create_session_sync(
        app_name="responsive_agentic_rag",
        user_id="demo_user",
        session_id="demo_session",
    )

    # -------------------------------------------------------------------------
    # Runner
    # -------------------------------------------------------------------------

    runner = Runner(
        app_name="responsive_agentic_rag",
        agent=agent,
        session_service=session_service,
    )

    # -------------------------------------------------------------------------
    # Interview question
    # -------------------------------------------------------------------------

    question = (
        "What is the standard settlement processing target?"
    )

    print("\n" + "-" * 80)
    print(
        f"QUESTION: {question}"
    )
    print("-" * 80)

    # -------------------------------------------------------------------------
    # ADK message
    # -------------------------------------------------------------------------

    message = types.Content(
        role="user",
        parts=[
            types.Part(
                text=question
            )
        ],
    )

    final_text = None

    # -------------------------------------------------------------------------
    # Invoke ADK
    # -------------------------------------------------------------------------

    try:

        for event in runner.run(
            user_id="demo_user",
            session_id=session.id,
            new_message=message,
        ):

            if event.is_final_response():

                if (
                    event.content
                    and event.content.parts
                ):

                    text_parts = []

                    for part in event.content.parts:

                        if part.text:
                            text_parts.append(
                                part.text
                            )

                    if text_parts:

                        final_text = "\n".join(
                            text_parts
                        )

    except Exception as exc:

        print("\nADK ERROR:")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        return

    # -------------------------------------------------------------------------
    # Print final response
    # -------------------------------------------------------------------------

    print("\nANSWER")
    print("=" * 80)

    if final_text:

        print(final_text)

    else:

        print(
            "No final text response was returned "
            "by the ADK Runner."
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("\n" + "=" * 80)
    print("RESPONSIVE AGENTIC RAG — DEMO")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load corpus
    # -------------------------------------------------------------------------

    chunks = load_chunks()

    # -------------------------------------------------------------------------
    # Build retrieval stack
    # -------------------------------------------------------------------------

    orchestrator = build_retrieval_stack(
        chunks
    )

    # -------------------------------------------------------------------------
    # Deterministic retrieval demonstration
    # -------------------------------------------------------------------------

    run_deterministic_demo(
        orchestrator
    )

    # -------------------------------------------------------------------------
    # Real ADK / Gemini demonstration
    # -------------------------------------------------------------------------

    run_real_adk_demo(
        orchestrator
    )


if __name__ == "__main__":
    main()