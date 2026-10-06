from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from responsive_agentic_rag.ingestion.chunker import TextChunker
from responsive_agentic_rag.ingestion.embedding_service import EmbeddingService
from responsive_agentic_rag.ingestion.normalizer import IngestionNormalizer
from responsive_agentic_rag.retrieval.qdrant_store import QdrantVectorStore

from responsive_agentic_rag.ingestion.runtime_ingestion import RuntimeIngestionService
from runtime_rag import RuntimeRetrievalService
from adk_gemini_service import GeminiADKAnswerService


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION_NAME = "demo_knowledge_chunks"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Responsive Agentic RAG",
    page_icon="🔎",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached application services
# ---------------------------------------------------------------------------

@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )


@st.cache_resource
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        model_name=EMBEDDING_MODEL,
    )


@st.cache_resource
def get_vector_store() -> QdrantVectorStore:
    client = get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        vector_dimension=EMBEDDING_DIMENSION,
    )

    vector_store.ensure_collection()

    return vector_store


@st.cache_resource
def get_ingestion_service() -> RuntimeIngestionService:
    return RuntimeIngestionService(
        normalizer=IngestionNormalizer(),
        chunker=TextChunker(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@st.cache_resource
def get_retrieval_service() -> RuntimeRetrievalService:
    return RuntimeRetrievalService(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@st.cache_resource
def get_answer_service() -> GeminiADKAnswerService:
    return GeminiADKAnswerService()


# ---------------------------------------------------------------------------
# Application services
# ---------------------------------------------------------------------------

ingestion_service = get_ingestion_service()
retrieval_service = get_retrieval_service()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔎 Responsive Agentic RAG")

st.caption(
    "Multi-source enterprise knowledge retrieval with deterministic "
    "hybrid search and bounded Google ADK answer generation."
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

ingest_tab, ask_tab = st.tabs(
    [
        "📥 Knowledge Ingestion",
        "💬 Ask Knowledge",
    ]
)


# ===========================================================================
# TAB 1 — INGESTION
# ===========================================================================

with ingest_tab:

    st.subheader("Upload Knowledge")

    st.write(
        "Upload a PDF or HTML document. The file will go through the "
        "existing normalization, chunking, embedding and Qdrant pipeline."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or HTML",
        type=["pdf", "html", "htm"],
    )

    col1, col2 = st.columns(2)

    with col1:

        document_id = st.text_input(
            "Document ID",
            placeholder="e.g. Live-Settlement-003",
        )

        version = st.text_input(
            "Version",
            placeholder="e.g. v1.0",
        )

        owner = st.text_input(
            "Owner",
            placeholder="e.g. Payments Operations",
        )

    with col2:

        approval_status = st.selectbox(
            "Approval Status",
            [
                "approved",
                "pending",
                "review",
                "draft",
                "deprecated",
            ],
            index=0,
        )

        authority = st.slider(
            "Authority",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1,
        )

    title = st.text_input(
        "Title",
        placeholder="Optional title",
    )

    ingest_clicked = st.button(
        "🚀 Ingest Document",
        type="primary",
        disabled=uploaded_file is None,
    )

    if ingest_clicked and uploaded_file is not None:

        start_time = time.perf_counter()

        try:

            content = uploaded_file.getvalue()

            with st.spinner(
                "Processing → Normalizing → Chunking → Embedding → Qdrant..."
            ):

                chunks = ingestion_service.ingest_uploaded_file(
                    file_name=uploaded_file.name,
                    content=content,
                    title=title,
                    document_id=document_id,
                    version=version,
                    owner=owner,
                    approval_status=approval_status,
                    authority=authority,
                )

                # BM25 is in-memory, so refresh it after ingestion.
                total_chunks = retrieval_service.add_uploaded_chunks(
                    chunks
                )

            elapsed = time.perf_counter() - start_time

            st.success(
                f"Successfully ingested **{uploaded_file.name}** "
                f"({len(chunks)} chunks)."
            )

            st.info(
                f"BM25 index refreshed. Total indexed chunks: "
                f"**{total_chunks}**. "
                f"Processing time: **{elapsed:.2f}s**."
            )

        except Exception as exc:

            st.error(
                f"Ingestion failed: {exc}"
            )


# ===========================================================================
# TAB 2 — ASK
# ===========================================================================

with ask_tab:

    st.subheader("Ask the Knowledge Base")

    question = st.text_area(
        "Question",
        placeholder=(
            "Example: What is the standard settlement processing target?"
        ),
        height=100,
    )

    ask_clicked = st.button(
        "🔍 Ask",
        type="primary",
        disabled=not question.strip(),
    )

    if ask_clicked:

        # ---------------------------------------------------------------
        # STEP 1 — Deterministic retrieval
        # ---------------------------------------------------------------

        retrieval_start = time.perf_counter()

        try:

            with st.spinner(
                "Searching semantic + BM25 + RRF..."
            ):

                trace = retrieval_service.retrieve(
                    question=question,
                    semantic_limit=10,
                    lexical_limit=10,
                    final_limit=10,
                )

            retrieval_latency = (
                time.perf_counter()
                - retrieval_start
            )

        except Exception as exc:

            st.error(
                f"Retrieval failed: {exc}"
            )
            st.stop()

        # ---------------------------------------------------------------
        # STEP 2 — LLM answer generation
        # ---------------------------------------------------------------

        answer_service = None

        try:

            answer_service = get_answer_service()

            with st.spinner(
                "Generating grounded answer..."
            ):

                answer_result = answer_service.answer(
                    question=question,
                    retrieval_trace=trace,
                )

        except Exception as exc:

            st.error(
                f"Agent execution failed: {exc}"
            )
            st.stop()

        # ---------------------------------------------------------------
        # ANSWER
        # ---------------------------------------------------------------

        st.markdown("## Answer")

        if answer_result.fallback_used:

            st.warning(
                f"Primary model unavailable/slow. "
                f"Answer generated using fallback model "
                f"**{answer_result.model}**."
            )

        else:

            st.success(
                f"Answer generated using **{answer_result.model}**."
            )

        st.markdown(
            answer_result.answer
        )

        # ---------------------------------------------------------------
        # Citations
        # ---------------------------------------------------------------

        st.markdown("## 📚 Evidence / Citations")

        final_candidates = trace["final"]

        if not final_candidates:

            st.info(
                "No supporting evidence was retrieved."
            )

        else:

            for index, candidate in enumerate(
                final_candidates[:6],
                start=1,
            ):

                with st.expander(
                    f"[{index}] {candidate.source_location}"
                ):

                    st.write(
                        f"**Source type:** "
                        f"{candidate.source_type}"
                    )

                    st.write(
                        f"**Source ID:** "
                        f"{candidate.source_id}"
                    )

                    st.write(
                        f"**Authority:** "
                        f"{candidate.authority_score:.2f}"
                    )

                    st.write(
                        f"**Combined score:** "
                        f"{candidate.combined_score:.4f}"
                    )

                    st.write(
                        candidate.content
                    )

        # ---------------------------------------------------------------
        # Retrieval details
        # ---------------------------------------------------------------

        with st.expander(
            "🔬 Retrieval Details"
        ):

            st.write(
                f"**Retrieval latency:** "
                f"{retrieval_latency:.2f}s"
            )

            st.write(
                f"**Final candidates:** "
                f"{len(trace['final'])}"
            )

            st.markdown(
                "### Top Semantic Results"
            )

            for index, candidate in enumerate(
                trace["semantic"][:5],
                start=1,
            ):

                st.write(
                    f"{index}. "
                    f"`{candidate.source_id}` — "
                    f"{candidate.content[:300]}..."
                )

            st.markdown(
                "### Top BM25 Results"
            )

            for index, candidate in enumerate(
                trace["lexical"][:5],
                start=1,
            ):

                st.write(
                    f"{index}. "
                    f"`{candidate.source_id}` — "
                    f"{candidate.content[:300]}..."
                )

            st.markdown(
                "### Final RRF / Ranked Results"
            )

            for index, candidate in enumerate(
                trace["final"][:10],
                start=1,
            ):

                st.write(
                    f"{index}. "
                    f"`{candidate.source_id}` — "
                    f"score={candidate.combined_score:.4f}"
                )

        # ---------------------------------------------------------------
        # Agent / Model Trace
        # ---------------------------------------------------------------

        with st.expander(
            "🤖 Agent / Model Trace"
        ):

            st.write(
                f"**Model:** "
                f"{answer_result.model}"
            )

            st.write(
                f"**Fallback used:** "
                f"{answer_result.fallback_used}"
            )

            st.write(
                f"**ADK events:** "
                f"{answer_result.events}"
            )

            st.write(
                f"**LLM latency:** "
                f"{answer_result.latency_seconds:.2f}s"
            )

            st.write(
                f"**Total retrieval + LLM latency:** "
                f"{retrieval_latency + answer_result.latency_seconds:.2f}s"
            )

            st.write(
                "Agent boundary: deterministic retrieval → "
                "retrieved evidence → ADK/Gemini → grounded answer."
            )