from responsive_agentic_rag.agents.tools import (
    RetrievalTool,
    RetrievalToolContract,
)
from responsive_agentic_rag.models.retrieval import (
    EvidenceSet,
    RetrievalCandidate,
    RetrievalQuery,
)
from responsive_agentic_rag.retrieval.evidence_sufficiency import (
    EvidenceSufficiency,
)


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.received_text: str | None = None

    def embed(self, text: str) -> list[float]:
        self.received_text = text
        return [0.1, 0.2, 0.3]


class FakeRetriever:
    def __init__(self) -> None:
        self.received_query: str | None = None
        self.received_query_vector: list[float] | None = None
        self.received_limit: int | None = None

    def search(
        self,
        query: str,
        query_vector: list[float],
        limit: int,
    ) -> list[RetrievalCandidate]:
        self.received_query = query
        self.received_query_vector = query_vector
        self.received_limit = limit

        return [
            RetrievalCandidate(
                chunk_id="chunk-1",
                source_id="source-1",
                source_type="document",
                source_location="test.pdf",
                content="Test evidence",
                semantic_score=0.9,
                lexical_score=0.8,
                combined_score=0.85,
                authority_score=0.9,
            ),
        ]


def test_retrieval_tool_embeds_query_and_calls_retriever():
    embedding_service = FakeEmbeddingService()
    retriever = FakeRetriever()

    tool = RetrievalTool(
        retriever=retriever,
        embedding_service=embedding_service,
        evidence_sufficiency=EvidenceSufficiency(),
    )

    query = RetrievalQuery(
        query_text="What is the settlement policy?",
        max_results=5,
    )

    tool.retrieve(query)

    assert embedding_service.received_text == query.query_text
    assert retriever.received_query == query.query_text
    assert retriever.received_query_vector == [0.1, 0.2, 0.3]
    assert retriever.received_limit == 5


def test_retrieval_tool_returns_evidence_set():
    tool = RetrievalTool(
        retriever=FakeRetriever(),
        embedding_service=FakeEmbeddingService(),
        evidence_sufficiency=EvidenceSufficiency(),
    )

    query = RetrievalQuery(
        query_text="What is the settlement policy?",
    )

    result = tool.retrieve(query)

    assert isinstance(result, EvidenceSet)
    assert result.query_id == query.query_text
    assert len(result.candidates) == 1
    assert result.candidates[0].chunk_id == "chunk-1"


def test_retrieval_tool_collects_unique_supporting_sources():
    embedding_service = FakeEmbeddingService()
    retriever = FakeRetriever()

    retriever.search = lambda query, query_vector, limit: [
        RetrievalCandidate(
            chunk_id="chunk-1",
            source_id="source-1",
            source_type="document",
            source_location="test-1.pdf",
            content="Evidence one",
            combined_score=0.8,
        ),
        RetrievalCandidate(
            chunk_id="chunk-2",
            source_id="source-1",
            source_type="document",
            source_location="test-1.pdf",
            content="Evidence two",
            combined_score=0.7,
        ),
        RetrievalCandidate(
            chunk_id="chunk-3",
            source_id="source-2",
            source_type="website",
            source_location="https://example.com",
            content="Evidence three",
            combined_score=0.6,
        ),
    ]

    tool = RetrievalTool(
        retriever=retriever,
        embedding_service=embedding_service,
        evidence_sufficiency=EvidenceSufficiency(),
    )

    query = RetrievalQuery(
        query_text="Find supporting information",
    )

    result = tool.retrieve(query)

    assert result.supporting_sources == ["source-1", "source-2"]


def test_retrieval_tool_implements_contract():
    tool = RetrievalTool(
        retriever=FakeRetriever(),
        embedding_service=FakeEmbeddingService(),
        evidence_sufficiency=EvidenceSufficiency(),
    )

    assert isinstance(tool, RetrievalToolContract)


def test_retrieval_tool_populates_sufficiency_assessment():
    tool = RetrievalTool(
        retriever=FakeRetriever(),
        embedding_service=FakeEmbeddingService(),
        evidence_sufficiency=EvidenceSufficiency(
            minimum_candidates=1,
            minimum_score=0.5,
        ),
    )

    query = RetrievalQuery(
        query_text="What is the settlement policy?",
    )

    result = tool.retrieve(query)

    assert result.sufficiency_assessment is True