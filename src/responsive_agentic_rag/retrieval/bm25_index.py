import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from responsive_agentic_rag.models.knowledge import ChunkRecord


class BM25Index:
    """Deterministic BM25 lexical index over chunk content."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._vocabulary: set[str] = set()
        self._tokenized_documents: list[list[str]] = []

    def build(self, chunks: Sequence[ChunkRecord]) -> None:
        """Build or rebuild the BM25 index from chunk records."""

        self._bm25 = None
        self._chunk_ids = []
        self._vocabulary = set()
        self._tokenized_documents = []

        if not chunks:
            return

        for chunk in chunks:
            tokens = self._tokenize(chunk.content)

            self._chunk_ids.append(chunk.chunk_id)
            self._tokenized_documents.append(tokens)
            self._vocabulary.update(tokens)

        self._bm25 = BM25Okapi(
            self._tokenized_documents
        )

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Return chunk IDs and BM25 scores for the query."""

        if not query.strip():
            raise ValueError("query cannot be empty")

        if limit < 1:
            raise ValueError("limit must be greater than 0")

        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        if not any(
            token in self._vocabulary
            for token in query_tokens
        ):
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_results = sorted(
            zip(
                self._chunk_ids,
                scores,
                self._tokenized_documents,
            ),
            key=lambda item: (-item[1], item[0]),
        )

        positive_results = [
            (chunk_id, float(score))
            for chunk_id, score, _ in ranked_results
            if score > 0.0
        ]

        if positive_results:
            return positive_results[:limit]

        # BM25 can assign a zero score when the query term has
        # zero IDF, particularly with very small corpora.
        # Preserve exact lexical matches in that case.
        matching_results = [
            (chunk_id, float(score))
            for chunk_id, score, document_tokens in ranked_results
            if any(
                token in document_tokens
                for token in query_tokens
            )
        ]

        return matching_results[:limit]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into deterministic BM25 tokens."""

        return re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )