import hashlib
import re

from responsive_agentic_rag.models.knowledge import (
    ChunkRecord,
    NormalizedSourceRecord,
)


MAX_CHUNK_CHARACTERS = 1200
OVERLAP_CHARACTERS = 150


class TextChunker:
    """Create deterministic, paragraph-aware chunks."""

    def __init__(
        self,
        max_chunk_characters: int = MAX_CHUNK_CHARACTERS,
        overlap_characters: int = OVERLAP_CHARACTERS,
    ) -> None:
        if max_chunk_characters <= 0:
            raise ValueError(
                "max_chunk_characters must be positive"
            )

        if overlap_characters < 0:
            raise ValueError(
                "overlap_characters cannot be negative"
            )

        if overlap_characters >= max_chunk_characters:
            raise ValueError(
                "overlap_characters must be smaller than "
                "max_chunk_characters"
            )

        self.max_chunk_characters = max_chunk_characters
        self.overlap_characters = overlap_characters

    def chunk(
        self,
        record: NormalizedSourceRecord,
    ) -> list[ChunkRecord]:
        """Split a normalized record into deterministic chunks."""

        paragraphs = self._split_paragraphs(record.body_text)

        if not paragraphs:
            return []

        chunks: list[ChunkRecord] = []
        current_text = ""
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_parts = self._split_large_text(paragraph)

            for part in paragraph_parts:
                candidate = self._append_text(
                    current_text,
                    part,
                )

                if (
                    current_text
                    and len(candidate) > self.max_chunk_characters
                ):
                    chunks.append(
                        self._build_chunk(
                            record,
                            current_text,
                            chunk_index,
                        )
                    )

                    chunk_index += 1

                    overlap = self._get_overlap(current_text)

                    current_text = self._append_text(
                        overlap,
                        part,
                    )

                    if len(current_text) > self.max_chunk_characters:
                        split_parts = self._split_large_text(
                            current_text
                        )

                        current_text = split_parts[0]

                        for extra_part in split_parts[1:]:
                            chunks.append(
                                self._build_chunk(
                                    record,
                                    current_text,
                                    chunk_index,
                                )
                            )

                            chunk_index += 1
                            current_text = self._get_overlap(
                                current_text
                            )

                            current_text = self._append_text(
                                current_text,
                                extra_part,
                            )
                else:
                    current_text = candidate

        if current_text:
            chunks.append(
                self._build_chunk(
                    record,
                    current_text,
                    chunk_index,
                )
            )

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into non-empty paragraphs."""

        return [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                text,
            )
            if paragraph.strip()
        ]

    def _split_large_text(self, text: str) -> list[str]:
        """
        Split oversized text into bounded word-aware segments.

        Words are kept intact where possible. If an individual word
        exceeds the configured limit, it is split at the character
        boundary as a final fallback.
        """

        text = text.strip()

        if len(text) <= self.max_chunk_characters:
            return [text]

        words = text.split()
        parts: list[str] = []
        current = ""

        for word in words:
            if len(word) > self.max_chunk_characters:
                if current:
                    parts.append(current)
                    current = ""

                for start in range(
                    0,
                    len(word),
                    self.max_chunk_characters,
                ):
                    parts.append(
                        word[
                            start:start
                            + self.max_chunk_characters
                        ]
                    )

                continue

            candidate = (
                word
                if not current
                else f"{current} {word}"
            )

            if len(candidate) <= self.max_chunk_characters:
                current = candidate
            else:
                parts.append(current)
                current = word

        if current:
            parts.append(current)

        return parts

    def _append_text(
        self,
        current_text: str,
        text: str,
    ) -> str:
        """Append text while preserving paragraph separation."""

        if not current_text:
            return text

        return f"{current_text}\n\n{text}"

    def _get_overlap(self, text: str) -> str:
        """Return a word-aware overlap from the end of a chunk."""

        if self.overlap_characters == 0:
            return ""

        overlap = text[-self.overlap_characters:]

        if len(overlap) == len(text):
            return overlap

        first_space = overlap.find(" ")

        if first_space >= 0:
            overlap = overlap[first_space + 1:]

        return overlap.strip()

    def _build_chunk(
        self,
        record: NormalizedSourceRecord,
        content: str,
        chunk_index: int,
    ) -> ChunkRecord:
        """Build a deterministic ChunkRecord."""

        content = content.strip()

        chunk_id = self._build_chunk_id(
            record,
            content,
            chunk_index,
        )

        return ChunkRecord(
            chunk_id=chunk_id,
            source_id=record.source_id,
            record_id=record.record_id,
            content=content,
            chunk_index=chunk_index,
            start_offset=None,
            end_offset=None,
            section_path=record.section_path.copy(),
            token_count=self._estimate_token_count(content),
            embedding=None,
            metadata=dict(record.metadata),
            authority_score=0.0,
        )

    def _build_chunk_id(
        self,
        record: NormalizedSourceRecord,
        content: str,
        chunk_index: int,
    ) -> str:
        """Generate a deterministic identifier for a chunk."""

        payload = (
            f"{record.record_id}|"
            f"{chunk_index}|"
            f"{content}"
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for metadata.

        This is intentionally a lightweight estimate rather than a
        model-specific tokenizer. Exact tokenizer usage belongs to
        the embedding implementation.
        """

        if not text.strip():
            return 0

        return len(text.split())
