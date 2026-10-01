import hashlib
from io import BytesIO

from bs4 import BeautifulSoup
from pypdf import PdfReader

from responsive_agentic_rag.models.knowledge import (
    KnowledgeSource,
    NormalizedSourceRecord,
)


class IngestionNormalizer:
    """Normalize source-specific raw content into canonical records."""

    def normalize(
        self,
        source: KnowledgeSource,
        raw_content: bytes,
    ) -> list[NormalizedSourceRecord]:
        """Convert raw source content into normalized records."""

        if source.source_type == "document":
            text = self._extract_pdf_text(raw_content)
        elif source.source_type == "website":
            text = self._extract_html_text(raw_content)
        else:
            raise ValueError(
                f"Unsupported source type: {source.source_type}"
            )

        text = self._clean_text(text)

        if not text:
            return []

        return [
            NormalizedSourceRecord(
                record_id=self._build_record_id(source, text),
                source_id=source.source_id,
                source_type=source.source_type,
                title=source.title,
                body_text=text,
                metadata={
                    **source.metadata,
                    "source_type": source.source_type,
                    "source_location": source.source_location
                    }
            )
        ]

    def _extract_pdf_text(self, raw_content: bytes) -> str:
        reader = PdfReader(BytesIO(raw_content))

        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        return "\n".join(pages)

    def _extract_html_text(self, raw_content: bytes) -> str:
        soup = BeautifulSoup(raw_content, "html.parser")

        for element in soup(
            ["script", "style", "noscript"]
        ):
            element.decompose()

        return soup.get_text(separator="\n")

    def _clean_text(self, text: str) -> str:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines)

    def _build_record_id(
        self,
        source: KnowledgeSource,
        text: str,
    ) -> str:
        payload = (
            f"{source.source_id}|"
            f"{source.source_location}|"
            f"{text}"
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()
