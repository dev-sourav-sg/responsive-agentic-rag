from pathlib import Path

from responsive_agentic_rag.connectors.document_connector import DocumentConnector
from responsive_agentic_rag.ingestion.normalizer import IngestionNormalizer
from responsive_agentic_rag.models.knowledge import KnowledgeSource


def make_document_source() -> KnowledgeSource:
    return KnowledgeSource(
        source_id="doc-001",
        source_type="document",
        title="Payment Policy",
        source_location="data/documents/payment_policy.pdf",
        authority=0.95,
        metadata={
            "file_name": "payment_policy.pdf",
            "owner": "Payments Operations",
        },
    )


def make_website_source() -> KnowledgeSource:
    return KnowledgeSource(
        source_id="web-001",
        source_type="website",
        title="Settlement Policy",
        source_location="https://example.com/settlement",
        authority=0.90,
        metadata={
            "url": "https://example.com/settlement",
            "owner": "Settlement Operations",
        },
    )


def test_normalizer_extracts_pdf_text():
    connector = DocumentConnector(Path("data/documents"))
    source_config = connector.list_sources()[0]

    raw_content = connector.fetch_content(source_config)

    source = KnowledgeSource(
        source_id="doc-001",
        source_type="document",
        title="Payment Policy",
        source_location=source_config.source_location,
        metadata=source_config.metadata,
    )

    normalizer = IngestionNormalizer()

    records = normalizer.normalize(source, raw_content)

    assert len(records) == 1
    assert records[0].source_id == "doc-001"
    assert records[0].source_type == "document"
    assert records[0].body_text
    assert records[0].record_id


def test_normalizer_extracts_html_text():
    normalizer = IngestionNormalizer()
    source = make_website_source()

    raw_content = b"""
        <html>
            <head>
                <title>Settlement</title>
                <script>alert("ignore me")</script>
            </head>
            <body>
                <h1>Settlement Policy</h1>
                <p>Settlement requires successful reconciliation.</p>
                <style>.hidden { display: none; }</style>
            </body>
        </html>
    """

    records = normalizer.normalize(source, raw_content)

    assert len(records) == 1
    assert "Settlement Policy" in records[0].body_text
    assert "Settlement requires successful reconciliation." in records[0].body_text
    assert "alert" not in records[0].body_text
    assert "display: none" not in records[0].body_text


def test_normalizer_preserves_source_metadata():
    normalizer = IngestionNormalizer()
    source = make_website_source()

    raw_content = b"<html><body>Test content</body></html>"

    records = normalizer.normalize(source, raw_content)

    assert records[0].metadata["url"] == (
        "https://example.com/settlement"
    )
    assert records[0].metadata["owner"] == "Settlement Operations"


def test_normalizer_generates_deterministic_record_id():
    normalizer = IngestionNormalizer()
    source = make_website_source()

    raw_content = b"""
        <html>
            <body>
                Same content every time.
            </body>
        </html>
    """

    first = normalizer.normalize(source, raw_content)
    second = normalizer.normalize(source, raw_content)

    assert first[0].record_id == second[0].record_id


def test_normalizer_returns_empty_for_empty_content():
    normalizer = IngestionNormalizer()
    source = make_website_source()

    raw_content = b"<html><body></body></html>"

    records = normalizer.normalize(source, raw_content)

    assert records == []