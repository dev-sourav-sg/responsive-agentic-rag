from responsive_agentic_rag.ingestion.chunker import (
    TextChunker,
)
from responsive_agentic_rag.models.knowledge import (
    NormalizedSourceRecord,
)


def make_record(
    body_text: str,
) -> NormalizedSourceRecord:
    return NormalizedSourceRecord(
        record_id="record-001",
        source_id="source-001",
        source_type="document",
        title="Test Document",
        body_text=body_text,
        section_path=["Policy"],
        metadata={
            "owner": "Test Owner",
        },
    )


def test_chunker_creates_chunk():
    chunker = TextChunker(
        max_chunk_characters=100,
        overlap_characters=20,
    )

    record = make_record(
        "This is a short paragraph containing useful "
        "information about payment processing."
    )

    chunks = chunker.chunk(record)

    assert len(chunks) == 1
    assert chunks[0].content == record.body_text
    assert chunks[0].source_id == "source-001"
    assert chunks[0].record_id == "record-001"
    assert chunks[0].chunk_index == 0


def test_chunker_splits_multiple_paragraphs():
    chunker = TextChunker(
        max_chunk_characters=80,
        overlap_characters=10,
    )

    record = make_record(
        "First paragraph contains payment information.\n\n"
        "Second paragraph contains settlement information.\n\n"
        "Third paragraph contains reconciliation information."
    )

    chunks = chunker.chunk(record)

    assert len(chunks) >= 2
    assert all(chunk.content for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(
        range(len(chunks))
    )


def test_chunker_preserves_metadata_and_section_path():
    chunker = TextChunker()

    record = make_record(
        "Payment processing information."
    )

    chunks = chunker.chunk(record)

    assert chunks[0].metadata["owner"] == "Test Owner"
    assert chunks[0].section_path == ["Policy"]


def test_chunker_generates_deterministic_ids():
    chunker = TextChunker(
        max_chunk_characters=100,
        overlap_characters=20,
    )

    record = make_record(
        "Deterministic chunk identifiers are important."
    )

    first = chunker.chunk(record)
    second = chunker.chunk(record)

    assert first[0].chunk_id == second[0].chunk_id


def test_chunker_records_token_count():
    chunker = TextChunker()

    record = make_record(
        "Payment settlement requires reconciliation."
    )

    chunks = chunker.chunk(record)

    assert chunks[0].token_count == 4


def test_chunker_handles_empty_record():
    chunker = TextChunker()

    record = make_record("")

    chunks = chunker.chunk(record)

    assert chunks == []


def test_chunker_rejects_invalid_configuration():
    try:
        TextChunker(max_chunk_characters=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        TextChunker(
            max_chunk_characters=100,
            overlap_characters=100,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_chunker_splits_oversized_paragraph():
    chunker = TextChunker(
        max_chunk_characters=50,
        overlap_characters=10,
    )

    record = make_record(
        "This is a deliberately long paragraph containing "
        "many words so that the chunker must split it into "
        "multiple bounded pieces without losing the content."
    )

    chunks = chunker.chunk(record)

    assert len(chunks) > 1
    assert all(
        len(chunk.content) <= 50
        for chunk in chunks
    )
