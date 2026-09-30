# Quickstart: Local Validation

## Prerequisites

- Python 3.12+
- `uv` installed
- Docker available for local vector store and optional LocalStack
- Access to a curated set of document files and website URLs

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Create a local environment file from the project template if present:
   ```bash
   cp .env.example .env
   ```

3. Start the supporting local infrastructure, if required by the environment configuration:
   ```bash
   docker compose up -d
   ```

4. Confirm the vector store is reachable and the application settings are loaded.

## Ingestion validation

Run the ingestion pipeline against a small demo corpus:

```bash
python -m responsive_agentic_rag.cli ingest --documents ./data/documents --websites https://example.com/docs
```

Expected outcome:
- Documents and website content are normalized.
- Chunk records are created.
- Embeddings are generated and stored.
- Metadata is available for retrieval and citation.

## Retrieval validation

Run a retrieval check against the corpus:

```bash
python -m responsive_agentic_rag.cli query --question "What policy governs the approval workflow?"
```

Expected outcome:
- Ranked evidence is returned.
- Source IDs, URLs, and metadata are visible.
- Retrieval scores and ranking metadata are available.

## Answer validation

Query the bounded ADK-enabled workflow:

```bash
python -m responsive_agentic_rag.cli answer --question "What policy governs the approval workflow?"
```

Expected outcome:
- The system grounds its answer in retrieved evidence.
- The response includes citations.
- If evidence is insufficient, the system states the information is unavailable instead of inventing content.

## Evaluation validation

Run the evaluation dataset:

```bash
pytest tests/eval -q
```

Expected outcome:
- Retrieval metrics and answer-quality metrics are generated.
- Results can be compared across retrieval or ranking changes.
