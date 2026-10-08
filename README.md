# Responsive Agentic RAG

A specification-driven, source-agnostic enterprise RAG platform that ingests documents and web content, preserves source provenance, performs deterministic hybrid retrieval, and uses a bounded Google ADK agent to generate grounded answers with citations.

The project is designed around a simple principle:

> **Make retrieval deterministic and inspectable; use agentic reasoning only where it adds value.**

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Goals](#goals)
- [Architecture](#architecture)
- [Core Design Principles](#core-design-principles)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Knowledge Ingestion](#knowledge-ingestion)
- [Retrieval and Answer Generation](#retrieval-and-answer-generation)
- [Hybrid Retrieval](#hybrid-retrieval)
- [Authority-Aware Ranking](#authority-aware-ranking)
- [Grounding and Citations](#grounding-and-citations)
- [Agentic Layer](#agentic-layer)
- [Specification-Driven Development](#specification-driven-development)
- [Testing](#testing)
- [Demo Walkthrough](#demo-walkthrough)
- [Engineering Trade-offs](#engineering-trade-offs)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Production Considerations](#production-considerations)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Conclusion](#conclusion)

---

# Overview

Responsive Agentic RAG is an enterprise-oriented Retrieval-Augmented Generation platform built to demonstrate how deterministic retrieval and bounded agentic reasoning can work together.

The platform currently supports:

- PDF document ingestion
- HTML/web content ingestion
- Common source normalization
- Deterministic chunking
- Local sentence-transformer embeddings
- Persistent Qdrant vector storage
- BM25 lexical retrieval
- Semantic retrieval
- Reciprocal Rank Fusion (RRF)
- Authority-aware ranking
- Google ADK-based answer generation
- Gemini-powered grounded responses
- Source citations
- Runtime document uploads
- Streamlit-based interactive UI
- Retrieval and model execution traces

The architecture intentionally separates the deterministic retrieval system from the generative agent.

```text
                    User
                     |
                     v
              +--------------+
              |  Streamlit   |
              |      UI      |
              +------+-------+
                     |
          +----------+----------+
          |                     |
          v                     v
     Ingestion Flow        Query Flow
          |                     |
          v                     v
   Normalize / Chunk       Hybrid Retrieval
   Embed / Persist         Semantic + BM25
          |                     |
          v                     v
       Qdrant                  RRF
          |                     |
          |                     v
          |              Authority-aware
          |                  Ranking
          |                     |
          |                     v
          |               Evidence Set
          |                     |
          |                     v
          |                 Google ADK
          |                     |
          |                     v
          |                  Gemini
          |                     |
          |                     v
          +------------> Grounded Answer
                              |
                              v
                         Citations