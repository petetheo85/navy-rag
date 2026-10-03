# Navy RAG: Recruiting Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/vectorstore-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A retrieval-augmented generation (RAG) system for U.S. Navy recruiting questions, backed by published Navy instructions and recruiting regulations. Designed to assist prospective applicants and recruiters with inquiries regarding enlistment standards, moral and medical waivers, tattoo policies, and rating requirements through verified policy citations.

> ⚠️ **Project Status & Disclaimer:** This project is an active engineering prototype under development. The core orchestration, retrieval engine, and user interfaces are implemented, while document corpus expansion and integration testing across external tool protocols (MCP) remain ongoing. This is an unofficial demonstration project and is not affiliated with or endorsed by the U.S. Navy or the Department of Defense.

---

## Architecture Overview

```
User / MCP Client
       │
       ▼
FastAPI (/docs)  ←──→  Streamlit Web UI
       │
       ▼
LangGraph Agentic State Machine
  ├── Scope Classification (in-scope vs. out-of-scope)
  ├── Dense & MMR Retrieval (ChromaDB)
  ├── Retrieval Evaluation & Query Reformulation
  ├── Grounded Synthesis (Gemini / OpenAI / Anthropic)
  └── Citation Verification & Recruiter Referral Fallback
```

### Key Engineering Decisions

| Component | Technology | Rationale |
|---|---|---|
| **Orchestration** | LangGraph + LCEL | Stateful routing, conditional retries, and strict verification loops rather than brittle linear chains. |
| **Model Portability** | Multi-Provider via `init_chat_model` | Decoupled provider abstraction enabling hot-swapping between Gemini, OpenAI, and Anthropic via configuration. |
| **Vector Storage** | ChromaDB (Local / Embedded) | Zero-infrastructure local vector storage with metadata filtering by audience and instruction source. |
| **Batch Processing** | PySpark *(Optional)* | Parallelized text extraction and chunking for handling large multi-hundred-page military instruction PDFs. |
| **Agent Protocol** | FastMCP (Model Context Protocol) | Exposes internal knowledge retrieval and document metadata inspection as standardized tools for Claude Desktop and Cursor. |
| **Frontend** | Streamlit + Custom CSS | Custom CNRC-compliant design system providing a branded interface with verified citation inspection. |

---

## Guardrails & Agentic Workflow

Unlike naive RAG pipelines that blindly retrieve and answer, this system enforces policy guardrails via a deterministic LangGraph workflow:

1. **Scope Classification:** Evaluates incoming questions before retrieval. Out-of-scope inquiries (e.g., general trivia, unrelated defense topics) are gracefully deflected without executing vector search.
2. **Evaluation & Reformulation:** If initial retrieval returns insufficient context, the query is rewritten into formal military terminology for a second retrieval attempt.
3. **Citation Anchoring:** The LLM is constrained to cite specific document page numbers. Responses that fail citation verification or lack explicit policy groundings are routed to a standardized recruiter referral fallback.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/petetheo85/navy-rag.git
cd navy-rag

# Create and activate virtual environment
python -m venv venv && source venv/bin/activate

# Install package and development dependencies
pip install -e ".[dev]"
```

### 2. Environment Configuration

Copy the sample environment file and configure your API keys:

```bash
cp .env.example .env
```

At minimum, provide an API key for your chosen provider (defaults to Google Gemini):
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Ingestion

Instruction PDFs are stored by audience classification in `data/public/` (applicant-facing) and `data/internal/` (recruiter-specific).

Run the ingestion pipeline to parse, chunk, and embed documents into ChromaDB:

```bash
# Standard local ingestion
navy ingest

# Batch processing via PySpark for larger corpora
navy ingest --spark --parallelism 4

# Force re-indexing
navy ingest --force
```

---

## Running the Application

### Streamlit Web Dashboard
Launches the interactive recruiting intelligence interface with source citation badges and system status indicators:
```bash
streamlit run app.py
```

### FastAPI Backend
Provides REST endpoints and token streaming with interactive OpenAPI documentation available at `http://localhost:8000/docs`:
```bash
uvicorn navy_rag.api.main:app --reload --port 8000
```
- `POST /api/chat`: Process inquiries with optional streaming (`stream=true`).
- `GET /api/health`: Health check and active collection metadata.
- `POST /api/ingest`: Trigger on-demand document indexing.

### MCP Server (Claude Desktop / Cursor)
Exposes the RAG system as a standardized Model Context Protocol tool provider:
```bash
python -m navy_rag.mcp.server
```

**Claude Desktop Configuration (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "navy-rag": {
      "command": "python",
      "args": ["-m", "navy_rag.mcp.server"],
      "cwd": "/path/to/navy-rag",
      "env": {
        "GOOGLE_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

**Available MCP Tools:**
- `search_navy_instructions`: Policy inquiry tool returning grounded answers with page-level citations.
- `list_available_topics`: Catalog of covered policy areas (waivers, ASVAB, ratings, pay).
- `get_suggested_inquiries`: Common sample inquiries matching the recruiting dashboard.
- `get_document_summary`: Metadata lookup and text preview for specific instruction documents.

---

## LLM Provider Switching

Switch model providers cleanly in `.env` without altering application code:

```bash
# Google Gemini (Default)
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-3.6-flash

# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-latest
```

---

## Testing & Quality Assurance

Unit and regression tests mock external API calls to enable fast, offline execution:

```bash
# Run test suite
pytest tests/ -v

# Run with test coverage report
pytest tests/ --cov=src/navy_rag --cov-report=term-missing
```

Code formatting and linting are configured with Ruff:
```bash
ruff check .
```

---

## Project Structure

```
navy-rag/
├── src/navy_rag/
│   ├── config.py               # Centralized Pydantic Settings
│   ├── llm/provider.py         # Multi-provider LLM factory (Gemini, OpenAI, Anthropic)
│   ├── ingestion/
│   │   ├── loader.py           # Document loading and metadata normalization
│   │   ├── chunker.py          # Character and token-aware text splitting
│   │   ├── embedder.py         # ChromaDB vector store client & embedding manager
│   │   └── spark_processor.py  # PySpark parallelized batch ingestion
│   ├── retrieval/retriever.py  # Dense and MMR retrieval with audience filtering
│   ├── chain/
│   │   ├── prompts.py          # Grounded and classification prompt templates
│   │   ├── rag_chain.py        # LCEL retrieval chain & streaming handlers
│   │   └── graph.py            # LangGraph state machine & guardrail routing
│   ├── cli/
│   │   ├── main.py             # Unified CLI entry point (`navy`)
│   │   └── ingest.py           # Document ingestion subcommand (`navy ingest`)
│   ├── api/
│   │   ├── main.py             # FastAPI application and route handlers
│   │   └── models.py           # Pydantic request and response schemas
│   └── mcp/server.py           # FastMCP server for IDE and desktop agent integration
├── assets/                     # America's Navy design system & static assets
│   ├── style.css               # CNRC brand CSS styling and layout rules
│   └── *.png                   # Brand graphics and iconography
├── data/
│   └── public/                 # Instructions and policy documents (PDF)
├── app.py                      # Streamlit interactive application
└── tests/                      # Pytest suite
```
