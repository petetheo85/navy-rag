# Navy RAG

[![Live Demo](https://img.shields.io/badge/Demo-Google%20Cloud%20Run-4285F4?logo=googlecloud&logoColor=white)](https://navy-rag-676509381448.us-central1.run.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Retrieval-augmented generation system for U.S. Navy recruiting policies, built with Google Gemini, LangGraph, and ChromaDB. Indexes 18 core naval and defense directives (COMNAVCRUITCOMINST, MANMED, DoDI, OPNAVINST, SECNAV, BUPERSINST), 49 MILPERSMAN articles, and 7 NAVPERS 15665J uniform regulation chapters (12,424 chunks) to return grounded answers with page-level citations.

> **Disclaimer:** Unofficial demonstration system. Not affiliated with, endorsed by, or representing the U.S. Navy or the Department of Defense. Verify official policy through authorized recruiting and command channels.

## Setup

```bash
git clone https://github.com/petetheo85/navy-rag.git
cd navy-rag

python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Configure environment variables:

```bash
cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`:

```bash
GOOGLE_API_KEY=your_gemini_api_key
```

## Usage

### Web UI & API

```bash
# Streamlit web interface
streamlit run app.py

# Headless FastAPI server
uvicorn navy_rag.api.main:app --host 127.0.0.1 --port 8000
```

### CLI Commands

```bash
# Differential policy sync (respects weekly/monthly/quarterly cadences)
navy-rag sync --respect-cadence

# Probe all upstream directives and crawlers immediately
navy-rag sync --all

# Dry-run upstream check without modifying files or vector store
navy-rag sync --dry-run

# Run the 50-query retrieval benchmark
navy-rag eval

# Run unit tests and linter
pytest tests/
ruff check src/ tests/ app.py
```

## Architecture

```
User Inquiry
     │
     ▼
Streamlit Web UI / FastAPI
     │
     ▼
LangGraph State Machine
  ├── Scope Evaluation (recruiting policy vs. out-of-scope deflection)
  ├── Lexicon Expansion (military acronyms & conversational context)
  ├── Hybrid Retrieval (Dense + BM25 RRF with directive diversity)
  ├── Lexical Grounding Check (relevance verification & referral routing)
  └── Grounded Generation (deterministic [DOC-N] page-level citations)
```

### Hybrid Retrieval & Corpus Skew

Two occupational manuals (`NAVPERS 18068F` Volumes I and II) make up 50% of the vector index (6,206 of 12,424 chunks). To prevent high-volume manuals from crowding out smaller directives during dense similarity search:

- **Weighted Reciprocal Rank Fusion (RRF):** Combines dense vector search (`0.60` weight) with metadata-enriched BM25 keyword search (`0.40` weight).
- **Directive Diversity Cap:** Restricts the top-3 results to distinct directives and caps total chunks per directive at 3.
- **Lexicon Expansion:** Expands Navy acronyms and designators (e.g., `RE-3`, `MEPS`, `PRT`, `CWT`, `BUMED`) into full directive terminology prior to retrieval.

Evaluated against `data/evals/golden_dataset.yaml` (50 verified queries across enlistment standards, medical waivers, tattoos, PRT, ASVAB, and security clearances):

| Metric | Dense MMR | Hybrid RRF |
|---|---|---|
| **Recall@3** | 86.0% | **100.0%** |
| **Recall@5** | 92.0% | **100.0%** |
| **MRR** | 0.860 | **0.933** |

### Differential Policy Synchronization

`navy-rag sync` keeps the corpus aligned with upstream DoD and MyNavyHR publications without re-embedding unchanged documents:

- **Tiered Cadences:** Directives are scheduled by update frequency (`weekly`, `monthly`, `quarterly`).
- **Conditional Probing:** Sends HTTP `HEAD` requests (`ETag`, `Last-Modified`, `Content-Length`) and uses `curl_cffi` TLS fingerprint impersonation for Akamai-protected `.navy.mil` endpoints. Falls back to existing local PDFs if an upstream portal is unavailable.
- **Hash-Aware Re-Indexing:** Computes `SHA-256` digests on downloaded files and purges/re-embeds only the modified directive's chunks in ChromaDB.
- **State Persistence:** Scheduled GitHub Actions workflows (`.github/workflows/policy-sync.yml`) hydrate `data/raw/` and `chroma_db/` from Google Cloud Storage, execute differential sync and regression tests, and trigger Cloud Build deployments only when policy changes occur.

## Project Structure

```
navy-rag/
├── .github/workflows/
│   └── policy-sync.yml         # Scheduled GitHub Actions sync & deployment pipeline
├── src/navy_rag/
│   ├── config.py               # Environment & settings configuration
│   ├── llm/provider.py         # LLM client initialization
│   ├── ingestion/
│   │   ├── downloader.py       # Bronze-layer downloader & HTTP conditional probing
│   │   ├── loader.py           # PDF loader, metadata normalization & SHA-256 hashing
│   │   ├── chunker.py          # Token-aware recursive text chunking
│   │   ├── embedder.py         # ChromaDB client & differential chunk updates
│   │   ├── sync.py             # Differential policy sync orchestrator
│   │   └── crawlers/           # MILPERSMAN and NAVPERS 15665J crawlers
│   ├── retrieval/
│   │   ├── retriever.py        # Retriever factory with audience filtering
│   │   ├── hybrid.py           # Weighted RRF & directive diversity enforcement
│   │   └── query_processor.py  # Military acronym expander & query contextualizer
│   ├── chain/
│   │   ├── prompts.py          # System and classification prompts
│   │   ├── rag_chain.py        # LCEL synthesis chain & [DOC-N] citation parser
│   │   └── graph.py            # LangGraph state machine & guardrails
│   ├── evals/                  # Benchmark evaluation harness (Recall@K, MRR)
│   ├── cli/                    # CLI subcommand adapters
│   └── api/                    # FastAPI route handlers
├── assets/                     # UI assets and stylesheet
├── data/
│   ├── manifests/              # Declarative corpus manifest (YAML)
│   ├── evals/                  # 50-query golden benchmark dataset
│   └── raw/                    # Source PDFs and crawled sub-corpora
├── app.py                      # Streamlit web interface
└── tests/                      # Pytest suite
```
