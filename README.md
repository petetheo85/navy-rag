# Navy RAG: Recruiting Intelligence

[![Live Demo](https://img.shields.io/badge/Demo-Google%20Cloud%20Run-4285F4?logo=googlecloud&logoColor=white)](https://navy-rag-676509381448.us-central1.run.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A retrieval-augmented generation system for U.S. Navy recruiting intelligence, built with Google Gemini, LangGraph, and ChromaDB. Grounded in authoritative naval directives (COMNAVCRUITCOMINST, MILPERSMAN, MANMED, DoDI, and NAVPERS uniform regulations) to provide answers with page-level citations for recruiters and applicants.

> **Disclaimer:** Unofficial demonstration system. Not affiliated with, endorsed by, or representing the U.S. Navy or the Department of Defense. Official policy inquiries must be verified through authorized recruiting and command channels.

---

## Technical Architecture

```
User Inquiry
     │
     ▼
Streamlit Web UI / FastAPI
     │
     ▼
LangGraph Agent State Machine
  ├── Scope Evaluation (recruiting policy vs. out-of-scope deflection)
  ├── Lexicon Expansion (military acronyms & conversational context)
  ├── Hybrid Dense + BM25 Retrieval (Weighted RRF with directive diversity)
  ├── Lexical Grounding Check (relevance verification & referral routing)
  └── Grounded Generation (deterministic [DOC-N] page-level citations)
```

---

## Engineering Decisions & Trade-Offs

### 1. Corpus Skew and Hybrid Retrieval (Dense + BM25 RRF)
The indexed corpus contains 18 core policy directives, 49 crawled MILPERSMAN articles, and 7 compiled NAVPERS 15665J chapters (12,424 total chunks).

Two manuals alone—`NAVPERS 18068F` Volumes I and II—account for 6,206 chunks (50.0% of the entire vector database). Under pure dense vector search, high-dimensional cosine similarity suffered from vector hubness: queries regarding niche medical disqualifications or reenlistment codes were frequently crowded out by occupational classification pages.

To resolve this:
- **Weighted Reciprocal Rank Fusion (RRF):** Dense vector similarity ($0.60$ weight) is fused with metadata-enriched BM25 keyword matching ($0.40$ weight).
- **Top-3 Directive Diversity:** The retriever enforces that no single directive can occupy more than one slot in the top-3 results, capping subsequent slots at 3 chunks per document.
- **Military Lexicon Expansion:** Acronyms and designators (e.g., RE-3, MEPS, PRT, CWT, BUMED) are deterministically expanded into their authoritative directive titles before retrieval.

### 2. Automated Policy Synchronization (`navy-rag sync`)
Naval instructions update at different intervals. Recruiting manuals update frequently via policy letters, while MILPERSMAN articles change through formal quarterly transmittals.

- **Tiered Cadence:** Directives are categorized into synchronization cadences (`weekly`, `monthly`, `quarterly`).
- **HTTP Conditional Probing:** Instead of re-downloading PDFs, the sync engine sends HTTP `HEAD` requests checking `ETag`, `Last-Modified`, and `Content-Length` headers against an audit manifest (`_ingestion_manifest.json`). Unchanged directives are verified in <100ms.
- **Hash-Aware Differential Re-Indexing:** When an upstream directive changes, its incoming `SHA-256` hash is compared against the stored chunk metadata in ChromaDB. Old chunks for that specific document are atomically purged and re-embedded without re-processing the remaining corpus.
- **GitOps Regression Gate:** A weekly GitHub Actions workflow (`.github/workflows/policy-sync.yml`) runs the sync engine. If policy updates are detected, it runs the test suite and evaluation benchmark before committing changes and triggering Cloud Build.

### 3. Zero-Cost Stateless Deployment
- **Bundled Vector Store:** ChromaDB and raw source directives are packaged directly inside the Docker container image. This eliminates cold-start network roundtrips to an external vector database and avoids monthly infrastructure costs.
- **CUDA Pruning:** Multi-stage Docker builds pre-install CPU-only PyTorch wheels and strip unused NVIDIA CUDA runtime binaries, reducing the final container image from 4.39 GB down to 1.58 GB.
- **Cloud Run Concurrency:** The container runs on Google Cloud Run with `--min-instances 0` and `--max-instances 1`, scaling to zero when idle for minimal hosting expense.

---

## Evaluation Benchmark

The retrieval engine is verified against a 50-query golden ground truth dataset (`data/evals/golden_dataset.yaml`) covering enlistment qualifications, medical disqualifications, tattoo regulations, physical readiness tables, ASVAB line scores, and moral waivers.

| Metric | Baseline (MMR) | Hybrid RRF (Current) |
|---|---|---|
| **Recall@3** | 86.0% | **100.0%** |
| **Recall@5** | 92.0% | **100.0%** |
| **MRR (Mean Reciprocal Rank)** | 0.860 | **0.933** |

Run the benchmark locally:
```bash
navy-rag eval
```

---

## Quickstart

### 1. Setup

```bash
git clone https://github.com/petetheo85/navy-rag.git
cd navy-rag

python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Configure your API key:
```bash
cp .env.example .env
```
Edit `.env` and set:
```bash
GOOGLE_API_KEY=your_gemini_api_key
```

### 2. Run the Web Application
```bash
streamlit run app.py
```
Includes dynamic Gemini quota cooldown monitoring, system configuration inspection, and PDF excerpt preview.

*(Optional headless API server: `uvicorn navy_rag.api.main:app --host 127.0.0.1 --port 8000`)*

### 3. Core Commands

```bash
# Automated differential sync (respects weekly/monthly/quarterly cadences)
navy-rag sync --respect-cadence

# Probe all upstream directives and run web crawlers immediately
navy-rag sync --all

# Dry-run upstream check without writing files or modifying database
navy-rag sync --dry-run

# Run the 50-query retrieval benchmark
navy-rag eval

# Run test suite and linter
pytest tests/
ruff check src/ tests/ app.py
```

---

## Repository Structure

```
navy-rag/
├── .github/workflows/
│   └── policy-sync.yml         # Scheduled & manual GitHub Actions sync pipeline
├── src/navy_rag/
│   ├── config.py               # Pydantic Settings & system configuration specs
│   ├── llm/provider.py         # Google Gemini chat model initialization
│   ├── ingestion/
│   │   ├── downloader.py       # Bronze-layer downloader & HTTP conditional probing
│   │   ├── loader.py           # PDF loader, metadata normalization & SHA-256
│   │   ├── chunker.py          # Token-aware recursive text chunking
│   │   ├── embedder.py         # ChromaDB client & hash-aware differential re-indexing
│   │   ├── sync.py             # Differential policy sync orchestrator
│   │   └── crawlers/           # MILPERSMAN and NAVPERS 15665J web scrapers
│   ├── retrieval/
│   │   ├── retriever.py        # Retriever factory with audience filtering
│   │   ├── hybrid.py           # Weighted RRF retriever & directive diversity cap
│   │   └── query_processor.py  # Lexicon query expander & conversational contextualizer
│   ├── chain/
│   │   ├── prompts.py          # Grounded and classification prompt templates
│   │   ├── rag_chain.py        # LCEL retrieval chain & [DOC-N] citation attribution
│   │   └── graph.py            # LangGraph state machine & guardrail routing
│   ├── evals/                  # Quantitative benchmark scoring (Recall@K, MRR)
│   ├── cli/                    # CLI subcommands registered in main.py
│   └── api/                    # Optional FastAPI headless route handlers
├── assets/                     # Design system assets and custom CSS
├── data/
│   ├── manifests/              # Declarative corpus manifest (YAML)
│   ├── evals/                  # 50-query golden benchmark dataset
│   └── raw/                    # Source directives and crawled sub-corpora
├── app.py                      # Streamlit interactive application
└── tests/                      # Pytest test suite
```
