# Agent Guidelines: Navy RAG Knowledge Base

## Overview
Navy RAG is an enterprise-grade retrieval-augmented generation system for U.S. Navy recruiting intelligence. It indexes authoritative naval and defense directives (COMNAVCRUITCOMINST, MILPERSMAN, MANMED, DoDI, NAVPERS uniform regulations) to provide grounded answers with page-level citations for recruiters and applicants.

## Core Commands
- **Run Unit Tests:** `pytest tests/`
- **Run Linter:** `ruff check src/ tests/`
- **Bronze Layer Download:** `navy-rag download`
- **Vector Ingestion:** `navy-rag ingest` (or `navy-rag ingest --spark --parallelism 4`)
- **MILPERSMAN Crawler:** `navy-rag crawl-milpersman`
- **Uniform Regulations Crawler:** `navy-rag crawl-uniforms`
- **Benchmark Evaluation:** `navy-rag eval` (or `navy-rag eval --category medical_standards`)
- **FastAPI Backend:** `uvicorn navy_rag.api.main:app --host 127.0.0.1 --port 8000`
- **Streamlit Frontend:** `streamlit run app.py`

## Architecture & Directory Conventions
- `src/navy_rag/config.py`: Single source of truth for configuration and environment variables. Never read `os.environ` outside `config.py`.
- `src/navy_rag/ingestion/`: Ingestion pipelines (loader, chunker, embedder, PySpark processor, and web crawlers).
- `src/navy_rag/retrieval/`: Dense vector, MMR, and hybrid retrieval logic.
- `src/navy_rag/chain/`: LangGraph agent state machine, guardrails, and grounded synthesis chains.
- `src/navy_rag/evals/`: Quantitative evaluation benchmark harness (Recall@K, MRR).
- `src/navy_rag/cli/`: Subcommand adapters registered in `main.py`. Keep business logic in core packages.
- `data/manifests/corpus_manifest.yaml`: Declarative bronze-layer document catalog.
- `data/raw/`: Immutable source PDFs. Crawled sub-corpora live in `data/raw/milpersman/` and `data/raw/uniform_regs/`.
- `data/evals/golden_dataset.yaml`: 50-query verified ground truth benchmark dataset.

## Engineering Standards
- **Scope & Minimalism:** Do exactly what was asked. Avoid speculative hooks, premature abstractions, or unrelated reformatting.
- **Comments:** Explain *why*, not *what*. Never generate decorative banners (`====`, `----`), step narration, TODOs, or emojis.
- **Code Quality:** Small single-purpose functions, explicit type hints on all signatures, no dead imports, and logging over `print` outside CLI tools.
- **Configuration & Security:** Never hardcode secrets. Listeners must bind to `127.0.0.1`.
- **Policy Integrity:** RAG responses must be strictly anchored to verified instruction citations. If a policy is ambiguous or unindexed, route to an authorized recruiter referral rather than speculating.
- **Verification:** Run `ruff check` and `pytest tests/` before declaring work complete.
