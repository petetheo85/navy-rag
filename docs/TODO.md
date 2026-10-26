# Navy RAG: Data Engineering & Pipeline Roadmap

## Planned Features & Architectural Enhancements

### 1. Automated Policy Synchronization & Drift Detection (`navy-rag sync`)

#### Background & Motivation
Official Navy recruiting standards, uniform regulations, and medical qualification manuals undergo periodic policy revisions through NAVADMIN messages, BUMED changes, and MILPERSMAN updates. To prevent knowledge staleness and ensure recruiters never quote superseded policies, the ingestion pipeline requires an automated Change Data Capture (CDC) and drift detection mechanism.

#### Production Architecture & Cadence
- **Cadence:** Weekly scheduled cron job (`0 3 * * 0` - every Sunday at 0300 UTC) or CI/CD workflow (GitHub Actions scheduled runner). Military directives do not mutate intra-day; a weekly cadence minimizes network overhead while capturing quarterly and monthly policy updates.
- **Header-Based Lightweight Check (HTTP HEAD):**
  - Query upstream URLs with `If-Modified-Since` and `If-None-Match` using cached `Last-Modified` timestamps and `ETag` headers stored in `corpus_manifest.yaml`.
  - Avoid re-downloading large multi-megabyte PDFs (e.g., 50MB NEOCS or 30MB MANMED) when headers confirm no change (`304 Not Modified`).
- **Cryptographic Verification (SHA-256):**
  - For servers that strip or randomize ETags (common with DoD DotNetNuke and SharePoint CDNs), stream the body through SHA-256 hashing without buffering the full file into memory.
  - Compare calculated SHA-256 against the recorded hash in `data/manifests/corpus_manifest.yaml`.
- **Atomic Vector Ingestion on Drift:**
  - When an upstream document hash changes:
    1. Quarantine and archive the previous PDF revision in `data/archive/<doc_id>/<timestamp>/`.
    2. Download and validate the new revision.
    3. Delete existing document chunks from ChromaDB by metadata filter: `{"doc_id": doc_id}`.
    4. Chunk, embed, and insert the updated document chunks with the new revision timestamp.
- **Recruiter Policy Alerting:**
  - Output an audit diff report (`logs/sync_audit_<date>.json`) detailing which instructions changed, page count deltas, and chunk delta counts.
  - Optional notification hook (Slack, email, or CLI warning in the Streamlit UI banner: *"Warning: NAVPERS 15665J was updated on [Date]. Review changes to Chapter 2 Grooming."*).

---

### 2. Full Corpus Expansion & Indexing
- [x] Crawl and compile NAVPERS 15665J Chapters 1–7 into structured military directive PDFs.
- [ ] Ingest NAVPERS 15665J Chapter PDFs into ChromaDB vector database.
- [ ] Ingest all 25 downloaded MILPERSMAN 1100-series recruiting policy articles into ChromaDB.
- [ ] Crawl MILPERSMAN 1300 (Assignment) and 1900 (Separation / Reenlistment codes RE-1 through RE-4).

---

### 3. Comprehensive Evaluation Benchmark Suite
- [ ] Create `data/evals/golden_dataset.json` containing 50+ grounded recruiter and applicant queries with source documents, article numbers, page ranges, and ground-truth answer keys.
- [ ] Build automated evaluation harness (`navy-rag eval`) measuring:
  - **Retrieval Recall@K** (k=3, k=5)
  - **Mean Reciprocal Rank (MRR)**
  - **Citation Precision & Authenticity**
  - **Groundedness / Faithfulness (LLM-as-a-judge & deterministic checks)**

---

### 4. Advanced Retrieval Optimization (Post-Quota Refresh)
- [ ] Implement Hybrid Search: Dense Vector (Google embeddings) + Sparse BM25 (lexical exact match).
- [ ] Cross-encoder re-ranking for Top-K refinement.
- [ ] Recruiter domain synonym and slang query expansion ("weed waiver" -> "cannabinoid moral waiver").
