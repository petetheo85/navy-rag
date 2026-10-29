"""Hybrid dense + sparse BM25 retriever using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from navy_rag.config import settings
from navy_rag.retrieval.query_processor import expand_military_query

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\-\./][a-z0-9]+)*")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "may",
        "navy",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
    }
)


@dataclass
class _BM25CorpusIndex:
    bm25: BM25Okapi
    documents: list[Document]


_BM25_CACHE: dict[str, _BM25CorpusIndex] = {}


def clear_bm25_cache() -> None:
    """Clear the in-memory BM25 corpus index cache."""
    _BM25_CACHE.clear()


def tokenize_military_text(text: str) -> list[str]:
    """Tokenize text while preserving military instruction codes and alphanumeric designators."""
    lowered = text.lower()
    raw_tokens = _TOKEN_RE.findall(lowered)
    tokens: list[str] = []
    for tok in raw_tokens:
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
        # Also index sub-parts of hyphenated/dotted designators (e.g. 1133-020 -> 1133, 020)
        if any(sep in tok for sep in ("-", ".", "/")):
            for part in re.split(r"[\-\./]+", tok):
                if len(part) >= 2 and part not in _STOPWORDS:
                    tokens.append(part)
    return tokens


def _document_to_bm25_tokens(doc: Document) -> list[str]:
    """Build metadata-enriched token list for a document chunk."""
    meta = doc.metadata or {}
    inst = str(meta.get("instruction_number") or "")
    doc_id = str(meta.get("doc_id") or "").replace("_", " ")
    title = str(meta.get("source") or meta.get("title") or "")
    header = f"{inst} {inst} {doc_id} {title}"
    return tokenize_military_text(f"{header}\n{doc.page_content}")


def _chunk_signature(doc: Document) -> tuple[str, int, int, str]:
    """Return a deterministic key identifying a unique document chunk."""
    meta = doc.metadata or {}
    filename = str(meta.get("filename") or meta.get("doc_id") or "")
    page = int(meta.get("page") or 0)
    chunk_idx = int(meta.get("chunk_index") or 0)
    prefix = doc.page_content[:80]
    return (filename, page, chunk_idx, prefix)


def _get_or_build_bm25_index(
    vectorstore: Any,
    audience: str | None = None,
) -> _BM25CorpusIndex | None:
    """Load corpus chunks from ChromaDB and cache an in-memory BM25Okapi index."""
    cache_key = f"{settings.chroma_db_dir}_{settings.chroma_collection}_{audience or 'all'}"
    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key]

    collection = getattr(vectorstore, "_collection", None)
    if collection is None or not hasattr(collection, "get"):
        return None

    try:
        where_filter = {"audience": audience} if audience else None
        raw = collection.get(where=where_filter, include=["documents", "metadatas"])
    except Exception as exc:
        logger.debug("Skipping BM25 index build (collection unavailable): %s", exc)
        return None

    if not isinstance(raw, dict):
        return None

    raw_docs = raw.get("documents") or []
    raw_metas = raw.get("metadatas") or []
    if not raw_docs:
        return None

    documents: list[Document] = []
    tokenized_corpus: list[list[str]] = []

    for content, meta in zip(raw_docs, raw_metas, strict=False):
        if not content:
            continue
        doc = Document(page_content=content, metadata=dict(meta or {}))
        tokens = _document_to_bm25_tokens(doc)
        if tokens:
            documents.append(doc)
            tokenized_corpus.append(tokens)

    if not documents:
        return None

    index = _BM25CorpusIndex(bm25=BM25Okapi(tokenized_corpus), documents=documents)
    _BM25_CACHE[cache_key] = index
    logger.info("Cached BM25 lexical index over %d chunks (audience=%s)", len(documents), audience)
    return index


class HybridRetriever(BaseRetriever):
    """Combines dense vector search with metadata-aware BM25 via Reciprocal Rank Fusion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_retriever: BaseRetriever
    vectorstore: Any
    audience: str | None = None
    k: int = 6
    fetch_k: int = 25
    dense_weight: float = 0.60
    bm25_weight: float = 0.40
    rrf_k: int = 60
    max_chunks_per_doc: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        dense_docs = self.dense_retriever.invoke(query)

        bm25_index = _get_or_build_bm25_index(self.vectorstore, audience=self.audience)
        if bm25_index is None:
            return dense_docs[: self.k]

        expanded_query = expand_military_query(query)
        query_tokens = tokenize_military_text(expanded_query)
        if not query_tokens:
            return dense_docs[: self.k]

        bm25_scores = bm25_index.bm25.get_scores(query_tokens)

        ranked_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda idx: bm25_scores[idx],
            reverse=True,
        )[: self.fetch_k]

        bm25_docs: list[Document] = [
            bm25_index.documents[idx] for idx in ranked_bm25_indices if bm25_scores[idx] > 0.0
        ]

        if not bm25_docs:
            return dense_docs[: self.k]

        # Fuse rankings using weighted Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[tuple[str, int, int, str], float] = {}
        doc_map: dict[tuple[str, int, int, str], Document] = {}
        bm25_hit_sigs: set[tuple[str, int, int, str]] = set()

        for rank, doc in enumerate(dense_docs[: self.fetch_k], start=1):
            sig = _chunk_signature(doc)
            doc_map[sig] = doc
            rrf_scores[sig] = rrf_scores.get(sig, 0.0) + (
                self.dense_weight / (self.rrf_k + rank)
            )

        for rank, doc in enumerate(bm25_docs, start=1):
            sig = _chunk_signature(doc)
            bm25_hit_sigs.add(sig)
            if sig not in doc_map:
                doc_map[sig] = doc
            rrf_scores[sig] = rrf_scores.get(sig, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank)
            )

        sorted_sigs = sorted(rrf_scores.keys(), key=lambda s: rrf_scores[s], reverse=True)

        enriched_candidates: list[tuple[str, Document]] = []
        for sig in sorted_sigs:
            candidate = doc_map[sig]
            meta = dict(candidate.metadata or {})
            meta["rrf_score"] = round(rrf_scores[sig], 6)
            meta["bm25_hit"] = sig in bm25_hit_sigs
            doc_key = str(meta.get("doc_id") or meta.get("filename") or "unknown")
            enriched_candidates.append(
                (doc_key, Document(page_content=candidate.page_content, metadata=meta))
            )

        # Stage 1: Ensure top-3 slots represent distinct directives when multiple match
        selected: list[Document] = []
        deferred: list[tuple[str, Document]] = []
        doc_counts: dict[str, int] = {}
        distinct_top_n = min(3, self.k)

        for doc_key, doc in enriched_candidates:
            count = doc_counts.get(doc_key, 0)
            limit = 1 if len(selected) < distinct_top_n else self.max_chunks_per_doc
            if count < limit:
                selected.append(doc)
                doc_counts[doc_key] = count + 1
                if len(selected) >= self.k:
                    break
            else:
                deferred.append((doc_key, doc))

        # Stage 2: Backfill remaining slots up to max_chunks_per_doc, then any overflow
        if len(selected) < self.k and deferred:
            overflow: list[Document] = []
            for doc_key, doc in deferred:
                count = doc_counts.get(doc_key, 0)
                if count < self.max_chunks_per_doc:
                    selected.append(doc)
                    doc_counts[doc_key] = count + 1
                    if len(selected) >= self.k:
                        break
                else:
                    overflow.append(doc)

            if len(selected) < self.k and overflow:
                selected.extend(overflow[: self.k - len(selected)])

        return selected
