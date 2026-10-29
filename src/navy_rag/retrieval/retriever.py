"""Retriever instantiation supporting hybrid (Dense + BM25 RRF), MMR, and similarity search."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.retrievers import BaseRetriever

from navy_rag.config import settings
from navy_rag.ingestion.embedder import get_vectorstore
from navy_rag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


def get_retriever(
    *,
    audience: str | None = None,
    k: int | None = None,
    method: str | None = None,
) -> BaseRetriever:
    """Return a BaseRetriever configured for hybrid RRF, MMR, or similarity search."""
    vectorstore = get_vectorstore()
    resolved_k = k or settings.retrieval_k
    resolved_method = (method or settings.retrieval_method).lower()

    search_kwargs: dict[str, Any] = {"k": resolved_k}
    if audience:
        search_kwargs["filter"] = {"audience": audience}

    if resolved_method == "hybrid":
        dense_kwargs = dict(search_kwargs)
        dense_kwargs["fetch_k"] = max(settings.mmr_fetch_k, resolved_k * 3)
        dense_kwargs["lambda_mult"] = settings.mmr_lambda
        dense_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=dense_kwargs,
        )
        logger.debug(
            "Configured Hybrid RRF retriever: k=%d, fetch_k=%d, weights=(%.2f, %.2f), audience=%s",
            resolved_k,
            settings.mmr_fetch_k,
            settings.hybrid_dense_weight,
            settings.hybrid_bm25_weight,
            audience,
        )
        return HybridRetriever(
            dense_retriever=dense_retriever,
            vectorstore=vectorstore,
            audience=audience,
            k=resolved_k,
            fetch_k=settings.mmr_fetch_k,
            dense_weight=settings.hybrid_dense_weight,
            bm25_weight=settings.hybrid_bm25_weight,
            rrf_k=settings.hybrid_rrf_k,
            max_chunks_per_doc=settings.max_chunks_per_doc,
        )

    if resolved_method == "mmr":
        search_kwargs["fetch_k"] = settings.mmr_fetch_k
        search_kwargs["lambda_mult"] = settings.mmr_lambda
        logger.debug(
            "Configured MMR retriever: k=%d, fetch_k=%d, lambda=%.2f, audience=%s",
            resolved_k,
            settings.mmr_fetch_k,
            settings.mmr_lambda,
            audience,
        )
    else:
        logger.debug("Configured similarity retriever: k=%d, audience=%s", resolved_k, audience)

    return vectorstore.as_retriever(
        search_type=resolved_method,
        search_kwargs=search_kwargs,
    )
