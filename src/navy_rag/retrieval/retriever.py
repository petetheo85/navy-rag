"""Retriever instantiation backed by ChromaDB vector store."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.retrievers import BaseRetriever

from navy_rag.config import settings
from navy_rag.ingestion.embedder import get_vectorstore

logger = logging.getLogger(__name__)


def get_retriever(
    *,
    audience: str | None = None,
    k: int | None = None,
    method: str | None = None,
) -> BaseRetriever:
    """Return a BaseRetriever with optional audience metadata filtering and MMR parameters."""
    vectorstore = get_vectorstore()
    resolved_k = k or settings.retrieval_k
    resolved_method = method or settings.retrieval_method

    search_kwargs: dict[str, Any] = {"k": resolved_k}
    if audience:
        search_kwargs["filter"] = {"audience": audience}

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
