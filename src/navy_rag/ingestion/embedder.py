"""ChromaDB vector store client and batch embedding management."""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Sequence

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from navy_rag.config import settings

logger = logging.getLogger(__name__)


_EMBEDDINGS_CACHE: dict[str, Embeddings] = {}
_VECTORSTORE_CACHE: dict[str, Chroma] = {}


def clear_vectorstore_cache() -> None:
    """Clear cached vector store and embeddings client instances."""
    _VECTORSTORE_CACHE.clear()
    _EMBEDDINGS_CACHE.clear()


def _build_embeddings() -> Embeddings:
    cache_key = f"{settings.embedding_provider}_{settings.embedding_model}"
    if cache_key in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[cache_key]

    if settings.embedding_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        inst = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
    elif settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        inst = OpenAIEmbeddings(model=settings.embedding_model)
    elif settings.embedding_provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        inst = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")

    _EMBEDDINGS_CACHE[cache_key] = inst
    return inst


def get_vectorstore() -> Chroma:
    """Return a cached Chroma client bound to the configured persistent directory."""
    cache_key = f"{settings.chroma_db_dir}_{settings.chroma_collection}"
    if cache_key in _VECTORSTORE_CACHE:
        return _VECTORSTORE_CACHE[cache_key]

    client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
    embeddings = _build_embeddings()
    store = Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
    )
    _VECTORSTORE_CACHE[cache_key] = store
    return store


def generate_chunk_id(chunk: Document) -> str:
    """Generate deterministic versioned chunk ID: {doc_id}_rev{rev}_ch{change}_{chunk_index:04d}."""
    doc_id = chunk.metadata.get("doc_id") or "doc"
    raw_rev = chunk.metadata.get("revision")
    rev = "0" if (not raw_rev or raw_rev == "BASE") else str(raw_rev)
    change = int(chunk.metadata.get("change", 0))
    chunk_index = int(chunk.metadata.get("chunk_index", 0))
    return f"{doc_id}_rev{rev}_ch{change}_{chunk_index:04d}"


def _get_ingested_count(vectorstore: Chroma, filename: str) -> int:
    result = vectorstore._collection.get(
        where={"filename": filename},
        include=[],
    )
    return len(result["ids"])


def _add_documents_with_retry(
    vectorstore: Chroma,
    batch: list[Document],
    ids: list[str] | None = None,
    *,
    max_retries: int = 5,
    initial_backoff: float = 35.0,
) -> None:
    """Add a batch of documents to Chroma with rate-limit and backoff retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            vectorstore.add_documents(batch, ids=ids)
            return
        except Exception as exc:
            err_msg = str(exc)
            is_rate_limit = (
                "429" in err_msg
                or "RESOURCE_EXHAUSTED" in err_msg
                or "quota" in err_msg.lower()
            )
            if not is_rate_limit or attempt == max_retries:
                raise

            backoff = initial_backoff * attempt
            delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
            if delay_match:
                backoff = max(backoff, float(delay_match.group(1)) + 3.0)

            logger.warning(
                "Rate limit encountered (429). Backing off %.1fs (attempt %d/%d)...",
                backoff,
                attempt,
                max_retries,
            )
            time.sleep(backoff)


def embed_and_store(
    chunks: Sequence[Document],
    *,
    skip_existing: bool = True,
    batch_size: int = 50,
) -> int:
    """
    Embed and persist document chunks into ChromaDB in batches.

    Returns the total number of chunks stored.
    """
    if not chunks:
        logger.warning("No chunks provided for embedding.")
        return 0

    vectorstore = get_vectorstore()

    by_file: dict[str, list[Document]] = {}
    for chunk in chunks:
        fname = chunk.metadata.get("filename", "unknown")
        by_file.setdefault(fname, []).append(chunk)

    to_store: list[Document] = []
    for fname, file_chunks in by_file.items():
        existing_count = _get_ingested_count(vectorstore, fname)
        if skip_existing and existing_count >= len(file_chunks):
            logger.info("Skipping already indexed file: %s (%d chunks)", fname, existing_count)
            continue

        if existing_count > 0:
            logger.info(
                "File %s is partially indexed (%d/%d chunks). Re-indexing cleanly...",
                fname,
                existing_count,
                len(file_chunks),
            )
            vectorstore._collection.delete(where={"filename": fname})

        to_store.extend(file_chunks)

    if not to_store:
        logger.info("All documents are up-to-date in collection '%s'.", settings.chroma_collection)
        return 0

    stored = 0
    total_batches = math.ceil(len(to_store) / batch_size)
    for i in range(0, len(to_store), batch_size):
        batch = to_store[i : i + batch_size]
        for c in batch:
            if "base_ssic" not in c.metadata or not c.metadata["base_ssic"]:
                c.metadata["base_ssic"] = c.metadata.get("instruction_number", "")
            if "revision" not in c.metadata or c.metadata["revision"] is None:
                c.metadata["revision"] = "BASE"
            if "change" not in c.metadata:
                c.metadata["change"] = 0
            else:
                c.metadata["change"] = int(c.metadata["change"])
            if "is_latest" not in c.metadata:
                c.metadata["is_latest"] = True
            else:
                c.metadata["is_latest"] = bool(c.metadata["is_latest"])

        ids = [generate_chunk_id(chunk) for chunk in batch]
        _add_documents_with_retry(vectorstore, batch, ids=ids)
        stored += len(batch)
        logger.info(
            "Persisted batch %d/%d (%d chunks)",
            i // batch_size + 1,
            total_batches,
            len(batch),
        )
        time.sleep(0.4)

    logger.info("Stored %d new chunk(s) in '%s'.", stored, settings.chroma_collection)
    return stored
