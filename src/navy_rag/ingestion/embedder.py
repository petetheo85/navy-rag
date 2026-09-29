"""ChromaDB vector store client and batch embedding management."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from navy_rag.config import settings

logger = logging.getLogger(__name__)


def _build_embeddings() -> Embeddings:
    if settings.embedding_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)

    if settings.embedding_provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def get_vectorstore() -> Chroma:
    """Return a Chroma client bound to the configured persistent directory."""
    client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
    embeddings = _build_embeddings()
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
    )


def _already_ingested(vectorstore: Chroma, filename: str) -> bool:
    result = vectorstore._collection.get(
        where={"filename": filename},
        limit=1,
        include=[],
    )
    return len(result["ids"]) > 0


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
        if skip_existing and _already_ingested(vectorstore, fname):
            logger.info("Skipping already indexed file: %s", fname)
            continue
        to_store.extend(file_chunks)

    if not to_store:
        logger.info("All documents are up-to-date in collection '%s'.", settings.chroma_collection)
        return 0

    stored = 0
    total_batches = math.ceil(len(to_store) / batch_size)
    for i in range(0, len(to_store), batch_size):
        batch = to_store[i : i + batch_size]
        vectorstore.add_documents(batch)
        stored += len(batch)
        logger.info(
            "Persisted batch %d/%d (%d chunks)",
            i // batch_size + 1,
            total_batches,
            len(batch),
        )

    logger.info("Stored %d new chunk(s) in '%s'.", stored, settings.chroma_collection)
    return stored
