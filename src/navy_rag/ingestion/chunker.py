"""Recursive character chunking with document index preservation."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from navy_rag.config import settings

logger = logging.getLogger(__name__)


def chunk_documents(
    docs: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents using recursive character boundaries and assign sequential chunk indices."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    logger.info(
        "Split %d pages into %d chunks (chunk_size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        size,
        overlap,
    )
    return chunks
