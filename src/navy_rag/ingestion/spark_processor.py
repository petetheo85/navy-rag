"""PySpark batch extraction and chunking for large document sets."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _pdf_to_row(pdf_path_str: str) -> list[dict]:
    """Worker task: parse and chunk a single PDF into serializable dicts."""
    from navy_rag.ingestion.chunker import chunk_documents
    from navy_rag.ingestion.loader import load_pdf

    path = Path(pdf_path_str)
    try:
        docs = load_pdf(path)
        chunks = chunk_documents(docs)
        return [{"page_content": c.page_content, **c.metadata} for c in chunks]
    except Exception as exc:
        logging.getLogger(__name__).error("Worker failed processing %s: %s", path.name, exc)
        return []


def process_with_spark(
    pdf_paths: list[Path],
    *,
    parallelism: int = 4,
) -> int:
    """Execute parallel document extraction via PySpark and persist generated chunks."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise RuntimeError(
            "PySpark is required for distributed ingestion. Install with: pip install '.[spark]'"
        )

    from langchain_core.documents import Document

    from navy_rag.ingestion.embedder import embed_and_store

    logger.info("Initializing SparkSession (local[%d])", parallelism)
    spark = (
        SparkSession.builder.master(f"local[{parallelism}]")
        .appName("NavyRAG-Ingestion")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        path_strings = [str(p) for p in pdf_paths]
        rdd = spark.sparkContext.parallelize(path_strings, numSlices=parallelism)
        chunk_dicts = rdd.flatMap(_pdf_to_row).collect()
    finally:
        spark.stop()

    logger.info("Spark extraction finished with %d chunks", len(chunk_dicts))
    if not chunk_dicts:
        logger.warning("No chunks generated from input PDFs.")
        return 0

    docs = [
        Document(
            page_content=d.pop("page_content"),
            metadata=d,
        )
        for d in chunk_dicts
    ]
    return embed_and_store(docs)
