"""Ingestion sub-command implementation."""

from __future__ import annotations

import argparse
import logging

from navy_rag.config import settings
from navy_rag.ingestion.chunker import chunk_documents
from navy_rag.ingestion.embedder import embed_and_store
from navy_rag.ingestion.loader import discover_pdfs, load_all

logger = logging.getLogger(__name__)


def run_standard(*, force: bool = False) -> int:
    docs = load_all()
    if not docs:
        logger.warning("No documents found in %s", settings.data_dir)
        return 0

    chunks = chunk_documents(docs)
    return embed_and_store(chunks, skip_existing=not force)


def run_spark(*, parallelism: int = 4, force: bool = False) -> int:
    from navy_rag.ingestion.spark_processor import process_with_spark

    pdfs = discover_pdfs()
    if not pdfs:
        logger.warning("No PDFs discovered in %s", settings.data_dir)
        return 0

    return process_with_spark(pdfs, parallelism=parallelism)


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "ingest",
        help="Extract, chunk, and embed instruction PDFs into ChromaDB",
        description="Ingest Navy and DoD PDF instructions into the vector store.",
    )
    parser.add_argument(
        "--spark",
        action="store_true",
        help="Distribute PDF extraction and chunking across local Spark workers",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of Spark worker partitions (default: 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index all documents, bypassing existing filename deduplication",
    )
    parser.set_defaults(handler=execute)


def execute(args: argparse.Namespace) -> int:
    logger.info(
        "Starting ingestion (provider: %s, collection: %s)",
        settings.llm_provider,
        settings.chroma_collection,
    )

    if args.spark:
        logger.info("Execution engine: PySpark (parallelism=%d)", args.parallelism)
        stored = run_spark(parallelism=args.parallelism, force=args.force)
    else:
        logger.info("Execution engine: standard (single-process)")
        stored = run_standard(force=args.force)

    logger.info("Ingestion complete. Stored %d chunks.", stored)
    return 0
