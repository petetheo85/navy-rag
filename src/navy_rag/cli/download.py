"""Download sub-command implementation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from navy_rag.ingestion.downloader import sync_corpus

logger = logging.getLogger(__name__)


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download",
        help="Fetch authoritative Navy and DoD policy PDFs into Bronze raw storage",
        description=(
            "Download policy PDFs defined in the declarative manifest and emit audit sidecar."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to custom corpus_manifest.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass hash checks and force re-downloading all documents",
    )
    parser.set_defaults(handler=execute)


def execute(args: argparse.Namespace) -> int:
    logger.info("Executing Bronze corpus download...")
    records = sync_corpus(manifest_path=args.manifest, force=args.force)

    logger.info("--- Ingestion Results ---")
    for r in records:
        logger.info(
            "[%s] %s -> %s (%d bytes, sha256=%s)",
            r.status.upper(),
            r.doc_id,
            r.filename,
            r.file_size,
            r.sha256[:12],
        )

    logger.info("Corpus synchronization complete. Processed %d documents.", len(records))
    return 0
