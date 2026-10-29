"""CLI sub-command for automated differential policy synchronization."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from navy_rag.ingestion.sync import run_policy_sync

logger = logging.getLogger(__name__)


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sync",
        help="Probe upstream Navy/DoD directives and differentially re-index changed PDFs",
        description=(
            "Check upstream policy portals by cadence tier, download updated directives, "
            "and atomically re-index modified documents in ChromaDB."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to custom corpus_manifest.yaml",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include MILPERSMAN and NAVPERS 15665J Uniform Regulations web crawlers",
    )
    parser.add_argument(
        "--respect-cadence",
        action="store_true",
        help="Skip directives whose configured sync_cadence interval has not yet elapsed",
    )
    parser.add_argument(
        "--cadence",
        choices=["daily", "weekly", "monthly", "quarterly"],
        default=None,
        help="Only check directives matching the specified cadence tier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe upstream headers and report drift without modifying files or ChromaDB",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full download and vector re-indexing regardless of hashes or cadence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON PolicySyncReport",
    )
    parser.set_defaults(handler=execute)


def execute(args: argparse.Namespace) -> int:
    logger.info(
        "Starting automated policy sync (all=%s, respect_cadence=%s, dry_run=%s, force=%s)...",
        args.all,
        args.respect_cadence,
        args.dry_run,
        args.force,
    )
    report = run_policy_sync(
        manifest_path=args.manifest,
        report_path=args.output,
        include_crawlers=args.all,
        respect_cadence=args.respect_cadence,
        cadence_filter=args.cadence,
        dry_run=args.dry_run,
        force=args.force,
    )

    logger.info(
        "Policy sync complete: checked=%d, downloaded=%d, updated=%d, skipped=%d, "
        "deferred=%d, failed=%d, chunks_reindexed=%d",
        report.checked_count,
        len(report.downloaded_docs),
        len(report.updated_docs),
        len(report.skipped_docs),
        len(report.deferred_by_cadence),
        len(report.failed_docs),
        report.chunks_reindexed,
    )
    return 0 if not report.failed_docs else 1
