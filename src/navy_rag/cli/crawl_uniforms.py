"""CLI sub-command for crawling and compiling NAVPERS 15665J Chapter PDFs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from navy_rag.config import settings
from navy_rag.ingestion.crawlers.uniform_regs import crawl_uniform_regs

logger = logging.getLogger(__name__)


def execute(args: argparse.Namespace) -> int:
    """Execute the multi-chapter NAVPERS 15665J crawler and typesetter."""
    default_dir = settings.raw_data_dir / "uniform_regs"
    output_dir = Path(args.output_dir) if args.output_dir else default_dir
    target_chapters = args.chapters

    logger.info("Starting NAVPERS 15665J synchronization...")
    try:
        manifest = crawl_uniform_regs(
            output_dir=output_dir,
            target_chapters=target_chapters,
            force=args.force,
        )
        compiled = sum(1 for r in manifest.records if r.status == "compiled")
        skipped = sum(1 for r in manifest.records if r.status == "skipped")
        failed = sum(1 for r in manifest.records if r.status == "failed")

        logger.info(
            "Crawl complete. Total: %d, Compiled: %d, Skipped: %d, Failed: %d",
            manifest.chapter_count,
            compiled,
            skipped,
            failed,
        )
        return 0 if failed == 0 else 1
    except Exception as exc:
        logger.exception("Failed synchronizing uniform regulations: %s", exc)
        return 1


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "crawl-uniforms",
        help="Crawl NAVPERS 15665J chapters and compile into formatted PDFs",
        description="Scrape official online uniform regulations into authentic chapter PDFs.",
    )
    parser.add_argument(
        "--chapters",
        nargs="+",
        default=None,
        help="Specific chapters to crawl (e.g. Chapter-1 Chapter-2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for compiled chapter PDFs (default: data/raw/uniform_regs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-compilation even if upstream content hash has not changed",
    )
    parser.set_defaults(handler=execute)
