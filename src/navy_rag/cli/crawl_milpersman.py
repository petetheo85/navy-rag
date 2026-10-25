"""CLI sub-command for crawling MILPERSMAN policy articles."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from navy_rag.config import settings
from navy_rag.ingestion.crawlers.milpersman import DEFAULT_CATEGORIES, crawl_milpersman

logger = logging.getLogger(__name__)


def execute(args: argparse.Namespace) -> int:
    """Execute the MILPERSMAN crawler."""
    categories = args.categories or list(DEFAULT_CATEGORIES)
    raw_dir = settings.raw_data_dir / "milpersman"
    output_dir = Path(args.output_dir) if args.output_dir else raw_dir

    logger.info("Starting MILPERSMAN crawl for categories: %s", categories)
    manifest = crawl_milpersman(
        categories=categories,
        output_dir=output_dir,
        force=args.force,
    )

    downloaded = sum(1 for r in manifest.records if r.status == "downloaded")
    skipped = sum(1 for r in manifest.records if r.status == "skipped")
    failed = sum(1 for r in manifest.records if r.status == "failed")

    logger.info(
        "Crawl complete. Total: %d, Downloaded: %d, Skipped: %d, Failed: %d",
        manifest.article_count,
        downloaded,
        skipped,
        failed,
    )
    return 0 if failed == 0 else 1


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "crawl-milpersman",
        help="Crawl and download MILPERSMAN articles from MyNavy HR",
        description="Discover and download recruiting/separation MILPERSMAN articles.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific categories to crawl (default: 1100-Recruiting 1900-Separation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for downloaded PDFs (default: data/raw/milpersman)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local files exist",
    )
    parser.set_defaults(handler=execute)
