"""Unified CLI entry point for the navy-rag suite."""

from __future__ import annotations

import argparse
import logging
import sys

from navy_rag.cli import ingest
from navy_rag.config import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="navy",
        description="Command-line interface for the Navy RAG pipeline.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=settings.log_level,
        help="Logging verbosity (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest.register_subcommand(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.logs_dir / "navy.log"),
        ],
    )

    handler = getattr(args, "handler", None)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
