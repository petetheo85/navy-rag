"""CLI subcommand for running evaluation benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from navy_rag.evals.evaluator import (
    evaluate_retriever,
    format_benchmark_report,
    load_benchmark_dataset,
)
from navy_rag.retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)


def execute(args: argparse.Namespace) -> int:
    """Run retrieval benchmark evaluation against the golden dataset."""
    dataset_path = Path(args.dataset) if args.dataset else None
    try:
        queries = load_benchmark_dataset(dataset_path)
    except Exception as exc:
        logger.error("Failed loading benchmark dataset: %s", exc)
        return 1

    if args.category:
        queries = [q for q in queries if q.category == args.category]
        logger.info("Filtered benchmark to category '%s' (%d queries)", args.category, len(queries))

    if not queries:
        logger.warning("No evaluation queries to execute.")
        return 1

    logger.info("Initializing retriever (method=%s, k=%d)...", args.method, args.k)
    try:
        retriever = get_retriever(method=args.method, k=args.k)
        summary = evaluate_retriever(retriever, queries, max_k=args.k)
    except Exception as exc:
        logger.error("Evaluation execution failed: %s", exc)
        return 1

    report_text = format_benchmark_report(summary)
    print("\n" + report_text + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(summary.model_dump(), fh, indent=2)
        logger.info("Saved evaluation JSON metrics to %s", out_path)

    return 0


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eval",
        help="Run quantitative retrieval benchmarks against the golden dataset",
        description="Evaluate retriever accuracy (Recall@K, MRR) against verified ground truth.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset YAML (default: data/evals/golden_dataset.yaml)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K retrieval depth for evaluation (default: 5)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter evaluation to a specific policy category",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["hybrid", "similarity", "mmr"],
        help="Retrieval method to benchmark (hybrid, similarity, or mmr)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save full JSON evaluation results",
    )
    parser.set_defaults(handler=execute)
