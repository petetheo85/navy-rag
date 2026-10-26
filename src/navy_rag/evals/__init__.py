"""Evaluation package for benchmarking Navy RAG retrieval and generation accuracy."""

from navy_rag.evals.evaluator import (
    evaluate_query,
    evaluate_retriever,
    format_benchmark_report,
    load_benchmark_dataset,
)
from navy_rag.evals.models import (
    BenchmarkQuery,
    BenchmarkSummary,
    CategoryMetrics,
    QueryEvalResult,
)

__all__ = [
    "BenchmarkQuery",
    "BenchmarkSummary",
    "CategoryMetrics",
    "QueryEvalResult",
    "evaluate_query",
    "evaluate_retriever",
    "format_benchmark_report",
    "load_benchmark_dataset",
]
