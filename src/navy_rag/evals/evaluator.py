"""Core evaluation harness for measuring retrieval accuracy and benchmark metrics."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from navy_rag.config import settings
from navy_rag.evals.models import (
    BenchmarkQuery,
    BenchmarkSummary,
    CategoryMetrics,
    QueryEvalResult,
)

logger = logging.getLogger(__name__)


def load_benchmark_dataset(dataset_path: Path | None = None) -> list[BenchmarkQuery]:
    """Load and parse the ground truth evaluation dataset."""
    path = dataset_path or (settings.base_dir / "data" / "evals" / "golden_dataset.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw_data = yaml.safe_load(fh) or {}

    queries_raw = raw_data.get("queries", [])
    queries = [BenchmarkQuery(**item) for item in queries_raw]
    logger.info("Loaded %d benchmark queries from %s", len(queries), path.name)
    return queries


def is_document_match(doc: Document, query: BenchmarkQuery) -> bool:
    """Determine whether a retrieved chunk matches ground truth targets."""
    doc_id = (doc.metadata.get("doc_id") or "").lower().strip()
    instruction_num = (doc.metadata.get("instruction_number") or "").lower().strip()
    source_title = (doc.metadata.get("source") or "").lower().strip()
    filename = (doc.metadata.get("filename") or "").lower().strip()
    combined_doc = f"{doc_id} {instruction_num} {source_title} {filename}".lower()

    expected_ids = {eid.lower().strip() for eid in query.expected_doc_ids}
    if doc_id and doc_id in expected_ids:
        return True

    # Manifest identifiers use underscores while crawled document filenames use hyphens.
    for eid in expected_ids:
        norm_eid = eid.replace("_", "-")
        if eid in combined_doc or norm_eid in combined_doc:
            return True

    for exp_cit in query.expected_citations:
        norm_cit = exp_cit.lower().strip()
        if norm_cit and norm_cit in combined_doc:
            return True
        base_inst = norm_cit.split(" ch")[0].split(" vol")[0].strip()
        if base_inst and instruction_num and base_inst in instruction_num:
            return True

    return False


def evaluate_query(
    query: BenchmarkQuery,
    retriever: BaseRetriever | Any,
    max_k: int = 5,
) -> QueryEvalResult:
    """Evaluate retrieval performance for an individual query."""
    try:
        retrieved_docs: list[Document] = retriever.invoke(query.query)
    except Exception as exc:
        logger.error("Error retrieving documents for query '%s': %s", query.id, exc)
        retrieved_docs = []

    retrieved_doc_ids: list[str] = []
    retrieved_instructions: list[str] = []

    first_hit_rank: int | None = None

    for rank, doc in enumerate(retrieved_docs[:max_k], start=1):
        d_id = doc.metadata.get("doc_id") or doc.metadata.get("filename", "")
        inst = doc.metadata.get("instruction_number") or doc.metadata.get("source", "")
        retrieved_doc_ids.append(d_id)
        retrieved_instructions.append(inst)

        if first_hit_rank is None and is_document_match(doc, query):
            first_hit_rank = rank

    hit_at_3 = first_hit_rank is not None and first_hit_rank <= 3
    hit_at_5 = first_hit_rank is not None and first_hit_rank <= 5
    rr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0

    return QueryEvalResult(
        query_id=query.id,
        category=query.category,
        query=query.query,
        expected_doc_ids=query.expected_doc_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        retrieved_instructions=retrieved_instructions,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        reciprocal_rank=rr,
    )


def evaluate_retriever(
    retriever: BaseRetriever | Any,
    queries: list[BenchmarkQuery],
    max_k: int = 5,
) -> BenchmarkSummary:
    """Run full benchmark evaluation across all queries and aggregate metrics."""
    results: list[QueryEvalResult] = []
    category_map: dict[str, list[QueryEvalResult]] = defaultdict(list)

    for q in queries:
        res = evaluate_query(q, retriever, max_k=max_k)
        results.append(res)
        category_map[res.category].append(res)

    total = len(results)
    if total == 0:
        return BenchmarkSummary(
            total_queries=0,
            overall_recall_at_3=0.0,
            overall_recall_at_5=0.0,
            overall_mrr=0.0,
            category_metrics={},
            query_results=[],
        )

    overall_recall_3 = sum(1 for r in results if r.hit_at_3) / total
    overall_recall_5 = sum(1 for r in results if r.hit_at_5) / total
    overall_mrr = sum(r.reciprocal_rank for r in results) / total

    category_metrics: dict[str, CategoryMetrics] = {}
    for cat_name, cat_results in category_map.items():
        cat_total = len(cat_results)
        c_rec_3 = sum(1 for r in cat_results if r.hit_at_3) / cat_total
        c_rec_5 = sum(1 for r in cat_results if r.hit_at_5) / cat_total
        c_mrr = sum(r.reciprocal_rank for r in cat_results) / cat_total
        category_metrics[cat_name] = CategoryMetrics(
            category=cat_name,
            total_queries=cat_total,
            recall_at_3=c_rec_3,
            recall_at_5=c_rec_5,
            mrr=c_mrr,
        )

    return BenchmarkSummary(
        total_queries=total,
        overall_recall_at_3=overall_recall_3,
        overall_recall_at_5=overall_recall_5,
        overall_mrr=overall_mrr,
        category_metrics=category_metrics,
        query_results=results,
    )


def format_benchmark_report(summary: BenchmarkSummary) -> str:
    """Format evaluation summary into an authentic ASCII/Markdown report table."""
    lines: list[str] = [
        "==================================================================",
        "                 NAVY RAG BENCHMARK EVALUATION REPORT             ",
        "==================================================================",
        f"Total Queries Evaluated: {summary.total_queries}",
        f"Overall Recall@3:        {summary.overall_recall_at_3 * 100:.1f}%",
        f"Overall Recall@5:        {summary.overall_recall_at_5 * 100:.1f}%",
        f"Overall MRR:             {summary.overall_mrr:.3f}",
        "------------------------------------------------------------------",
        f"{'Category':<26} | {'Count':<5} | {'Recall@3':<8} | {'Recall@5':<8} | {'MRR':<5}",
        "------------------------------------------------------------------",
    ]

    for cat, m in sorted(summary.category_metrics.items()):
        rec3_str = f"{m.recall_at_3 * 100:>7.1f}%"
        rec5_str = f"{m.recall_at_5 * 100:>7.1f}%"
        lines.append(
            f"{cat:<26} | {m.total_queries:<5} | {rec3_str} | {rec5_str} | {m.mrr:>5.3f}"
        )

    lines.append("------------------------------------------------------------------")

    misses = [r for r in summary.query_results if not r.hit_at_5]
    if misses:
        lines.append(f"\nRetrieval Misses ({len(misses)} queries):")
        for miss in misses:
            lines.append(f"  - [{miss.query_id}] ({miss.category}) \"{miss.query}\"")
            lines.append(f"      Expected:  {miss.expected_doc_ids}")
            lines.append(f"      Retrieved: {miss.retrieved_doc_ids[:3]}")

    return "\n".join(lines)
