"""Data models for RAG evaluation benchmarks."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkQuery(BaseModel):
    """A single ground-truth recruiting inquiry."""

    id: str
    category: str
    query: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    ground_truth_answer: str
    key_facts: list[str] = Field(default_factory=list)


class QueryEvalResult(BaseModel):
    """Retrieval evaluation result for an individual query."""

    query_id: str
    category: str
    query: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    retrieved_instructions: list[str]
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float


class CategoryMetrics(BaseModel):
    """Aggregate metrics for a specific domain category."""

    category: str
    total_queries: int
    recall_at_3: float
    recall_at_5: float
    mrr: float


class BenchmarkSummary(BaseModel):
    """Comprehensive evaluation summary across the benchmark dataset."""

    total_queries: int
    overall_recall_at_3: float
    overall_recall_at_5: float
    overall_mrr: float
    category_metrics: dict[str, CategoryMetrics]
    query_results: list[QueryEvalResult]
