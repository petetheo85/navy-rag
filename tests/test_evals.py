"""Unit tests for the evaluation benchmark suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from navy_rag.cli.main import build_parser
from navy_rag.evals.evaluator import (
    evaluate_query,
    evaluate_retriever,
    format_benchmark_report,
    is_document_match,
    load_benchmark_dataset,
)
from navy_rag.evals.models import BenchmarkQuery


class TestEvaluationDataset:
    def test_golden_dataset_loads_all_50_queries(self):
        queries = load_benchmark_dataset()
        assert len(queries) == 50

        categories = {q.category for q in queries}
        assert "medical_standards" in categories
        assert "grooming_tattoos" in categories
        assert "moral_legal_waivers" in categories
        assert "physical_readiness" in categories
        assert "asvab_ratings_education" in categories
        assert "prior_service_accessions" in categories

        for q in queries:
            assert q.id.strip() != ""
            assert len(q.query.strip()) > 10
            assert len(q.expected_doc_ids) > 0
            assert len(q.ground_truth_answer.strip()) > 20
            assert len(q.key_facts) > 0


class TestEvaluationMatching:
    def test_is_document_match_by_doc_id(self):
        q = BenchmarkQuery(
            id="T1",
            category="medical",
            query="test query",
            expected_doc_ids=["dodi_6130_03_v1"],
            expected_citations=["DoDI 6130.03 Vol. 1"],
            ground_truth_answer="answer",
            key_facts=["fact"],
        )
        doc = Document(page_content="text", metadata={"doc_id": "dodi_6130_03_v1"})
        assert is_document_match(doc, q) is True

    def test_is_document_match_by_citation_substring(self):
        q = BenchmarkQuery(
            id="T2",
            category="grooming",
            query="test query",
            expected_doc_ids=["navpers_15665j_ch2"],
            expected_citations=["NAVPERS 15665J Ch. 2"],
            ground_truth_answer="answer",
            key_facts=["fact"],
        )
        doc = Document(
            page_content="text",
            metadata={"instruction_number": "NAVPERS 15665J", "source": "Chapter 2 Grooming"},
        )
        assert is_document_match(doc, q) is True

    def test_is_document_match_miss(self):
        q = BenchmarkQuery(
            id="T3",
            category="prt",
            query="test query",
            expected_doc_ids=["opnavinst_6110_1k"],
            expected_citations=["OPNAVINST 6110.1K"],
            ground_truth_answer="answer",
            key_facts=["fact"],
        )
        doc = Document(
            page_content="text",
            metadata={
                "doc_id": "comnavcruitcominst_1130_8",
                "instruction_number": "COMNAVCRUITCOMINST 1130.8",
            },
        )
        assert is_document_match(doc, q) is False


class TestEvaluatorExecution:
    def test_evaluate_query_hit_at_rank_1(self):
        q = BenchmarkQuery(
            id="Q1",
            category="medical",
            query="sample query",
            expected_doc_ids=["target_doc"],
            expected_citations=["Target Doc"],
            ground_truth_answer="answer",
            key_facts=["fact"],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="c1", metadata={"doc_id": "target_doc"}),
            Document(page_content="c2", metadata={"doc_id": "other"}),
        ]

        result = evaluate_query(q, mock_retriever, max_k=5)
        assert result.hit_at_3 is True
        assert result.hit_at_5 is True
        assert result.reciprocal_rank == 1.0

    def test_evaluate_query_hit_at_rank_4(self):
        q = BenchmarkQuery(
            id="Q2",
            category="medical",
            query="sample query",
            expected_doc_ids=["target_doc"],
            expected_citations=["Target Doc"],
            ground_truth_answer="answer",
            key_facts=["fact"],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="c1", metadata={"doc_id": "other1"}),
            Document(page_content="c2", metadata={"doc_id": "other2"}),
            Document(page_content="c3", metadata={"doc_id": "other3"}),
            Document(page_content="c4", metadata={"doc_id": "target_doc"}),
        ]

        result = evaluate_query(q, mock_retriever, max_k=5)
        assert result.hit_at_3 is False
        assert result.hit_at_5 is True
        assert result.reciprocal_rank == 0.25

    def test_evaluate_retriever_aggregates_metrics(self):
        q1 = BenchmarkQuery(
            id="QA",
            category="cat1",
            query="query a",
            expected_doc_ids=["doc_a"],
            ground_truth_answer="ans",
            key_facts=["f"],
        )
        q2 = BenchmarkQuery(
            id="QB",
            category="cat1",
            query="query b",
            expected_doc_ids=["doc_b"],
            ground_truth_answer="ans",
            key_facts=["f"],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = [
            [Document(page_content="c", metadata={"doc_id": "doc_a"})],
            [Document(page_content="c", metadata={"doc_id": "nomatch"})],
        ]

        summary = evaluate_retriever(mock_retriever, [q1, q2], max_k=5)
        assert summary.total_queries == 2
        assert summary.overall_recall_at_3 == 0.5
        assert summary.overall_recall_at_5 == 0.5
        assert summary.overall_mrr == 0.5
        assert "cat1" in summary.category_metrics

        report = format_benchmark_report(summary)
        assert "NAVY RAG BENCHMARK EVALUATION REPORT" in report
        assert "cat1" in report


class TestEvalCLI:
    def test_cli_parser_registers_eval(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["eval", "--help"])
        assert exc.value.code == 0
