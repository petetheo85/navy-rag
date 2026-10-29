"""Unit tests for context formatting, scope routing, and citation cleanup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from navy_rag.chain.rag_chain import _filter_sources_and_clean_answer, _format_context


class TestFormatContext:
    def _doc(self, content: str, **meta) -> Document:
        return Document(page_content=content, metadata=meta)

    def test_includes_instruction_number(self):
        doc = self._doc("Pay is $2,000/mo.", instruction_number="MILPERSMAN 7000-010", page=3)
        result = _format_context([doc])
        assert "MILPERSMAN 7000-010" in result
        assert "p.3" in result

    def test_falls_back_to_source_when_no_instruction_number(self):
        doc = self._doc("Some content.", source="Navy Pay Guide", page=1)
        result = _format_context([doc])
        assert "Navy Pay Guide" in result

    def test_multiple_docs_separated(self):
        docs = [
            self._doc("Content A.", source="Doc A"),
            self._doc("Content B.", source="Doc B"),
        ]
        result = _format_context(docs)
        assert "---" in result
        assert "Content A." in result
        assert "Content B." in result

    def test_empty_docs_returns_empty_string(self):
        assert _format_context([]) == ""


class TestScopeClassification:
    @patch("navy_rag.chain.graph.get_llm")
    def test_in_scope_label_passes_through(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.__or__ = lambda self, other: MagicMock(invoke=lambda x: "in_scope_with_docs")
        mock_get_llm.return_value = mock_llm

        from navy_rag.chain.graph import classify_scope

        state = {
            "question": "What is the Navy tattoo policy?",
            "scope": "",
            "retrieved_docs": [],
            "retrieval_sufficient": False,
            "answer": "",
            "sources": [],
            "chat_history": [],
            "audience": None,
            "retry_count": 0,
        }
        with patch("navy_rag.chain.graph.SCOPE_CLASSIFIER_PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "in_scope_with_docs"
            mock_prompt.__or__ = lambda self, other: mock_chain

            result = classify_scope(state)
            assert result == {"scope": "in_scope_with_docs"}


class TestRecruiterReferralOnWeakRetrieval:
    @patch("navy_rag.chain.graph.invoke_rag")
    def test_uncited_grounded_response_refers_to_recruiter(self, mock_invoke):
        from navy_rag.chain.graph import RECRUITER_REFERRAL_MESSAGE, generate_grounded

        mock_invoke.return_value = {
            "answer": "Some answer without citations.",
            "sources": [],
            "has_context": False,
        }
        state = {
            "question": "What is the enlistment bonus for nuclear field?",
            "scope": "in_scope_with_docs",
            "retrieved_docs": [],
            "retrieval_sufficient": True,
            "answer": "",
            "sources": [],
            "chat_history": [],
            "audience": "public",
            "retry_count": 0,
        }
        result = generate_grounded(state)
        assert result["scope"] == "referred"
        assert result["sources"] == []
        assert "recruiter" in result["answer"].lower()
        assert result["answer"] == RECRUITER_REFERRAL_MESSAGE

    def test_route_after_evaluation_refers_on_insufficient_retrieval(self):
        from navy_rag.chain.graph import route_after_evaluation

        state_first_attempt = {
            "retrieval_sufficient": False,
            "retry_count": 0,
        }
        assert route_after_evaluation(state_first_attempt) == "reformulate_query"

        state_retried = {
            "retrieval_sufficient": False,
            "retry_count": 1,
        }
        assert route_after_evaluation(state_retried) == "refer_recruiter"

    def test_refer_recruiter_node(self):
        from navy_rag.chain.graph import RECRUITER_REFERRAL_MESSAGE, refer_recruiter

        state = {"question": "Any question"}
        result = refer_recruiter(state)
        assert result["scope"] == "referred"
        assert result["sources"] == []
        assert result["answer"] == RECRUITER_REFERRAL_MESSAGE


class TestContextFormatEdgeCases:
    def test_missing_page_omitted(self):
        doc = Document(page_content="text", metadata={"source": "DocA"})
        assert "p." not in _format_context([doc])


class TestFilterSourcesAndCleanAnswer:
    def test_filters_to_cited_pages_from_sources_used(self):
        raw_answer = (
            "The maximum enlistment age is 41 for active duty.\n\nSOURCES_USED: p. 137, 392, 431"
        )
        sources = [
            {"source": "COMNAVCRUITCOMINST 1130.8K", "filename": "doc.pdf", "page": 52},
            {"source": "COMNAVCRUITCOMINST 1130.8K", "filename": "doc.pdf", "page": 137},
            {"source": "COMNAVCRUITCOMINST 1130.8K", "filename": "doc.pdf", "page": 392},
            {"source": "COMNAVCRUITCOMINST 1130.8K", "filename": "doc.pdf", "page": 431},
            {"source": "COMNAVCRUITCOMINST 1130.8K", "filename": "doc.pdf", "page": 537},
        ]
        answer, filtered = _filter_sources_and_clean_answer(raw_answer, sources)
        assert answer == "The maximum enlistment age is 41 for active duty."
        assert len(filtered) == 3
        assert [s["page"] for s in filtered] == [137, 392, 431]

    def test_cleans_trailing_parenthetical_source(self):
        raw_answer = (
            "You can enlist up to age 41.\n\n(Source: COMNAVCRUITCOMINST 1130.8K, p. 137, 392)"
        )
        sources = [
            {"source": "COMNAVCRUITCOMINST 1130.8K", "filename": "doc.pdf", "page": 52},
            {"source": "COMNAVCRUITCOMINST 1130.8K", "filename": "doc.pdf", "page": 137},
            {"source": "COMNAVCRUITCOMINST 1130.8K", "filename": "doc.pdf", "page": 392},
        ]
        answer, filtered = _filter_sources_and_clean_answer(raw_answer, sources)
        assert answer == "You can enlist up to age 41."
        assert len(filtered) == 2
        assert [s["page"] for s in filtered] == [137, 392]

    def test_handles_page_range(self):
        raw_answer = "Policy details.\n\nSOURCES_USED: pp. 10-12"
        sources = [
            {"source": "Doc", "filename": "doc.pdf", "page": 9},
            {"source": "Doc", "filename": "doc.pdf", "page": 10},
            {"source": "Doc", "filename": "doc.pdf", "page": 11},
            {"source": "Doc", "filename": "doc.pdf", "page": 12},
            {"source": "Doc", "filename": "doc.pdf", "page": 13},
        ]
        answer, filtered = _filter_sources_and_clean_answer(raw_answer, sources)
        assert [s["page"] for s in filtered] == [10, 11, 12]

    def test_handles_none_sources(self):
        raw_answer = "No policy was found.\n\nSOURCES_USED: none"
        sources = [{"source": "Doc", "filename": "doc.pdf", "page": 5}]
        answer, filtered = _filter_sources_and_clean_answer(raw_answer, sources)
        assert answer == "No policy was found."
        assert filtered == []

    def test_fallback_when_no_citations(self):
        raw_answer = "General statement without citations."
        sources = [{"source": "Doc", "filename": "doc.pdf", "page": 5}]
        answer, filtered = _filter_sources_and_clean_answer(raw_answer, sources)
        assert answer == "General statement without citations."
        assert filtered == sources

    def test_filters_by_doc_id_preventing_same_page_collision(self):
        raw_answer = "RE-4 is non-waiverable [DOC-1].\n\nSOURCES_USED: DOC-1"
        sources = [
            {"source": "MILPERSMAN 1133-020", "filename": "milpersman.pdf", "page": 2},
            {"source": "DoDI 6130.03 Vol. 1", "filename": "dodi.pdf", "page": 2},
        ]
        answer, filtered = _filter_sources_and_clean_answer(raw_answer, sources)
        assert answer == "RE-4 is non-waiverable."
        assert len(filtered) == 1
        assert filtered[0]["source"] == "MILPERSMAN 1133-020"


class TestConsolidationAndEvaluation:
    def test_consolidate_docs_merges_same_page_in_chunk_order(self):
        from navy_rag.chain.rag_chain import _consolidate_docs_and_sources

        d2 = Document(
            page_content="Second half of table.",
            metadata={"filename": "prt.pdf", "page": 4, "chunk_index": 11, "source": "PRT"},
        )
        d1 = Document(
            page_content="First half of table.",
            metadata={"filename": "prt.pdf", "page": 4, "chunk_index": 10, "source": "PRT"},
        )
        consolidated, sources = _consolidate_docs_and_sources([d2, d1])
        assert len(consolidated) == 1
        assert len(sources) == 1
        assert consolidated[0].metadata["doc_ref"] == "DOC-1"
        assert consolidated[0].page_content.startswith("First half of table.")
        assert "Second half of table." in consolidated[0].page_content

    def test_evaluate_retrieval_rejects_unmatched_hybrid_chunks(self):
        from navy_rag.chain.graph import evaluate_retrieval

        unmatched_docs = [
            Document(
                page_content="Completely unrelated culinary text.",
                metadata={"bm25_hit": False, "rrf_score": 0.008},
            ),
            Document(
                page_content="Another unrelated baking paragraph.",
                metadata={"bm25_hit": False, "rrf_score": 0.007},
            ),
        ]
        state = {
            "question": "What are the Nuclear Field ASVAB requirements?",
            "retrieved_docs": unmatched_docs,
        }
        assert evaluate_retrieval(state) == {"retrieval_sufficient": False}

    def test_system_specs_retrievable_from_settings(self):
        from navy_rag.config import settings

        specs = settings.get_system_specs()
        assert "model" in specs
        assert "embeddings" in specs
        assert "vector_store" in specs
        assert "retrieval" in specs
        assert "query_engine" in specs
        assert "citations" in specs
        assert "HYBRID RRF" in specs["retrieval"]
