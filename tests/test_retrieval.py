"""Unit tests for the vector store retriever factory, hybrid RRF, and query processor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from navy_rag.retrieval.hybrid import HybridRetriever, clear_bm25_cache, tokenize_military_text
from navy_rag.retrieval.query_processor import (
    contextualize_followup_query,
    expand_military_query,
)


class _DummyDenseRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self.docs


class TestRetrieverFactory:
    def _make_mock_vectorstore(self):
        mock_retriever = _DummyDenseRetriever(docs=[])
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        return mock_vs, mock_retriever

    @patch("navy_rag.retrieval.retriever.get_vectorstore")
    def test_returns_hybrid_retriever_by_default(self, mock_get_vs):
        mock_vs, mock_retriever = self._make_mock_vectorstore()
        mock_get_vs.return_value = mock_vs

        from navy_rag.retrieval.retriever import get_retriever

        retriever = get_retriever()

        mock_vs.as_retriever.assert_called_once()
        assert isinstance(retriever, HybridRetriever)
        assert retriever.dense_retriever is mock_retriever

    @patch("navy_rag.retrieval.retriever.get_vectorstore")
    def test_audience_filter_applied(self, mock_get_vs):
        mock_vs, _ = self._make_mock_vectorstore()
        mock_get_vs.return_value = mock_vs

        from navy_rag.retrieval.retriever import get_retriever

        get_retriever(audience="public")

        _, kwargs = mock_vs.as_retriever.call_args
        assert kwargs["search_kwargs"]["filter"] == {"audience": "public"}

    @patch("navy_rag.retrieval.retriever.get_vectorstore")
    def test_no_filter_when_no_audience(self, mock_get_vs):
        mock_vs, _ = self._make_mock_vectorstore()
        mock_get_vs.return_value = mock_vs

        from navy_rag.retrieval.retriever import get_retriever

        get_retriever()

        _, kwargs = mock_vs.as_retriever.call_args
        assert "filter" not in kwargs["search_kwargs"]

    @patch("navy_rag.retrieval.retriever.get_vectorstore")
    def test_custom_k(self, mock_get_vs):
        mock_vs, _ = self._make_mock_vectorstore()
        mock_get_vs.return_value = mock_vs

        from navy_rag.retrieval.retriever import get_retriever

        get_retriever(k=10)

        _, kwargs = mock_vs.as_retriever.call_args
        assert kwargs["search_kwargs"]["k"] == 10

    @patch("navy_rag.retrieval.retriever.get_vectorstore")
    def test_mmr_fetch_k_included(self, mock_get_vs):
        mock_vs, _ = self._make_mock_vectorstore()
        mock_get_vs.return_value = mock_vs

        from navy_rag.retrieval.retriever import get_retriever

        get_retriever(method="mmr")

        _, kwargs = mock_vs.as_retriever.call_args
        assert "fetch_k" in kwargs["search_kwargs"]
        assert kwargs["search_type"] == "mmr"

    @patch("navy_rag.retrieval.retriever.get_vectorstore")
    def test_similarity_method(self, mock_get_vs):
        mock_vs, _ = self._make_mock_vectorstore()
        mock_get_vs.return_value = mock_vs

        from navy_rag.retrieval.retriever import get_retriever

        get_retriever(method="similarity")

        _, kwargs = mock_vs.as_retriever.call_args
        assert kwargs["search_type"] == "similarity"
        assert "fetch_k" not in kwargs["search_kwargs"]


class TestHybridRRFAndDiversity:
    def setup_method(self):
        clear_bm25_cache()

    def test_military_tokenizer_preserves_designators(self):
        tokens = tokenize_military_text("Under MILPERSMAN 1133-020, RE-4 and NAVMED 6110/4 apply.")
        assert "1133-020" in tokens
        assert "1133" in tokens
        assert "020" in tokens
        assert "re-4" in tokens
        assert "6110/4" in tokens

    def test_hybrid_rrf_boosts_lexical_instruction_match_and_caps_per_doc(self):
        neocs_chunks = [
            Document(
                page_content=f"General occupational tasks for Sailor rating {i}.",
                metadata={
                    "doc_id": "neocs_vol_1",
                    "filename": "navpers-18068f-vol1.pdf",
                    "instruction_number": "NAVPERS 18068F Vol. I",
                    "page": i,
                    "chunk_index": i,
                },
            )
            for i in range(1, 6)
        ]
        re4_doc = Document(
            page_content="An RE-4 reenlistment code signifies ineligibility for reenlistment.",
            metadata={
                "doc_id": "milpersman_1133_020",
                "filename": "milpersman-1133-020.pdf",
                "instruction_number": "MILPERSMAN 1133-020",
                "page": 2,
                "chunk_index": 1,
            },
        )

        dense_retriever = _DummyDenseRetriever(docs=[*neocs_chunks, re4_doc])
        mock_vs = MagicMock()
        all_docs = [*neocs_chunks, re4_doc]
        mock_vs._collection.get.return_value = {
            "documents": [d.page_content for d in all_docs],
            "metadatas": [d.metadata for d in all_docs],
        }

        hybrid = HybridRetriever(
            dense_retriever=dense_retriever,
            vectorstore=mock_vs,
            k=4,
            max_chunks_per_doc=2,
        )

        results = hybrid.invoke("What does an RE-4 reenlistment code mean?")
        assert len(results) == 4
        # RE-4 chunk is boosted into the top results and marked with bm25_hit=True
        re4_hits = [d for d in results if d.metadata.get("doc_id") == "milpersman_1133_020"]
        assert len(re4_hits) == 1
        assert re4_hits[0].metadata["bm25_hit"] is True
        assert re4_hits[0].metadata["rrf_score"] > 0


class TestQueryProcessor:
    def test_expand_military_query_adds_authoritative_directives(self):
        expanded = expand_military_query("What is the Navy neck tattoo policy?")
        assert "NAVPERS" in expanded
        assert "15665J" in expanded

    def test_contextualize_followup_query_prepends_prior_topic(self):
        history = [
            {"role": "user", "content": "Is asthma after age 13 disqualifying?"},
            {"role": "assistant", "content": "Yes, under DoDI 6130.03."},
        ]
        result = contextualize_followup_query("Can that be waived?", history)
        assert "asthma" in result.lower()
        assert "Can that be waived?" in result

    def test_standalone_query_unchanged_with_history(self):
        history = [
            {"role": "user", "content": "Is asthma after age 13 disqualifying?"},
            {"role": "assistant", "content": "Yes, under DoDI 6130.03."},
        ]
        standalone = "What are the minimum ASVAB line scores for Nuclear Field pipeline?"
        assert contextualize_followup_query(standalone, history) == standalone
