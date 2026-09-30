"""Unit tests for the vector store retriever factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestRetrieverFactory:
    def _make_mock_vectorstore(self):
        mock_retriever = MagicMock()
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        return mock_vs, mock_retriever

    @patch("navy_rag.retrieval.retriever.get_vectorstore")
    def test_returns_retriever(self, mock_get_vs):
        mock_vs, mock_retriever = self._make_mock_vectorstore()
        mock_get_vs.return_value = mock_vs

        from navy_rag.retrieval.retriever import get_retriever

        retriever = get_retriever()

        mock_vs.as_retriever.assert_called_once()
        assert retriever is mock_retriever

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
