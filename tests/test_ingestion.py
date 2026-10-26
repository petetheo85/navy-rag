"""Unit tests for PDF loading, instruction number extraction, and chunking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from navy_rag.ingestion.chunker import chunk_documents
from navy_rag.ingestion.loader import (
    _audience_from_path,
    _derive_subcorpora_metadata,
    _extract_instruction_number,
    discover_pdfs,
)


class TestInstructionNumberExtraction:
    def test_milpersman_pattern(self):
        text = "Per MILPERSMAN 1000-010, members must..."
        assert _extract_instruction_number(text) == "MILPERSMAN 1000-010"

    def test_bupersinst_pattern(self):
        text = "See BUPERSINST 1430.16G for advancement criteria."
        assert _extract_instruction_number(text) == "BUPERSINST 1430.16G"

    def test_no_match_returns_none(self):
        assert _extract_instruction_number("No instruction here.") is None


class TestAudienceFromPath:
    def test_public_folder(self, tmp_path):
        pdf = tmp_path / "public" / "test.pdf"
        pdf.parent.mkdir()
        pdf.touch()
        with patch("navy_rag.ingestion.loader.settings") as mock_settings:
            mock_settings.allowed_audiences = ["public", "internal"]
            assert _audience_from_path(pdf) == "public"

    def test_raw_folder(self, tmp_path):
        pdf = tmp_path / "raw" / "test.pdf"
        pdf.parent.mkdir()
        pdf.touch()
        assert _audience_from_path(pdf) == "public"

    def test_unknown_folder_raises(self, tmp_path):
        pdf = tmp_path / "classified" / "test.pdf"
        pdf.parent.mkdir()
        pdf.touch()
        with patch("navy_rag.ingestion.loader.settings") as mock_settings:
            mock_settings.allowed_audiences = ["public", "internal"]
            with pytest.raises(ValueError, match="unrecognized folder"):
                _audience_from_path(pdf)


class TestChunking:
    def _make_doc(self, content: str, filename: str = "test.pdf") -> Document:
        return Document(
            page_content=content,
            metadata={"filename": filename, "audience": "public", "page": 1},
        )

    def test_short_doc_produces_one_chunk(self):
        doc = self._make_doc("Short content.")
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_long_doc_splits(self):
        doc = self._make_doc("word " * 500)
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_metadata_preserved_on_chunks(self):
        doc = self._make_doc("word " * 500)
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert chunk.metadata["filename"] == "test.pdf"
            assert chunk.metadata["audience"] == "public"

    def test_chunk_index_assigned(self):
        doc = self._make_doc("word " * 500)
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_input(self):
        assert chunk_documents([]) == []


class TestDiscoverPdfs:
    def test_discovers_pdfs(self, tmp_path):
        pub = tmp_path / "public"
        pub.mkdir()
        (pub / "doc1.pdf").touch()
        (pub / "doc2.pdf").touch()
        (pub / "not_a_pdf.txt").touch()

        with patch("navy_rag.ingestion.loader.settings") as mock_settings:
            mock_settings.allowed_audiences = ["public"]
            mock_settings.data_dir = tmp_path
            pdfs = discover_pdfs(tmp_path)

        assert len(pdfs) == 2
        assert all(p.suffix == ".pdf" for p in pdfs)

    def test_discovers_pdfs_in_raw(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "instruction1.pdf").touch()
        (raw / "instruction2.pdf").touch()
        (raw / "notes.txt").touch()

        with patch("navy_rag.ingestion.loader.settings") as mock_settings:
            mock_settings.allowed_audiences = ["public"]
            mock_settings.data_dir = tmp_path
            pdfs = discover_pdfs(tmp_path)

        assert len(pdfs) == 2
        assert all(p.suffix == ".pdf" for p in pdfs)

    def test_discovers_subcorpora_pdfs(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "root.pdf").touch()
        uniform = raw / "uniform_regs"
        uniform.mkdir()
        (uniform / "navpers-15665j-ch2.pdf").touch()
        mil = raw / "milpersman"
        mil.mkdir()
        (mil / "milpersman-1100-011.pdf").touch()

        with patch("navy_rag.ingestion.loader.settings") as mock_settings:
            mock_settings.allowed_audiences = ["public"]
            mock_settings.data_dir = tmp_path
            pdfs = discover_pdfs(tmp_path)

        assert len(pdfs) == 3
        filenames = {p.name for p in pdfs}
        assert filenames == {"root.pdf", "navpers-15665j-ch2.pdf", "milpersman-1100-011.pdf"}

    def test_missing_folder_returns_empty(self, tmp_path):
        with patch("navy_rag.ingestion.loader.settings") as mock_settings:
            mock_settings.allowed_audiences = ["public"]
            mock_settings.data_dir = tmp_path
            assert discover_pdfs(tmp_path) == []


class TestSubcorporaMetadata:
    def test_uniform_regs_metadata(self):
        meta = _derive_subcorpora_metadata(Path("data/raw/uniform_regs/navpers-15665j-ch2.pdf"))
        assert meta["doc_id"] == "navpers_15665j_ch2"
        assert meta["base_ssic"] == "NAVPERS 15665"
        assert meta["revision"] == "J"
        assert meta["instruction_number"] == "NAVPERS 15665J Ch. 2"
        assert meta["is_latest"] is True

    def test_milpersman_metadata(self):
        meta = _derive_subcorpora_metadata(Path("data/raw/milpersman/milpersman-1100-011.pdf"))
        assert meta["doc_id"] == "milpersman_1100_011"
        assert meta["base_ssic"] == "MILPERSMAN 1100-011"
        assert meta["revision"] == "BASE"
        assert meta["instruction_number"] == "MILPERSMAN 1100-011"
