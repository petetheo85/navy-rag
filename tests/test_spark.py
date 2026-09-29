"""Smoke tests for the distributed PySpark extraction module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pypdf import PdfWriter

pyspark = pytest.importorskip("pyspark")


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample_test_doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


def test_pdf_to_row_worker_function(sample_pdf: Path):
    from navy_rag.ingestion.spark_processor import _pdf_to_row

    rows = _pdf_to_row(str(sample_pdf))
    assert isinstance(rows, list)


@patch("navy_rag.ingestion.embedder.embed_and_store")
def test_process_with_spark_smoke(mock_embed, sample_pdf: Path):
    from navy_rag.ingestion.spark_processor import process_with_spark

    mock_embed.return_value = 0
    stored = process_with_spark([sample_pdf], parallelism=1)
    assert stored == 0
