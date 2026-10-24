"""Unit tests for the PDF page slicer and citation inspection utility."""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from navy_rag.ingestion.pdf_slicer import (
    extract_pdf_pages,
    get_page_text,
    resolve_pdf_path,
)


def _create_dummy_pdf(path: Path, num_pages: int = 3) -> None:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as fh:
        writer.write(fh)


def test_resolve_pdf_path(tmp_path: Path, monkeypatch):
    from navy_rag.config import settings

    monkeypatch.setattr(settings, "base_dir", tmp_path)

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pdf = raw_dir / "comnavcruitcominst-1130-8.pdf"
    pdf.touch()

    assert resolve_pdf_path("comnavcruitcominst-1130-8.pdf") == pdf
    # Normalized alias should resolve
    assert resolve_pdf_path("comnavcruitcominst-11308k.pdf") == pdf
    assert resolve_pdf_path("nonexistent.pdf") is None


def test_get_page_text_returns_string_or_none(tmp_path: Path, monkeypatch):
    from navy_rag.config import settings

    monkeypatch.setattr(settings, "base_dir", tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    dummy_pdf = raw_dir / "sample.pdf"
    _create_dummy_pdf(dummy_pdf, num_pages=2)

    # Empty blank page returns empty string, not None
    assert get_page_text("sample.pdf", 1) == ""
    assert get_page_text("sample.pdf", 99) is None
    assert get_page_text("missing.pdf", 1) is None


def test_extract_pdf_pages_single_page(tmp_path: Path, monkeypatch):
    from navy_rag.config import settings

    monkeypatch.setattr(settings, "base_dir", tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    dummy_pdf = raw_dir / "multi_page.pdf"
    _create_dummy_pdf(dummy_pdf, num_pages=5)

    sliced_bytes = extract_pdf_pages("multi_page.pdf", [3])
    assert sliced_bytes is not None

    reader = PdfReader(io.BytesIO(sliced_bytes))
    assert len(reader.pages) == 1


def test_extract_pdf_pages_multiple_pages(tmp_path: Path, monkeypatch):
    from navy_rag.config import settings

    monkeypatch.setattr(settings, "base_dir", tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    dummy_pdf = raw_dir / "multi_page.pdf"
    _create_dummy_pdf(dummy_pdf, num_pages=5)

    sliced_bytes = extract_pdf_pages("multi_page.pdf", [2, 4])
    assert sliced_bytes is not None

    reader = PdfReader(io.BytesIO(sliced_bytes))
    assert len(reader.pages) == 2


def test_extract_pdf_pages_out_of_bounds(tmp_path: Path, monkeypatch):
    from navy_rag.config import settings

    monkeypatch.setattr(settings, "base_dir", tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    dummy_pdf = raw_dir / "multi_page.pdf"
    _create_dummy_pdf(dummy_pdf, num_pages=2)

    assert extract_pdf_pages("multi_page.pdf", [10]) is None
    assert extract_pdf_pages("multi_page.pdf", []) is None
    assert extract_pdf_pages("missing.pdf", [1]) is None
