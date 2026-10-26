"""On-demand PDF page extraction for policy citation inspection."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import yaml
from pypdf import PdfReader, PdfWriter

from navy_rag.config import settings

logger = logging.getLogger(__name__)


def _clean_stem(val: str) -> str:
    return re.sub(r"[^a-z0-9]", "", val.lower())


def get_document_metadata(filename: str) -> dict[str, str]:
    """Retrieve canonical title and instruction number from manifest by filename."""
    manifest_path = settings.manifest_path
    if not manifest_path.is_file() or not filename:
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        docs = data.get("documents", [])
        target_clean = _clean_stem(Path(filename).stem)

        for d in docs:
            cand_fn = d.get("filename", "")
            cand_clean = _clean_stem(Path(cand_fn).stem)
            if (
                cand_fn == filename
                or cand_clean == target_clean
                or cand_clean in target_clean
                or target_clean in cand_clean
            ):
                return {
                    "title": d.get("title", ""),
                    "instruction_number": d.get("instruction_number", ""),
                    "source_agency": d.get("source_agency", ""),
                }
    except Exception as exc:
        logger.debug("Failed reading manifest for %s: %s", filename, exc)

    from navy_rag.ingestion.loader import _derive_subcorpora_metadata

    sub_meta = _derive_subcorpora_metadata(Path(filename))
    if sub_meta:
        return {
            "title": sub_meta.get("title", ""),
            "instruction_number": sub_meta.get("instruction_number", ""),
            "source_agency": sub_meta.get("source_agency", ""),
        }

    return {}


def render_pdf_page_image(filename: str, page_num: int, scale: float = 2.0) -> Any | None:
    """Render a 1-indexed PDF page as a high-resolution PIL Image."""
    path = resolve_pdf_path(filename)
    if not path:
        return None

    try:
        doc = pdfium.PdfDocument(str(path))
        if 1 <= page_num <= len(doc):
            page = doc[page_num - 1]
            return page.render(scale=scale).to_pil()
        logger.warning("Page %d out of range (1-%d) for %s", page_num, len(doc), filename)
    except Exception as exc:
        logger.error("Failed rendering page %d of %s to image: %s", page_num, filename, exc)
    return None


def resolve_pdf_path(filename: str) -> Path | None:
    """Locate a PDF file in Bronze raw storage, supporting aliases and normalized filenames."""
    if not filename:
        return None

    # Direct match in raw storage, crawled sub-corpora, or data directories
    candidates = [
        settings.raw_data_dir / filename,
        settings.raw_data_dir / "milpersman" / filename,
        settings.raw_data_dir / "uniform_regs" / filename,
        settings.data_dir / "public" / filename,
        settings.data_dir / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path

    target_clean = _clean_stem(Path(filename).stem)
    if not target_clean:
        return None

    # Search raw storage and sub-corpora for matching or normalized stem (e.g. 11308k vs 1130-8)
    if settings.raw_data_dir.is_dir():
        for path in settings.raw_data_dir.rglob("*.pdf"):
            cand_clean = _clean_stem(path.stem)
            if (
                cand_clean == target_clean
                or cand_clean in target_clean
                or target_clean in cand_clean
            ):
                return path

    return None


def get_page_text(filename: str, page_num: int) -> str | None:
    """Extract clean text directly from a specified 1-indexed PDF page."""
    path = resolve_pdf_path(filename)
    if not path:
        return None

    try:
        reader = PdfReader(str(path))
        if 1 <= page_num <= len(reader.pages):
            return reader.pages[page_num - 1].extract_text().strip()
    except Exception as exc:
        logger.error("Failed extracting text from %s page %d: %s", filename, page_num, exc)
    return None


def extract_pdf_pages(filename: str, pages: list[int]) -> bytes | None:
    """Extract specified 1-indexed pages from a PDF and return the sliced PDF bytes."""
    path = resolve_pdf_path(filename)
    if not path:
        logger.warning("PDF file '%s' not found for page extraction", filename)
        return None

    valid_pages = [p for p in pages if isinstance(p, int) and p > 0]
    if not valid_pages:
        return None

    try:
        reader = PdfReader(str(path))
        writer = PdfWriter()
        total_pages = len(reader.pages)

        pages_added = 0
        for page_num in valid_pages:
            zero_idx = page_num - 1
            if 0 <= zero_idx < total_pages:
                writer.add_page(reader.pages[zero_idx])
                pages_added += 1
            else:
                logger.warning(
                    "Page %d out of bounds for %s (total: %d)",
                    page_num,
                    filename,
                    total_pages,
                )

        if pages_added == 0:
            return None

        out_buffer = io.BytesIO()
        writer.write(out_buffer)
        return out_buffer.getvalue()
    except Exception as exc:
        logger.error("Failed extracting pages %s from %s: %s", valid_pages, filename, exc)
        return None
