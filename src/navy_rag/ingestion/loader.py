"""PDF document loader and metadata extraction for naval instructions."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

# TODO: migrate to langchain-pypdf once langchain-community splits
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pypdf import PdfReader

from navy_rag.config import settings

logger = logging.getLogger(__name__)

# Standard Navy/DoD instruction numbering formats
_INSTRUCTION_RE = re.compile(
    r"\b(MILPERSMAN|BUPERSINST|OPNAVINST|SECNAVINST|COMNAVCRUITCOMINST|NAVADMIN|"
    r"DODI|DODD|JAGINST|BUMEDINST|NAVPERS)\s*[\d\-\.]+[A-Z]?\b",
    re.IGNORECASE,
)


def _extract_instruction_number(text: str) -> str | None:
    match = _INSTRUCTION_RE.search(text)
    return match.group(0).upper() if match else None


def _pdf_title(path: Path) -> str | None:
    try:
        reader = PdfReader(str(path))
        meta = reader.metadata
        if meta and meta.title and meta.title.strip().lower() not in ("", "untitled"):
            return meta.title.strip()
    except Exception as exc:
        logger.debug("Failed reading PDF metadata from %s: %s", path.name, exc)
    return None


def _audience_from_path(path: Path) -> str:
    folder = path.parent.name
    if folder not in settings.allowed_audiences:
        raise ValueError(
            f"'{path}' is in an unrecognized folder '{folder}'. "
            f"Move it to one of: {settings.allowed_audiences}"
        )
    return folder


def load_pdf(path: Path) -> list[Document]:
    """Parse a single PDF into 1-indexed page Documents with source metadata."""
    logger.info("Loading %s", path.name)

    audience = _audience_from_path(path)
    title = _pdf_title(path) or path.stem

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    if not pages:
        logger.warning("No extractable pages in %s; skipping", path.name)
        return []

    first_page_text = pages[0].page_content if pages else ""
    instruction_number = _extract_instruction_number(first_page_text)

    base_metadata = {
        "audience": audience,
        "source": title,
        "filename": path.name,
        "filepath": str(path.relative_to(settings.base_dir)),
        "instruction_number": instruction_number or "",
        "ingested_at": datetime.now(UTC).isoformat(),
    }

    docs: list[Document] = []
    for page in pages:
        page_metadata = {
            **base_metadata,
            "page": page.metadata.get("page", 0) + 1,
        }
        docs.append(Document(page_content=page.page_content, metadata=page_metadata))

    logger.info("Loaded %d pages from %s", len(docs), path.name)
    return docs


def discover_pdfs(data_dir: Path | None = None) -> list[Path]:
    """Collect PDF paths across all configured audience directories."""
    root = data_dir or settings.data_dir
    found: list[Path] = []

    for audience in settings.allowed_audiences:
        audience_dir = root / audience
        if not audience_dir.exists():
            continue
        pdfs = sorted(audience_dir.glob("*.pdf"))
        logger.info("Discovered %d PDF(s) in %s/", len(pdfs), audience)
        found.extend(pdfs)

    return found


def load_all(data_dir: Path | None = None) -> list[Document]:
    """Discover and parse all PDFs across audience directories."""
    pdfs = discover_pdfs(data_dir)
    if not pdfs:
        logger.warning("No PDF documents discovered in %s", data_dir or settings.data_dir)
        return []

    docs: list[Document] = []
    for pdf_path in pdfs:
        try:
            docs.extend(load_pdf(pdf_path))
        except Exception as exc:
            logger.error("Failed loading %s: %s", pdf_path.name, exc)

    logger.info("Loaded %d total pages across %d document(s)", len(docs), len(pdfs))
    return docs
