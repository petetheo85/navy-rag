"""PDF document loader and metadata extraction for naval instructions."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

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


@lru_cache(maxsize=1)
def _load_manifest_metadata() -> dict[str, dict[str, Any]]:
    manifest_path = settings.manifest_path
    if not manifest_path.is_file():
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        docs = raw.get("documents", [])
        return {
            doc["filename"]: {
                "title": doc.get("title", ""),
                "instruction_number": doc.get("instruction_number", ""),
                "source_agency": doc.get("source_agency", ""),
                "audience": doc.get("audience", "public"),
                "doc_id": doc.get("doc_id", ""),
                "base_ssic": doc.get("base_ssic", doc.get("instruction_number", "")),
                "revision": doc.get("revision") if doc.get("revision") is not None else "BASE",
                "change": int(doc.get("change", 0)),
                "is_latest": bool(doc.get("is_latest", True)),
            }
            for doc in docs
            if "filename" in doc
        }
    except Exception as exc:
        logger.debug("Failed reading manifest metadata from %s: %s", manifest_path, exc)
        return {}


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
    if folder in ("raw", "uniform_regs", "milpersman") or "raw" in path.parts:
        return "public"
    if folder not in settings.allowed_audiences:
        raise ValueError(
            f"'{path}' is in an unrecognized folder '{folder}'. "
            f"Move it to one of: {settings.allowed_audiences} or raw"
        )
    return folder


def _derive_subcorpora_metadata(path: Path) -> dict[str, Any]:
    name = path.stem.lower()
    parent_folder = path.parent.name

    # NAVPERS 15665J Uniform Regulations chapters (e.g., navpers-15665j-ch2.pdf)
    ch_match = re.search(r"navpers-15665j-ch(\d+)", name) or re.search(r"ch(\d+)", name)
    if ch_match and (parent_folder == "uniform_regs" or "navpers-15665j" in name):
        ch_num = ch_match.group(1)
        return {
            "title": f"NAVPERS 15665J Uniform Regulations - Chapter {ch_num}",
            "instruction_number": f"NAVPERS 15665J Ch. {ch_num}",
            "source_agency": "MyNavyHR Uniform Regulations",
            "audience": "public",
            "doc_id": f"navpers_15665j_ch{ch_num}",
            "base_ssic": "NAVPERS 15665",
            "revision": "J",
            "change": 0,
            "is_latest": True,
        }

    # MILPERSMAN Articles (e.g., milpersman-1100-011.pdf)
    mil_match = re.search(r"milpersman-(\d{4}-\d{3})", name)
    if mil_match or parent_folder == "milpersman":
        art_num = mil_match.group(1) if mil_match else name.replace("milpersman-", "")
        return {
            "title": f"MILPERSMAN {art_num}",
            "instruction_number": f"MILPERSMAN {art_num}",
            "source_agency": "MyNavyHR",
            "audience": "public",
            "doc_id": f"milpersman_{art_num.replace('-', '_')}",
            "base_ssic": f"MILPERSMAN {art_num}",
            "revision": "BASE",
            "change": 0,
            "is_latest": True,
        }

    return {}


def load_pdf(path: Path) -> list[Document]:
    """Parse a single PDF into 1-indexed page Documents with source metadata."""
    logger.info("Loading %s", path.name)

    manifest_meta = _load_manifest_metadata().get(path.name) or _derive_subcorpora_metadata(path)

    audience = manifest_meta.get("audience") or _audience_from_path(path)
    title = manifest_meta.get("title") or _pdf_title(path) or path.stem

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    if not pages:
        logger.warning("No extractable pages in %s; skipping", path.name)
        return []

    first_page_text = pages[0].page_content if pages else ""
    instruction_number = (
        manifest_meta.get("instruction_number")
        or _extract_instruction_number(first_page_text)
        or ""
    )

    base_metadata = {
        "audience": audience,
        "source": title,
        "filename": path.name,
        "filepath": str(path.relative_to(settings.base_dir)),
        "instruction_number": instruction_number,
        "source_agency": manifest_meta.get("source_agency", ""),
        "doc_id": manifest_meta.get("doc_id", ""),
        "base_ssic": manifest_meta.get("base_ssic", instruction_number),
        "revision": manifest_meta.get("revision", "BASE"),
        "change": manifest_meta.get("change", 0),
        "is_latest": manifest_meta.get("is_latest", True),
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
    """Collect PDF paths across raw Bronze storage and audience directories."""
    root = data_dir or settings.data_dir
    found: list[Path] = []
    seen_names: set[str] = set()

    raw_dir = root / "raw"
    if raw_dir.exists():
        raw_pdfs = sorted(raw_dir.glob("*.pdf"))
        logger.info("Discovered %d PDF(s) in raw/", len(raw_pdfs))
        for p in raw_pdfs:
            found.append(p)
            seen_names.add(p.name)

        for sub_name in ("uniform_regs", "milpersman"):
            sub_dir = raw_dir / sub_name
            if sub_dir.is_dir():
                sub_pdfs = sorted(sub_dir.glob("*.pdf"))
                new_sub = [p for p in sub_pdfs if p.name not in seen_names]
                if new_sub:
                    logger.info("Discovered %d PDF(s) in raw/%s/", len(new_sub), sub_name)
                    for p in new_sub:
                        found.append(p)
                        seen_names.add(p.name)

    for audience in settings.allowed_audiences:
        audience_dir = root / audience
        if not audience_dir.exists():
            continue
        pdfs = sorted(audience_dir.glob("*.pdf"))
        new_pdfs = [p for p in pdfs if p.name not in seen_names]
        if new_pdfs:
            logger.info("Discovered %d PDF(s) in %s/", len(new_pdfs), audience)
            for p in new_pdfs:
                found.append(p)
                seen_names.add(p.name)

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
