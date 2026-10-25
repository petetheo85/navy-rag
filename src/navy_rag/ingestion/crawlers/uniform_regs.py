"""Automated recursive crawler and structured PDF typesetter for NAVPERS 15665J."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin

import requests
from fpdf import FPDF
from pydantic import BaseModel, Field
from pypdf import PdfReader

from navy_rag.config import settings
from navy_rag.ingestion.downloader import _build_http_session, resolve_ca_bundle

logger = logging.getLogger(__name__)

MYNAVYHR_BASE_URL = "https://www.mynavyhr.navy.mil"
UNIFORM_TOC_URL = (
    "https://www.mynavyhr.navy.mil/References/US-Navy-Uniforms/Uniform-Regulations/Table-of-Contents/"
)
_TIMEOUT_SECONDS = 30

CHAPTER_NAMES = {
    "Chapter-1": "CHAPTER 1: GENERAL UNIFORM REGULATIONS",
    "Chapter-2": "CHAPTER 2: GROOMING STANDARDS",
    "Chapter-3": "CHAPTER 3: UNIFORM COMPONENTS",
    "Chapter-4": "CHAPTER 4: RANK AND RATE INSIGNIA",
    "Chapter-5": "CHAPTER 5: IDENTIFICATION BADGES AND AWARDS",
    "Chapter-6": "CHAPTER 6: SPECIAL UNIFORM SITUATIONS",
    "Chapter-7": "CHAPTER 7: CIVILIAN CLOTHING REGULATIONS",
}


class UniformSection(BaseModel):
    title: str
    url: str
    chapter_key: str
    slug: str


class UniformChapterAuditRecord(BaseModel):
    chapter_key: str
    chapter_title: str
    filename: str
    file_size: int
    page_count: int
    section_count: int
    sha256: str
    timestamp: str
    status: Literal["compiled", "skipped", "failed"]
    error: str | None = None


class UniformRegsAuditManifest(BaseModel):
    generated_at: str
    chapter_count: int
    records: list[UniformChapterAuditRecord] = Field(default_factory=list)


def _sanitize_for_pdf(text: str) -> str:
    """Normalize Unicode characters to Latin-1 compatible representations for PDF typography."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u00a0": " ",
        "\u2022": "*",
        "\u2026": "...",
        "\u00bd": " 1/2",
        "\u00bc": " 1/4",
        "\u00be": " 3/4",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def discover_uniform_sections(
    session: requests.Session | None = None,
) -> dict[str, list[UniformSection]]:
    """Crawl the Table of Contents to discover all chapter subsections and nested article pages."""
    http_session = session or _build_http_session(resolve_ca_bundle())
    logger.info("Discovering NAVPERS 15665J structure from %s", UNIFORM_TOC_URL)
    resp = http_session.get(UNIFORM_TOC_URL, timeout=_TIMEOUT_SECONDS)
    resp.raise_for_status()

    chapters: dict[str, list[UniformSection]] = {}
    seen_urls: set[str] = set()

    for match in re.finditer(r"<a[^>]+href=[\"']([^#\"']+)[\"'][^>]*>(.*?)</a>", resp.text):
        raw_href = match.group(1).split("?")[0].rstrip("/")
        full_url = urljoin(MYNAVYHR_BASE_URL, raw_href)

        if "Uniform-Regulations" not in full_url or "Table-of-Contents" in full_url:
            continue

        raw_title = re.sub(r"<[^>]+>", " ", match.group(2)).strip()
        clean_title = " ".join(unescape(raw_title).replace("&nbsp;", " ").split())
        if not clean_title or clean_title.lower() == "table of contents":
            continue

        # Extract Chapter key (e.g. Chapter-1 through Chapter-7)
        ch_match = re.search(r"Chapter-(\d)", full_url, re.IGNORECASE)
        if not ch_match:
            continue

        ch_key = f"Chapter-{ch_match.group(1)}"
        slug = full_url.rstrip("/").split("/")[-1]

        # Ignore redundant root chapter page links
        if slug.lower() == ch_key.lower():
            continue

        normalized_url = full_url.rstrip("/") + "/"
        if normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)
        section = UniformSection(
            title=clean_title,
            url=normalized_url,
            chapter_key=ch_key,
            slug=slug,
        )
        chapters.setdefault(ch_key, []).append(section)

    for ch_key, sections in sorted(chapters.items()):
        logger.info("Discovered %d section(s) in %s", len(sections), ch_key)

    return chapters


def extract_section_blocks(raw_html: str) -> list[str]:
    """Parse clean article and paragraph blocks from MyNavy HR HTML markup."""
    # Find start of body content (strip site navigation and table-of-contents lists)
    match = re.search(
        r"(SECTION \d+.*?|\d{4}\..*?)(?=LEGAL & ADMINISTRATIVE LINKS|<div class=[\"']dnn_layout|$)",
        raw_html,
        re.DOTALL | re.IGNORECASE,
    )
    body_raw = match.group(1) if match else raw_html

    clean = re.sub(r"<script.*?</script>", "", body_raw, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<style.*?</style>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<br\s*/?>", "\n", clean, flags=re.IGNORECASE)
    clean = re.sub(r"</p>", "\n\n", clean, flags=re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = unescape(clean)

    blocks = [" ".join(line.split()) for line in clean.splitlines() if line.strip()]
    return blocks


class NavyDirectivePDF(FPDF):
    """Clean directive-styled PDF with running Navy header and footer."""

    def __init__(self, chapter_title: str) -> None:
        super().__init__(orientation="P", unit="mm", format="letter")
        self.chapter_title = chapter_title

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(2, 42, 58)
        header_text = f"NAVPERS 15665J | {self.chapter_title.upper()}"
        self.cell(0, 5, _sanitize_for_pdf(header_text), align="L")
        self.ln(5)
        self.set_draw_color(218, 165, 32)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(120, 120, 120)
        footer_msg = f"Page {self.page_no()} | Official Navy Uniform Regulations (MyNavy HR)"
        self.cell(0, 5, footer_msg, align="C")


def compile_chapter_pdf(
    chapter_key: str,
    sections_data: list[tuple[str, list[str]]],
    output_path: Path,
) -> Path:
    """Format and compile structured chapter sections into an authentic Navy instruction PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chapter_title = CHAPTER_NAMES.get(chapter_key, f"CHAPTER {chapter_key}")

    pdf = NavyDirectivePDF(chapter_title=chapter_title)
    pdf.set_auto_page_break(auto=True, margin=15)
    base_margin = 18.0
    pdf.set_margins(base_margin, 15, base_margin)
    pdf.add_page()

    # Title Banner Block
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(2, 42, 58)
    pdf.cell(0, 8, "NAVPERS 15665J", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(180, 130, 20)
    pdf.cell(
        0,
        6,
        "UNITED STATES NAVY UNIFORM REGULATIONS",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(
        0,
        6,
        _sanitize_for_pdf(chapter_title),
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_font("Helvetica", "I", 7)
    sync_date = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(
        0,
        5,
        f"Synchronized from MyNavy HR on {sync_date}",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)

    for sec_header, blocks in sections_data:
        # Section Header with Gold Accent Rule
        pdf.ln(3)
        pdf.set_left_margin(base_margin)
        pdf.set_x(base_margin)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(2, 42, 58)
        pdf.cell(0, 7, _sanitize_for_pdf(sec_header), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(218, 165, 32)
        pdf.set_line_width(0.4)
        pdf.line(base_margin, pdf.get_y(), pdf.w - base_margin, pdf.get_y())
        pdf.ln(3)

        for b in blocks:
            clean = _sanitize_for_pdf(b)

            article_match = re.match(r"^(\d+\.\s+[A-Z\s/]{3,}\.?)\s*(.*)", clean)
            major_match = re.match(r"^(ARTICLE\s+\d+|[1-7]\d{3}\.\s+.*)", clean)

            if article_match:
                hdr_label = article_match.group(1).strip()
                body_intro = article_match.group(2).strip()

                pdf.ln(2.5)
                pdf.set_left_margin(base_margin)
                pdf.set_x(base_margin)
                pdf.set_fill_color(240, 244, 248)
                pdf.set_font("Helvetica", "B", 9.5)
                pdf.set_text_color(2, 42, 58)
                pdf.cell(0, 6, f"  {hdr_label}", fill=True, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)

                if body_intro:
                    pdf.set_font("Helvetica", size=8.5)
                    pdf.set_text_color(25, 25, 25)
                    pdf.write(4.3, body_intro)
                    pdf.ln(5)

            elif major_match:
                pdf.ln(2.5)
                pdf.set_left_margin(base_margin)
                pdf.set_x(base_margin)
                pdf.set_fill_color(240, 244, 248)
                pdf.set_font("Helvetica", "B", 9.5)
                pdf.set_text_color(2, 42, 58)
                pdf.cell(0, 6, f"  {clean}", fill=True, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)

            elif re.match(r"^[a-z]\.\s+", clean):
                pdf.set_left_margin(base_margin + 6)
                pdf.set_x(base_margin + 6)
                m = re.match(r"^([a-z]\.(?:\s+[A-Za-z]+:?)?)\s*(.*)", clean)
                prefix = m.group(1) if m else clean[:3]
                body_text = m.group(2) if m else clean[3:]

                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_text_color(2, 42, 58)
                pdf.write(4.3, f"{prefix} ")
                pdf.set_font("Helvetica", size=8.5)
                pdf.set_text_color(25, 25, 25)
                pdf.write(4.3, body_text)
                pdf.ln(5)

            elif re.match(r"^\(\d+\)\s+", clean):
                pdf.set_left_margin(base_margin + 12)
                pdf.set_x(base_margin + 12)
                m = re.match(r"^(\(\d+\)(?:\s+[A-Za-z\s]+:?)?)\s*(.*)", clean)
                prefix = m.group(1) if m else clean[:4]
                body_text = m.group(2) if m else clean[4:]

                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_text_color(2, 42, 58)
                pdf.write(4.3, f"{prefix} ")
                pdf.set_font("Helvetica", size=8.5)
                pdf.set_text_color(25, 25, 25)
                pdf.write(4.3, body_text)
                pdf.ln(5)

            else:
                pdf.set_left_margin(base_margin)
                pdf.set_x(base_margin)
                pdf.set_font("Helvetica", size=8.5)
                pdf.set_text_color(25, 25, 25)
                pdf.write(4.3, clean)
                pdf.ln(5)

    pdf.set_left_margin(base_margin)
    pdf.output(str(output_path))
    logger.info(
        "Compiled %s (%d pages, %d bytes)",
        output_path.name,
        pdf.page_no(),
        output_path.stat().st_size,
    )
    return output_path


def crawl_uniform_regs(
    output_dir: Path | None = None,
    target_chapters: list[str] | None = None,
    force: bool = False,
) -> UniformRegsAuditManifest:
    """Crawl NAVPERS 15665J TOC, crawl all sub-pages, and compile formatted chapter PDFs."""
    target_dir = output_dir or (settings.raw_data_dir / "uniform_regs")
    target_dir.mkdir(parents=True, exist_ok=True)

    session = _build_http_session(resolve_ca_bundle())
    all_chapters = discover_uniform_sections(session=session)

    requested = set(target_chapters) if target_chapters else set(all_chapters.keys())
    records: list[UniformChapterAuditRecord] = []

    for ch_key in sorted(all_chapters.keys()):
        if ch_key not in requested:
            continue

        sections = all_chapters[ch_key]
        ch_num = ch_key.split("-")[-1]
        pdf_filename = f"navpers-15665j-ch{ch_num}.pdf"
        target_pdf = target_dir / pdf_filename
        hash_sidecar = target_pdf.with_suffix(".sha256")
        timestamp = datetime.now(UTC).isoformat()
        ch_title = CHAPTER_NAMES.get(ch_key, ch_key)

        logger.info("Processing %s (%d section pages)...", ch_title, len(sections))

        sections_data: list[tuple[str, list[str]]] = []
        hasher = hashlib.sha256()

        try:
            for sec in sections:
                resp = session.get(sec.url, timeout=_TIMEOUT_SECONDS)
                resp.raise_for_status()

                blocks = extract_section_blocks(resp.text)
                sec_header = f"{sec.title.upper()} ({sec.slug})"
                sections_data.append((sec_header, blocks))
                hasher.update(" ".join(blocks).encode("utf-8"))

            content_hash = hasher.hexdigest()

            if target_pdf.is_file() and hash_sidecar.is_file() and not force:
                saved_hash = hash_sidecar.read_text(encoding="utf-8").strip()
                if saved_hash == content_hash:
                    logger.info("Chapter %s is already up to date (%s)", ch_num, pdf_filename)
                    reader = PdfReader(str(target_pdf))
                    records.append(
                        UniformChapterAuditRecord(
                            chapter_key=ch_key,
                            chapter_title=ch_title,
                            filename=pdf_filename,
                            file_size=target_pdf.stat().st_size,
                            page_count=len(reader.pages),
                            section_count=len(sections),
                            sha256=content_hash,
                            timestamp=timestamp,
                            status="skipped",
                        )
                    )
                    continue

            compiled_path = compile_chapter_pdf(ch_key, sections_data, target_pdf)
            hash_sidecar.write_text(content_hash, encoding="utf-8")
            reader = PdfReader(str(compiled_path))

            records.append(
                UniformChapterAuditRecord(
                    chapter_key=ch_key,
                    chapter_title=ch_title,
                    filename=pdf_filename,
                    file_size=compiled_path.stat().st_size,
                    page_count=len(reader.pages),
                    section_count=len(sections),
                    sha256=content_hash,
                    timestamp=timestamp,
                    status="compiled",
                )
            )

        except Exception as exc:
            logger.error("Failed processing %s: %s", ch_key, exc)
            records.append(
                UniformChapterAuditRecord(
                    chapter_key=ch_key,
                    chapter_title=ch_title,
                    filename=pdf_filename,
                    file_size=0,
                    page_count=0,
                    section_count=len(sections),
                    sha256="",
                    timestamp=timestamp,
                    status="failed",
                    error=str(exc),
                )
            )

    manifest = UniformRegsAuditManifest(
        generated_at=datetime.now(UTC).isoformat(),
        chapter_count=len(records),
        records=records,
    )

    manifest_file = target_dir / "_uniform_regs_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as fh:
        json.dump(manifest.model_dump(), fh, indent=2)

    logger.info("Uniform regulations crawl complete. Processed %d chapters.", len(records))
    return manifest
