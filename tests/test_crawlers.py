"""Unit tests for automated MILPERSMAN and NAVPERS 15665J policy crawlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pypdf import PdfReader

from navy_rag.cli.main import build_parser
from navy_rag.ingestion.crawlers.milpersman import (
    MilpersmanArticle,
    download_article,
    extract_articles_from_html,
)
from navy_rag.ingestion.crawlers.uniform_regs import (
    _sanitize_for_pdf,
    compile_chapter_pdf,
    extract_section_blocks,
)


class TestMilpersmanCrawler:
    SAMPLE_HTML = """
    <html>
      <body>
        <div class="content">
          <a href="/Portals/55/Reference/MILPERSMAN/1000/1100Recruiting/1100-011.pdf?ver=123">
            <span>1100-011&nbsp;</span> Selection of Personnel for Nuclear Propulsion Training
          </a>
          <a href="/Portals/55/Reference/MILPERSMAN/1000/1100Recruiting/1133-020.pdf">
            1133-020 Eligibility Requirements for Enlistment
          </a>
          <a href="/NotAPdf/page.html">Irrelevant Link</a>
        </div>
      </body>
    </html>
    """

    def test_extract_articles_from_html(self):
        articles = extract_articles_from_html(self.SAMPLE_HTML, category="1100-Recruiting")
        assert len(articles) == 2

        a1, a2 = articles[0], articles[1]
        assert a1.article_number == "1100-011"
        assert a1.filename == "milpersman-1100-011.pdf"
        assert a1.category == "1100-Recruiting"
        assert "1100-011.pdf" in a1.url

        assert a2.article_number == "1133-020"
        assert a2.filename == "milpersman-1133-020.pdf"

    def test_download_article_idempotent(self, tmp_path: Path):
        session = MagicMock()
        response = MagicMock()
        response.iter_content.return_value = [b"%PDF-1.4 dummy pdf bytes for test"]
        response.raise_for_status.return_value = None
        session.get.return_value = response

        article = MilpersmanArticle(
            article_number="1133-020",
            title="Eligibility Requirements",
            category="1100-Recruiting",
            url="https://example.mil/1133-020.pdf",
            filename="milpersman-1133-020.pdf",
        )

        # Initial download
        rec1 = download_article(article, tmp_path, session)
        assert rec1.status == "downloaded"
        assert (tmp_path / "milpersman-1133-020.pdf").is_file()
        assert rec1.file_size > 0

        # Second download should skip
        rec2 = download_article(article, tmp_path, session)
        assert rec2.status == "skipped"
        assert rec2.sha256 == rec1.sha256


class TestUniformRegsCrawler:
    SAMPLE_HTML = """
    <html>
      <head><style>body { font: 12px; }</style></head>
      <body>
        <script>var x = 1;</script>
        <div id="dnn_Content">
          SECTION 2: PERSONAL APPEARANCE
          <p>2201. PERSONAL APPEARANCE. Leaders enforce standard grooming.</p>
          <p>7. TATTOOS/BODY ART/BRANDS. Tattoos are authorized with restrictions.</p>
          <p>a. Content: Tattoos must not be prejudicial.</p>
          <p>(1) Subparagraph one.</p>
          LEGAL & ADMINISTRATIVE LINKS
          <a href="/footer">Footer</a>
        </div>
      </body>
    </html>
    """

    def test_sanitize_for_pdf(self):
        raw = "Smart \u201cquotes\u201d and \u2018single\u2019 -- \u2014 emdash & \u00bd fraction"
        sanitized = _sanitize_for_pdf(raw)
        assert '"quotes"' in sanitized
        assert "'single'" in sanitized
        assert "1/2 fraction" in sanitized

    def test_extract_section_blocks(self):
        blocks = extract_section_blocks(self.SAMPLE_HTML)
        assert len(blocks) >= 3
        assert any("2201. PERSONAL APPEARANCE" in b for b in blocks)
        assert any("7. TATTOOS/BODY ART/BRANDS" in b for b in blocks)
        assert all("LEGAL & ADMINISTRATIVE LINKS" not in b for b in blocks)
        assert all("var x = 1" not in b for b in blocks)

    def test_compile_chapter_pdf(self, tmp_path: Path):
        sec1_blocks = [
            "2101. GENERAL INFORMATION",
            "1. GENERAL. Grooming standards ensure neatness.",
            "a. Cleanliness. Personnel shall maintain cleanliness.",
        ]
        sec2_blocks = [
            "2201. PERSONAL APPEARANCE",
            "7. TATTOOS/BODY ART/BRANDS. Tattoos must comply with location rules.",
            "a. Content: Prohibited content includes gang symbols.",
            "(1) Prohibited extremist affiliation.",
        ]
        sections = [
            ("SECTION 1: GENERAL INFORMATION (2101)", sec1_blocks),
            ("SECTION 2: PERSONAL APPEARANCE (2201)", sec2_blocks),
        ]
        output_pdf = tmp_path / "navpers-15665j-ch2.pdf"
        compiled = compile_chapter_pdf("Chapter-2", sections, output_pdf)

        assert compiled.is_file()
        reader = PdfReader(str(compiled))
        assert len(reader.pages) >= 1
        page_text = reader.pages[0].extract_text()
        assert "NAVPERS 15665J" in page_text
        assert "GROOMING STANDARDS" in page_text


class TestCrawlerCLI:
    def test_cli_parser_registers_crawl_subcommands(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["crawl-milpersman", "--help"])
        assert exc.value.code == 0

        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["crawl-uniforms", "--help"])
        assert exc.value.code == 0
