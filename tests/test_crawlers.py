"""Unit tests for automated MILPERSMAN and NAVPERS 15665J policy crawlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from navy_rag.cli.main import build_parser
from navy_rag.ingestion.crawlers.milpersman import (
    MilpersmanArticle,
    download_article,
    extract_articles_from_html,
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


class TestCrawlerCLI:
    def test_cli_parser_registers_crawl_subcommands(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["crawl-milpersman", "--help"])
        assert exc.value.code == 0
