"""Automated crawler for MyNavy HR MILPERSMAN policy articles."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, Field

from navy_rag.config import settings
from navy_rag.ingestion.downloader import (
    _build_http_session,
    compute_file_sha256,
    resolve_ca_bundle,
)

logger = logging.getLogger(__name__)

MYNAVYHR_BASE_URL = "https://www.mynavyhr.navy.mil"
MILPERSMAN_CATEGORY_BASE = (
    "https://www.mynavyhr.navy.mil/References/MILPERSMAN/1000-Military-Personnel/"
)
DEFAULT_CATEGORIES = ("1100-Recruiting", "1900-Separation")
_HASH_CHUNK_BYTES = 65536
_TIMEOUT_SECONDS = 30

_ARTICLE_LINK_RE = re.compile(
    r"<a[^>]*href=[\"']([^\"']+\.pdf[^\"']*)[\"'][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_ARTICLE_NUM_RE = re.compile(r"\b(\d{4}-\d{3})\b")


class MilpersmanArticle(BaseModel):
    article_number: str
    title: str
    category: str
    url: str
    filename: str


class MilpersmanAuditRecord(BaseModel):
    article_number: str
    category: str
    url: str
    filename: str
    file_size: int
    sha256: str
    timestamp: str
    status: Literal["downloaded", "skipped", "updated", "failed"]
    error: str | None = None


class MilpersmanAuditManifest(BaseModel):
    generated_at: str
    article_count: int
    records: list[MilpersmanAuditRecord] = Field(default_factory=list)


def extract_articles_from_html(html_text: str, category: str) -> list[MilpersmanArticle]:
    """Parse article numbers, titles, and PDF URLs from category HTML markup."""
    articles: list[MilpersmanArticle] = []
    seen_articles: set[str] = set()

    for match in _ARTICLE_LINK_RE.finditer(html_text):
        raw_href = match.group(1).strip()
        link_text = re.sub(r"<[^>]+>", " ", match.group(2))
        clean_text = " ".join(link_text.replace("&nbsp;", " ").split())

        num_match = _ARTICLE_NUM_RE.search(clean_text) or _ARTICLE_NUM_RE.search(raw_href)
        if not num_match:
            continue

        art_num = num_match.group(1)
        if art_num in seen_articles:
            continue

        seen_articles.add(art_num)
        full_url = urljoin(MYNAVYHR_BASE_URL, raw_href)
        filename = f"milpersman-{art_num}.pdf"

        articles.append(
            MilpersmanArticle(
                article_number=art_num,
                title=clean_text or f"MILPERSMAN {art_num}",
                category=category,
                url=full_url,
                filename=filename,
            )
        )

    return sorted(articles, key=lambda a: a.article_number)


def discover_articles(
    category: str,
    session: requests.Session | None = None,
) -> list[MilpersmanArticle]:
    """Fetch category index page and extract all referenced MILPERSMAN article PDFs."""
    http_session = session or _build_http_session(resolve_ca_bundle())
    cat_url = f"{MILPERSMAN_CATEGORY_BASE.rstrip('/')}/{category.strip('/')}/"

    logger.info("Discovering MILPERSMAN articles in category '%s' from %s", category, cat_url)
    resp = http_session.get(cat_url, timeout=_TIMEOUT_SECONDS)
    resp.raise_for_status()

    articles = extract_articles_from_html(resp.text, category)
    logger.info("Discovered %d article(s) in category '%s'", len(articles), category)
    return articles


def download_article(
    article: MilpersmanArticle,
    output_dir: Path,
    session: requests.Session,
    force: bool = False,
) -> MilpersmanAuditRecord:
    """Download a single MILPERSMAN PDF idempotently with SHA-256 verification."""
    target_path = output_dir / article.filename
    timestamp = datetime.now(UTC).isoformat()

    if target_path.is_file() and not force:
        file_size = target_path.stat().st_size
        file_sha256 = compute_file_sha256(target_path)
        return MilpersmanAuditRecord(
            article_number=article.article_number,
            category=article.category,
            url=article.url,
            filename=article.filename,
            file_size=file_size,
            sha256=file_sha256,
            timestamp=timestamp,
            status="skipped",
        )

    try:
        resp = session.get(article.url, stream=True, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(dir=output_dir, delete=False, suffix=".tmp") as tmp_file:
            tmp_path = Path(tmp_file.name)
            for chunk in resp.iter_content(chunk_size=_HASH_CHUNK_BYTES):
                tmp_file.write(chunk)

        file_size = tmp_path.stat().st_size
        file_sha256 = compute_file_sha256(tmp_path)
        tmp_path.replace(target_path)

        logger.info(
            "Downloaded MILPERSMAN %s -> %s (%d bytes)",
            article.article_number,
            target_path.name,
            file_size,
        )
        return MilpersmanAuditRecord(
            article_number=article.article_number,
            category=article.category,
            url=article.url,
            filename=article.filename,
            file_size=file_size,
            sha256=file_sha256,
            timestamp=timestamp,
            status="downloaded",
        )
    except Exception as exc:
        logger.error("Failed downloading MILPERSMAN %s: %s", article.article_number, exc)
        return MilpersmanAuditRecord(
            article_number=article.article_number,
            category=article.category,
            url=article.url,
            filename=article.filename,
            file_size=0,
            sha256="",
            timestamp=timestamp,
            status="failed",
            error=str(exc),
        )


def crawl_milpersman(
    categories: list[str] | tuple[str, ...] | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> MilpersmanAuditManifest:
    """Crawl specified categories, download PDFs, and record audit manifest."""
    target_categories = list(categories) if categories else list(DEFAULT_CATEGORIES)
    target_dir = output_dir or (settings.raw_data_dir / "milpersman")
    target_dir.mkdir(parents=True, exist_ok=True)

    session = _build_http_session(resolve_ca_bundle())
    records: list[MilpersmanAuditRecord] = []

    for cat in target_categories:
        try:
            articles = discover_articles(cat, session=session)
        except Exception as exc:
            logger.error("Failed discovering articles for category '%s': %s", cat, exc)
            continue

        for art in articles:
            rec = download_article(art, target_dir, session, force=force)
            records.append(rec)

    manifest = MilpersmanAuditManifest(
        generated_at=datetime.now(UTC).isoformat(),
        article_count=len(records),
        records=records,
    )

    manifest_file = target_dir / "_milpersman_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as fh:
        json.dump(manifest.model_dump(), fh, indent=2)

    logger.info(
        "MILPERSMAN crawl complete. Processed %d articles across %d categories.",
        len(records),
        len(target_categories),
    )
    return manifest
