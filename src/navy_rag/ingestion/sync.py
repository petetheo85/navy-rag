"""Automated differential policy synchronization and vector re-indexing."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from navy_rag.config import settings
from navy_rag.ingestion.chunker import chunk_documents
from navy_rag.ingestion.downloader import (
    IngestionAuditRecord,
    SyncCadence,
    _build_http_session,
    ingest_document,
    is_cadence_due,
    load_corpus_manifest,
    load_previous_audit_records,
    resolve_ca_bundle,
    write_audit_manifest,
)
from navy_rag.ingestion.embedder import embed_and_store
from navy_rag.ingestion.loader import load_all

logger = logging.getLogger(__name__)


class PolicySyncReport(BaseModel):
    started_at: str
    completed_at: str
    dry_run: bool = False
    respect_cadence: bool = False
    checked_count: int = 0
    downloaded_docs: list[str] = Field(default_factory=list)
    updated_docs: list[str] = Field(default_factory=list)
    skipped_docs: list[str] = Field(default_factory=list)
    deferred_by_cadence: list[str] = Field(default_factory=list)
    failed_docs: list[str] = Field(default_factory=list)
    crawlers_executed: list[str] = Field(default_factory=list)
    chunks_reindexed: int = 0


def _read_manifest_generated_at(manifest_file: Path) -> str | None:
    if not manifest_file.is_file():
        return None
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        val = payload.get("generated_at")
        return val if isinstance(val, str) else None
    except (json.JSONDecodeError, OSError):
        return None


def write_sync_report(report: PolicySyncReport, report_path: Path | None = None) -> Path:
    """Persist the latest synchronization telemetry report atomically."""
    destination = report_path or settings.sync_report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.parent / f".{destination.name}.tmp"
    temp_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(destination)
    return destination


def run_policy_sync(
    *,
    manifest_path: Path | None = None,
    raw_dir: Path | None = None,
    audit_path: Path | None = None,
    report_path: Path | None = None,
    include_crawlers: bool = False,
    respect_cadence: bool = False,
    cadence_filter: SyncCadence | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> PolicySyncReport:
    """Probe upstream directives by cadence tier and differentially re-index changed PDFs."""
    started_at = datetime.now(UTC).isoformat()
    corpus = load_corpus_manifest(manifest_path)
    target_raw_dir = raw_dir or settings.raw_data_dir
    target_audit_path = audit_path or (target_raw_dir / "_ingestion_manifest.json")
    prior_records = load_previous_audit_records(target_audit_path)
    verify_tls = resolve_ca_bundle()

    downloaded_docs: list[str] = []
    updated_docs: list[str] = []
    skipped_docs: list[str] = []
    deferred_by_cadence: list[str] = []
    failed_docs: list[str] = []
    crawlers_executed: list[str] = []
    changed_filenames: set[str] = set()
    audit_records: list[IngestionAuditRecord] = []
    checked_count = 0

    with _build_http_session(verify_tls) as session:
        for doc in corpus.documents:
            prior = prior_records.get(doc.doc_id)
            target_file = target_raw_dir / doc.filename

            if cadence_filter and doc.sync_cadence != cadence_filter:
                deferred_by_cadence.append(doc.doc_id)
                if prior:
                    audit_records.append(prior)
                continue

            if (
                respect_cadence
                and not force
                and prior is not None
                and target_file.is_file()
                and not is_cadence_due(prior.timestamp, doc.sync_cadence)
            ):
                logger.info(
                    "Deferring %s (%s cadence not yet due; last checked %s)",
                    doc.doc_id,
                    doc.sync_cadence,
                    prior.timestamp[:10],
                )
                deferred_by_cadence.append(doc.doc_id)
                audit_records.append(prior)
                continue

            checked_count += 1
            record = ingest_document(
                doc,
                session=session,
                raw_dir=target_raw_dir,
                prior_records=prior_records,
                s3_bucket=settings.bronze_s3_bucket,
                s3_prefix=settings.bronze_s3_prefix,
                force=force,
                check_upstream=True,
                dry_run=dry_run,
                timeout_seconds=settings.download_timeout_seconds,
            )
            audit_records.append(record)

            if record.status in ("downloaded", "fallback"):
                downloaded_docs.append(doc.doc_id)
                changed_filenames.add(doc.filename)
            elif record.status == "updated":
                updated_docs.append(doc.doc_id)
                changed_filenames.add(doc.filename)
            elif record.status == "skipped":
                skipped_docs.append(doc.doc_id)
            elif record.status == "failed":
                failed_docs.append(doc.doc_id)

    if not dry_run and audit_records:
        write_audit_manifest(
            audit_records,
            target_audit_path,
            s3_bucket=settings.bronze_s3_bucket,
            s3_prefix=settings.bronze_s3_prefix,
        )

    if include_crawlers:
        uniform_dir = target_raw_dir / "uniform_regs"
        uniform_manifest_path = uniform_dir / "_uniform_regs_manifest.json"
        uniform_cadence = settings.uniform_regs_sync_cadence
        uniform_last = _read_manifest_generated_at(uniform_manifest_path)
        uniform_due = (cadence_filter is None or cadence_filter == uniform_cadence) and (
            force or not respect_cadence or is_cadence_due(uniform_last, uniform_cadence)
        )
        if uniform_due and not dry_run:
            from navy_rag.ingestion.crawlers.uniform_regs import crawl_uniform_regs

            crawlers_executed.append("uniform_regs")
            u_manifest = crawl_uniform_regs(output_dir=uniform_dir, force=force)
            for u_rec in u_manifest.records:
                checked_count += 1
                doc_key = f"uniform_regs:{u_rec.chapter_key}"
                if u_rec.status == "compiled":
                    updated_docs.append(doc_key)
                    changed_filenames.add(u_rec.filename)
                elif u_rec.status == "skipped":
                    skipped_docs.append(doc_key)
                elif u_rec.status == "failed":
                    failed_docs.append(doc_key)
        elif not uniform_due:
            deferred_by_cadence.append("crawler:uniform_regs")

        mil_dir = target_raw_dir / "milpersman"
        mil_manifest_path = mil_dir / "_milpersman_manifest.json"
        mil_cadence = settings.milpersman_sync_cadence
        mil_due = (
            (cadence_filter is None or cadence_filter == mil_cadence)
            and (
                force
                or not respect_cadence
                or is_cadence_due(_read_manifest_generated_at(mil_manifest_path), mil_cadence)
            )
        )
        if mil_due and not dry_run:
            from navy_rag.ingestion.crawlers.milpersman import DEFAULT_CATEGORIES, crawl_milpersman

            crawlers_executed.append("milpersman")
            m_manifest = crawl_milpersman(
                categories=list(DEFAULT_CATEGORIES),
                output_dir=mil_dir,
                force=force,
                check_upstream=True,
            )
            for m_rec in m_manifest.records:
                checked_count += 1
                doc_key = f"milpersman:{m_rec.article_number}"
                if m_rec.status == "downloaded":
                    downloaded_docs.append(doc_key)
                    changed_filenames.add(m_rec.filename)
                elif m_rec.status == "updated":
                    updated_docs.append(doc_key)
                    changed_filenames.add(m_rec.filename)
                elif m_rec.status == "skipped":
                    skipped_docs.append(doc_key)
                elif m_rec.status == "failed":
                    failed_docs.append(doc_key)
        elif not mil_due:
            deferred_by_cadence.append("crawler:milpersman")

    chunks_reindexed = 0
    if not dry_run and (changed_filenames or force):
        data_root = target_raw_dir.parent if target_raw_dir.name == "raw" else settings.data_dir
        docs = load_all(
            data_dir=data_root,
            only_filenames=None if force else changed_filenames,
        )
        if docs:
            chunks = chunk_documents(docs)
            chunks_reindexed = embed_and_store(
                chunks,
                skip_existing=not force,
                changed_filenames=changed_filenames,
            )

    report = PolicySyncReport(
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        dry_run=dry_run,
        respect_cadence=respect_cadence,
        checked_count=checked_count,
        downloaded_docs=downloaded_docs,
        updated_docs=updated_docs,
        skipped_docs=skipped_docs,
        deferred_by_cadence=deferred_by_cadence,
        failed_docs=failed_docs,
        crawlers_executed=crawlers_executed,
        chunks_reindexed=chunks_reindexed,
    )

    if not dry_run or report_path is not None:
        write_sync_report(report, report_path)

    return report
