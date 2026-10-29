"""Idempotent Bronze-layer downloader for authoritative Navy and DoD policy PDFs."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import requests
import yaml
from pydantic import BaseModel, Field, model_validator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from navy_rag.config import settings

logger = logging.getLogger(__name__)

# 64 KiB read buffer balances syscall overhead and memory footprint for large manuals.
_HASH_CHUNK_BYTES = 65536
_DEFAULT_DOD_CA_FILENAME = "dod_root_ca.pem"
_HTTP_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

SyncCadence = Literal["daily", "weekly", "monthly", "quarterly"]


class CorpusDocument(BaseModel):
    doc_id: str
    instruction_number: str
    title: str
    source_agency: str
    url: str
    filename: str
    audience: str = "public"
    sha256: str | None = None
    fallback_path: Path | None = None
    base_ssic: str = ""
    revision: str | None = None
    change: int = 0
    is_latest: bool = True
    last_verified: str = "2026-10-26"
    sync_cadence: SyncCadence = "monthly"

    @model_validator(mode="before")
    @classmethod
    def _populate_base_ssic(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("base_ssic"):
            data["base_ssic"] = data.get("instruction_number", "")
        return data


class CorpusManifest(BaseModel):
    schema_version: str = "1.0"
    documents: list[CorpusDocument] = Field(default_factory=list)


class IngestionAuditRecord(BaseModel):
    doc_id: str
    url: str
    filename: str
    file_size: int
    sha256: str
    timestamp: str
    status: Literal["downloaded", "skipped", "updated", "fallback", "failed"]
    previous_sha256: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None


class IngestionManifest(BaseModel):
    generated_at: str
    document_count: int
    records: list[IngestionAuditRecord]


def load_corpus_manifest(manifest_path: Path | None = None) -> CorpusManifest:
    """Parse and validate the declarative YAML corpus manifest."""
    path = manifest_path or settings.manifest_path
    if not path.exists():
        raise FileNotFoundError(f"Corpus manifest not found at {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw_manifest = yaml.safe_load(fh) or {}

    return CorpusManifest.model_validate(raw_manifest)


def resolve_ca_bundle(
    configured_bundle: Path | None = None,
    certs_dir: Path | None = None,
) -> str | bool:
    """Resolve the CA certificate bundle for .mil TLS verification.

    Never disables TLS verification. Uses an explicit DoD CA bundle when configured
    or present in certs/dod_root_ca.pem, falling back to the standard system/certifi
    trust store for publicly cross-signed .mil endpoints.
    """
    explicit_bundle = configured_bundle if configured_bundle is not None else settings.dod_ca_bundle
    if explicit_bundle is not None:
        resolved = (
            explicit_bundle
            if explicit_bundle.is_absolute()
            else settings.base_dir / explicit_bundle
        )
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Configured DoD CA bundle does not exist: {resolved}"
            )
        return str(resolved)

    cert_directory = certs_dir or settings.certs_dir
    default_dod_bundle = cert_directory / _DEFAULT_DOD_CA_FILENAME
    if default_dod_bundle.is_file():
        return str(default_dod_bundle)

    return True


def compute_file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_previous_audit_records(audit_path: Path) -> dict[str, IngestionAuditRecord]:
    if not audit_path.is_file():
        return {}

    try:
        with audit_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        manifest = IngestionManifest.model_validate(payload)
        return {record.doc_id: record for record in manifest.records}
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("Ignoring unreadable audit manifest at %s: %s", audit_path, exc)
        return {}


def _build_http_session(verify_tls: str | bool) -> requests.Session:
    session = requests.Session()
    session.verify = verify_tls
    session.headers.update({"User-Agent": _HTTP_USER_AGENT})

    # .mil portals behind Akamai/Cloudflare intermittently drop idle connections or throttle bursts.
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=_RETRY_STATUS_CODES,
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _s3_object_matches_hash(bucket: str, key: str, expected_sha256: str) -> tuple[bool, int | None]:
    import boto3
    from botocore.exceptions import ClientError

    s3_client = boto3.client("s3")
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchKey", "NotFound"):
            return False, None
        raise

    remote_sha256 = (response.get("Metadata") or {}).get("sha256", "")
    content_length = response.get("ContentLength")
    return remote_sha256.lower() == expected_sha256.lower(), content_length


def _upload_file_to_s3(
    local_path: Path,
    bucket: str,
    key: str,
    sha256_digest: str,
    doc_id: str,
) -> None:
    import boto3

    s3_client = boto3.client("s3")
    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
        ExtraArgs={
            "ContentType": "application/pdf",
            "Metadata": {"sha256": sha256_digest, "doc_id": doc_id},
        },
    )


def _expected_hash_for_doc(
    doc: CorpusDocument,
    prior_records: dict[str, IngestionAuditRecord],
) -> str | None:
    if doc.sha256:
        return doc.sha256.lower()
    prior = prior_records.get(doc.doc_id)
    return prior.sha256.lower() if prior else None


def is_cadence_due(
    last_timestamp: str | None,
    cadence: SyncCadence,
    *,
    now: datetime | None = None,
) -> bool:
    """Return True when the document's configured sync cadence interval has elapsed."""
    if not last_timestamp:
        return True
    try:
        last_dt = datetime.fromisoformat(last_timestamp)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=UTC)
    except ValueError:
        return True

    current = now or datetime.now(UTC)
    interval_days = settings.sync_cadence_days.get(cadence, 30)
    elapsed_days = (current - last_dt).total_seconds() / 86400.0
    return elapsed_days >= interval_days


def _header_str(headers: Any, key: str) -> str | None:
    if headers is None or not hasattr(headers, "get"):
        return None
    val = headers.get(key)
    return val.strip() if isinstance(val, str) and val.strip() else None


def _probe_upstream_changed(
    session: requests.Session,
    url: str,
    local_path: Path,
    prior_record: IngestionAuditRecord | None,
    timeout_seconds: int,
) -> tuple[bool, str | None, str | None]:
    """Probe upstream HTTP headers via HEAD to detect policy updates without full download."""
    req_headers: dict[str, str] = {}
    if prior_record and prior_record.etag:
        req_headers["If-None-Match"] = prior_record.etag
    if prior_record and prior_record.last_modified:
        req_headers["If-Modified-Since"] = prior_record.last_modified

    try:
        resp = session.head(
            url,
            headers=req_headers or None,
            allow_redirects=True,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        logger.debug("HEAD probe failed for %s (%s); falling back to GET", url, exc)
        return True, None, None

    status_code = getattr(resp, "status_code", 200)
    if status_code == 304:
        return (
            False,
            prior_record.etag if prior_record else None,
            prior_record.last_modified if prior_record else None,
        )
    if isinstance(status_code, int) and status_code >= 400:
        return True, None, None

    headers = getattr(resp, "headers", None)
    remote_etag = _header_str(headers, "ETag")
    remote_lm = _header_str(headers, "Last-Modified")
    raw_cl = _header_str(headers, "Content-Length")
    remote_size = int(raw_cl) if raw_cl and raw_cl.isdigit() else None
    local_size = local_path.stat().st_size

    if remote_size is not None and remote_size != local_size:
        return True, remote_etag, remote_lm

    if prior_record and prior_record.etag and remote_etag:
        return remote_etag != prior_record.etag, remote_etag, remote_lm

    if prior_record and prior_record.last_modified and remote_lm:
        return remote_lm != prior_record.last_modified, remote_etag, remote_lm

    if remote_size is not None and remote_size == local_size:
        return False, remote_etag, remote_lm

    return True, remote_etag, remote_lm


def _resolve_fallback(doc: CorpusDocument) -> Path | None:
    if doc.fallback_path:
        path = (
            doc.fallback_path
            if doc.fallback_path.is_absolute()
            else settings.base_dir / doc.fallback_path
        )
        if path.is_file():
            return path

    candidate = settings.data_dir / "public" / doc.filename
    if candidate.is_file():
        return candidate

    return None


def _stream_download_to_temp(
    session: requests.Session,
    url: str,
    part_path: Path,
    timeout_seconds: int,
) -> tuple[str, int, str | None, str | None]:
    digest = hashlib.sha256()
    bytes_written = 0

    with session.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        headers = getattr(response, "headers", None)
        etag = _header_str(headers, "ETag")
        last_modified = _header_str(headers, "Last-Modified")
        with part_path.open("wb") as out_fh:
            for chunk in response.iter_content(chunk_size=_HASH_CHUNK_BYTES):
                if not chunk:
                    continue
                out_fh.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)

    return digest.hexdigest(), bytes_written, etag, last_modified


def ingest_document(
    doc: CorpusDocument,
    *,
    session: requests.Session,
    raw_dir: Path,
    prior_records: dict[str, IngestionAuditRecord],
    s3_bucket: str | None = None,
    s3_prefix: str = "",
    force: bool = False,
    check_upstream: bool = False,
    dry_run: bool = False,
    timeout_seconds: int = 60,
) -> IngestionAuditRecord:
    """Download a single corpus document idempotently and return its audit record."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / doc.filename
    s3_key = f"{s3_prefix.strip('/')}/{doc.filename}".lstrip("/")
    prior = prior_records.get(doc.doc_id)
    expected_hash = _expected_hash_for_doc(doc, prior_records)
    now_iso = datetime.now(UTC).isoformat()

    if not force and expected_hash:
        if target_path.is_file():
            local_hash = compute_file_sha256(target_path)
            if local_hash == expected_hash:
                if not check_upstream or doc.sha256:
                    logger.info(
                        "Skipping %s (%s): local SHA-256 matches %s",
                        doc.doc_id,
                        doc.filename,
                        local_hash[:12],
                    )
                    return IngestionAuditRecord(
                        doc_id=doc.doc_id,
                        url=doc.url,
                        filename=doc.filename,
                        file_size=target_path.stat().st_size,
                        sha256=local_hash,
                        timestamp=now_iso,
                        status="skipped",
                        etag=prior.etag if prior else None,
                        last_modified=prior.last_modified if prior else None,
                    )

                changed, remote_etag, remote_lm = _probe_upstream_changed(
                    session=session,
                    url=doc.url,
                    local_path=target_path,
                    prior_record=prior,
                    timeout_seconds=timeout_seconds,
                )
                if not changed:
                    logger.info(
                        "Skipping %s (%s): upstream headers unchanged (%s)",
                        doc.doc_id,
                        doc.filename,
                        local_hash[:12],
                    )
                    return IngestionAuditRecord(
                        doc_id=doc.doc_id,
                        url=doc.url,
                        filename=doc.filename,
                        file_size=target_path.stat().st_size,
                        sha256=local_hash,
                        timestamp=now_iso,
                        status="skipped",
                        etag=remote_etag or (prior.etag if prior else None),
                        last_modified=remote_lm or (prior.last_modified if prior else None),
                    )
                if dry_run:
                    logger.info(
                        "[DRY-RUN] Upstream change detected for %s (%s)",
                        doc.doc_id,
                        doc.filename,
                    )
                    return IngestionAuditRecord(
                        doc_id=doc.doc_id,
                        url=doc.url,
                        filename=doc.filename,
                        file_size=target_path.stat().st_size,
                        sha256=local_hash,
                        timestamp=now_iso,
                        status="updated",
                        previous_sha256=local_hash,
                        etag=remote_etag,
                        last_modified=remote_lm,
                    )
        elif s3_bucket and not check_upstream:
            matched, s3_size = _s3_object_matches_hash(s3_bucket, s3_key, expected_hash)
            if matched and s3_size is not None:
                logger.info(
                    "Skipping %s (%s): S3 SHA-256 matches %s",
                    doc.doc_id,
                    s3_key,
                    expected_hash[:12],
                )
                return IngestionAuditRecord(
                    doc_id=doc.doc_id,
                    url=doc.url,
                    filename=doc.filename,
                    file_size=s3_size,
                    sha256=expected_hash,
                    timestamp=now_iso,
                    status="skipped",
                )

    if dry_run:
        return IngestionAuditRecord(
            doc_id=doc.doc_id,
            url=doc.url,
            filename=doc.filename,
            file_size=target_path.stat().st_size if target_path.is_file() else 0,
            sha256=expected_hash or "",
            timestamp=now_iso,
            status="updated" if target_path.is_file() else "downloaded",
            previous_sha256=expected_hash if target_path.is_file() else None,
        )

    part_path = raw_dir / f".{doc.filename}.part"
    try:
        downloaded_hash, byte_count, resp_etag, resp_lm = _stream_download_to_temp(
            session=session,
            url=doc.url,
            part_path=part_path,
            timeout_seconds=timeout_seconds,
        )
    except requests.RequestException as exc:
        part_path.unlink(missing_ok=True)
        fallback = _resolve_fallback(doc)
        if fallback and fallback.is_file():
            logger.warning(
                "Upstream download failed for %s (%s). Using local fallback: %s",
                doc.doc_id,
                exc,
                fallback.name,
            )
            fallback_bytes = fallback.read_bytes()
            target_path.write_bytes(fallback_bytes)
            downloaded_hash = hashlib.sha256(fallback_bytes).hexdigest()
            byte_count = len(fallback_bytes)

            if s3_bucket:
                _upload_file_to_s3(
                    local_path=target_path,
                    bucket=s3_bucket,
                    key=s3_key,
                    sha256_digest=downloaded_hash,
                    doc_id=doc.doc_id,
                )

            return IngestionAuditRecord(
                doc_id=doc.doc_id,
                url=doc.url,
                filename=doc.filename,
                file_size=byte_count,
                sha256=downloaded_hash,
                timestamp=now_iso,
                status="fallback",
            )

        if target_path.is_file():
            existing_hash = compute_file_sha256(target_path)
            logger.warning(
                "Upstream fetch failed for %s (%s); retaining existing file (%s)",
                doc.doc_id,
                exc,
                existing_hash[:12],
            )
            return IngestionAuditRecord(
                doc_id=doc.doc_id,
                url=doc.url,
                filename=doc.filename,
                file_size=target_path.stat().st_size,
                sha256=existing_hash,
                timestamp=now_iso,
                status="skipped",
                etag=prior.etag if prior else None,
                last_modified=prior.last_modified if prior else None,
            )

        logger.error("Failed to download %s from %s: %s", doc.doc_id, doc.url, exc)
        return IngestionAuditRecord(
            doc_id=doc.doc_id,
            url=doc.url,
            filename=doc.filename,
            file_size=0,
            sha256="",
            timestamp=now_iso,
            status="failed",
            error=str(exc),
        )
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    try:
        if doc.sha256 and downloaded_hash != doc.sha256.lower():
            expected_lower = doc.sha256.lower()
            raise ValueError(
                f"SHA-256 mismatch for {doc.doc_id}: expected {expected_lower}, "
                f"got {downloaded_hash}"
            )

        existing_hash: str | None = None
        if target_path.is_file():
            existing_hash = compute_file_sha256(target_path)
        elif doc.doc_id in prior_records:
            existing_hash = prior_records[doc.doc_id].sha256.lower()

        if not force and existing_hash == downloaded_hash and target_path.is_file():
            part_path.unlink(missing_ok=True)
            logger.info("No upstream changes for %s (%s)", doc.doc_id, downloaded_hash[:12])
            return IngestionAuditRecord(
                doc_id=doc.doc_id,
                url=doc.url,
                filename=doc.filename,
                file_size=byte_count,
                sha256=downloaded_hash,
                timestamp=now_iso,
                status="skipped",
                etag=resp_etag or (prior.etag if prior else None),
                last_modified=resp_lm or (prior.last_modified if prior else None),
            )

        # Atomic rename prevents partial PDFs from being visible to downstream readers.
        part_path.replace(target_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    if s3_bucket:
        _upload_file_to_s3(
            local_path=target_path,
            bucket=s3_bucket,
            key=s3_key,
            sha256_digest=downloaded_hash,
            doc_id=doc.doc_id,
        )

    if existing_hash and existing_hash != downloaded_hash:
        logger.info(
            "Detected upstream revision for %s: %s -> %s",
            doc.doc_id,
            existing_hash[:12],
            downloaded_hash[:12],
        )
        return IngestionAuditRecord(
            doc_id=doc.doc_id,
            url=doc.url,
            filename=doc.filename,
            file_size=byte_count,
            sha256=downloaded_hash,
            timestamp=now_iso,
            status="updated",
            previous_sha256=existing_hash,
            etag=resp_etag,
            last_modified=resp_lm,
        )

    logger.info("Downloaded %s (%d bytes, sha256=%s)", doc.doc_id, byte_count, downloaded_hash[:12])
    return IngestionAuditRecord(
        doc_id=doc.doc_id,
        url=doc.url,
        filename=doc.filename,
        file_size=byte_count,
        sha256=downloaded_hash,
        timestamp=now_iso,
        status="downloaded",
        etag=resp_etag,
        last_modified=resp_lm,
    )


def write_audit_manifest(
    records: list[IngestionAuditRecord],
    audit_path: Path | None = None,
    *,
    s3_bucket: str | None = None,
    s3_prefix: str = "",
) -> Path:
    """Write the sidecar _ingestion_manifest.json audit log atomically."""
    destination = audit_path or settings.ingestion_manifest_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    manifest = IngestionManifest(
        generated_at=datetime.now(UTC).isoformat(),
        document_count=len(records),
        records=records,
    )

    temp_manifest = destination.parent / f".{destination.name}.tmp"
    temp_manifest.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    temp_manifest.replace(destination)

    if s3_bucket:
        import boto3

        s3_key = f"{s3_prefix.strip('/')}/{destination.name}".lstrip("/")
        boto3.client("s3").upload_file(
            Filename=str(destination),
            Bucket=s3_bucket,
            Key=s3_key,
            ExtraArgs={"ContentType": "application/json"},
        )

    return destination


def sync_corpus(
    *,
    manifest_path: Path | None = None,
    raw_dir: Path | None = None,
    audit_path: Path | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    force: bool = False,
    check_upstream: bool = False,
    dry_run: bool = False,
) -> list[IngestionAuditRecord]:
    """Synchronize all documents in the corpus manifest and emit the audit log."""
    corpus = load_corpus_manifest(manifest_path)
    target_raw_dir = raw_dir or settings.raw_data_dir
    target_audit_path = audit_path or (target_raw_dir / "_ingestion_manifest.json")
    bucket = s3_bucket if s3_bucket is not None else settings.bronze_s3_bucket
    prefix = s3_prefix if s3_prefix is not None else settings.bronze_s3_prefix

    prior_records = load_previous_audit_records(target_audit_path)
    verify_tls = resolve_ca_bundle()

    records: list[IngestionAuditRecord] = []
    with _build_http_session(verify_tls) as session:
        for doc in corpus.documents:
            record = ingest_document(
                doc,
                session=session,
                raw_dir=target_raw_dir,
                prior_records=prior_records,
                s3_bucket=bucket,
                s3_prefix=prefix,
                force=force,
                check_upstream=check_upstream,
                dry_run=dry_run,
                timeout_seconds=settings.download_timeout_seconds,
            )
            records.append(record)

    if not dry_run:
        write_audit_manifest(
            records,
            target_audit_path,
            s3_bucket=bucket,
            s3_prefix=prefix,
        )
    return records


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="downloader",
        description="Fetch authoritative Navy/DoD policy documents into Bronze raw storage.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to corpus_manifest.yaml (default: from config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading even if existing content matches hash.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting Bronze layer corpus synchronization...")
    records = sync_corpus(manifest_path=args.manifest, force=args.force)

    logger.info("--- Ingestion Summary ---")
    for r in records:
        logger.info(
            "[%s] %s -> %s (%d bytes, sha256=%s)",
            r.status.upper(),
            r.doc_id,
            r.filename,
            r.file_size,
            r.sha256[:12],
        )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

