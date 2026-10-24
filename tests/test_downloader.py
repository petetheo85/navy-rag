"""Unit tests for the Bronze-layer corpus downloader and sidecar audit log."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from navy_rag.ingestion.downloader import (
    CorpusDocument,
    IngestionAuditRecord,
    compute_file_sha256,
    ingest_document,
    load_corpus_manifest,
    load_previous_audit_records,
    resolve_ca_bundle,
    sync_corpus,
    write_audit_manifest,
)


def _mock_stream_response(payload: bytes) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.iter_content.return_value = [payload]
    response.raise_for_status.return_value = None
    return response


class TestCorpusManifest:
    def test_loads_committed_corpus_manifest(self):
        manifest = load_corpus_manifest()
        assert len(manifest.documents) == 4
        doc_ids = {doc.doc_id for doc in manifest.documents}
        assert doc_ids == {
            "comnavcruitcominst_1130_8",
            "cnrc_orlmm",
            "milpersman_1133_020",
            "dodi_1304_26",
        }

    def test_missing_manifest_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Corpus manifest not found"):
            load_corpus_manifest(tmp_path / "nonexistent.yaml")


class TestDodTlsVerification:
    def test_defaults_to_system_trust_store_when_no_dod_bundle(self, tmp_path: Path):
        with patch("navy_rag.ingestion.downloader.settings") as mock_settings:
            mock_settings.dod_ca_bundle = None
            mock_settings.certs_dir = tmp_path
            assert resolve_ca_bundle(certs_dir=tmp_path) is True

    def test_uses_dod_root_ca_when_present_in_certs_dir(self, tmp_path: Path):
        ca_file = tmp_path / "dod_root_ca.pem"
        ca_file.write_text("-----BEGIN CERTIFICATE-----\n...\n", encoding="utf-8")
        with patch("navy_rag.ingestion.downloader.settings") as mock_settings:
            mock_settings.dod_ca_bundle = None
            mock_settings.certs_dir = tmp_path
            assert resolve_ca_bundle(certs_dir=tmp_path) == str(ca_file)

    def test_missing_explicit_ca_bundle_fails_fast(self, tmp_path: Path):
        missing_bundle = tmp_path / "missing_ca.pem"
        with pytest.raises(FileNotFoundError, match="Configured DoD CA bundle does not exist"):
            resolve_ca_bundle(configured_bundle=missing_bundle)


class TestIdempotentIngestion:
    def _sample_doc(self, sha256: str | None = None) -> CorpusDocument:
        return CorpusDocument(
            doc_id="dodi_1304_26",
            instruction_number="DoDI 1304.26",
            title="Qualification Standards",
            source_agency="DoD ESD WHS",
            url="https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodi/130426p.pdf",
            filename="dodi-1304-26.pdf",
            audience="public",
            sha256=sha256,
        )

    def test_downloads_new_pdf_and_verifies_sha256(self, tmp_path: Path):
        pdf_bytes = b"%PDF-1.7 sample navy instruction content"
        expected_sha = hashlib.sha256(pdf_bytes).hexdigest()

        session = MagicMock()
        session.get.return_value = _mock_stream_response(pdf_bytes)

        doc = self._sample_doc()
        record = ingest_document(
            doc,
            session=session,
            raw_dir=tmp_path,
            prior_records={},
        )

        assert record.status == "downloaded"
        assert record.sha256 == expected_sha
        assert record.file_size == len(pdf_bytes)
        assert (tmp_path / doc.filename).read_bytes() == pdf_bytes
        assert compute_file_sha256(tmp_path / doc.filename) == expected_sha

    def test_skips_download_when_local_file_matches_manifest_hash(self, tmp_path: Path):
        pdf_bytes = b"%PDF-1.7 cached policy"
        expected_sha = hashlib.sha256(pdf_bytes).hexdigest()

        doc = self._sample_doc(sha256=expected_sha)
        local_file = tmp_path / doc.filename
        local_file.write_bytes(pdf_bytes)

        session = MagicMock()
        record = ingest_document(
            doc,
            session=session,
            raw_dir=tmp_path,
            prior_records={},
        )

        assert record.status == "skipped"
        assert record.sha256 == expected_sha
        session.get.assert_not_called()

    def test_skips_download_when_local_file_matches_prior_audit_record(self, tmp_path: Path):
        pdf_bytes = b"%PDF-1.7 previously audited policy"
        expected_sha = hashlib.sha256(pdf_bytes).hexdigest()

        doc = self._sample_doc()
        local_file = tmp_path / doc.filename
        local_file.write_bytes(pdf_bytes)

        prior = {
            doc.doc_id: IngestionAuditRecord(
                doc_id=doc.doc_id,
                url=doc.url,
                filename=doc.filename,
                file_size=len(pdf_bytes),
                sha256=expected_sha,
                timestamp="2026-10-24T00:00:00+00:00",
                status="downloaded",
            )
        }

        session = MagicMock()
        record = ingest_document(
            doc,
            session=session,
            raw_dir=tmp_path,
            prior_records=prior,
        )

        assert record.status == "skipped"
        session.get.assert_not_called()

    def test_detects_upstream_revision_when_hash_changes(self, tmp_path: Path):
        old_bytes = b"%PDF-1.7 version 1"
        new_bytes = b"%PDF-1.7 version 2 with change transmittal"
        old_sha = hashlib.sha256(old_bytes).hexdigest()
        new_sha = hashlib.sha256(new_bytes).hexdigest()

        doc = self._sample_doc()
        local_file = tmp_path / doc.filename
        local_file.write_bytes(old_bytes)

        session = MagicMock()
        session.get.return_value = _mock_stream_response(new_bytes)

        record = ingest_document(
            doc,
            session=session,
            raw_dir=tmp_path,
            prior_records={},
            force=True,
        )

        assert record.status == "updated"
        assert record.previous_sha256 == old_sha
        assert record.sha256 == new_sha
        assert local_file.read_bytes() == new_bytes

    def test_raises_on_pinned_sha256_mismatch(self, tmp_path: Path):
        pdf_bytes = b"%PDF-1.7 corrupted or unexpected payload"
        doc = self._sample_doc(sha256="0" * 64)

        session = MagicMock()
        session.get.return_value = _mock_stream_response(pdf_bytes)

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            ingest_document(
                doc,
                session=session,
                raw_dir=tmp_path,
                prior_records={},
            )

        assert not (tmp_path / doc.filename).exists()

    def test_skips_when_s3_object_hash_matches(self, tmp_path: Path):
        expected_sha = "a" * 64
        doc = self._sample_doc(sha256=expected_sha)
        session = MagicMock()

        with patch(
            "navy_rag.ingestion.downloader._s3_object_matches_hash",
            return_value=(True, 4096),
        ) as mock_s3_check:
            record = ingest_document(
                doc,
                session=session,
                raw_dir=tmp_path,
                prior_records={},
                s3_bucket="navy-rag-bronze",
                s3_prefix="bronze/raw",
            )

        assert record.status == "skipped"
        assert record.file_size == 4096
        assert record.sha256 == expected_sha
        mock_s3_check.assert_called_once_with(
            "navy-rag-bronze",
            "bronze/raw/dodi-1304-26.pdf",
            expected_sha,
        )
        session.get.assert_not_called()


class TestAuditManifestSidecar:
    def test_writes_and_reloads_ingestion_manifest_json(self, tmp_path: Path):
        audit_path = tmp_path / "_ingestion_manifest.json"
        record = IngestionAuditRecord(
            doc_id="comnavcruitcominst_1130_8",
            url="https://www.cnrc.navy.mil/1130.8K.pdf",
            filename="comnavcruitcominst-1130-8.pdf",
            file_size=102400,
            sha256="b" * 64,
            timestamp="2026-10-24T15:00:00+00:00",
            status="downloaded",
        )

        written = write_audit_manifest([record], audit_path)
        assert written == audit_path
        assert audit_path.is_file()

        raw_json = json.loads(audit_path.read_text(encoding="utf-8"))
        assert raw_json["document_count"] == 1
        entry = raw_json["records"][0]
        assert entry["doc_id"] == "comnavcruitcominst_1130_8"
        assert entry["url"] == "https://www.cnrc.navy.mil/1130.8K.pdf"
        assert entry["file_size"] == 102400
        assert entry["sha256"] == "b" * 64
        assert entry["timestamp"] == "2026-10-24T15:00:00+00:00"

        reloaded = load_previous_audit_records(audit_path)
        assert "comnavcruitcominst_1130_8" in reloaded
        assert reloaded["comnavcruitcominst_1130_8"].sha256 == "b" * 64

    def test_sync_corpus_end_to_end(self, tmp_path: Path):
        manifest_file = tmp_path / "corpus_manifest.yaml"
        manifest_file.write_text(
            """
schema_version: "1.0"
documents:
  - doc_id: "test_doc"
    instruction_number: "DoDI 1304.26"
    title: "Test Doc"
    source_agency: "ESD WHS"
    url: "https://www.esd.whs.mil/test.pdf"
    filename: "test.pdf"
    audience: "public"
""",
            encoding="utf-8",
        )

        raw_dir = tmp_path / "raw"
        pdf_bytes = b"%PDF-1.7 test end to end"

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.get.return_value = _mock_stream_response(pdf_bytes)

        with patch("navy_rag.ingestion.downloader._build_http_session", return_value=mock_session):
            records_first = sync_corpus(manifest_path=manifest_file, raw_dir=raw_dir)
            records_second = sync_corpus(manifest_path=manifest_file, raw_dir=raw_dir)

        assert len(records_first) == 1
        assert records_first[0].status == "downloaded"
        assert len(records_second) == 1
        assert records_second[0].status == "skipped"
        assert (raw_dir / "_ingestion_manifest.json").is_file()
