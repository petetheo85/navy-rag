"""Unit tests for automated differential policy sync, cadence tiers, and hash-aware re-indexing."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from navy_rag.ingestion.downloader import (
    CorpusDocument,
    IngestionAuditRecord,
    ingest_document,
    is_cadence_due,
    load_corpus_manifest,
)
from navy_rag.ingestion.embedder import embed_and_store
from navy_rag.ingestion.sync import run_policy_sync


def _mock_stream_response(payload: bytes, headers: dict[str, str] | None = None) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.iter_content.return_value = [payload]
    response.raise_for_status.return_value = None
    response.headers = headers or {}
    return response


class TestSyncCadenceTiers:
    def test_manifest_assigns_tiered_sync_cadences(self):
        manifest = load_corpus_manifest()
        by_id = {d.doc_id: d for d in manifest.documents}

        assert by_id["comnavcruitcominst_1130_8"].sync_cadence == "weekly"
        assert by_id["srb_eligibility_chart"].sync_cadence == "weekly"
        assert by_id["dodi_6130_03_v1"].sync_cadence == "monthly"
        assert by_id["manmed_ch15"].sync_cadence == "monthly"
        assert by_id["milpersman_1133_020"].sync_cadence == "quarterly"

    def test_is_cadence_due_respects_tier_intervals(self):
        now = datetime(2026, 10, 29, 12, 0, 0, tzinfo=UTC)

        five_days_ago = (now - timedelta(days=5)).isoformat()
        ten_days_ago = (now - timedelta(days=10)).isoformat()
        forty_days_ago = (now - timedelta(days=40)).isoformat()
        hundred_days_ago = (now - timedelta(days=100)).isoformat()

        assert is_cadence_due(None, "quarterly", now=now) is True
        assert is_cadence_due(five_days_ago, "weekly", now=now) is False
        assert is_cadence_due(ten_days_ago, "weekly", now=now) is True
        assert is_cadence_due(ten_days_ago, "monthly", now=now) is False
        assert is_cadence_due(forty_days_ago, "monthly", now=now) is True
        assert is_cadence_due(forty_days_ago, "quarterly", now=now) is False
        assert is_cadence_due(hundred_days_ago, "quarterly", now=now) is True


class TestUpstreamConditionalProbing:
    def _sample_doc(self) -> CorpusDocument:
        return CorpusDocument(
            doc_id="srb_eligibility_chart",
            instruction_number="SRB Chart FY27 CH1",
            title="SRB Award Chart",
            source_agency="MyNavyHR",
            url="https://www.mynavyhr.navy.mil/srb.pdf",
            filename="srb-eligibility-chart.pdf",
            sync_cadence="weekly",
        )

    def test_skips_download_when_head_etag_matches(self, tmp_path: Path):
        pdf_bytes = b"%PDF-1.7 existing srb chart"
        sha = hashlib.sha256(pdf_bytes).hexdigest()
        doc = self._sample_doc()
        local_file = tmp_path / doc.filename
        local_file.write_bytes(pdf_bytes)

        prior = {
            doc.doc_id: IngestionAuditRecord(
                doc_id=doc.doc_id,
                url=doc.url,
                filename=doc.filename,
                file_size=len(pdf_bytes),
                sha256=sha,
                timestamp="2026-10-20T00:00:00+00:00",
                status="downloaded",
                etag='"v1-etag"',
            )
        }

        head_resp = MagicMock()
        head_resp.status_code = 200
        head_resp.headers = {"ETag": '"v1-etag"', "Content-Length": str(len(pdf_bytes))}

        session = MagicMock()
        session.head.return_value = head_resp

        record = ingest_document(
            doc,
            session=session,
            raw_dir=tmp_path,
            prior_records=prior,
            check_upstream=True,
        )

        assert record.status == "skipped"
        assert record.etag == '"v1-etag"'
        session.head.assert_called_once()
        session.get.assert_not_called()

    def test_detects_upstream_update_without_force_when_etag_changes(self, tmp_path: Path):
        old_bytes = b"%PDF-1.7 srb chart ch1"
        new_bytes = b"%PDF-1.7 srb chart ch2 updated award multiples"
        old_sha = hashlib.sha256(old_bytes).hexdigest()
        new_sha = hashlib.sha256(new_bytes).hexdigest()

        doc = self._sample_doc()
        local_file = tmp_path / doc.filename
        local_file.write_bytes(old_bytes)

        prior = {
            doc.doc_id: IngestionAuditRecord(
                doc_id=doc.doc_id,
                url=doc.url,
                filename=doc.filename,
                file_size=len(old_bytes),
                sha256=old_sha,
                timestamp="2026-10-20T00:00:00+00:00",
                status="downloaded",
                etag='"v1-etag"',
            )
        }

        head_resp = MagicMock()
        head_resp.status_code = 200
        head_resp.headers = {"ETag": '"v2-etag"', "Content-Length": str(len(new_bytes))}

        session = MagicMock()
        session.head.return_value = head_resp
        session.get.return_value = _mock_stream_response(
            new_bytes,
            headers={"ETag": '"v2-etag"', "Last-Modified": "Wed, 29 Oct 2026 10:00:00 GMT"},
        )

        record = ingest_document(
            doc,
            session=session,
            raw_dir=tmp_path,
            prior_records=prior,
            check_upstream=True,
            force=False,
        )

        assert record.status == "updated"
        assert record.previous_sha256 == old_sha
        assert record.sha256 == new_sha
        assert record.etag == '"v2-etag"'
        assert local_file.read_bytes() == new_bytes

    def test_dry_run_reports_update_without_modifying_disk(self, tmp_path: Path):
        old_bytes = b"%PDF-1.7 original"
        old_sha = hashlib.sha256(old_bytes).hexdigest()
        doc = self._sample_doc()
        local_file = tmp_path / doc.filename
        local_file.write_bytes(old_bytes)

        prior = {
            doc.doc_id: IngestionAuditRecord(
                doc_id=doc.doc_id,
                url=doc.url,
                filename=doc.filename,
                file_size=len(old_bytes),
                sha256=old_sha,
                timestamp="2026-10-20T00:00:00+00:00",
                status="downloaded",
                etag='"v1-etag"',
            )
        }

        head_resp = MagicMock()
        head_resp.status_code = 200
        head_resp.headers = {"ETag": '"v2-etag"', "Content-Length": "99999"}

        session = MagicMock()
        session.head.return_value = head_resp

        record = ingest_document(
            doc,
            session=session,
            raw_dir=tmp_path,
            prior_records=prior,
            check_upstream=True,
            dry_run=True,
        )

        assert record.status == "updated"
        assert local_file.read_bytes() == old_bytes
        session.get.assert_not_called()


class TestHashAwareDifferentialEmbedding:
    @patch("navy_rag.ingestion.embedder.get_vectorstore")
    @patch("navy_rag.retrieval.hybrid.clear_bm25_cache")
    def test_reindexes_file_when_sha256_changes_with_identical_chunk_count(
        self,
        mock_clear_bm25,
        mock_get_vs,
    ):
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs

        def fake_collection_get(*, where, include, limit=None):
            fname = where["filename"]
            if fname == "unchanged.pdf":
                if include == ["metadatas"]:
                    return {"metadatas": [{"file_sha256": "sha_same"}]}
                return {"ids": ["id1", "id2"]}
            if fname == "updated.pdf":
                if include == ["metadatas"]:
                    return {"metadatas": [{"file_sha256": "sha_old"}]}
                return {"ids": ["id3", "id4"]}
            return {"ids": [], "metadatas": []}

        mock_vs._collection.get.side_effect = fake_collection_get

        chunks = [
            Document(
                page_content="Unchanged chunk 1",
                metadata={
                    "doc_id": "doc_u",
                    "filename": "unchanged.pdf",
                    "file_sha256": "sha_same",
                    "chunk_index": 0,
                },
            ),
            Document(
                page_content="Unchanged chunk 2",
                metadata={
                    "doc_id": "doc_u",
                    "filename": "unchanged.pdf",
                    "file_sha256": "sha_same",
                    "chunk_index": 1,
                },
            ),
            Document(
                page_content="Updated chunk 1",
                metadata={
                    "doc_id": "doc_c",
                    "filename": "updated.pdf",
                    "file_sha256": "sha_new",
                    "chunk_index": 0,
                },
            ),
            Document(
                page_content="Updated chunk 2",
                metadata={
                    "doc_id": "doc_c",
                    "filename": "updated.pdf",
                    "file_sha256": "sha_new",
                    "chunk_index": 1,
                },
            ),
        ]

        with patch("navy_rag.ingestion.embedder.time.sleep"):
            stored = embed_and_store(chunks, skip_existing=True)

        assert stored == 2
        mock_vs._collection.delete.assert_called_once_with(where={"filename": "updated.pdf"})
        mock_vs.add_documents.assert_called_once()
        added_docs = mock_vs.add_documents.call_args[0][0]
        assert len(added_docs) == 2
        assert all(d.metadata["filename"] == "updated.pdf" for d in added_docs)
        mock_clear_bm25.assert_called_once()


class TestPolicySyncOrchestrator:
    def test_respects_cadence_and_reindexes_only_updated_docs(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        manifest_path = tmp_path / "corpus_manifest.yaml"
        manifest_path.write_text(
            """
schema_version: "1.0"
documents:
  - doc_id: "weekly_doc"
    instruction_number: "SRB Chart"
    title: "Weekly SRB Chart"
    source_agency: "MyNavyHR"
    url: "https://example.mil/weekly.pdf"
    filename: "weekly.pdf"
    sync_cadence: "weekly"
  - doc_id: "quarterly_doc"
    instruction_number: "MILPERSMAN 1133-020"
    title: "Quarterly MILPERSMAN"
    source_agency: "MyNavyHR"
    url: "https://example.mil/quarterly.pdf"
    filename: "quarterly.pdf"
    sync_cadence: "quarterly"
""",
            encoding="utf-8",
        )

        old_weekly = b"%PDF-1.7 old weekly"
        new_weekly = b"%PDF-1.7 new weekly updated"
        quarterly_bytes = b"%PDF-1.7 quarterly stable"

        (raw_dir / "weekly.pdf").write_bytes(old_weekly)
        (raw_dir / "quarterly.pdf").write_bytes(quarterly_bytes)

        now = datetime.now(UTC)
        ten_days_ago = (now - timedelta(days=10)).isoformat()

        from navy_rag.ingestion.downloader import write_audit_manifest

        audit_path = raw_dir / "_ingestion_manifest.json"
        write_audit_manifest(
            [
                IngestionAuditRecord(
                    doc_id="weekly_doc",
                    url="https://example.mil/weekly.pdf",
                    filename="weekly.pdf",
                    file_size=len(old_weekly),
                    sha256=hashlib.sha256(old_weekly).hexdigest(),
                    timestamp=ten_days_ago,
                    status="downloaded",
                    etag='"w1"',
                ),
                IngestionAuditRecord(
                    doc_id="quarterly_doc",
                    url="https://example.mil/quarterly.pdf",
                    filename="quarterly.pdf",
                    file_size=len(quarterly_bytes),
                    sha256=hashlib.sha256(quarterly_bytes).hexdigest(),
                    timestamp=ten_days_ago,
                    status="downloaded",
                    etag='"q1"',
                ),
            ],
            audit_path,
        )

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        head_resp = MagicMock()
        head_resp.status_code = 200
        head_resp.headers = {"ETag": '"w2"', "Content-Length": str(len(new_weekly))}
        mock_session.head.return_value = head_resp
        mock_session.get.return_value = _mock_stream_response(
            new_weekly,
            headers={"ETag": '"w2"'},
        )

        fake_doc = Document(
            page_content="Updated SRB multiples content",
            metadata={"filename": "weekly.pdf", "doc_id": "weekly_doc"},
        )

        with (
            patch(
                "navy_rag.ingestion.sync._build_http_session",
                return_value=mock_session,
            ),
            patch(
                "navy_rag.ingestion.sync.load_all",
                return_value=[fake_doc],
            ) as mock_load_all,
            patch(
                "navy_rag.ingestion.sync.embed_and_store",
                return_value=1,
            ) as mock_embed,
        ):
            report = run_policy_sync(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                audit_path=audit_path,
                report_path=raw_dir / "_sync_report.json",
                respect_cadence=True,
            )

        assert report.checked_count == 1
        assert report.updated_docs == ["weekly_doc"]
        assert report.deferred_by_cadence == ["quarterly_doc"]
        assert report.chunks_reindexed == 1
        mock_load_all.assert_called_once_with(
            data_dir=tmp_path,
            only_filenames={"weekly.pdf"},
        )
        mock_embed.assert_called_once()
        assert (raw_dir / "_sync_report.json").is_file()
