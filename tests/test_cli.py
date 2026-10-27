"""Unit tests for the unified navy-rag command-line interface."""
from __future__ import annotations

import pytest

from navy_rag.cli.main import build_parser


def test_cli_parser_builds():
    parser = build_parser()
    assert parser.prog == "navy-rag"


def test_cli_help_exits_cleanly():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["ingest", "--help"])
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["download", "--help"])
    assert exc_info.value.code == 0


def test_cli_download_invokes_handler():
    from unittest.mock import patch

    from navy_rag.cli.main import main

    with patch("navy_rag.cli.download.sync_corpus", return_value=[]) as mock_sync:
        code = main(["download", "--force"])
        assert code == 0
        mock_sync.assert_called_once()


def test_cli_download_all_invokes_crawlers():
    from unittest.mock import patch

    from navy_rag.cli.main import main

    with (
        patch("navy_rag.cli.download.sync_corpus", return_value=[]),
        patch("navy_rag.ingestion.crawlers.milpersman.crawl_milpersman") as mock_milpersman,
        patch("navy_rag.ingestion.crawlers.uniform_regs.crawl_uniform_regs") as mock_uniforms,
    ):
        code = main(["download", "--all"])
        assert code == 0
        mock_milpersman.assert_called_once()
        mock_uniforms.assert_called_once()

