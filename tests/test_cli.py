"""Unit tests for the unified navy command-line interface."""
from __future__ import annotations

import pytest

from navy_rag.cli.main import build_parser


def test_cli_parser_builds():
    parser = build_parser()
    assert parser.prog == "navy"


def test_cli_help_exits_cleanly():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["ingest", "--help"])
    assert exc_info.value.code == 0
