"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_real_api_calls(monkeypatch):
    """
    Guard against accidentally hitting a live API during unit tests.

    Integration tests that need a real LLM or vector store should opt out
    by marking the test with @pytest.mark.integration.
    """
    # This fixture is intentionally left as a placeholder — individual tests
    # mock the LLM via unittest.mock. Add monkeypatch.setenv calls here if
    # you need to set test-safe API keys across the board.
