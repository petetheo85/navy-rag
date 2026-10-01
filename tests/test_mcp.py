"""Smoke tests for Model Context Protocol server registration and tools."""

from __future__ import annotations


def test_mcp_server_imports_and_registers_tools():
    from navy_rag.mcp.server import (
        get_document_summary,
        list_available_topics,
        mcp,
        search_navy_instructions,
    )

    assert mcp.name == "Navy RAG"

    topics_output = list_available_topics()
    assert "Topics covered" in topics_output
    assert "Pay grades" in topics_output
    assert "Enlistment eligibility" in topics_output

    assert callable(search_navy_instructions)
    assert callable(get_document_summary)
