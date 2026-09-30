"""Retrieval-augmented generation execution and citation parsing."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from navy_rag.chain.prompts import RAG_PROMPT
from navy_rag.llm.provider import get_llm
from navy_rag.retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)


def _format_context(docs: list[Document]) -> str:
    """Format retrieved chunks into tagged context blocks for generation."""
    if not docs:
        return ""
    parts: list[str] = []
    for doc in docs:
        meta = doc.metadata
        source = meta.get("instruction_number") or meta.get("source", "Unknown")
        page = meta.get("page")
        header = f"[{source}, p.{page}]" if page else f"[{source}]"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _filter_sources_and_clean_answer(
    raw_answer: str, sources: list[dict]
) -> tuple[str, list[dict]]:
    """Strip trailer citations from raw response and filter sources to cited pages."""
    answer = raw_answer.strip()
    cited_pages: set[int] = set()
    cited_text = ""
    is_none = False

    sources_used_match = re.search(r"\n*SOURCES_USED:\s*(.+)$", answer, re.IGNORECASE)
    if sources_used_match:
        cited_text = sources_used_match.group(1).strip()
        answer = answer[: sources_used_match.start()].strip()
        if re.search(r"\bnone\b", cited_text, re.IGNORECASE):
            is_none = True

    trailing_match = re.search(
        r"(?:\n+|\s+)(?:[\(\[]\s*)?(?:Sources?|References?)\s*:\s*([^()\[\]\n]+(?:\n(?!\n)[^()\[\]\n]+)*)(?:[\)\]])?\s*$",
        answer,
        re.IGNORECASE,
    )
    if trailing_match:
        if not cited_text:
            cited_text = trailing_match.group(1).strip()
        answer = answer[: trailing_match.start()].strip()

    inline_match = re.findall(
        r"[\(\[]\s*(?:Sources?|References?)\s*:\s*([^()\[\]]+)[\)\]]",
        answer,
        re.IGNORECASE,
    )
    if inline_match and not cited_text:
        cited_text = " ".join(inline_match)

    answer = re.sub(
        r"[\(\[]\s*(?:Sources?|References?)\s*:\s*[^()\[\]]+[\)\]]",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()

    if is_none:
        return answer, []

    if cited_text:
        page_matches = re.findall(r"(?:pp?\.?|pages?)\s*([0-9,\s\-–]+)", cited_text, re.IGNORECASE)
        sections = page_matches if page_matches else [cited_text]
        for sec in sections:
            for start_str, end_str in re.findall(r"\b(\d+)\s*[-–]\s*(\d+)\b", sec):
                start, end = int(start_str), int(end_str)
                if start <= end and end - start <= 50:
                    cited_pages.update(range(start, end + 1))
            for num_str in re.findall(r"\b\d+\b", sec):
                cited_pages.add(int(num_str))

    if cited_pages:
        filtered = [s for s in sources if s.get("page") in cited_pages]
        if filtered:
            return answer, filtered

    return answer, sources


def invoke_rag(
    question: str,
    *,
    audience: str | None = None,
    chat_history: list | None = None,
) -> dict:
    """Execute synchronous grounded generation for a single query."""
    retriever = get_retriever(audience=audience)
    retrieved_docs = retriever.invoke(question)

    context = _format_context(retrieved_docs)
    llm = get_llm()

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history or [],
        }
    )

    sources = [
        {
            "source": d.metadata.get("instruction_number") or d.metadata.get("source"),
            "filename": d.metadata.get("filename"),
            "page": d.metadata.get("page"),
        }
        for d in retrieved_docs
    ]

    seen: set[tuple[str | None, int | None]] = set()
    unique_sources: list[dict] = []
    for s in sources:
        key = (s["filename"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    clean_answer, filtered_sources = _filter_sources_and_clean_answer(answer, unique_sources)
    return {
        "answer": clean_answer,
        "sources": filtered_sources,
        "has_context": bool(retrieved_docs),
    }


async def stream_rag(
    question: str,
    *,
    audience: str | None = None,
    chat_history: list | None = None,
) -> AsyncIterator[str]:
    """Stream grounded response tokens as they arrive from the provider."""
    retriever = get_retriever(audience=audience)
    retrieved_docs = retriever.invoke(question)
    context = _format_context(retrieved_docs)

    llm = get_llm(streaming=True)
    chain = RAG_PROMPT | llm | StrOutputParser()

    async for token in chain.astream(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history or [],
        }
    ):
        yield token
