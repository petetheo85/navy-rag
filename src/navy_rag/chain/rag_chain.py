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


def _consolidate_docs_and_sources(
    retrieved_docs: list[Document],
) -> tuple[list[Document], list[dict]]:
    """Merge same-page chunks in reading order and build 1-to-1 indexed source metadata."""
    grouped: dict[tuple[str | None, int | None], list[tuple[int, Document]]] = {}
    for order_idx, doc in enumerate(retrieved_docs):
        meta = doc.metadata or {}
        filename = meta.get("filename")
        page = meta.get("page")
        key = (
            (f"_anon_{order_idx}", None)
            if (filename is None and page is None)
            else (filename, page)
        )
        grouped.setdefault(key, []).append((order_idx, doc))

    consolidated_docs: list[Document] = []
    unique_sources: list[dict] = []

    for doc_num, (_, group) in enumerate(grouped.items(), start=1):
        # Reconstruct split tables or paragraphs on the same page in chunk_index order
        sorted_group = sorted(
            group,
            key=lambda item: (
                int(item[1].metadata.get("chunk_index", item[0]))
                if item[1].metadata
                else item[0]
            ),
        )
        first_meta = dict(sorted_group[0][1].metadata or {})
        excerpts: list[str] = []
        for _, chunk_doc in sorted_group:
            text = chunk_doc.page_content.strip()
            if text and not any(text in existing for existing in excerpts):
                excerpts.append(text)

        merged_text = "\n\n...\n\n".join(excerpts)
        doc_ref = f"DOC-{doc_num}"
        first_meta["doc_ref"] = doc_ref

        consolidated_docs.append(Document(page_content=merged_text, metadata=first_meta))
        unique_sources.append(
            {
                "source": first_meta.get("instruction_number") or first_meta.get("source"),
                "filename": first_meta.get("filename"),
                "page": first_meta.get("page"),
                "title": first_meta.get("source") or first_meta.get("title") or "",
                "excerpt": merged_text,
            }
        )

    return consolidated_docs, unique_sources


def _format_context(docs: list[Document]) -> str:
    """Format retrieved chunks into tagged [DOC-N] context blocks for generation."""
    if not docs:
        return ""
    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("instruction_number") or meta.get("source", "Unknown")
        page = meta.get("page")
        doc_tag = meta.get("doc_ref") or f"DOC-{idx}"
        header = f"[{doc_tag} | {source}, p.{page}]" if page else f"[{doc_tag} | {source}]"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _filter_sources_and_clean_answer(
    raw_answer: str, sources: list[dict]
) -> tuple[str, list[dict]]:
    """Strip trailer citations from raw response and filter sources by DOC-N or page number."""
    answer = raw_answer.strip()
    cited_pages: set[int] = set()
    cited_doc_nums: set[int] = set()
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

    inline_doc_matches = re.findall(
        r"[\(\[]\s*(DOC-\d+(?:\s*,\s*DOC-\d+)*)\s*[\)\]]",
        answer,
        re.IGNORECASE,
    )
    if inline_doc_matches and not cited_text:
        cited_text = " ".join(inline_doc_matches)

    answer = re.sub(
        r"[\(\[]\s*(?:Sources?|References?)\s*:\s*[^()\[\]]+[\)\]]",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"\s*[\(\[]\s*DOC-\d+(?:\s*,\s*DOC-\d+)*\s*[\)\]]",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()

    if is_none:
        return answer, []

    if cited_text:
        for doc_num_str in re.findall(r"\bDOC-(\d+)\b", cited_text, re.IGNORECASE):
            cited_doc_nums.add(int(doc_num_str))

        if cited_doc_nums:
            filtered_by_doc = [
                s for idx, s in enumerate(sources, start=1) if idx in cited_doc_nums
            ]
            if filtered_by_doc:
                return answer, filtered_by_doc

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
    docs: list[Document] | None = None,
) -> dict:
    """Execute synchronous grounded generation for a single query."""
    if docs is not None:
        retrieved_docs = docs
    else:
        retriever = get_retriever(audience=audience)
        retrieved_docs = retriever.invoke(question)

    consolidated_docs, unique_sources = _consolidate_docs_and_sources(retrieved_docs)
    context = _format_context(consolidated_docs)
    llm = get_llm()

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history or [],
        }
    )

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
    docs: list[Document] | None = None,
) -> AsyncIterator[str]:
    """Stream grounded response tokens as they arrive from the provider."""
    if docs is not None:
        retrieved_docs = docs
    else:
        retriever = get_retriever(audience=audience)
        retrieved_docs = retriever.invoke(question)

    consolidated_docs, _ = _consolidate_docs_and_sources(retrieved_docs)
    context = _format_context(consolidated_docs)

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
