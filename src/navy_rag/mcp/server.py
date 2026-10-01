"""Model Context Protocol (MCP) server exposing retrieval tools."""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from navy_rag.chain.graph import run_graph
from navy_rag.config import settings
from navy_rag.ingestion.embedder import get_vectorstore

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="Navy RAG",
    instructions=(
        "Demonstration intelligence and policy knowledge base for Navy recruiting. "
        "Answers questions about enlistment eligibility, moral/medical waivers, physical fitness standards, "
        "tattoo/appearance policies, ratings, ASVAB requirements, pay, and benefits using published Navy/DoD instructions. "
        "Unofficial demonstration project; not affiliated with or endorsed by the U.S. Navy."
    ),
)


def consolidate_sources(sources: list[dict]) -> list[dict]:
    """Group citations by document and consolidate page numbers (e.g., pp. 5, 12)."""
    grouped: dict[str, dict] = {}
    for src in sources:
        doc_name = (src.get("source") or "").strip()
        filename = (src.get("filename") or "").strip()
        key = filename or doc_name or "document"
        display_name = doc_name or filename or "Navy Instruction"

        if key not in grouped:
            grouped[key] = {
                "name": display_name,
                "pages": set(),
            }
        elif doc_name and grouped[key]["name"] == filename:
            grouped[key]["name"] = doc_name

        page = src.get("page")
        if page is not None:
            try:
                p_num = int(page)
                if p_num > 0:
                    grouped[key]["pages"].add(p_num)
            except (ValueError, TypeError):
                p_str = str(page).strip()
                if p_str:
                    grouped[key]["pages"].add(p_str)

    consolidated = []
    for item in grouped.values():
        pages = item["pages"]
        int_pages = sorted([p for p in pages if isinstance(p, int)])
        str_pages = sorted([str(p) for p in pages if not isinstance(p, int)])
        all_pages = [str(p) for p in int_pages] + str_pages

        if len(all_pages) == 1:
            page_text = f"p. {all_pages[0]}"
        elif len(all_pages) > 1:
            page_text = f"pp. {', '.join(all_pages)}"
        else:
            page_text = ""

        consolidated.append(
            {
                "name": item["name"],
                "page_text": page_text,
            }
        )
    return consolidated


@mcp.tool()
def search_navy_instructions(
    question: str,
    audience: str = "public",
    chat_history: list[dict] | None = None,
) -> str:
    """Query the Navy Recruiting Intelligence knowledge base and return a grounded response with citations."""
    if audience not in settings.allowed_audiences:
        return f"Invalid audience '{audience}'. Valid choices: {settings.allowed_audiences}"

    try:
        result = run_graph(question, audience=audience, chat_history=chat_history or [])
        answer = result["answer"]
        scope = result.get("scope", "")
        sources = result.get("sources", [])
    except Exception as exc:
        logger.exception("Inference failed for question: %s", question)
        return "An error occurred while querying the Navy intelligence knowledge base. Please try again."

    if scope == "in_scope_with_docs" and not sources:
        scope = "referred"

    labels = {
        "in_scope_with_docs": "Sourced from Navy Instructions",
        "in_scope_no_docs": "General Knowledge: No Policy Match",
        "out_of_scope": "Outside Navy Recruiting Scope",
        "referred": "Recruiter Referral: No Policy Match",
    }

    footer_lines = []
    scope_text = labels.get(scope)
    if scope_text:
        footer_lines.append(f"[{scope_text}]")

    if scope == "in_scope_with_docs" and sources:
        consolidated = consolidate_sources(sources)
        for item in consolidated:
            page_suffix = f" · {item['page_text']}" if item["page_text"] else ""
            footer_lines.append(f"  • {item['name']}{page_suffix}")

    if footer_lines:
        answer += "\n\n" + "\n".join(footer_lines)

    return answer


@mcp.tool()
def list_available_topics() -> str:
    """Return catalog of topic domains covered by the knowledge base."""
    topics = [
        "Enlistment eligibility — age, education, citizenship",
        "Moral character and criminal history waivers",
        "Medical conditions and physical fitness standards (PRT)",
        "Tattoo, body art, and appearance policy",
        "Navy ratings (job specialties) and ASVAB qualification scores",
        "Pay grades and basic pay (enlisted and officer)",
        "Basic Allowance for Housing (BAH) and Basic Allowance for Subsistence (BAS)",
        "Enlistment bonuses and special incentive pay",
        "TRICARE health insurance and dental/vision benefits",
        "Servicemembers Group Life Insurance (SGLI)",
        "Thrift Savings Plan (TSP) and Blended Retirement System (BRS)",
        "Dependent benefits and family support",
        "Recruit Training Command (boot camp) overview and requirements",
    ]
    return "Topics covered by the Navy RAG knowledge base:\n" + "\n".join(f"- {t}" for t in topics)


@mcp.tool()
def get_suggested_inquiries() -> list[str]:
    """Return suggested inquiries matching the recruiting intelligence interface."""
    return [
        "What is the Navy's tattoo policy?",
        "What medical conditions require a waiver?",
        "Can I join the Navy with a misdemeanor?",
        "What are the physical fitness standards?",
        "What are the age limits to enlist?",
        "Can someone with a GED enlist?",
    ]


@mcp.tool()
def get_document_summary(filename: str) -> str:
    """Retrieve metadata and sample content for a specific instruction document by filename."""
    vectorstore = get_vectorstore()
    results = vectorstore._collection.get(
        where={"filename": filename},
        limit=3,
        include=["documents", "metadatas"],
    )

    if not results["documents"]:
        return f"No document found with filename '{filename}' in the knowledge base."

    meta = results["metadatas"][0] if results["metadatas"] else {}
    source = meta.get("instruction_number") or meta.get("source") or filename

    preview = results["documents"][0][:500].strip()
    return (
        f"Document: {source}\n"
        f"Filename: {filename}\n"
        f"Audience: {meta.get('audience', 'unknown')}\n"
        f"Last ingested: {meta.get('ingested_at', 'unknown')}\n\n"
        f"Preview:\n{preview}..."
    )


if __name__ == "__main__":
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        transport = sys.argv[idx + 1]

    mcp.run(transport=transport)
