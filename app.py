"""
Streamlit chat interface for the Navy RAG system.
Fully styled in compliance with the America's Navy Brand Guidelines (CNRC).
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import streamlit.components.v1 as components

from navy_rag.chain.graph import run_graph
from navy_rag.config import settings
from navy_rag.ingestion.pdf_slicer import (
    extract_pdf_pages,
    get_document_metadata,
    render_pdf_page_image,
)

# Page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="America's Navy | Recruiting Intelligence",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "collapse_sidebar" not in st.session_state:
    st.session_state.collapse_sidebar = False

# Clean up any trailing unfulfilled user message from an interrupted rerun
while st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
    st.session_state.messages.pop()

is_busy = bool(st.session_state.is_generating)


# Static asset helpers and theme loading
def get_base64_image(image_path: str | Path) -> str:
    path = Path(image_path)
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/png;base64,{encoded}"
    return ""


ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_WHITE = get_base64_image(ASSETS_DIR / "navy_logo_white.png")
EAGLE_GOLD = get_base64_image(ASSETS_DIR / "navy_eagle_gold.png")
LOGO_TAGLINE_WHITE = get_base64_image(ASSETS_DIR / "navy_logo_tagline_white.png")

STYLE_FILE = ASSETS_DIR / "style.css"
if STYLE_FILE.exists():
    st.markdown(f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# Sidebar navigation and controls
with st.sidebar:
    # America's Navy brand lockup
    if LOGO_WHITE:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <img src="{LOGO_WHITE}" class="sidebar-logo" alt="America's Navy" />
                <p class="sidebar-tagline">Recruiting Intelligence System</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2 style="font-family:'Montserrat', sans-serif; letter-spacing:0.1em; margin:0;">AMERICA'S NAVY</h2>
                <p class="sidebar-tagline">Recruiting Intelligence System</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<p class="sidebar-section-title">Suggested Inquiries</p>', unsafe_allow_html=True)

    examples = [
        "What is the Navy's tattoo policy?",
        "What medical conditions disqualify applicants at MEPS?",
        "What are the PRT push-up and plank standards?",
        "Can I join the Navy with a GED instead of a diploma?",
        "Can someone enlist with a misdemeanor record?",
        "What physical standards apply to SEAL and Diver programs?",
        "What ASVAB scores are required for Cyber (CWT)?",
        "What is the maximum age limit to enlist in the Navy?",
        "What are the body fat percentage limits for recruits?",
        "What is the waiting period after PRK or LASIK eye surgery?",
        "Can prior-service veterans with an RE-3 code reenlist?",
        "What are the requirements for Officer Candidate School (OCS)?",
    ]
    with st.container(key="sidebar_questions"):
        for i, q in enumerate(examples):
            if st.button(q, key=f"ex_{i}", width="stretch", disabled=is_busy):
                st.session_state.pending_question = q
                st.session_state.collapse_sidebar = True

    # Distinct Reset Briefing Button
    if st.button(
        "↺ Reset Briefing / Clear Chat",
        key="btn_reset_briefing",
        type="primary",
        width="stretch",
        disabled=is_busy,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.is_generating = False
        st.rerun()

    # Anchored sidebar footer: project disclaimer and system metadata
    st.markdown(
        f"""
        <div class="sidebar-footer-anchor">
            <div class="system-specs-card">
                <div class="specs-header">
                    <span class="specs-indicator"></span> System Online
                </div>
                <div class="specs-row">
                    <span class="specs-label">Model:</span>
                    <span class="specs-val">{settings.llm_model}</span>
                </div>
                <div class="specs-row">
                    <span class="specs-label">Embeddings:</span>
                    <span class="specs-val">{settings.embedding_model}</span>
                </div>
                <div class="specs-row">
                    <span class="specs-label">Vector Store:</span>
                    <span class="specs-val">ChromaDB</span>
                </div>
                <div class="specs-row">
                    <span class="specs-label">Retrieval:</span>
                    <span class="specs-val">{settings.retrieval_method.upper()} (k={settings.retrieval_k})</span>
                </div>
            </div>
            <p class="brand-disclaimer">* Unofficial demonstration project. Not affiliated with or endorsed by the U.S. Navy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



# Header and branding
header_logo_tag = (
    f'<img src="{LOGO_TAGLINE_WHITE}" class="header-brand-logo" alt="America\'s Navy" />'
    if LOGO_TAGLINE_WHITE
    else '<h1 style="margin:0; font-size:1.6rem; letter-spacing:0.08em;">AMERICA\'S NAVY</h1>'
)

pin_img_tag = (
    f'<img src="{EAGLE_GOLD}" class="header-eagle-pin" alt="Navy Eagle" />'
    if EAGLE_GOLD
    else '<span style="color:var(--navy-gold); font-size:0.85rem;">⚓</span>'
)

st.markdown(
    f"""
    <div class="header-container">
        {header_logo_tag}
        <div class="header-subtitle">Recruiting Intelligence System</div>
        <div class="navy-chevron-bar">
            <div class="chevron-line-left"></div>
            <div class="chevron-stripes">
                <div class="chevron-stripe"></div>
                <div class="chevron-stripe"></div>
                <div class="chevron-stripe"></div>
            </div>
            {pin_img_tag}
            <div class="chevron-stripes">
                <div class="chevron-stripe"></div>
                <div class="chevron-stripe"></div>
                <div class="chevron-stripe"></div>
            </div>
            <div class="chevron-line"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Source citation consolidation and rendering
def consolidate_sources(sources: list[dict]) -> list[dict]:
    """Group citations by document and consolidate page numbers, titles, and excerpts."""
    grouped: dict[str, dict] = {}
    for src in sources:
        doc_name = (src.get("source") or "").strip()
        filename = (src.get("filename") or "").strip()
        title = (src.get("title") or "").strip()
        key = filename or doc_name or "document"
        display_name = doc_name or filename or "Navy Instruction"

        if key not in grouped:
            grouped[key] = {
                "name": display_name,
                "filename": filename,
                "title": title or display_name,
                "pages": set(),
                "excerpts": {},
            }
        elif doc_name and grouped[key]["name"] == filename:
            grouped[key]["name"] = doc_name

        page = src.get("page")
        excerpt = src.get("excerpt", "")
        if page is not None:
            try:
                p_num = int(page)
                if p_num > 0:
                    grouped[key]["pages"].add(p_num)
                    if excerpt:
                        existing = grouped[key]["excerpts"].get(p_num, "")
                        if excerpt not in existing:
                            grouped[key]["excerpts"][p_num] = (
                                f"{existing}\n\n...\n\n{excerpt}" if existing else excerpt
                            )
            except (ValueError, TypeError):
                p_str = str(page).strip()
                if p_str:
                    grouped[key]["pages"].add(p_str)
                    if excerpt:
                        grouped[key]["excerpts"][p_str] = excerpt

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
                "filename": item["filename"],
                "title": item["title"],
                "pages": int_pages,
                "page_text": page_text,
                "excerpts": item["excerpts"],
            }
        )
    return consolidated


@st.dialog("Official Policy Reference", width="large")
def show_policy_dialog(item: dict) -> None:
    """Display grounded official document page for a citation."""
    filename = item.get("filename", "")
    pages = [p for p in item.get("pages", []) if isinstance(p, int) and p > 0]

    manifest_meta = get_document_metadata(filename)
    title = manifest_meta.get("title") or item.get("title") or item.get("name") or "Navy Instruction"
    instruction_num = manifest_meta.get("instruction_number") or item.get("name") or ""

    # Avoid duplicating instruction number if title and instruction are identical
    has_distinct_instruction = (
        instruction_num
        and instruction_num.strip().lower() != title.strip().lower()
    )

    instruction_html = (
        f'<div class="dialog-doc-instruction">{html.escape(instruction_num)}</div>'
        if has_distinct_instruction
        else ""
    )

    page_summary = (
        f"Page {pages[0]}"
        if len(pages) == 1
        else f"Pages {', '.join(str(p) for p in pages)}"
    ) if pages else ""

    badge_html = f'<div class="dialog-page-badge">{html.escape(page_summary)}</div>' if page_summary else ""

    st.markdown(
        f"""
        <div class="dialog-header-card">
            <div class="dialog-header-top">
                <div class="dialog-doc-title">⚓ {html.escape(title)}</div>
                {badge_html}
            </div>
            {instruction_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pages:
        st.warning("No specific page numbers available for this reference.")
        return

    # If multiple pages are cited, provide a page selector
    if len(pages) > 1:
        selected_page = st.radio(
            "Select cited page to inspect:",
            pages,
            horizontal=True,
            format_func=lambda p: f"Page {p}",
            key=f"modal_page_select_{filename}",
        )
    else:
        selected_page = pages[0]

    page_img = render_pdf_page_image(filename, selected_page, scale=2.0)
    if page_img:
        st.image(page_img, width="stretch")
        pdf_slice_bytes = extract_pdf_pages(filename, [selected_page])
        if pdf_slice_bytes:
            stem = Path(filename).stem
            st.download_button(
                label=f"⬇ Download Page {selected_page} (PDF)",
                data=pdf_slice_bytes,
                file_name=f"{stem}_page_{selected_page}.pdf",
                mime="application/pdf",
                key=f"dl_{filename}_{selected_page}",
                width="stretch",
            )
    else:
        st.warning(f"Could not render image for page {selected_page} of {filename}.")


def render_footer(scope: str, sources: list[dict], msg_idx: int = 0) -> None:
    """Render the verification scope indicator and cited policy sources with reference link."""
    # If no sources supported the answer, treat as referred
    if scope == "in_scope_with_docs" and not sources:
        scope = "referred"

    labels = {
        "in_scope_with_docs": ("grounded", "Sourced from Navy Instructions"),
        "in_scope_no_docs": ("general", "General Knowledge: No Policy Match"),
        "out_of_scope": ("refused", "Outside Navy Recruiting Scope"),
        "referred": ("refused", "Recruiter Referral: No Policy Match"),
    }
    if scope not in labels:
        return

    cls, text = labels[scope]

    st.markdown(
        f"""
        <div class="scope-footer {cls}">
            <div class="scope-meta">
                <span class="scope-text">{text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if scope == "in_scope_with_docs" and sources:
        consolidated = consolidate_sources(sources)
        for idx, item in enumerate(consolidated):
            page_part = f"&nbsp;·&nbsp;{item['page_text']}" if item["page_text"] else ""
            col_text, col_btn = st.columns([0.76, 0.24])
            with col_text:
                st.markdown(
                    f'<div class="source-text-item">'
                    f'<span class="source-name">{html.escape(item["name"])}</span>'
                    f'<span class="source-pages">{page_part}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_btn:
                btn_key = f"cite_ref_{msg_idx}_{idx}_{item['filename']}"
                if st.button("View Reference ↗", key=btn_key, type="tertiary"):
                    show_policy_dialog(item)


# User input processing
new_prompt = st.session_state.pending_question
st.session_state.pending_question = None

if user_input := st.chat_input(
    "Ask about enlistment standards, eligibility, waivers, or ratings…",
    disabled=is_busy,
):
    new_prompt = user_input

# Append user prompt to state before rendering
if new_prompt and not is_busy:
    st.session_state.messages.append({"role": "user", "content": new_prompt})

if st.session_state.get("collapse_sidebar"):
    st.session_state.collapse_sidebar = False
    components.html(
        """
        <script>
        (function() {
            try {
                if (window.parent.innerWidth < 992) {
                    const doc = window.parent.document;
                    const btn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                             || doc.querySelector('[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button')
                             || doc.querySelector('button[aria-label="Close sidebar"]')
                             || doc.querySelector('[data-testid="stBaseButton-headerNoPadding"]');
                    if (btn) {
                        btn.click();
                    }
                }
            } catch (e) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


# Initial briefing card (shown before any user turns)
if not st.session_state.messages:
    st.markdown(
        """
        <div class="briefing-card">
            <div class="briefing-title">
                Mission Briefing: Ready for Orders
            </div>
            <div class="briefing-desc">
                Welcome to the Navy Recruiting Intelligence demonstration project.
                Inquire about enlistment eligibility criteria, education and citizenship standards,
                moral and medical waiver policies, tattoo and appearance regulations, and rating qualification guidelines.
                All responses are retrieved and cited directly from published Navy Recruiting Command instructions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Conversation history
assistant_avatar = (
    str(ASSETS_DIR / "navy_eagle_gold.png")
    if (ASSETS_DIR / "navy_eagle_gold.png").exists()
    else "⚓"
)
user_avatar = "👤"

# Completed conversation pairs
completed_messages = (
    st.session_state.messages[:-1]
    if (st.session_state.messages and st.session_state.messages[-1].get("role") == "user")
    else st.session_state.messages
)

for msg_idx, msg in enumerate(completed_messages):
    avatar = assistant_avatar if msg["role"] == "assistant" else user_avatar
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_footer(msg.get("scope", ""), msg.get("sources", []), msg_idx=msg_idx)

# Active turn execution
if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
    current_question = st.session_state.messages[-1]["content"]

    # Active user prompt
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(current_question)

    # Active assistant response generation
    with st.chat_message("assistant", avatar=assistant_avatar):
        st.markdown('<div class="click-shield"></div>', unsafe_allow_html=True)
        with st.spinner("Reviewing Navy policies…"):
            # Guarantee strict alternating turns (user/assistant) in model input
            sanitized_history = []
            for m in st.session_state.messages[:-1]:
                if m.get("role") in ("user", "assistant") and m.get("content"):
                    if not sanitized_history or sanitized_history[-1]["role"] != m["role"]:
                        sanitized_history.append({"role": m["role"], "content": m["content"]})
            if sanitized_history and sanitized_history[-1]["role"] == "user":
                sanitized_history.pop()

            try:
                st.session_state.is_generating = True
                result = run_graph(current_question, audience="public", chat_history=sanitized_history)
                answer = result["answer"]
                sources = result.get("sources", [])
                scope = result.get("scope", "")
            except Exception as exc:
                print(f"QUERY ERROR: {type(exc).__name__}: {exc}", flush=True)
                err_text = str(exc).lower()
                if any(k in err_text for k in ("resource_exhausted", "429", "quota", "rate limit")):
                    answer = (
                        "**System Advisory: Demonstration Quota Limit Reached**\n\n"
                        "This is an independent, unfunded demonstration project and has temporarily "
                        "reached its automated inquiry verification threshold.\n\n"
                        "Please pause for 30–60 seconds and submit your inquiry again, or consult with an authorized "
                        "Navy Recruiter for immediate guidance."
                    )
                else:
                    answer = (
                        "**System Advisory: Knowledge Base Temporarily Unavailable**\n\n"
                        "An unexpected processing error occurred while reviewing authoritative Navy Directives. "
                        "Please re-submit your inquiry. If this notice persists, refer to the published Navy Instructions "
                        "catalog or contact an authorized recruiter."
                    )
                sources = []
                scope = "error"
            finally:
                st.session_state.is_generating = False

        st.markdown(answer)
        render_footer(scope, sources, msg_idx=len(st.session_state.messages))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "scope": scope,
        }
    )
    st.rerun()
