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

from navy_rag.chain.graph import run_graph
from navy_rag.config import settings

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
        "What medical conditions require a waiver?",
        "Can I join the Navy with a misdemeanor?",
        "What are the physical fitness standards?",
        "What are the age limits to enlist?",
        "Can someone with a GED enlist?",
    ]
    for q in examples:
        if st.button(q, key=f"ex_{q[:15]}", use_container_width=True, disabled=is_busy):
            st.session_state.pending_question = q

    # Distinct Reset Briefing Button
    if st.button(
        "↺ Reset Briefing / Clear Chat",
        key="btn_reset_briefing",
        type="primary",
        use_container_width=True,
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


def render_footer(scope: str, sources: list[dict]) -> None:
    """Render the verification scope indicator and cited policy sources."""
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

    sources_html = ""
    if scope == "in_scope_with_docs" and sources:
        consolidated = consolidate_sources(sources)
        if consolidated:
            items_html = ""
            for item in consolidated:
                page_part = (
                    f'<span class="source-pages">&nbsp;·&nbsp;{item["page_text"]}</span>'
                    if item["page_text"]
                    else ""
                )
                items_html += (
                    f'<div class="source-text-item">'
                    f'<span class="source-name">{html.escape(item["name"])}</span>'
                    f"{page_part}"
                    f"</div>"
                )
            sources_html = f'<div class="sources-text-list">{items_html}</div>'

    st.markdown(
        f"""
        <div class="scope-footer {cls}">
            <div class="scope-meta">
                <span class="scope-text">{text}</span>
            </div>
            {sources_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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

for msg in completed_messages:
    avatar = assistant_avatar if msg["role"] == "assistant" else user_avatar
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_footer(msg.get("scope", ""), msg.get("sources", []))

# Active turn execution
if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
    current_question = st.session_state.messages[-1]["content"]

    # Active user prompt
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(current_question)

    # Active assistant response generation
    with st.chat_message("assistant", avatar=assistant_avatar):
        st.markdown('<div class="click-shield"></div>', unsafe_allow_html=True)
        with st.spinner("Retrieving Navy recruiting instructions…"):
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
                answer = "An error occurred while querying the Navy intelligence knowledge base. Please try again."
                sources = []
                scope = "error"
                st.error(str(exc))
            finally:
                st.session_state.is_generating = False

        st.markdown(answer)
        render_footer(scope, sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "scope": scope,
        }
    )
    st.rerun()
