"""State machine orchestration for scope classification, retrieval, and generation."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from navy_rag.chain.prompts import (
    GENERAL_KNOWLEDGE_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    QUERY_REWRITE_PROMPT,
    RECRUITER_REFERRAL_MESSAGE,
    SCOPE_CLASSIFIER_PROMPT,
)
from navy_rag.chain.rag_chain import invoke_rag
from navy_rag.llm.provider import get_llm
from navy_rag.retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    question: str
    scope: str
    retrieved_docs: list[Document]
    retrieval_sufficient: bool
    answer: str
    sources: list[dict]
    chat_history: list[Any]
    audience: str | None
    retry_count: int


def classify_scope(state: RAGState) -> dict:
    """Classify incoming query intent into supported routing categories."""
    llm = get_llm(temperature=0.0)
    chain = SCOPE_CLASSIFIER_PROMPT | llm | StrOutputParser()
    scope = chain.invoke({"question": state["question"]}).strip().lower()

    valid_scopes = {"in_scope_with_docs", "in_scope_no_docs", "out_of_scope"}
    if scope not in valid_scopes:
        logger.warning("Unrecognized scope '%s'; defaulting to 'in_scope_with_docs'", scope)
        scope = "in_scope_with_docs"

    return {"scope": scope}


def retrieve(state: RAGState) -> dict:
    """Execute vector retrieval against configured audience index."""
    retriever = get_retriever(audience=state.get("audience"))
    docs = retriever.invoke(state["question"])
    return {"retrieved_docs": docs}


def evaluate_retrieval(state: RAGState) -> dict:
    """Evaluate retrieval coverage against minimal threshold."""
    docs = state.get("retrieved_docs", [])
    sufficient = len(docs) >= 2
    return {"retrieval_sufficient": sufficient}


def reformulate_query(state: RAGState) -> dict:
    """Rephrase query using formal military terminology on initial retrieval failure."""
    llm = get_llm(temperature=0.2)
    chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
    rephrased = chain.invoke({"question": state["question"]})
    return {"question": rephrased, "retry_count": state.get("retry_count", 0) + 1}


def generate_grounded(state: RAGState) -> dict:
    """Execute grounded generation and ensure claims are anchored in source text."""
    result = invoke_rag(
        state["question"],
        audience=state.get("audience"),
        chat_history=state.get("chat_history", []),
    )
    if not result["sources"]:
        logger.info("No sources cited from context; routing to recruiter referral")
        return {
            "answer": RECRUITER_REFERRAL_MESSAGE,
            "sources": [],
            "scope": "referred",
        }
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "scope": "in_scope_with_docs",
    }


def generate_general(state: RAGState) -> dict:
    """Generate ungrounded response with mandatory disclaimer for broad queries."""
    llm = get_llm()
    chain = GENERAL_KNOWLEDGE_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "question": state["question"],
            "chat_history": state.get("chat_history", []),
        }
    )
    return {"answer": answer, "sources": []}


def refer_recruiter(state: RAGState) -> dict:
    """Terminal node providing recruiter contact guidance on missing context."""
    return {
        "answer": RECRUITER_REFERRAL_MESSAGE,
        "sources": [],
        "scope": "referred",
    }


def refuse(state: RAGState) -> dict:
    """Terminal node rejecting queries outside naval recruiting domain."""
    llm = get_llm(temperature=0.3)
    chain = OUT_OF_SCOPE_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"]})
    return {"answer": answer, "sources": []}


def route_after_classify(state: RAGState) -> Literal["retrieve", "generate_general", "refuse"]:
    scope = state["scope"]
    if scope == "in_scope_with_docs":
        return "retrieve"
    if scope == "in_scope_no_docs":
        return "generate_general"
    return "refuse"


def route_after_evaluation(
    state: RAGState,
) -> Literal["generate_grounded", "reformulate_query", "refer_recruiter"]:
    if state["retrieval_sufficient"]:
        return "generate_grounded"
    if state.get("retry_count", 0) < 1:
        return "reformulate_query"
    return "refer_recruiter"


def _build_graph() -> Any:
    graph = StateGraph(RAGState)

    graph.add_node("classify_scope", classify_scope)
    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate_retrieval", evaluate_retrieval)
    graph.add_node("reformulate_query", reformulate_query)
    graph.add_node("generate_grounded", generate_grounded)
    graph.add_node("generate_general", generate_general)
    graph.add_node("refer_recruiter", refer_recruiter)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "classify_scope")
    graph.add_conditional_edges("classify_scope", route_after_classify)
    graph.add_edge("retrieve", "evaluate_retrieval")
    graph.add_conditional_edges("evaluate_retrieval", route_after_evaluation)
    graph.add_edge("reformulate_query", "retrieve")
    graph.add_edge("generate_grounded", END)
    graph.add_edge("generate_general", END)
    graph.add_edge("refer_recruiter", END)
    graph.add_edge("refuse", END)

    return graph.compile()


navy_rag_graph = _build_graph()


def run_graph(
    question: str,
    *,
    audience: str | None = None,
    chat_history: list | None = None,
) -> dict:
    """Execute the RAG workflow graph for a user query."""
    initial_state: RAGState = {
        "question": question,
        "scope": "",
        "retrieved_docs": [],
        "retrieval_sufficient": False,
        "answer": "",
        "sources": [],
        "chat_history": chat_history or [],
        "audience": audience,
        "retry_count": 0,
    }

    final_state = navy_rag_graph.invoke(initial_state)
    return {
        "answer": final_state["answer"],
        "sources": final_state.get("sources", []),
        "scope": final_state["scope"],
    }
