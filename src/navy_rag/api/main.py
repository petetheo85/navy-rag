"""FastAPI application exposing chat, health check, and administrative routes."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from navy_rag.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    SourceDocument,
    SyncRequest,
    SyncResponse,
)
from navy_rag.chain.graph import run_graph
from navy_rag.chain.rag_chain import stream_rag
from navy_rag.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Navy RAG API",
    description=(
        "Knowledge retrieval and grounded Q&A service backed by published naval instructions."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return backend service status and active index details."""
    return HealthResponse(
        llm_provider=settings.llm_provider,
        collection=settings.chroma_collection,
        retrieval_method=settings.retrieval_method,
        system_specs=settings.get_system_specs(),
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["rag"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Process an inquiry through the RAG workflow with optional token streaming."""
    if request.stream:

        async def token_stream():
            async for token in stream_rag(
                request.question,
                audience=request.audience,
                chat_history=request.chat_history,
            ):
                yield token

        return StreamingResponse(token_stream(), media_type="text/plain")

    try:
        result = run_graph(
            request.question,
            audience=request.audience,
            chat_history=request.chat_history,
        )
    except Exception as exc:
        logger.exception("Inference failed for question: %s", request.question)
        raise HTTPException(
            status_code=500,
            detail="Failed to process question. Please retry.",
        ) from exc

    return ChatResponse(
        answer=result["answer"],
        scope=result["scope"],
        sources=[SourceDocument(**s) for s in result.get("sources", [])],
    )


@app.post("/api/ingest", response_model=IngestResponse, tags=["admin"])
async def ingest() -> IngestResponse:
    """Trigger manual re-indexing of documents in the data directory."""
    from navy_rag.ingestion.chunker import chunk_documents
    from navy_rag.ingestion.embedder import embed_and_store
    from navy_rag.ingestion.loader import load_all

    docs = load_all()
    if not docs:
        return IngestResponse(chunks_stored=0, message="No documents found in data directory.")

    chunks = chunk_documents(docs)
    stored = embed_and_store(chunks)
    return IngestResponse(
        chunks_stored=stored,
        message=f"Ingestion complete. {stored} chunk(s) stored in '{settings.chroma_collection}'.",
    )


@app.post("/api/sync", response_model=SyncResponse, tags=["admin"])
async def sync(request: SyncRequest | None = None) -> SyncResponse:
    """Probe upstream directives by cadence and differentially re-index changed PDFs."""
    from navy_rag.ingestion.sync import run_policy_sync

    req = request or SyncRequest()
    report = run_policy_sync(
        include_crawlers=req.include_crawlers,
        respect_cadence=req.respect_cadence,
        cadence_filter=req.cadence_filter,
        dry_run=req.dry_run,
        force=req.force,
    )
    return SyncResponse(
        checked_count=report.checked_count,
        downloaded_docs=report.downloaded_docs,
        updated_docs=report.updated_docs,
        skipped_docs=report.skipped_docs,
        deferred_by_cadence=report.deferred_by_cadence,
        failed_docs=report.failed_docs,
        chunks_reindexed=report.chunks_reindexed,
        message=(
            f"Sync complete: {len(report.updated_docs)} updated, "
            f"{len(report.downloaded_docs)} downloaded, "
            f"{report.chunks_reindexed} chunk(s) re-indexed."
        ),
    )
