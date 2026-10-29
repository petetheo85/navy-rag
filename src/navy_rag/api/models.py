"""Pydantic schemas defining the API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    audience: Literal["public", "internal"] = "public"
    stream: bool = False
    chat_history: list[dict] = Field(default_factory=list)


class SourceDocument(BaseModel):
    source: str | None
    filename: str | None
    page: int | None


class ChatResponse(BaseModel):
    answer: str
    scope: str
    sources: list[SourceDocument] = Field(default_factory=list)


class IngestResponse(BaseModel):
    chunks_stored: int
    message: str


class SyncRequest(BaseModel):
    include_crawlers: bool = False
    respect_cadence: bool = True
    cadence_filter: Literal["daily", "weekly", "monthly", "quarterly"] | None = None
    dry_run: bool = False
    force: bool = False


class SyncResponse(BaseModel):
    checked_count: int
    downloaded_docs: list[str] = Field(default_factory=list)
    updated_docs: list[str] = Field(default_factory=list)
    skipped_docs: list[str] = Field(default_factory=list)
    deferred_by_cadence: list[str] = Field(default_factory=list)
    failed_docs: list[str] = Field(default_factory=list)
    chunks_reindexed: int = 0
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_provider: str
    collection: str
    retrieval_method: str = "hybrid"
    system_specs: dict[str, str] = Field(default_factory=dict)
