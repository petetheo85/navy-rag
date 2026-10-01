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


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_provider: str
    collection: str
