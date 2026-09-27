"""Application configuration via Pydantic BaseSettings."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_dir: Path = Field(default_factory=lambda: Path(__file__).parents[2])

    @computed_field
    @property
    def data_dir(self) -> Path:
        p = self.base_dir / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @computed_field
    @property
    def chroma_db_dir(self) -> Path:
        p = self.base_dir / "chroma_db"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @computed_field
    @property
    def logs_dir(self) -> Path:
        p = self.base_dir / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Provider API keys
    google_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Observability
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "navy-rag"

    # Model configuration
    llm_provider: Literal["google_genai", "openai", "anthropic"] = "google_genai"
    llm_model: str = "gemini-3.6-flash"
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=2048, gt=0)

    # Embeddings
    embedding_provider: Literal["google", "openai", "huggingface"] = "google"
    embedding_model: str = "gemini-embedding-001"

    # Vector store
    chroma_collection: str = "navy_docs"

    # Retrieval parameters
    retrieval_k: int = Field(default=5, gt=0)
    retrieval_method: Literal["similarity", "mmr"] = "mmr"
    mmr_fetch_k: int = Field(default=20, gt=0)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)

    # Ingestion chunking
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    allowed_audiences: list[str] = ["public", "internal"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @model_validator(mode="after")
    def _validate_credentials(self) -> Settings:
        key_map = {
            "google_genai": self.google_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        if not key_map.get(self.llm_provider):
            warnings.warn(
                f"No API key configured for provider '{self.llm_provider}'.",
                stacklevel=2,
            )
        return self


settings = Settings()
