"""Chat model initialization across configured LLM providers."""

from __future__ import annotations

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from navy_rag.config import settings

logger = logging.getLogger(__name__)


def get_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a chat model, falling back to application settings for unspecified parameters."""
    resolved_provider = provider or settings.llm_provider
    resolved_model = model or settings.llm_model
    resolved_temp = temperature if temperature is not None else settings.llm_temperature
    resolved_max = max_tokens or settings.llm_max_tokens

    model_id = f"{resolved_provider}:{resolved_model}"
    logger.debug(
        "Initializing LLM: %s (temperature=%.2f, streaming=%s)", model_id, resolved_temp, streaming
    )

    return init_chat_model(
        model_id,
        temperature=resolved_temp,
        max_tokens=resolved_max,
        streaming=streaming,
        **kwargs,
    )
