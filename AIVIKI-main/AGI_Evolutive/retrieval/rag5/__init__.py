"""Helper utilities to expose RAG5 LLM integrations."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


def request_llm_guidance(
    question: str,
    *,
    config: Mapping[str, Any] | None = None,
    expansions: list[str] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> Optional[Mapping[str, Any]]:
    """Return optional LLM recommendations for the retrieval controller."""

    payload: Dict[str, Any] = {
        "question": question,
        "config": dict(config or {}),
        "expansions": list(expansions or []),
        "diagnostics": dict(diagnostics or {}),
    }

    response = try_call_llm_dict(
        "rag5_controller",
        input_payload=payload,
        logger=LOGGER,
    )

    if isinstance(response, Mapping):
        return dict(response)
    return None


__all__ = ["request_llm_guidance"]
