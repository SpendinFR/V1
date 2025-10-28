import datetime
import json
import os
from typing import Any

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_client import (
    JSON_ONLY_DIRECTIVE,
    LLMCallError,
    LLMResult,
    OllamaLLMClient,
    OllamaModelConfig,
    build_json_prompt,
)
from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMIntegrationManager,
    LLMInvocation,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
    set_llm_manager,
)
from AGI_Evolutive.utils.llm_specs import (
    AVAILABLE_MODELS,
    LLMIntegrationSpec,
    LLM_INTEGRATION_SPECS,
    SPEC_BY_KEY,
    get_spec,
)

__all__ = [
    "AVAILABLE_MODELS",
    "JSON_ONLY_DIRECTIVE",
    "LLMCallError",
    "LLMIntegrationError",
    "LLMIntegrationManager",
    "LLMInvocation",
    "LLMResult",
    "LLMUnavailableError",
    "LLMIntegrationSpec",
    "LLM_INTEGRATION_SPECS",
    "OllamaLLMClient",
    "OllamaModelConfig",
    "get_llm_manager",
    "SPEC_BY_KEY",
    "build_json_prompt",
    "get_spec",
    "is_llm_enabled",
    "json_sanitize",
    "now_iso",
    "safe_write_json",
    "set_llm_manager",
]


def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_sanitize(obj), handle, ensure_ascii=False, indent=2)
