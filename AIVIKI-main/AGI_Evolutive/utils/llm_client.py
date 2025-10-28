"""Utilities for calling local LLMs (Ollama-based) with JSON-only outputs."""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest


JSON_ONLY_DIRECTIVE = (
    "Réponds UNIQUEMENT par du JSON brut.\n"
    "N’utilise AUCUNE balise Markdown : pas de ```json ni de ``` .\n"
    "N’ajoute aucun préfixe/texte. Le 1er caractère doit être { et le dernier }"
)


def _ensure_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_json_prompt(
    task_description: str,
    *,
    input_data: Optional[Any] = None,
    extra_instructions: Optional[Sequence[str]] = None,
    example_output: Optional[Any] = None,
) -> str:
    """Compose a prompt enforcing the JSON-only instruction and optional example."""

    parts: list[str] = []
    description = task_description.strip()
    if not description:
        raise ValueError("task_description must not be empty")
    parts.append(description)

    if input_data is not None:
        serialized = _ensure_text(input_data).strip()
        if serialized:
            parts.append(f"Données disponibles:\n{serialized}")

    directive_lines = [JSON_ONLY_DIRECTIVE]
    if extra_instructions:
        directive_lines.extend(instruction.strip() for instruction in extra_instructions if instruction and instruction.strip())
    parts.append("\n".join(directive_lines))

    if example_output is not None:
        example_text = _ensure_text(example_output).strip()
        if example_text:
            parts.append(f"Exemple de sortie JSON attendue :\n{example_text}")

    return "\n\n".join(parts)


@dataclass(frozen=True)
class OllamaModelConfig:
    """Configuration describing one local Ollama model."""

    name: str
    temperature: float = 0.2
    top_p: float = 0.9
    request_timeout: float = 300.0
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class LLMResult:
    """Wrapper around the structured response produced by the LLM call."""

    parsed: Any
    raw: str
    response: Mapping[str, Any]


class LLMCallError(RuntimeError):
    """Raised when the LLM call fails or returns malformed data."""


TransportCallable = Callable[[str, bytes, float], tuple[int, bytes]]


class OllamaLLMClient:
    """Minimal client to call local Ollama models and enforce JSON outputs."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        transport: Optional[TransportCallable] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport or self._http_transport
        self._logger = logger or logging.getLogger(__name__)

    def generate_json(
        self,
        model: OllamaModelConfig,
        task_description: str,
        *,
        input_data: Optional[Any] = None,
        extra_instructions: Optional[Sequence[str]] = None,
        example_output: Optional[Any] = None,
        max_retries: int = 1,
    ) -> LLMResult:
        """Call the Ollama API and parse the JSON response."""

        prompt = build_json_prompt(
            task_description,
            input_data=input_data,
            extra_instructions=extra_instructions,
            example_output=example_output,
        )

        payload: MutableMapping[str, Any] = {
            "model": model.name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": model.temperature,
                "top_p": model.top_p,
            },
        }
        if model.system_prompt:
            payload["system"] = model.system_prompt

        attempts = max(0, int(max_retries)) + 1
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                response_data = self._post_json("/api/generate", payload, timeout=model.request_timeout)
                raw_response = response_data.get("response", "")
                if not isinstance(raw_response, str):
                    raise LLMCallError("Invalid response payload: missing 'response' text")
                parsed = json.loads(raw_response)
                return LLMResult(parsed=parsed, raw=raw_response, response=response_data)
            except (LLMCallError, json.JSONDecodeError) as exc:
                last_error = exc
                self._logger.warning("LLM call failed (attempt %s/%s): %s", attempt + 1, attempts, exc)
        raise LLMCallError("Failed to obtain valid JSON response from LLM") from last_error

    # -----------------
    # Internal helpers
    # -----------------
    def _http_transport(self, path: str, data: bytes, timeout: float) -> tuple[int, bytes]:
        url = f"{self.base_url}{path}"
        request = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlrequest.urlopen(request, timeout=timeout) as response:
                status = response.getcode()
                body = response.read()
                return status, body
        except urlerror.HTTPError as exc:  # pragma: no cover - simple passthrough
            body = exc.read() if hasattr(exc, "read") else b""
            raise LLMCallError(f"HTTP error {exc.code}: {body.decode('utf-8', errors='ignore')}") from exc
        except urlerror.URLError as exc:  # pragma: no cover - network failure
            raise LLMCallError(f"Connection error: {exc.reason}") from exc
        except socket.timeout as exc:  # pragma: no cover - timeout
            raise LLMCallError("Connection timed out") from exc
        except TimeoutError as exc:  # pragma: no cover - timeout
            raise LLMCallError("Connection timed out") from exc

    def _post_json(self, path: str, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        status, body = self._transport(path, encoded, timeout)
        if status < 200 or status >= 300:
            raise LLMCallError(f"Unexpected HTTP status {status}: {body.decode('utf-8', errors='ignore')}")
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise LLMCallError("Failed to decode Ollama response JSON") from exc


__all__ = [
    "JSON_ONLY_DIRECTIVE",
    "LLMCallError",
    "LLMResult",
    "OllamaLLMClient",
    "OllamaModelConfig",
    "build_json_prompt",
]

