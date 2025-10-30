"""Error taxonomy shared across the runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class AGIError(RuntimeError):
    """Base class for orchestrator/runtime errors."""

    code = "error.generic"

    def __init__(self, message: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.cause = cause


class RetryableError(AGIError):
    code = "error.retryable"


class FatalError(AGIError):
    code = "error.fatal"


class ValidationError(AGIError):
    code = "error.validation"


@dataclass
class ErrorReport:
    """Structured view propagated to telemetry."""

    code: str
    message: str
    details: Optional[dict] = None


__all__ = [
    "AGIError",
    "RetryableError",
    "FatalError",
    "ValidationError",
    "ErrorReport",
]
