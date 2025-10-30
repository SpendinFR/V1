"""Lightweight schema validation for orchestrator payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .errors import ValidationError


@dataclass
class IntentPayload:
    intent: str
    confidence: float


@dataclass
class PlanPayload:
    summary: str
    steps: list


@dataclass
class ActionPayload:
    type: str
    payload: Mapping[str, Any]


@dataclass
class ActionResultPayload:
    status: str
    details: Mapping[str, Any]


def _expect_mapping(name: str, payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValidationError(f"{name} payload must be a mapping")
    return payload


def _float01(value: Any, *, name: str) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be numeric") from None
    if not 0.0 <= num <= 1.0:
        raise ValidationError(f"{name} must be between 0 and 1")
    return num


def validate_intent(payload: Any) -> IntentPayload:
    data = _expect_mapping("Intent", payload)
    intent = data.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise ValidationError("Intent.intent must be a non empty string")
    confidence = _float01(data.get("confidence", 0.0), name="Intent.confidence")
    return IntentPayload(intent=intent.strip(), confidence=confidence)


def validate_plan(payload: Any) -> PlanPayload:
    data = _expect_mapping("Plan", payload)
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValidationError("Plan.summary must be non empty")
    steps = data.get("steps")
    if not isinstance(steps, list) or not all(isinstance(step, Mapping) for step in steps):
        raise ValidationError("Plan.steps must be a list of mappings")
    return PlanPayload(summary=summary.strip(), steps=list(steps))


def validate_action(payload: Any) -> ActionPayload:
    data = _expect_mapping("Action", payload)
    action_type = data.get("type")
    if not isinstance(action_type, str) or not action_type.strip():
        raise ValidationError("Action.type must be non empty")
    action_payload = data.get("payload") or {}
    if not isinstance(action_payload, Mapping):
        raise ValidationError("Action.payload must be a mapping")
    return ActionPayload(type=action_type.strip(), payload=dict(action_payload))


def validate_action_result(payload: Any) -> ActionResultPayload:
    data = _expect_mapping("ActionResult", payload)
    status = data.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValidationError("ActionResult.status must be non empty")
    details = data.get("details") or {}
    if not isinstance(details, Mapping):
        raise ValidationError("ActionResult.details must be a mapping")
    return ActionResultPayload(status=status.strip(), details=dict(details))


def validate_boundary(name: str, payload: Any, validator) -> Optional[Any]:
    if payload is None:
        return None
    try:
        return validator(payload)
    except ValidationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValidationError(f"{name} payload validation failed: {exc}")


__all__ = [
    "IntentPayload",
    "PlanPayload",
    "ActionPayload",
    "ActionResultPayload",
    "validate_intent",
    "validate_plan",
    "validate_action",
    "validate_action_result",
    "validate_boundary",
]
