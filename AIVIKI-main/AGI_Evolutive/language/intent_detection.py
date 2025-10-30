"""Placeholder intent detection module used by integration specs tests."""

from __future__ import annotations

from typing import Any, Mapping


def classify_intent(text: str) -> Mapping[str, Any]:
    """Return a static intent classification used in offline tests."""

    return {"text": text, "intent": "inform", "confidence": 0.5}
