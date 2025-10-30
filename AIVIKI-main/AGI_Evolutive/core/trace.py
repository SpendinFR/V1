"""Helpers to access the active trace context."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .session_context import get_current_session


def current_trace_id(default: Optional[str] = None) -> Optional[str]:
    """Return the currently active trace identifier if any."""

    session = get_current_session()
    if session is None:
        return default
    return session.last_trace_id or default


def annotate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the active trace identifier to ``payload`` when available."""

    trace_id = current_trace_id()
    if trace_id:
        payload.setdefault("trace_id", trace_id)
    return payload


__all__ = ["annotate", "current_trace_id"]
