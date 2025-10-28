import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


class TriggerType(Enum):
    NEED = auto()          # homeostasis (internal drives)
    GOAL = auto()          # explicit task/goal (incl. curiosity/meta-generated)
    CURIOSITY = auto()     # info gap not yet materialized as a goal
    THREAT = auto()        # danger/alert
    SIGNAL = auto()        # external interrupt / notification
    HABIT = auto()         # context-cued routine
    EMOTION = auto()       # affective peak drives priority
    MEMORY_ASSOC = auto()  # spontaneous recall/association
    SELF_JUDGMENT = auto()  # self-reflection on performance/behavior


@dataclass
class Trigger:
    type: TriggerType
    # importance, probability, reversibility, effort, uncertainty, immediacy,
    # habit_strength, deadline_ts, source, etc.
    meta: Dict[str, Any]
    payload: Optional[Dict[str, Any]] = None  # raw content (user text, goal id, memory id, etc.)


LOGGER = logging.getLogger(__name__)


def classify_trigger(
    event: Mapping[str, Any], *, default: TriggerType = TriggerType.SIGNAL
) -> Trigger:
    """Return a trigger enriched by the LLM when available."""

    heuristic = _heuristic_trigger(event, default)
    payload = {
        "event": dict(event),
        "heuristic_type": heuristic.type.name,
    }
    response = try_call_llm_dict(
        "trigger_classifier",
        input_payload=json_sanitize(payload),
        logger=LOGGER,
        max_retries=2,
    )
    if not isinstance(response, Mapping):
        return heuristic

    trigger_type = _parse_trigger_type(response.get("trigger_type")) or heuristic.type
    meta = dict(heuristic.meta)
    reason = _clean_text(response.get("reason"))
    if reason:
        meta["llm_reason"] = reason
    priority = response.get("priority")
    if isinstance(priority, (int, float)):
        meta["llm_priority"] = max(0.0, min(1.0, float(priority)))
    suggestions = response.get("suggested_actions")
    if isinstance(suggestions, (list, tuple)):
        actions = [_clean_text(item) for item in suggestions if _clean_text(item)]
        if actions:
            meta["llm_suggestions"] = actions
    notes = _clean_text(response.get("notes"))
    if notes:
        meta["llm_notes"] = notes

    return Trigger(type=trigger_type, meta=meta, payload=heuristic.payload)


def _heuristic_trigger(event: Mapping[str, Any], default: TriggerType) -> Trigger:
    text = _clean_text(event.get("text"))
    meta = dict(event.get("meta", {})) if isinstance(event.get("meta"), Mapping) else {}
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        payload = dict(payload)
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("alerte", "incident", "erreur", "panic")):
        ttype = TriggerType.THREAT
    elif any(keyword in lowered for keyword in ("besoin", "envie", "fatigue", "faim")):
        ttype = TriggerType.NEED
    elif any(keyword in lowered for keyword in ("objectif", "goal", "mission")):
        ttype = TriggerType.GOAL
    elif "curiosit" in lowered or "explorer" in lowered:
        ttype = TriggerType.CURIOSITY
    elif any(keyword in lowered for keyword in ("souvenir", "rappel", "mémoire")):
        ttype = TriggerType.MEMORY_ASSOC
    elif any(keyword in lowered for keyword in ("déçu", "joie", "colère", "stress")):
        ttype = TriggerType.EMOTION
    else:
        ttype = default
    if text:
        meta.setdefault("text", text)
    payload_dict = dict(payload) if isinstance(payload, Mapping) else None
    return Trigger(type=ttype, meta=meta, payload=payload_dict)


def _parse_trigger_type(value: Any) -> Optional[TriggerType]:
    if isinstance(value, TriggerType):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        for member in TriggerType:
            if member.name == normalized:
                return member
    return None


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return ""


__all__ = ["Trigger", "TriggerType", "classify_trigger"]
