import logging
from typing import Any, Dict, List, Mapping, Optional

from AGI_Evolutive.core.trigger_types import Trigger, TriggerType
from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_contracts import enforce_llm_contract
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


class TriggerRouter:
    def __init__(self) -> None:
        self.last_decision: Dict[str, Any] = {
            "pipelines": [],
            "secondary": [],
            "notes": None,
            "source": "fallback",
        }


    def select_pipeline(self, t: Trigger) -> str:
        fallback = self._fallback_pipeline(t)
        payload = {
            "trigger": t.type.name if isinstance(t.type, TriggerType) else str(t.type),
            "meta": json_sanitize(t.meta or {}),
            "payload": json_sanitize(t.payload or {}),
            "fallback": fallback,
        }

        llm_result = try_call_llm_dict(
            "trigger_router",
            input_payload=payload,
            logger=logger,
        )

        cleaned = enforce_llm_contract("trigger_router", llm_result)
        if cleaned is not None:
            llm_result = cleaned

        pipelines: List[str] = []
        secondary: List[str] = []
        notes: Optional[str] = None
        source = "fallback"

        if isinstance(llm_result, Mapping):
            raw_primary = llm_result.get("pipelines")
            if isinstance(raw_primary, (list, tuple)):
                pipelines = [str(item).strip() for item in raw_primary if str(item).strip()]
            raw_secondary = llm_result.get("secondary")
            if isinstance(raw_secondary, (list, tuple)):
                secondary = [str(item).strip() for item in raw_secondary if str(item).strip()]
            notes_val = llm_result.get("notes")
            if isinstance(notes_val, str) and notes_val.strip():
                notes = notes_val.strip()
            if pipelines or secondary or notes:
                source = "llm"

        if not pipelines:
            pipelines = [fallback]
        primary = pipelines[0]

        self.last_decision = {
            "trigger": payload["trigger"],
            "pipelines": pipelines,
            "secondary": secondary,
            "notes": notes,
            "source": source,
        }

        return primary

    def _fallback_pipeline(self, t: Trigger) -> str:
        if t.type is TriggerType.THREAT:
            return "THREAT"
        if t.type is TriggerType.NEED:
            return "NEED"
        if t.type is TriggerType.GOAL:
            return "GOAL"
        if t.type is TriggerType.CURIOSITY:
            return "CURIOSITY"
        if t.type is TriggerType.SIGNAL:
            return "SIGNAL"
        if t.type is TriggerType.HABIT:
            return "HABIT"
        if t.type is TriggerType.EMOTION:
            return "EMOTION"
        if t.type is TriggerType.MEMORY_ASSOC:
            return "MEMORY_ASSOC"
        if t.type is TriggerType.SELF_JUDGMENT:
            return "SELF_JUDGMENT"
        return t.type.name
