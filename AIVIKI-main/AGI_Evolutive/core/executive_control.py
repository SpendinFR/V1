from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


class ExecutiveControl:
    """Arbitrates whether to execute the next cognition cycle."""

    def __init__(self, arch: Any, *, logger: Optional[logging.Logger] = None) -> None:
        self.arch = arch
        self.logger = logger or LOGGER
        self._last_decision: Optional[Dict[str, Any]] = None

    def run_step(self, user_msg: Any = None, inbox_docs: Optional[list] = None) -> Any:
        """Run a control cycle, optionally deferring execution based on LLM advice."""

        inbox_docs = inbox_docs or []
        payload = {
            "user_message": user_msg,
            "inbox_count": len(inbox_docs),
            "pending_triggers": self._summarise_pending_triggers(),
            "system_load": self._read_runtime_load(),
            "last_decision": self._last_decision,
        }
        response = try_call_llm_dict(
            "executive_control",
            input_payload=payload,
            logger=self.logger,
            max_retries=2,
        )
        decision = (response or {}).get("decision", "execute")
        self._last_decision = dict(response) if response else {"decision": decision, "notes": "fallback"}

        if decision not in {"retarder", "annuler"}:
            result = self.arch.cycle(user_msg=user_msg, inbox_docs=inbox_docs)
            self._last_decision["result"] = "executed"
            return result

        self._last_decision["result"] = "skipped"
        return None

    def _summarise_pending_triggers(self) -> Dict[str, Any]:
        pending = getattr(self.arch, "pending_triggers", None)
        if not isinstance(pending, list):
            return {"count": 0}
        kinds: Dict[str, int] = {}
        for item in pending[:10]:
            kind = getattr(item, "kind", None) or getattr(item, "type", "unknown")
            kinds[kind] = kinds.get(kind, 0) + 1
        return {"count": len(pending), "by_kind": kinds}

    def _read_runtime_load(self) -> Dict[str, Any]:
        runtime = getattr(self.arch, "runtime", None)
        if runtime is None:
            return {}
        summary = {}
        for attr in ("interactive_load", "background_load", "latency"):
            value = getattr(runtime, attr, None)
            if value is not None:
                summary[attr] = value
        return summary


__all__ = ["ExecutiveControl"]
