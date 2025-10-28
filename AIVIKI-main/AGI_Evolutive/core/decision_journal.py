from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


class DecisionJournal:
    """Links triggers -> decision -> action -> outcome; persists in memory."""

    def __init__(self, memory_store=None) -> None:
        self.memory = memory_store
        self._open: Dict[str, Dict] = {}

    def new(self, decision_ctx: Dict) -> str:
        decision_id = f"dec:{uuid.uuid4().hex[:12]}"
        self._open[decision_id] = {
            "decision_id": decision_id,
            "ctx": decision_ctx,
            "ts_start": time.time(),
            "trace_id": None,
            "action": None,
            "expected_score": None,
        }
        return decision_id

    def attach_trace(self, decision_id: str, trace_id: str) -> None:
        if decision_id in self._open:
            self._open[decision_id]["trace_id"] = trace_id

    def commit_action(self, decision_id: str, action: Dict, expected_score: float) -> None:
        if decision_id in self._open:
            self._open[decision_id]["action"] = action
            self._open[decision_id]["expected_score"] = float(expected_score)

    def _llm_payload(self, decision: Dict[str, Any], obtained_score: float, latency_ms: float) -> Dict[str, Any]:
        ctx = decision.get("ctx", {}) if isinstance(decision.get("ctx"), dict) else {}
        action = decision.get("action") if isinstance(decision.get("action"), dict) else {}
        payload = {
            "trigger": ctx.get("trigger"),
            "mode": ctx.get("mode"),
            "context": {k: v for k, v in ctx.items() if k not in {"trigger", "mode"}},
            "action": action,
            "expected_score": decision.get("expected_score"),
            "obtained_score": obtained_score,
            "latency_ms": latency_ms,
        }
        trace_id = decision.get("trace_id")
        if trace_id:
            payload["trace_id"] = trace_id
        return payload

    def _llm_summary(
        self, decision: Dict[str, Any], obtained_score: float, latency_ms: float
    ) -> Optional[Dict[str, Any]]:
        response = try_call_llm_dict(
            "decision_journal",
            input_payload=self._llm_payload(decision, obtained_score, latency_ms),
            logger=LOGGER,
        )
        if isinstance(response, dict):
            return dict(response)
        return None

    def _fallback_summary(
        self, decision: Dict[str, Any], obtained_score: float, latency_ms: float
    ) -> Dict[str, Any]:
        action = decision.get("action") if isinstance(decision.get("action"), dict) else {}
        label = action.get("name") or action.get("type") or "action"
        reason = action.get("rationale") or action.get("reason")
        alternatives = []
        ctx = decision.get("ctx") if isinstance(decision.get("ctx"), dict) else {}
        fallback_reason = reason or (
            f"Décision '{label}' prise en mode {ctx.get('mode') or 'standard'}"
        )
        if isinstance(ctx.get("alternatives"), list):
            for entry in ctx["alternatives"]:
                if isinstance(entry, dict):
                    option = entry.get("option") or entry.get("action")
                    if option:
                        alternatives.append({
                            "option": option,
                            "reason": entry.get("reason") or entry.get("rationale"),
                        })
        return {
            "decision": label,
            "reason": fallback_reason,
            "expected_score": decision.get("expected_score"),
            "obtained_score": obtained_score,
            "alternatives": alternatives,
            "latency_ms": latency_ms,
            "notes": "",
        }

    def close(self, decision_id: str, obtained_score: float, latency_ms: Optional[float] = None) -> None:
        decision = self._open.get(decision_id)
        if not decision:
            return
        ts_end = time.time()
        if latency_ms is None:
            latency_ms = 1000.0 * (ts_end - decision["ts_start"])
        latency_ms = float(latency_ms)

        llm_bundle = self._llm_summary(decision, float(obtained_score), latency_ms)
        if not llm_bundle:
            llm_bundle = self._fallback_summary(decision, float(obtained_score), latency_ms)

        outcome = {
            "kind": "decision",
            "decision_id": decision_id,
            "trigger": decision["ctx"].get("trigger"),
            "mode": decision["ctx"].get("mode"),
            "action": decision.get("action"),
            "expected_score": decision.get("expected_score", 1.0),
            "obtained_score": float(obtained_score),
            "latency_ms": float(latency_ms),
            "trace_id": decision.get("trace_id"),
            "ts": ts_end,
            "llm": llm_bundle,
        }
        if self.memory is not None:
            self.memory.add(outcome)
        del self._open[decision_id]
