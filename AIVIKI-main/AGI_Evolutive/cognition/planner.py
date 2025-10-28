import json
import logging
import math
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Mapping

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_contracts import enforce_llm_contract
from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)

from AGI_Evolutive.core.config import cfg


def _rag_signals(rag_out: Dict[str, Any]) -> Dict[str, float]:
    if not rag_out or rag_out.get("status") != "ok":
        return {"rag_top1": 0.0, "rag_docs": 0.0, "rag_mean": 0.0, "rag_diversity": 0.0}
    cites = rag_out.get("citations") or []
    if not cites:
        return {"rag_top1": 0.0, "rag_docs": 0.0, "rag_mean": 0.0, "rag_diversity": 0.0}
    scores = [c.get("score", 0.0) for c in cites]
    top1 = max(scores) if scores else 0.0
    mean = sum(scores) / max(1, len(scores))
    div = len({c.get("doc_id") for c in cites if isinstance(c, dict)}) / max(1, len(cites))
    return {
        "rag_top1": float(top1),
        "rag_docs": float(len(cites)),
        "rag_mean": float(mean),
        "rag_diversity": float(div),
    }

_PLANS = cfg()["PLANS_PATH"]

DEFAULT_STOP_RULES: Dict[str, Any] = {"max_options": 3, "max_seconds": 900}

_BANDIT_BETAS: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
_MAX_DRIFTS = 32


LOGGER = logging.getLogger(__name__)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _ensure_dict_field(obj: Any, key: str) -> Dict[str, Any]:
    if isinstance(obj, dict):
        value = obj.get(key)
        if not isinstance(value, dict):
            value = {}
            obj[key] = value
        return value
    value = getattr(obj, key, None)
    if not isinstance(value, dict):
        value = {}
        setattr(obj, key, value)
    return value


def _set_field(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


class Planner:
    """Persisted goal planner managing micro-actions."""

    def __init__(self, architecture: Optional[Any] = None) -> None:
        self._lock = threading.RLock()
        self.state: Dict[str, Any] = {"plans": {}, "metrics": {}, "updated_at": time.time()}
        self.architecture: Optional[Any] = architecture
        self._load()
        self._ensure_metrics_structures()
        self._last_llm_plan: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Internal helpers for adaptive metrics / bandits
    # ------------------------------------------------------------------

    def _ensure_metrics_structures(self) -> None:
        with self._lock:
            metrics = self.state.setdefault("metrics", {})
            metrics.setdefault("signals", {})
            metrics.setdefault("outcomes", {"count": 0, "success": 0, "avg_duration": 0.0})
            metrics.setdefault("stop_rules", {})
            metrics.setdefault("drifts", [])

    def _signal_metrics(self, signal: str) -> Dict[str, Any]:
        metrics = self.state.setdefault("metrics", {})
        signals = metrics.setdefault("signals", {})
        data = signals.setdefault(signal, {"bandit": {}, "ema": {}, "history": []})
        bandit = data.setdefault("bandit", {})
        ema_store = data.setdefault("ema", {})
        for beta in _BANDIT_BETAS:
            key = f"{beta:.1f}"
            bandit.setdefault(key, {"alpha": 1.0, "beta": 1.0})
            ema_store.setdefault(key, 0.0)
        data.setdefault("last_used", {})
        return data

    def _select_beta(self, bandit_stats: Dict[str, Dict[str, float]]) -> str:
        best_beta = None
        best_sample = float("-inf")
        for beta_key, params in bandit_stats.items():
            alpha = max(1e-3, float(params.get("alpha", 1.0)))
            beta_param = max(1e-3, float(params.get("beta", 1.0)))
            sample = random.betavariate(alpha, beta_param)
            if sample > best_sample:
                best_sample = sample
                best_beta = beta_key
        return best_beta or next(iter(bandit_stats.keys()))

    def _apply_adaptive_ema(self, signal: str, raw_value: float) -> Tuple[float, str]:
        data = self._signal_metrics(signal)
        bandit_stats = data["bandit"]
        selected_key = self._select_beta(bandit_stats)
        beta_value = float(selected_key)
        ema_store = data["ema"]
        previous = float(ema_store.get(selected_key, raw_value))
        ema_value = beta_value * raw_value + (1.0 - beta_value) * previous
        ema_store[selected_key] = ema_value
        data["last_used"] = {
            "beta": selected_key,
            "raw": float(raw_value),
            "ema": float(ema_value),
            "ts": time.time(),
        }
        history = data.setdefault("history", [])
        history.append({"ts": time.time(), "raw": float(raw_value), "ema": float(ema_value)})
        if len(history) > 32:
            del history[0 : len(history) - 32]
        self._record_drift(signal, raw_value, ema_value)
        return float(ema_value), selected_key

    def _record_drift(self, signal: str, raw_value: float, ema_value: float) -> None:
        metrics = self.state.setdefault("metrics", {})
        drifts = metrics.setdefault("drifts", [])
        delta = abs(raw_value - ema_value)
        # Trigger drift log when EMA lags behind by more than 30% of raw magnitude
        threshold = 0.3 * (abs(raw_value) + 1e-9)
        if delta >= threshold:
            drifts.append(
                {
                    "ts": time.time(),
                    "signal": signal,
                    "raw": float(raw_value),
                    "ema": float(ema_value),
                    "delta": float(delta),
                }
            )
            if len(drifts) > _MAX_DRIFTS:
                del drifts[0 : len(drifts) - _MAX_DRIFTS]

    def _capture_signal_snapshot(self) -> Dict[str, Dict[str, Any]]:
        metrics = self.state.get("metrics", {}).get("signals", {})
        snapshot: Dict[str, Dict[str, Any]] = {}
        for signal, data in metrics.items():
            last = data.get("last_used") if isinstance(data, dict) else None
            if isinstance(last, dict) and last.get("beta") is not None:
                snapshot[signal] = {
                    "beta": last.get("beta"),
                    "ema": float(last.get("ema", 0.0)),
                    "raw": float(last.get("raw", 0.0)),
                    "ts": float(last.get("ts", time.time())),
                }
        return snapshot

    def _register_outcome(self, success: bool, elapsed: float) -> None:
        metrics = self.state.setdefault("metrics", {})
        outcomes = metrics.setdefault("outcomes", {"count": 0, "success": 0, "avg_duration": 0.0})
        outcomes["count"] = int(outcomes.get("count", 0)) + 1
        if success:
            outcomes["success"] = float(outcomes.get("success", 0.0)) + 1.0
        else:
            outcomes.setdefault("success", float(outcomes.get("success", 0.0)))
        avg_duration = float(outcomes.get("avg_duration", elapsed))
        beta = 0.3
        outcomes["avg_duration"] = beta * float(elapsed) + (1.0 - beta) * avg_duration
        history = outcomes.setdefault("history", [])
        history.append({"ts": time.time(), "success": bool(success), "duration": float(elapsed)})
        if len(history) > 64:
            del history[0 : len(history) - 64]

    def _update_signal_feedback(self, metrics: Dict[str, Any], success: bool) -> None:
        signals = metrics.get("signals") if isinstance(metrics, dict) else None
        if not isinstance(signals, dict):
            return
        for signal, payload in signals.items():
            if not isinstance(payload, dict):
                continue
            beta_key = payload.get("beta")
            estimate = payload.get("ema")
            if beta_key is None or estimate is None:
                continue
            self._register_signal_feedback(signal, str(beta_key), float(estimate), success)

    def _register_signal_feedback(
        self,
        signal: str,
        beta_key: str,
        estimate: float,
        success: bool,
    ) -> None:
        data = self._signal_metrics(signal)
        stats = data["bandit"].setdefault(beta_key, {"alpha": 1.0, "beta": 1.0})
        normalized = self._normalize_signal_estimate(signal, estimate)
        prediction = normalized >= 0.5
        if success == prediction:
            stats["alpha"] = float(stats.get("alpha", 1.0)) + 1.0
        else:
            stats["beta"] = float(stats.get("beta", 1.0)) + 1.0

    def _normalize_signal_estimate(self, signal: str, value: float) -> float:
        if signal in {"rag_docs"}:
            # more documents increases value but saturates quickly
            norm = 1.0 - math.exp(-max(0.0, value))
        elif signal in {"rag_diversity"}:
            norm = max(0.0, min(1.0, value))
        else:
            # assume scores in [0, 1] but clamp otherwise
            norm = max(0.0, min(1.0, value))
        return norm

    def _maybe_preplan_with_rag(self, frame: Any, architecture: Optional[Any]) -> Dict[str, Any]:
        out = {"rag_out": None, "rag_signals": {}, "grounded_context": []}
        try:
            arch = architecture or self.architecture
            rag = getattr(arch, "rag", None)
            if getattr(frame, "intent", None) in {"ask", "request", "summarize"} and rag:
                text = _get_field(frame, "text") or _get_field(frame, "surface_form")
                if not text:
                    return out
                rag_out = rag.ask(str(text))
                out["rag_out"] = rag_out
                if isinstance(rag_out, dict) and rag_out.get("status") == "ok":
                    cites = rag_out.get("citations") or []
                    scores = [c.get("score", 0.0) for c in cites if isinstance(c, dict)]
                    top1 = max(scores) if scores else 0.0
                    mean = sum(scores) / max(1, len(scores)) if scores else 0.0
                    div = (
                        len({c.get("doc_id") for c in cites if isinstance(c, dict)})
                        / max(1, len(cites))
                        if cites
                        else 0.0
                    )
                    raw_signals = {
                        "rag_top1": float(top1),
                        "rag_docs": float(len(cites)),
                        "rag_mean": float(mean),
                        "rag_diversity": float(div),
                    }
                    adapt_signals: Dict[str, float] = {}
                    signal_meta: Dict[str, Dict[str, Any]] = {}
                    for name, raw in raw_signals.items():
                        ema_value, beta_key = self._apply_adaptive_ema(name, raw)
                        adapt_signals[name] = float(ema_value)
                        signal_meta[name] = {
                            "raw": float(raw),
                            "ema": float(ema_value),
                            "beta": beta_key,
                        }
                    out["rag_signals"] = adapt_signals
                    out["rag_signal_meta"] = signal_meta
                    out["grounded_context"] = [
                        {
                            "doc_id": c.get("doc_id"),
                            "offsets": (c.get("start"), c.get("end")),
                            "snippet": c.get("snippet"),
                            "score": c.get("score"),
                        }
                        for c in cites
                        if isinstance(c, dict)
                    ]
        except Exception:
            pass
        return out

    def _load(self) -> None:
        if os.path.exists(_PLANS):
            try:
                with open(_PLANS, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with self._lock:
                    if isinstance(data, dict):
                        self.state.update(data)
                self._ensure_metrics_structures()
            except Exception:
                pass

    def _save(self) -> None:
        with self._lock:
            self.state["updated_at"] = time.time()
            os.makedirs(os.path.dirname(_PLANS), exist_ok=True)
            with open(_PLANS, "w", encoding="utf-8") as f:
                json.dump(json_sanitize(self.state), f, ensure_ascii=False, indent=2)

    def plan_for_goal(
        self, goal_id: str, description: str, steps: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        with self._lock:
            plan = self.state["plans"].get(goal_id)
            if not plan:
                plan = {
                    "goal_id": goal_id,
                    "description": description,
                    "steps": steps or [],
                    "created_at": time.time(),
                }
                self.state["plans"][goal_id] = plan
                self._save()
            return plan

    def _plan_with_llm(self, frame: Any) -> Optional[Dict[str, Any]]:
        if not _llm_enabled():
            return None

        goal_id = _get_field(frame, "goal_id")
        description = _get_field(frame, "description") or ""
        payload = {
            "goal": {"id": goal_id, "description": description},
            "signals": _get_field(frame, "signals", {}),
            "context": _get_field(frame, "context", {}),
        }

        try:
            response = _llm_manager().call_dict(
                "planner_support",
                input_payload=payload,
            )
            cleaned = enforce_llm_contract("planner_support", response)
            if cleaned is not None:
                response = cleaned
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM planner support unavailable", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        steps_payload = response.get("plan")
        if not isinstance(steps_payload, list):
            return None

        plan_steps: List[Dict[str, Any]] = []
        for idx, entry in enumerate(steps_payload):
            if not isinstance(entry, Mapping):
                continue
            desc = str(entry.get("description", "")).strip()
            if not desc:
                continue
            step_id = entry.get("id")
            if not isinstance(step_id, str) or not step_id:
                step_id = f"llm_step_{idx + 1}"
            depends_on = entry.get("depends_on") or []
            if isinstance(depends_on, (list, tuple)):
                deps_list = [str(dep).strip() for dep in depends_on if str(dep).strip()]
            else:
                deps_list = []
            try:
                priority = float(entry.get("priority", 0.5))
            except (TypeError, ValueError):
                priority = 0.5
            priority = max(0.05, min(1.0, priority))
            action_type = entry.get("action_type")
            if not isinstance(action_type, str) or not action_type.strip():
                action_type = self._infer_action_type(desc)
            step_context = entry.get("context") if isinstance(entry.get("context"), Mapping) else None
            action_payload = {
                "goal_id": goal_id,
                "step_id": step_id,
                "llm": True,
            }
            if step_context:
                action_payload["context"] = dict(step_context)
            plan_steps.append(
                {
                    "id": step_id,
                    "desc": desc,
                    "status": "todo",
                    "priority": priority,
                    "depends_on": deps_list,
                    "action": {"type": action_type, "payload": action_payload},
                }
            )

        if not plan_steps:
            return None

        plan_dict: Dict[str, Any] = {
            "type": "llm",
            "goal_id": goal_id,
            "description": description,
            "steps": plan_steps,
            "risks": response.get("risks"),
            "notes": response.get("notes"),
            "llm": response,
        }

        with self._lock:
            if goal_id:
                record = self.state["plans"].setdefault(
                    goal_id,
                    {"goal_id": goal_id, "description": description or "", "steps": []},
                )
                record["description"] = description or record.get("description", "")
                record["steps"] = plan_steps
                record["llm"] = response
                record["source"] = "llm"
                record["updated_at"] = time.time()
                self._save()

        self._last_llm_plan = plan_dict
        return plan_dict

    def _normalise_stop_rules(self, stop_rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        rules = dict(DEFAULT_STOP_RULES)
        metrics = self.state.get("metrics", {}).get("outcomes", {})
        count = int(metrics.get("count", 0))
        success = float(metrics.get("success", 0.0))
        success_rate = success / count if count else 0.0
        avg_duration = float(metrics.get("avg_duration", 0.0))

        # Adaptive tuning : more exploration when success rate is low
        if count and success_rate < 0.45:
            rules["max_options"] = min(6, int(round(rules["max_options"] * 1.5)))
        elif count and success_rate > 0.75:
            rules["max_options"] = max(2, int(round(rules["max_options"] * 0.8)))

        if avg_duration and avg_duration > 0:
            # keep some slack but avoid exploding runtime
            rules["max_seconds"] = max(
                120,
                int(min(1800, avg_duration * 3.0)),
            )

        if isinstance(stop_rules, dict):
            for key, value in stop_rules.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    rules[key] = value
                else:
                    rules[key] = value
        return rules

    def frame(
        self,
        goal: Any,
        stop_rules: Optional[Dict[str, Any]] = None,
        architecture: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Construit un cadre de planification pour un goal donné.

        Le ``goal`` peut être un ``GoalNode``, un dict ou simplement un identifiant.
        ``stop_rules`` est merge avec :data:`DEFAULT_STOP_RULES`.
        """

        rules = self._normalise_stop_rules(stop_rules)
        goal_id: Optional[str] = None
        description: Optional[str] = None
        context: Dict[str, Any] = {}

        if isinstance(goal, dict):
            goal_id = goal.get("id") or goal.get("goal_id") or goal.get("name")
            description = goal.get("description") or goal.get("desc")
            context = dict(goal.get("context", {})) if isinstance(goal.get("context"), dict) else {}
        elif hasattr(goal, "id"):
            goal_id = getattr(goal, "id", None)
            description = getattr(goal, "description", None)
        elif isinstance(goal, str):
            goal_id = goal

        plan: Optional[Dict[str, Any]] = None
        with self._lock:
            if goal_id:
                plan = self.state["plans"].get(goal_id)
        if goal_id and plan is None:
            desc = description or (goal_id.replace("_", " ") if isinstance(goal_id, str) else "")
            plan = self.plan_for_goal(goal_id, desc)

        frame: Dict[str, Any] = {
            "goal_id": goal_id,
            "description": description or (plan.get("description") if isinstance(plan, dict) else None),
            "stop_rules": rules,
        }
        if context:
            frame["context"] = context
        if plan:
            frame["plan"] = plan

        if architecture is not None:
            try:
                rag_bundle = self._maybe_preplan_with_rag(goal, architecture)
                if isinstance(rag_bundle, dict):
                    signals = rag_bundle.get("rag_signals") or {}
                    if signals:
                        frame.setdefault("signals", {}).update(signals)
                    meta = rag_bundle.get("rag_signal_meta")
                    if meta:
                        frame.setdefault("signals_meta", {}).update(meta)
                    grounded = rag_bundle.get("grounded_context")
                    if grounded:
                        frame.setdefault("context", {})["grounded_evidence"] = grounded
            except Exception:
                pass

        return frame

    def _infer_action_type(self, desc: str) -> str:
        text = (desc or "").lower()
        if any(token in text for token in ("question", "ask", "demande", "poser")):
            return "message_user"
        if any(token in text for token in ("observer", "observe", "regarder")):
            return "simulate"
        if any(token in text for token in ("écrire", "note", "consigner", "journaliser")):
            return "write_memory"
        if any(token in text for token in ("plan", "structurer", "organiser")):
            return "plan"
        if any(token in text for token in ("analyser", "analyze", "analyser")):
            return "reflect"
        return "reflect"

    def _step_to_action(self, goal_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
        desc = step.get("desc", "")
        action_spec = step.get("action") if isinstance(step.get("action"), dict) else {}
        payload = dict(action_spec.get("payload", {}))
        payload.setdefault("goal_id", goal_id)
        payload.setdefault("step_id", step.get("id"))
        if "context" in step and isinstance(step["context"], dict):
            payload.setdefault("context", step["context"])

        act_type = action_spec.get("type") or step.get("type") or self._infer_action_type(desc)
        action = {
            "type": act_type,
            "payload": payload,
        }
        if "mode" in action_spec:
            action["mode"] = action_spec["mode"]

        priority = step.get("priority")
        if not isinstance(priority, (int, float)):
            priority = payload.get("priority", 0.5)

        return {
            "id": step.get("id"),
            "goal_id": goal_id,
            "desc": desc,
            "status": step.get("status", "doing"),
            "priority": float(priority) if isinstance(priority, (int, float)) else 0.5,
            "action": action,
        }

    def plan(self, frame: Any, architecture: Optional[Any] = None) -> Dict[str, Any]:
        if architecture is not None:
            self.architecture = architecture
        arch = architecture or self.architecture
        rag_bundle = self._maybe_preplan_with_rag(frame, arch)

        if rag_bundle["rag_out"] and rag_bundle["rag_out"].get("status") == "refused":
            plan = {
                "type": "clarify",
                "reason": rag_bundle["rag_out"].get("reason"),
                "questions": [
                    "Peux-tu préciser la période/la source attendue ?",
                    "As-tu un lien ou document de référence ?",
                ],
            }
            _set_field(frame, "plan", plan)
            return plan

        if rag_bundle["grounded_context"]:
            context = _ensure_dict_field(frame, "context")
            context["grounded_evidence"] = rag_bundle["grounded_context"]
            signals = _ensure_dict_field(frame, "signals")
            signals.update(rag_bundle["rag_signals"])
            meta = rag_bundle.get("rag_signal_meta")
            if meta:
                _ensure_dict_field(frame, "signals_meta").update(meta)

        plan = _get_field(frame, "plan")
        if not isinstance(plan, dict) or not plan.get("steps"):
            llm_plan = self._plan_with_llm(frame)
            if llm_plan:
                _set_field(frame, "plan", llm_plan)
                return llm_plan
        if isinstance(plan, dict):
            return plan
        return {}

    def add_step(self, goal_id: str, desc: str) -> str:
        with self._lock:
            plan = self.state["plans"].setdefault(
                goal_id, {"goal_id": goal_id, "description": "", "steps": []}
            )
            step_id = f"s{len(plan['steps']) + 1}"
            plan["steps"].append({"id": step_id, "desc": desc, "status": "todo"})
            self._save()
            return step_id

    def pop_next_action(self, goal_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            plan = self.state["plans"].get(goal_id)
            if not plan:
                return None
            for step in plan["steps"]:
                status = step.get("status")
                if status not in {"todo", "blocked"}:
                    continue
                retry_at = step.get("retry_at")
                now = time.time()
                if isinstance(retry_at, (int, float)) and retry_at > now:
                    continue
                step["status"] = "doing"
                step.pop("retry_at", None)
                step["last_emitted_at"] = now
                action = self._step_to_action(goal_id, step)
                signal_snapshot = self._capture_signal_snapshot()
                if signal_snapshot:
                    step.setdefault("metrics", {})["signals"] = signal_snapshot
                self._save()
                return action
            return None

    def _result_indicates_success(self, result: Optional[Dict[str, Any]]) -> Optional[bool]:
        if not result:
            return None
        if not isinstance(result, dict):
            return None
        if result.get("success") is not None:
            return bool(result.get("success"))
        if result.get("ok") is False:
            return False
        if result.get("simulated") is False:
            # treat explicit simulator fallbacks as unsuccessful completions
            if result.get("reason") in {"simulator_unavailable", "simulator_error"}:
                return False
        if result.get("supported") is False:
            return False
        return True

    def _summarize_result(self, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        summary: Dict[str, Any] = {}
        for key in ("ok", "reason", "simulated", "supported"):
            if key in result:
                summary[key] = result[key]
        if "success" in result:
            summary["reported_success"] = result["success"]
        return summary

    def mark_action_done(
        self,
        goal_id: str,
        step_id: str,
        success: Optional[bool] = None,
        *,
        result: Optional[Dict[str, Any]] = None,
    ):
        with self._lock:
            plan = self.state.get("plans", {}).get(goal_id)
            if not plan:
                return
            for step in plan.get("steps", []):
                if step.get("id") == step_id:
                    attempts = int(step.get("attempts", 0))
                    derived = self._result_indicates_success(result)
                    success_flag: bool
                    if success is None:
                        success_flag = bool(derived if derived is not None else True)
                    else:
                        success_flag = bool(success)
                        if derived is not None:
                            success_flag = success_flag and bool(derived)
                    if success_flag:
                        step["status"] = "done"
                        step.pop("retry_at", None)
                    else:
                        attempts += 1
                        step["status"] = "todo"
                        cooldown = min(3600.0, 5.0 * (2 ** min(6, attempts - 1)))
                        step["retry_at"] = time.time() + cooldown
                    step["attempts"] = attempts
                    step["last_update"] = time.time()
                    history_entry = {
                        "ts": time.time(),
                        "event": "completed",
                        "success": success_flag,
                    }
                    summary = self._summarize_result(result)
                    if summary:
                        history_entry.update(summary)
                    if not success_flag and summary.get("reason"):
                        step["last_failure_reason"] = summary.get("reason")
                    step.setdefault("history", []).append(history_entry)
                    emitted_at = step.get("last_emitted_at")
                    if emitted_at:
                        elapsed = max(0.0, time.time() - emitted_at)
                        self._register_outcome(success_flag, elapsed)
                    if step.get("metrics"):
                        self._update_signal_feedback(step["metrics"], bool(success_flag))
                    break
            self._save()

    def pending_goals(self) -> List[Dict[str, Any]]:
        with self._lock:
            pending_statuses = {"todo", "doing", "blocked"}
            return [
                p
                for p in self.state.get("plans", {}).values()
                if any(s.get("status") in pending_statuses for s in p.get("steps", []))
            ]
    def simulate_action(self, step_desc: str) -> Dict[str, Any]:
        length = len(step_desc)
        success_prob = max(0.3, min(0.9, 1.0 - (length / 200.0)))
        expected_time = min(10.0, 1.0 + length / 80.0)
        return {"success_prob": success_prob, "expected_time": expected_time}

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def plans_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {gid: dict(plan) for gid, plan in self.state["plans"].items()}
