"""Cognition package: reward processing and related utilities."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

__all__ = [
    "reward_engine",
    "Planner",
    "Homeostasis",
    "Proposer",
    "summarize_cognition_state",
]

"""High-level cognition helpers (homeostasis, planning, metacognition, etc.)."""
"""Cognition subpackage exposing planner, homeostasis and proposer."""

from .planner import Planner  # noqa: F401  (re-export)
from .homeostasis import Homeostasis  # noqa: F401  (re-export)
from .proposer import Proposer  # noqa: F401  (re-export)


def _safe_len(value: Any) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except Exception:
        return 0


def _collect_planner_snapshot(planner: Any) -> Dict[str, Any]:
    if planner is None:
        return {}
    snapshot: Dict[str, Any] = {}
    plans: MutableMapping[str, Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] = {}
    state = getattr(planner, "state", None)
    if isinstance(state, Mapping):
        plans = state.get("plans") or {}
    if not isinstance(plans, Mapping) and isinstance(getattr(planner, "plans", None), Mapping):
        plans = getattr(planner, "plans")
    if isinstance(plans, Mapping):
        active = [
            str(pid)
            for pid, plan in plans.items()
            if isinstance(plan, Mapping) and plan.get("status") not in {"done", "archived"}
        ]
        snapshot["plans_active"] = active[:20]
        snapshot["counts"] = {
            "total": _safe_len(plans),
            "active": len(active),
            "completed": sum(
                1 for plan in plans.values() if isinstance(plan, Mapping) and plan.get("status") == "done"
            ),
        }
    metrics = getattr(planner, "metrics", None)
    if isinstance(metrics, Mapping):
        for key in ("success_rate", "latency_avg", "latency_p95"):
            if key in metrics:
                snapshot.setdefault("metrics", {})[key] = metrics[key]
    return snapshot


def _collect_feedback_snapshot(arch: Any) -> Dict[str, Any]:
    feedback = getattr(arch, "feedback", None)
    if isinstance(feedback, Mapping):
        recent = feedback.get("recent")
        if isinstance(recent, list):
            return {"recent": recent[-10:]}
    return {}


def summarize_cognition_state(arch: Any) -> Dict[str, Any]:
    """Produce an overview of cognition subsystems with an optional LLM summary."""

    planner_snapshot = _collect_planner_snapshot(getattr(arch, "planner", None))
    homeostasis = getattr(arch, "homeostasis", None)
    proposer = getattr(arch, "proposer", None)
    telemetry = getattr(arch, "telemetry", None)

    heuristics: Dict[str, Any] = {
        "planner": planner_snapshot,
        "homeostasis_state": getattr(homeostasis, "state", None),
        "proposer_backlog": _safe_len(getattr(proposer, "backlog", []) or []),
        "feedback": _collect_feedback_snapshot(arch),
    }
    if telemetry is not None:
        try:
            heuristics["telemetry_samples"] = telemetry.tail(5)  # type: ignore[attr-defined]
        except Exception:
            pass

    payload = {
        "timestamp": getattr(arch, "now", lambda: None)(),
        "heuristics": heuristics,
        "alerts": getattr(arch, "alerts", None),
    }

    logger = getattr(arch, "logger", None)
    llm_response = try_call_llm_dict(
        "cognition_overview",
        input_payload=payload,
        logger=logger,
    )

    summary = {
        "summary": "Synthèse heuristique des sous-systèmes cognition.",
        "recommended_focus": [],
        "confidence": 0.4,
        "notes": "LLM désactivé : résumé heuristique uniquement.",
        "details": heuristics,
    }

    if planner_snapshot:
        counts = planner_snapshot.get("counts", {})
        summary["recommended_focus"].append(
            f"Plans actifs: {counts.get('active', 0)}/{counts.get('total', 0)}"
        )
        summary["confidence"] = 0.55
        summary["notes"] = "Synthèse basée sur métriques internes."

    if llm_response:
        if isinstance(llm_response.get("summary"), str):
            summary["summary"] = llm_response["summary"].strip()
        if isinstance(llm_response.get("recommended_focus"), list):
            summary["recommended_focus"] = [
                str(item) for item in llm_response["recommended_focus"] if str(item).strip()
            ][:5]
        if "confidence" in llm_response:
            try:
                summary["confidence"] = max(
                    0.0, min(1.0, float(llm_response["confidence"]))
                )
            except (TypeError, ValueError):
                pass
        if isinstance(llm_response.get("notes"), str) and llm_response["notes"].strip():
            summary["notes"] = llm_response["notes"].strip()
        if llm_response.get("alerts"):
            summary["alerts"] = list(llm_response["alerts"])
        summary["llm"] = llm_response

    return summary
