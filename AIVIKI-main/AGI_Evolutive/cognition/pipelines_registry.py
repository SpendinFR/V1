"""Registry and policy helpers for cognition pipelines.

The historical implementation provided a static in-code mapping from trigger
types to ordered lists of stages.  This module keeps the original defaults but
augments them with:

* optional loading from a JSON configuration file so that pipeline definitions
  can be tweaked without redeploying code;
* a policy layer that chooses the most appropriate pipeline variant depending
  on trigger metadata (immediacy, importance, novelty, etc.);
* reusable condition helpers that can also be leveraged by the orchestrator for
  telemetry and monitoring.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


class Stage(Enum):
    PERCEIVE = auto()
    ATTEND = auto()
    INTERPRET = auto()
    EVALUATE = auto()
    REFLECT = auto()
    REASON = auto()
    DECIDE = auto()
    ACT = auto()
    FEEDBACK = auto()
    LEARN = auto()
    UPDATE = auto()


class ActMode(Enum):
    REFLEX = auto()
    HABIT = auto()
    DELIBERATE = auto()


def _ACT(mode_or_fn: Callable[[Dict[str, Any]], ActMode] | ActMode) -> Dict[str, Any]:
    """Utility kept for backward compatibility with inline definitions."""

    return {"stage": Stage.ACT, "mode": mode_or_fn}


# --- condition helpers ----------------------------------------------------

ConditionSpec = Dict[str, Any]


def _resolve_ctx_path(ctx: Dict[str, Any], path: str) -> Any:
    """Resolve dotted paths such as ``meta.immediacy`` within a context dict."""

    parts = path.split(".")
    current: Any = ctx
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _compare(value: Any, op: str, target: Any) -> bool:
    try:
        if op == "gte":
            return value is not None and float(value) >= float(target)
        if op == "lte":
            return value is not None and float(value) <= float(target)
        if op == "gt":
            return value is not None and float(value) > float(target)
        if op == "lt":
            return value is not None and float(value) < float(target)
        if op == "eq":
            return value == target
        if op == "neq":
            return value != target
        if op == "in":
            if isinstance(target, Iterable) and not isinstance(target, (str, bytes)):
                return value in target
            return False
        if op == "contains":
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                return target in value
            if isinstance(value, str):
                return str(target) in value
            return False
    except (TypeError, ValueError):
        return False
    raise ValueError(f"Unsupported comparator '{op}' in pipeline condition")


def evaluate_condition(condition: Any, ctx: Dict[str, Any]) -> bool:
    """Evaluate declarative conditions.

    ``condition`` may be:

    * a mapping of dotted paths to comparator specs (``{"meta.immediacy":
      {"gte": 0.8}}``)
    * contain ``any_of`` / ``all_of`` / ``not`` logical operators;
    * a string referencing a predefined condition.
    """

    if not condition:
        return True

    if isinstance(condition, str):
        preset = CONDITION_PRESETS.get(condition)
        return evaluate_condition(preset, ctx) if preset is not None else False

    if isinstance(condition, list):
        return all(evaluate_condition(item, ctx) for item in condition)

    if isinstance(condition, dict):
        if "any_of" in condition:
            clauses = condition.get("any_of") or []
            return any(evaluate_condition(clause, ctx) for clause in clauses)
        if "all_of" in condition:
            clauses = condition.get("all_of") or []
            return all(evaluate_condition(clause, ctx) for clause in clauses)
        if "not" in condition:
            return not evaluate_condition(condition.get("not"), ctx)

        for key, spec in condition.items():
            value = _resolve_ctx_path(ctx, key)
            if isinstance(spec, dict):
                for op, expected in spec.items():
                    if not _compare(value, op, expected):
                        return False
            else:
                if value != spec:
                    return False
        return True

    return False


def should_skip_stage(condition: Any, ctx: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Return whether a stage should be skipped and an optional reason."""

    try:
        matched = evaluate_condition(condition, ctx)
    except Exception as exc:  # defensive: conditions come from config
        return False, f"condition_error:{exc}"
    if matched:
        if isinstance(condition, str):
            return True, condition
        return True, None
    return False, None


CONDITION_PRESETS: Dict[str, ConditionSpec] = {
    "high_immediacy": {"meta.immediacy": {"gte": 0.8}},
    "medium_immediacy": {"meta.immediacy": {"gte": 0.4}},
    "low_importance": {"meta.importance": {"lte": 0.3}},
    "high_importance": {"meta.importance": {"gte": 0.7}},
    "habitual_context": {"scratch.habit_strength": {"gte": 0.6}},
}


# --- pipeline descriptors -------------------------------------------------


@dataclass
class PipelineDescriptor:
    name: str
    family: str
    steps: List[Dict[str, Any]]
    priority: int = 100
    condition: Optional[Any] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone_with_steps(self, steps: List[Dict[str, Any]]) -> "PipelineDescriptor":
        return PipelineDescriptor(
            name=self.name,
            family=self.family,
            steps=steps,
            priority=self.priority,
            condition=self.condition,
            description=self.description,
            metadata=dict(self.metadata),
        )


@dataclass
class PolicySelection:
    name: str
    family: str
    reason: str
    descriptor: PipelineDescriptor
    llm_meta: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "pipeline": self.name,
            "family": self.family,
            "reason": self.reason,
            "metadata": dict(self.descriptor.metadata),
        }
        if self.llm_meta:
            payload["llm"] = self.llm_meta
        return payload


class PipelinePolicy:
    """Select pipeline variants based on trigger context."""

    def __init__(self, descriptors: Iterable[PipelineDescriptor]):
        self._families: Dict[str, List[PipelineDescriptor]] = {}
        for desc in descriptors:
            self._families.setdefault(desc.family, []).append(desc)
        for family in self._families.values():
            family.sort(key=lambda d: d.priority)

    def _llm_refine_selection(self, selection: PolicySelection, ctx: Dict[str, Any]) -> PolicySelection:
        payload = {
            "family": selection.family,
            "candidate": {
                "name": selection.name,
                "reason": selection.reason,
                "metadata": selection.descriptor.metadata,
            },
            "context": ctx,
            "candidates": [
                {
                    "name": desc.name,
                    "priority": desc.priority,
                    "condition": desc.condition,
                    "description": desc.description,
                }
                for desc in self._families.get(selection.family, [])
            ],
        }
        response = try_call_llm_dict(
            "cognition_pipelines_registry",
            input_payload=payload,
            logger=getattr(self, "_logger", None),
        )
        if not isinstance(response, dict):
            return selection

        descriptor = selection.descriptor
        reason = selection.reason
        target_name = response.get("pipeline")
        if isinstance(target_name, str):
            for desc in self._families.get(selection.family, []):
                if desc.name == target_name:
                    descriptor = desc
                    reason = response.get("reason", desc.description or reason)
                    break
        elif isinstance(response.get("reason"), str) and response["reason"].strip():
            reason = response["reason"].strip()

        meta: Dict[str, Any] = {"llm": response}
        if isinstance(response.get("notes"), str) and response["notes"].strip():
            meta["notes"] = response["notes"].strip()
        if "confidence" in response:
            try:
                meta["confidence"] = max(0.0, min(1.0, float(response["confidence"])))
            except (TypeError, ValueError):
                pass
        if isinstance(response.get("actions"), list):
            meta["actions"] = [str(a) for a in response["actions"] if str(a).strip()][:5]

        return PolicySelection(
            name=descriptor.name,
            family=selection.family,
            reason=reason,
            descriptor=descriptor,
            llm_meta=meta,
        )

    def select(self, family_name: str, ctx: Dict[str, Any]) -> PolicySelection:
        candidates = self._families.get(family_name)
        fallback: Optional[PipelineDescriptor] = None
        if not candidates:
            desc = PipelineDescriptor(
                name=family_name,
                family=family_name,
                steps=[],
                priority=100,
                description="direct",
            )
            selection = PolicySelection(
                name=family_name,
                family=family_name,
                reason="direct",
                descriptor=desc,
            )
            return self._llm_refine_selection(selection, ctx)

        for desc in candidates:
            if desc.condition is None and fallback is None:
                fallback = desc
            if desc.condition is None:
                continue
            if evaluate_condition(desc.condition, ctx):
                reason = desc.description or "condition_matched"
                selection = PolicySelection(
                    name=desc.name,
                    family=desc.family,
                    reason=reason,
                    descriptor=desc,
                )
                return self._llm_refine_selection(selection, ctx)

        if fallback is None:
            fallback = candidates[0]
        reason = fallback.description or "default"
        selection = PolicySelection(
            name=fallback.name,
            family=fallback.family,
            reason=reason,
            descriptor=fallback,
        )
        return self._llm_refine_selection(selection, ctx)


# --- default pipelines ----------------------------------------------------


def _mode_reflex_if_immediate(ctx: Dict[str, Any]) -> ActMode:
    immediacy = 0.0
    try:
        immediacy = float((ctx.get("meta") or {}).get("immediacy", 0.0))
    except Exception:
        immediacy = 0.0
    return ActMode.REFLEX if immediacy >= 0.8 else ActMode.DELIBERATE


def _mode_habit_if_low_importance(ctx: Dict[str, Any]) -> ActMode:
    importance = 0.5
    try:
        importance = float((ctx.get("meta") or {}).get("importance", 0.5))
    except Exception:
        importance = 0.5
    return ActMode.HABIT if importance < 0.4 else ActMode.DELIBERATE


MODE_SELECTORS: Dict[str, Callable[[Dict[str, Any]], ActMode]] = {
    "reflex_if_immediate": _mode_reflex_if_immediate,
    "habit_if_low_importance": _mode_habit_if_low_importance,
}


def _normalize_skip(condition: Any) -> Any:
    if isinstance(condition, (str, dict, list)):
        return condition
    return None


def _normalize_steps(raw_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for raw in raw_steps:
        stage_name = raw.get("stage")
        if not stage_name:
            continue
        stage = Stage[stage_name]
        entry: Dict[str, Any] = {"stage": stage}
        if "skip_if" in raw:
            entry["skip_if"] = _normalize_skip(raw.get("skip_if"))
        if stage is Stage.ACT:
            if "mode" in raw:
                entry["mode"] = ActMode[raw["mode"]]
            elif "mode_selector" in raw:
                entry["mode"] = MODE_SELECTORS.get(raw["mode_selector"])
        if "metadata" in raw:
            entry["metadata"] = dict(raw["metadata"])
        steps.append(entry)
    return steps


def _default_descriptors() -> Dict[str, PipelineDescriptor]:
    descs = {
        "THREAT": PipelineDescriptor(
            name="THREAT",
            family="THREAT",
            priority=20,
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "EVALUATE"},
                    {"stage": "REFLECT", "skip_if": {"meta.immediacy": {"gte": 0.8}}},
                    {"stage": "REASON", "skip_if": {"meta.immediacy": {"gte": 0.8}}},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode_selector": "reflex_if_immediate"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
            description="default",
        ),
        "THREAT_HIGH_IMMEDIACY": PipelineDescriptor(
            name="THREAT_HIGH_IMMEDIACY",
            family="THREAT",
            priority=0,
            condition={"meta.immediacy": {"gte": 0.8}},
            description="immediacy>=0.8",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "EVALUATE"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "REFLEX"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "GOAL": PipelineDescriptor(
            name="GOAL",
            family="GOAL",
            priority=20,
            description="default",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "EVALUATE"},
                    {"stage": "REFLECT"},
                    {"stage": "REASON"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "DELIBERATE"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "GOAL_FAST_TRACK": PipelineDescriptor(
            name="GOAL_FAST_TRACK",
            family="GOAL",
            priority=5,
            condition={"meta.immediacy": {"gte": 0.7}},
            description="fast-track",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "EVALUATE"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode_selector": "reflex_if_immediate"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "GOAL_LOW_IMPORTANCE": PipelineDescriptor(
            name="GOAL_LOW_IMPORTANCE",
            family="GOAL",
            priority=10,
            condition="low_importance",
            description="low-importance",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode_selector": "habit_if_low_importance"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "CURIOSITY": PipelineDescriptor(
            name="CURIOSITY",
            family="CURIOSITY",
            priority=20,
            description="default",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "EVALUATE"},
                    {"stage": "REFLECT"},
                    {"stage": "REASON"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "DELIBERATE"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "NEED": PipelineDescriptor(
            name="NEED",
            family="NEED",
            priority=10,
            description="default",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "EVALUATE"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "HABIT"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "SIGNAL": PipelineDescriptor(
            name="SIGNAL",
            family="SIGNAL",
            priority=10,
            description="default",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "EVALUATE"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "HABIT"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "HABIT": PipelineDescriptor(
            name="HABIT",
            family="HABIT",
            priority=10,
            description="default",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "HABIT"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "EMOTION": PipelineDescriptor(
            name="EMOTION",
            family="EMOTION",
            priority=10,
            description="default",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "EVALUATE"},
                    {"stage": "REFLECT"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "HABIT"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "MEMORY_ASSOC": PipelineDescriptor(
            name="MEMORY_ASSOC",
            family="MEMORY_ASSOC",
            priority=10,
            description="default",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "ATTEND"},
                    {"stage": "INTERPRET"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "HABIT"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "SELF_JUDGMENT": PipelineDescriptor(
            name="SELF_JUDGMENT",
            family="SELF_JUDGMENT",
            priority=10,
            description="default",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "INTERPRET"},
                    {"stage": "EVALUATE"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "HABIT"},
                    {"stage": "FEEDBACK"},
                    {"stage": "LEARN"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
        "INTROSPECTION": PipelineDescriptor(
            name="INTROSPECTION",
            family="INTROSPECTION",
            priority=10,
            description="default",
            steps=_normalize_steps(
                [
                    {"stage": "PERCEIVE"},
                    {"stage": "INTERPRET"},
                    {"stage": "REFLECT"},
                    {"stage": "REASON"},
                    {"stage": "DECIDE"},
                    {"stage": "ACT", "mode": "DELIBERATE"},
                    {"stage": "FEEDBACK"},
                    {"stage": "UPDATE"},
                ]
            ),
        ),
    }
    return descs


# --- configuration loading ------------------------------------------------


CONFIG_ENV = "AGI_PIPELINES_CONFIG"
CONFIG_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "pipelines.json"


def _load_external_descriptors() -> Dict[str, PipelineDescriptor]:
    path_str = os.environ.get(CONFIG_ENV)
    if path_str:
        candidate = Path(path_str)
    else:
        candidate = CONFIG_DEFAULT_PATH
    if not candidate.exists():
        return {}
    try:
        with candidate.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}

    ext: Dict[str, PipelineDescriptor] = {}
    for entry in data.get("pipelines", []):
        try:
            name = entry["name"]
            family = entry.get("family", name)
            steps = _normalize_steps(entry.get("steps", []))
            priority = int(entry.get("priority", 100))
            condition = entry.get("condition")
            description = entry.get("description")
            metadata = entry.get("metadata") or {}
            ext[name] = PipelineDescriptor(
                name=name,
                family=family,
                steps=steps,
                priority=priority,
                condition=condition,
                description=description,
                metadata=dict(metadata),
            )
        except Exception:
            continue
    return ext


def _merge_descriptors() -> Dict[str, PipelineDescriptor]:
    defaults = _default_descriptors()
    overrides = _load_external_descriptors()
    defaults.update(overrides)
    return defaults


def _build_registry(descriptors: Dict[str, PipelineDescriptor]) -> Dict[str, List[Dict[str, Any]]]:
    return {name: desc.steps for name, desc in descriptors.items()}


_DESCRIPTORS = _merge_descriptors()
REGISTRY: Dict[str, List[Dict[str, Any]]] = _build_registry(_DESCRIPTORS)
PIPELINE_POLICY = PipelinePolicy(_DESCRIPTORS.values())


def describe_registry() -> Dict[str, Any]:
    """Return a lightweight description of the available pipelines."""

    overview: Dict[str, Dict[str, Any]] = {}
    for desc in _DESCRIPTORS.values():
        overview.setdefault(desc.family, {"variants": []})
        overview[desc.family].setdefault("default", desc.name if desc.condition is None else overview[desc.family].get("default"))
        overview[desc.family]["variants"].append(
            {
                "name": desc.name,
                "priority": desc.priority,
                "condition": desc.condition,
                "description": desc.description,
            }
        )
    return overview


__all__ = [
    "Stage",
    "ActMode",
    "REGISTRY",
    "PIPELINE_POLICY",
    "evaluate_condition",
    "should_skip_stage",
    "describe_registry",
]
