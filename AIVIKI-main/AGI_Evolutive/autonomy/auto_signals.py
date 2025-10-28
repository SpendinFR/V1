"""Registry for autonomous evaluation signals created by the agent.

This lightweight helper keeps track of every self-defined metric that an
autonomous intention introduces.  It allows any module to discover newly
created signals, push fresh observations and query the latest values without
requiring the metric to be pre-declared in the codebase.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


@dataclass
class AutoSignal:
    """Metadata tracked for a single autonomous signal."""

    action_type: str
    name: str
    metric: str
    target: float
    direction: str
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_value: Optional[float] = None
    last_source: Optional[str] = None
    last_updated: float = 0.0

    def to_observation(self) -> Optional[float]:
        if self.last_value is None:
            return None
        try:
            return float(self.last_value)
        except (TypeError, ValueError):
            return None


def extract_keywords(*chunks: str) -> List[str]:
    """Return distinct lowercase keywords from the provided chunks."""

    words: Dict[str, None] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for word in re.findall(r"[\w']+", chunk.lower()):
            if len(word) < 4:
                continue
            words.setdefault(word, None)
    return list(words.keys())


NEGATIVE_HINTS = {
    "risk",
    "stress",
    "fatigue",
    "burnout",
    "cost",
    "tension",
    "latency",
    "delay",
    "error",
    "debt",
    "conflict",
    "overload",
}


def _normalise_keyword(keyword: str) -> str:
    normalized = unicodedata.normalize("NFKD", keyword or "").encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _metric_from_keyword(token: str) -> str:
    intrinsic_suffixes = (
        "score",
        "rate",
        "ratio",
        "depth",
        "balance",
        "quality",
        "accuracy",
        "intensity",
        "alignment",
        "stability",
        "consistency",
        "fluency",
        "coverage",
        "confidence",
    )
    for suffix in intrinsic_suffixes:
        if token.endswith(suffix):
            return token

    derived_suffixes = (
        ("ship", "index"),
        ("ness", "index"),
        ("ment", "progress"),
        ("tion", "progress"),
        ("sion", "progress"),
        ("ance", "stability"),
        ("ence", "stability"),
        ("ity", "quality"),
        ("acy", "quality"),
        ("ure", "consistency"),
        ("ing", "consistency"),
    )
    for suffix, addition in derived_suffixes:
        if token.endswith(suffix):
            return f"{token}_{addition}"

    if len(token) <= 4:
        return f"{token}_score"
    return f"{token}_level"


def _direction_for_keyword(token: str) -> str:
    if any(hint in token for hint in NEGATIVE_HINTS):
        return "below"
    return "above"


def _target_for_keyword(token: str, position: int, total: int) -> float:
    length_factor = min(1.0, len(token) / 12.0)
    vowel_ratio = sum(1 for char in token if char in "aeiou") / max(1, len(token))
    order_bonus = 0.12 * (1.0 - (position / max(1, total)))
    base = 0.55 + 0.25 * length_factor + 0.1 * vowel_ratio + order_bonus
    return round(max(0.55, min(0.95, base)), 3)


def _weight_for_keyword(position: int, total: int) -> float:
    if total <= 0:
        return 1.0
    spread = max(0.0, (total - position - 1) / max(1, total - 1))
    return round(1.0 + 0.4 * spread, 3)


def _derive_signal_templates(
    base: str,
    keywords: Sequence[str],
    requirements: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    templates: List[Dict[str, Any]] = []
    seen_names: Dict[str, None] = {}

    ordered_tokens: List[tuple[str, str]] = []
    for raw_keyword in keywords:
        normalised = _normalise_keyword(raw_keyword)
        if not normalised:
            continue
        if normalised in seen_names:
            continue
        seen_names[normalised] = None
        ordered_tokens.append((raw_keyword, normalised))

    limit = min(len(ordered_tokens), 8)

    def add(
        token: str,
        metric: str,
        target: float,
        *,
        direction: str = "above",
        weight: float = 1.0,
        prefix: Optional[str] = None,
        source_keyword: Optional[str] = None,
    ) -> None:
        name_token = _normalise_keyword(token) or _normalise_keyword(metric) or "metric"
        if prefix:
            name_token = f"{prefix}_{name_token}"
        signal_name = f"{base}__{name_token}"
        if signal_name in seen_names:
            return
        seen_names[signal_name] = None
        entry = {
            "name": signal_name,
            "metric": metric,
            "target": target,
            "direction": direction,
            "weight": weight,
        }
        if source_keyword:
            entry["source_keyword"] = source_keyword
        templates.append(entry)

    for idx, (raw_keyword, token) in enumerate(ordered_tokens[:limit]):
        metric = _metric_from_keyword(token)
        direction = _direction_for_keyword(token)
        target = _target_for_keyword(token, idx, max(1, limit))
        weight = _weight_for_keyword(idx, max(1, limit))
        add(token, metric, target, direction=direction, weight=weight, source_keyword=raw_keyword)

    if requirements:
        seen_requirements: Dict[str, None] = {}
        for req in requirements:
            normalized_req = _normalise_keyword(str(req))
            if not normalized_req or normalized_req in seen_requirements:
                continue
            seen_requirements[normalized_req] = None
            metric = f"requirement:{normalized_req}"
            target = round(0.62 + 0.05 * min(len(seen_requirements), 4), 3)
            add(normalized_req, metric, target, prefix="requirement", source_keyword=str(req))

    return templates


def _heuristic_signal_derivation(
    action_type: str,
    description: str,
    *,
    requirements: Optional[Sequence[str]] = None,
    hints: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    base = str(action_type or "auto").replace(" ", "_")
    signals: List[Dict[str, Any]] = [
        {
            "name": f"{base}__success",
            "metric": "success_rate",
            "target": 0.8,
            "direction": "above",
        },
        {
            "name": f"{base}__confidence",
            "metric": "confidence",
            "target": 0.7,
            "direction": "above",
        },
        {
            "name": f"{base}__consistency",
            "metric": "consistency",
            "target": 0.6,
            "direction": "above",
        },
    ]

    chunks = [description]
    if requirements:
        chunks.append(" ".join(str(req) for req in requirements))
    if hints:
        chunks.append(" ".join(str(hint) for hint in hints))

    keywords = extract_keywords(*chunks)
    if not keywords:
        return signals

    extra = _derive_signal_templates(base, keywords, requirements=requirements)
    existing = {signal["name"] for signal in signals}
    for candidate in extra:
        if candidate.get("name") not in existing:
            signals.append(candidate)
            existing.add(candidate.get("name"))

    return signals


def _llm_signal_derivation(
    action_type: str,
    description: str,
    *,
    requirements: Optional[Sequence[str]] = None,
    hints: Optional[Sequence[str]] = None,
    baseline: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    payload = json_sanitize({
        "action_type": action_type,
        "description": description,
        "requirements": [str(req) for req in (requirements or [])],
        "hints": [str(hint) for hint in (hints or [])],
        "baseline": list(baseline or []),
    })
    response = try_call_llm_dict(
        "autonomy_signal_derivation",
        input_payload=payload,
        logger=logger,
    )
    if not isinstance(response, Mapping):
        return []

    candidates = response.get("signals")
    if not isinstance(candidates, list):
        return []

    parsed: List[Dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "").strip()
        metric = str(entry.get("metric") or name or "").strip()
        if not metric:
            continue
        if not name:
            name = metric
        try:
            target = float(entry.get("target", 0.7))
        except (TypeError, ValueError):
            target = 0.7
        try:
            weight = float(entry.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        direction = str(entry.get("direction", "above") or "above").lower()
        if direction not in {"above", "below"}:
            direction = "above"
        candidate: Dict[str, Any] = {
            "name": name,
            "metric": metric,
            "target": max(0.0, min(1.5, target)),
            "direction": direction,
            "weight": weight,
        }
        for optional_key in ("source_keyword", "notes", "rationale"):
            if entry.get(optional_key):
                candidate[optional_key] = entry[optional_key]
        parsed.append(candidate)
    return parsed


def derive_signals_for_description(
    action_type: str,
    description: str,
    *,
    requirements: Optional[Sequence[str]] = None,
    hints: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate autonomous signal definitions from textual hints."""

    baseline = _heuristic_signal_derivation(
        action_type,
        description,
        requirements=requirements,
        hints=hints,
    )
    llm_candidates = _llm_signal_derivation(
        action_type,
        description,
        requirements=requirements,
        hints=hints,
        baseline=baseline,
    )
    if not llm_candidates:
        return baseline

    existing = {(sig.get("name"), sig.get("metric")) for sig in baseline}
    for candidate in llm_candidates:
        key = (candidate.get("name"), candidate.get("metric"))
        if not candidate.get("metric"):
            continue
        if key in existing:
            continue
        baseline.append(candidate)
        existing.add(key)
    return baseline


class AutoSignalRegistry:
    """Dynamic catalogue of self-generated metrics."""

    def __init__(self) -> None:
        self._signals: Dict[str, Dict[str, AutoSignal]] = {}

    # ------------------------------------------------------------------
    def register(
        self,
        action_type: str,
        signals: Sequence[Mapping[str, Any]] | None,
        *,
        evaluation: Optional[Mapping[str, Any]] = None,
        blueprint: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
        requirements: Optional[Sequence[str]] = None,
        hints: Optional[Sequence[str]] = None,
    ) -> List[AutoSignal]:
        """Declare or update the signal definitions for an autonomous action."""

        registered: List[AutoSignal] = []
        if not action_type:
            return registered

        store = self._signals.setdefault(str(action_type), {})

        if (not signals or all(not signal for signal in signals)) and description:
            signals = derive_signals_for_description(
                action_type,
                description,
                requirements=requirements,
                hints=hints,
            )
        elif signals is None:
            signals = []

        baseline = None
        if isinstance(evaluation, Mapping):
            try:
                baseline = float(evaluation.get("score", evaluation.get("significance", 0.5)))
            except (TypeError, ValueError):
                baseline = None
        checkpoints = []
        if isinstance(blueprint, Mapping):
            checkpoints = blueprint.get("checkpoints", [])  # type: ignore[assignment]

        for signal in signals:
            if not isinstance(signal, Mapping):
                continue
            name = str(signal.get("name") or signal.get("metric") or "").strip()
            metric = str(signal.get("metric") or name).strip()
            if not metric:
                continue
            try:
                target = float(signal.get("target", 0.6) or 0.6)
            except (TypeError, ValueError):
                target = 0.6
            direction = str(signal.get("direction", "above") or "above").lower()
            try:
                weight = float(signal.get("weight", 1.0) or 1.0)
            except (TypeError, ValueError):
                weight = 1.0
            entry = store.get(metric)
            metadata = {
                "name": name or metric,
                "target": target,
                "direction": direction,
                "weight": weight,
                "source": str(signal.get("source") or "auto_evolution"),
            }
            for checkpoint in checkpoints:
                if isinstance(checkpoint, Mapping) and str(checkpoint.get("metric")) == metric:
                    metadata.setdefault("reward_if_met", checkpoint.get("reward_if_met"))
                    metadata.setdefault("penalty_if_missed", checkpoint.get("penalty_if_missed"))
            if entry is None:
                entry = AutoSignal(
                    action_type=str(action_type),
                    name=metadata["name"],
                    metric=metric,
                    target=target,
                    direction=direction,
                    weight=weight,
                    metadata=metadata,
                )
                if baseline is not None:
                    entry.last_value = baseline
                    entry.last_source = "baseline"
                    entry.last_updated = time.time()
                store[metric] = entry
            else:
                entry.target = target
                entry.direction = direction
                entry.weight = weight
                entry.metadata.update(metadata)
                if baseline is not None and entry.last_value is None:
                    entry.last_value = baseline
                    entry.last_source = "baseline"
                    entry.last_updated = time.time()
            registered.append(entry)
        return registered

    # ------------------------------------------------------------------
    def derive(
        self,
        action_type: str,
        description: str,
        *,
        requirements: Optional[Sequence[str]] = None,
        hints: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        return derive_signals_for_description(
            action_type,
            description,
            requirements=requirements,
            hints=hints,
        )
    # ------------------------------------------------------------------
    def record(
        self,
        action_type: str,
        metric: str,
        value: Any,
        *,
        source: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[AutoSignal]:
        """Store an observation for a metric, creating it on-the-fly if needed."""

        if not action_type or not metric:
            return None
        store = self._signals.setdefault(str(action_type), {})
        key = str(metric)
        entry = store.get(key)
        if entry is None:
            entry = AutoSignal(
                action_type=str(action_type),
                name=key,
                metric=key,
                target=0.6,
                direction="above",
                weight=1.0,
                metadata={"name": key, "target": 0.6, "direction": "above", "weight": 1.0},
            )
            store[key] = entry
        try:
            entry.last_value = float(value)
        except (TypeError, ValueError):
            return entry
        entry.last_source = source or entry.last_source
        entry.last_updated = timestamp or time.time()
        return entry

    # ------------------------------------------------------------------
    def bulk_record(
        self,
        action_type: str,
        metrics: Mapping[str, Any],
        *,
        source: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> List[AutoSignal]:
        """Convenience wrapper to record many metrics at once."""

        observations: List[AutoSignal] = []
        for key, value in metrics.items():
            if not key:
                continue
            entry = self.record(action_type, key, value, source=source, timestamp=timestamp)
            if entry is not None:
                observations.append(entry)
        return observations

    # ------------------------------------------------------------------
    def get_signals(self, action_type: str) -> List[AutoSignal]:
        store = self._signals.get(str(action_type))
        if not store:
            return []
        return list(store.values())

    # ------------------------------------------------------------------
    def get_observations(self, action_type: str) -> Dict[str, float]:
        observed: Dict[str, float] = {}
        for signal in self.get_signals(action_type):
            obs = signal.to_observation()
            if obs is not None:
                observed[signal.metric] = obs
        return observed

    # ------------------------------------------------------------------
    def actions(self) -> Iterable[str]:
        return list(self._signals.keys())


__all__ = [
    "AutoSignal",
    "AutoSignalRegistry",
    "derive_signals_for_description",
    "extract_keywords",
]

