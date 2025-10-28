"""Light-weight structural causal model helpers and counterfactual simulator."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from ..utils.llm_service import try_call_llm_dict
from .structures import CausalStore, DomainSimulator, SimulationResult


logger = logging.getLogger(__name__)


@dataclass
class CounterfactualReport:
    query: Dict[str, Any]
    supported: bool
    intervention: Dict[str, Any]
    evidence: Dict[str, Any]
    simulations: List[Dict[str, Any]]
    generated_at: float
    heuristic_summary: Optional[str] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    assumptions: Optional[List[str]] = None
    checks: Optional[List[Dict[str, Any]]] = None
    actions: Optional[List[Dict[str, Any]]] = None
    alerts: Optional[List[str]] = None
    notes: Optional[str] = None
    llm_guidance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the report."""

        payload = {
            "query": self.query,
            "supported": self.supported,
            "intervention": self.intervention,
            "evidence": self.evidence,
            "simulations": self.simulations,
            "generated_at": self.generated_at,
        }
        if self.heuristic_summary is not None:
            payload["heuristic_summary"] = self.heuristic_summary
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.assumptions is not None:
            payload["assumptions"] = list(self.assumptions)
        if self.checks is not None:
            payload["checks"] = list(self.checks)
        if self.actions is not None:
            payload["actions"] = list(self.actions)
        if self.alerts is not None:
            payload["alerts"] = list(self.alerts)
        if self.notes is not None:
            payload["notes"] = self.notes
        if self.llm_guidance is not None:
            payload["llm_guidance"] = self.llm_guidance
        return payload


_MAX_TEXT_LENGTH = 600
_MAX_LIST_ITEMS = 10


def _truncate_text(value: str, *, max_length: int = _MAX_TEXT_LENGTH) -> str:
    text = str(value).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def _sanitize_payload(value: Any, *, max_length: int = _MAX_TEXT_LENGTH, max_items: int = _MAX_LIST_ITEMS) -> Any:
    if isinstance(value, Mapping):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            sanitized[str(key)] = _sanitize_payload(item, max_length=max_length, max_items=max_items)
        return sanitized
    if isinstance(value, list):
        trimmed = value[:max_items]
        return [
            _sanitize_payload(item, max_length=max_length, max_items=max_items)
            for item in trimmed
        ]
    if isinstance(value, tuple):
        return tuple(
            _sanitize_payload(item, max_length=max_length, max_items=max_items)
            for item in list(value)[:max_items]
        )
    if isinstance(value, (str, bytes)):
        return _truncate_text(value.decode() if isinstance(value, bytes) else value, max_length=max_length)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate_text(str(value), max_length=max_length)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


class SCMStore(CausalStore):
    """Causal knowledge base derived from the belief graph."""

    def __init__(
        self,
        beliefs: Any,
        ontology: Any,
        *,
        decay: float = 0.02,
        min_decay: float = 0.001,
        max_decay: float = 0.25,
        max_update_step: float = 0.2,
        drift_threshold: float = 0.12,
        drift_log_size: int = 200,
    ) -> None:
        super().__init__()
        self.beliefs = beliefs
        self.ontology = ontology
        self.base_decay = decay
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.max_update_step = max(0.01, max_update_step)
        self.decay_learning_rate = 0.08
        self.interval_smoothing = 0.25
        self.hazard_blend = 0.35
        self.drift_threshold = drift_threshold
        self.support_threshold = 0.05
        self._link_metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._drift_log: Deque[Dict[str, Any]] = deque(maxlen=drift_log_size)
        self._bootstrap_from_beliefs()

    # ------------------------------------------------------------------
    def _bootstrap_from_beliefs(self) -> None:
        if not self.beliefs:
            return
        try:
            for belief in self.beliefs.query(relation="causes", active_only=False):
                self.register(
                    belief.subject,
                    belief.value,
                    strength=float(belief.confidence),
                    description=belief.relation_type,
                )
        except Exception:
            # Belief graph may not expose the required API yet; fail softly.
            return

    # ------------------------------------------------------------------
    def register(
        self,
        cause: str,
        effect: str,
        *,
        strength: float = 0.5,
        description: Optional[str] = None,
        conditions: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Register or update a causal relation with adaptive metadata."""

        key = (cause, effect)
        now = time.time()
        metadata = self._link_metadata.get(key)

        link = None
        for candidate in self._by_cause.get(cause, []):
            if candidate.effect == effect:
                link = candidate
                break

        if metadata is None or link is None:
            link = link or self._create_link(
                cause,
                effect,
                strength=float(strength),
                description=description,
                conditions=conditions,
            )
            self._link_metadata[key] = {
                "link": link,
                "strength": self._clamp01(float(strength)),
                "decay": float(self.base_decay),
                "last_update": now,
                "last_observation": now,
                "avg_interval": None,
                "usage_count": 1,
                "drift": 0.0,
                "confidence": float(confidence) if confidence is not None else float(strength),
            }
            return

        self._apply_time_decay(key, now)

        previous_strength = metadata["strength"]
        incoming_strength = float(strength)
        error = incoming_strength - previous_strength
        step = self._bounded_step(error)
        updated_strength = self._clamp01(previous_strength + step)
        metadata["strength"] = updated_strength
        metadata["confidence"] = (
            float(confidence)
            if confidence is not None
            else 0.7 * metadata.get("confidence", updated_strength) + 0.3 * incoming_strength
        )
        metadata["last_update"] = now

        previous_observation = metadata.get("last_observation", now)
        interval = now - previous_observation
        if interval > 0:
            avg_interval = metadata.get("avg_interval")
            if avg_interval is None:
                metadata["avg_interval"] = interval
            else:
                metadata["avg_interval"] = (1 - self.interval_smoothing) * avg_interval + self.interval_smoothing * interval
        metadata["last_observation"] = now
        metadata["usage_count"] = metadata.get("usage_count", 0) + 1

        self._update_decay(metadata, reinforcement=step)
        self._record_drift(key, updated_strength - previous_strength, reason="reinforcement")

        link.strength = metadata["strength"]
        if description:
            link.description = description
        if conditions is not None:
            link.conditions = list(conditions)

    # ------------------------------------------------------------------
    def _create_link(
        self,
        cause: str,
        effect: str,
        *,
        strength: float,
        description: Optional[str],
        conditions: Optional[List[str]],
    ):
        from .structures import CausalLink

        link = CausalLink(
            cause=cause,
            effect=effect,
            strength=strength,
            description=description,
            conditions=list(conditions) if conditions else None,
        )
        self.add_link(link)
        return link

    def _bounded_step(self, delta: float) -> float:
        if delta > self.max_update_step:
            return self.max_update_step
        if delta < -self.max_update_step:
            return -self.max_update_step
        return delta

    def _clamp01(self, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _update_decay(self, metadata: Dict[str, Any], *, reinforcement: float) -> None:
        current = metadata.get("decay", self.base_decay)
        adjusted = current - self.decay_learning_rate * reinforcement
        adjusted = max(self.min_decay, min(self.max_decay, adjusted))

        avg_interval = metadata.get("avg_interval")
        if avg_interval:
            hazard = 1.0 / max(avg_interval, 1e-6)
            adjusted = (1 - self.hazard_blend) * adjusted + self.hazard_blend * hazard
        metadata["decay"] = max(self.min_decay, min(self.max_decay, adjusted))

    def _apply_time_decay(self, key: Tuple[str, str], now: Optional[float] = None) -> None:
        metadata = self._link_metadata.get(key)
        if not metadata:
            return
        now = now or time.time()
        last_update = metadata.get("last_update", now)
        elapsed = max(0.0, now - last_update)
        if elapsed <= 0.0:
            return
        decay = metadata.get("decay", self.base_decay)
        decayed_strength = metadata["strength"] * math.exp(-decay * elapsed)
        decayed_strength = self._clamp01(decayed_strength)
        if abs(decayed_strength - metadata["strength"]) < 1e-6:
            metadata["last_update"] = now
            return

        previous_strength = metadata["strength"]
        metadata["strength"] = decayed_strength
        metadata["last_update"] = now
        metadata["link"].strength = decayed_strength
        self._record_drift(key, decayed_strength - previous_strength, reason="decay")

    def _record_drift(self, key: Tuple[str, str], delta: float, *, reason: str) -> None:
        metadata = self._link_metadata.get(key)
        if not metadata:
            return
        drift = metadata.get("drift", 0.0)
        metadata["drift"] = 0.8 * drift + 0.2 * abs(delta)
        magnitude = abs(delta)
        if magnitude < self.drift_threshold:
            return
        cause, effect = key
        self._drift_log.append(
            {
                "cause": cause,
                "effect": effect,
                "delta": delta,
                "reason": reason,
                "timestamp": time.time(),
                "strength": metadata.get("strength"),
                "decay": metadata.get("decay"),
            }
        )

    # ------------------------------------------------------------------
    def test_relation(
        self,
        cause: str,
        effect: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        now = time.time()
        links = [link for link in self.get_effects(cause) if link.effect == effect]
        enriched: List[Dict[str, Any]] = []
        supported = False
        for link in links:
            key = (link.cause, link.effect)
            self._apply_time_decay(key, now)
            payload = link.to_dict()
            metadata = self._link_metadata.get(key)
            if metadata:
                payload.update(
                    {
                        "decay": metadata.get("decay"),
                        "usage_count": metadata.get("usage_count"),
                        "drift": metadata.get("drift"),
                        "last_update": metadata.get("last_update"),
                        "avg_interval": metadata.get("avg_interval"),
                    }
                )
                supported = supported or metadata["strength"] >= self.support_threshold
            enriched.append(payload)

        satisfied: List[str] = []
        unsatisfied: List[str] = []
        for link in links:
            for cond in link.conditions or []:
                key, _, expected = cond.partition("=")
                key = key.strip()
                expected = expected.strip()
                value = str(context.get(key, ""))
                if expected:
                    (satisfied if value == expected else unsatisfied).append(cond)
                elif value:
                    satisfied.append(cond)
                else:
                    unsatisfied.append(cond)

        return {
            "cause": cause,
            "effect": effect,
            "supported": supported,
            "links": enriched,
            "satisfied_conditions": satisfied,
            "unsatisfied_conditions": unsatisfied,
        }

    # ------------------------------------------------------------------
    def intervention(self, cause: str, action: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        effects = self.get_effects(cause)
        predicted = []
        for link in effects:
            key = (link.cause, link.effect)
            self._apply_time_decay(key, now)
            metadata = self._link_metadata.get(key, {})
            predicted.append(
                {
                    "effect": link.effect,
                    "strength": float(link.strength),
                    "conditions": list(link.conditions or []),
                    "decay": metadata.get("decay"),
                    "drift": metadata.get("drift"),
                }
            )
        return {
            "cause": cause,
            "action": action or f"do({cause})",
            "predicted_effects": predicted,
        }

    # ------------------------------------------------------------------
    def ingest_simulation(
        self, cause: Optional[str], effect: Optional[str], result: Union[SimulationResult, Dict[str, Any], None]
    ) -> None:
        if not cause or not effect or result is None:
            return
        key = (cause, effect)
        if key not in self._link_metadata:
            return

        now = time.time()
        self._apply_time_decay(key, now)
        metadata = self._link_metadata[key]

        success = False
        confidence = 1.0
        if isinstance(result, SimulationResult):
            success = bool(result.success)
            if result.details and "confidence" in result.details:
                try:
                    confidence = float(result.details["confidence"])
                except (TypeError, ValueError):
                    confidence = 1.0
            if result.details and "likelihood" in result.details:
                try:
                    confidence = float(result.details["likelihood"])
                except (TypeError, ValueError):
                    pass
        elif isinstance(result, dict):
            success = bool(result.get("success"))
            if "confidence" in result:
                try:
                    confidence = float(result["confidence"])
                except (TypeError, ValueError):
                    confidence = 1.0
        else:
            success = bool(result)

        direction = 1.0 if success else -1.0
        feedback_scale = 0.05
        step = self._bounded_step(direction * confidence * feedback_scale)
        previous_strength = metadata["strength"]
        metadata["strength"] = self._clamp01(previous_strength + step)
        metadata["last_update"] = now
        metadata["usage_count"] = metadata.get("usage_count", 0) + 1
        metadata["link"].strength = metadata["strength"]

        previous_observation = metadata.get("last_observation", now)
        interval = now - previous_observation
        if interval > 0:
            avg_interval = metadata.get("avg_interval")
            if avg_interval is None:
                metadata["avg_interval"] = interval
            else:
                metadata["avg_interval"] = (1 - self.interval_smoothing) * avg_interval + self.interval_smoothing * interval
        metadata["last_observation"] = now

        self._update_decay(metadata, reinforcement=step)
        self._record_drift(key, metadata["strength"] - previous_strength, reason="simulation")

    # ------------------------------------------------------------------
    @property
    def drift_log(self) -> List[Dict[str, Any]]:
        return list(self._drift_log)

    def refresh_from_belief(self, belief: Any) -> None:
        if getattr(belief, "relation", "") != "causes":
            return
        self.register(
            belief.subject,
            belief.value,
            strength=float(getattr(belief, "confidence", 0.5)),
            description=getattr(belief, "relation_type", None),
        )


class CounterfactualSimulator:
    """Evaluate lightweight counterfactuals based on the SCM and domain sims."""

    def __init__(self, scm: SCMStore, domain: Optional[DomainSimulator] = None) -> None:
        self.scm = scm
        self.domain = domain or DomainSimulator()

    # ------------------------------------------------------------------
    def run(self, query: Dict[str, Any]) -> CounterfactualReport:
        cause = query.get("cause") or query.get("if") or query.get("action")
        effect = query.get("effect") or query.get("then")
        scenario = query.get("scenario") or {}
        evidence = self.scm.test_relation(cause, effect, context=scenario)
        intervention = self.scm.intervention(cause or "unknown", action=query.get("action"))
        simulations: List[Dict[str, Any]] = []

        domains: Iterable[str] = query.get("domains") or [cause or "generic"]
        for domain in domains:
            sim = self.domain.simulate(domain, {**scenario, "cause": cause, "effect": effect})
            self.scm.ingest_simulation(cause, effect, sim)
            if isinstance(sim, SimulationResult):
                simulations.append(sim.to_dict())
            elif isinstance(sim, dict):
                simulations.append(sim)
            else:
                simulations.append({"success": bool(sim), "outcome": str(sim)})

        heuristic_summary = self._summarise_counterfactual(
            cause,
            effect,
            evidence,
            intervention,
            simulations,
        )
        fallback_actions = self._fallback_actions(intervention)
        baseline_confidence = self._baseline_confidence(evidence, simulations)

        report = CounterfactualReport(
            query=query,
            supported=bool(evidence.get("supported")),
            intervention=intervention,
            evidence=evidence,
            simulations=simulations,
            generated_at=time.time(),
            heuristic_summary=heuristic_summary,
            summary=heuristic_summary,
            confidence=baseline_confidence,
            actions=fallback_actions if fallback_actions else None,
        )

        llm_payload = {
            "query": query,
            "heuristics": {
                "supported": report.supported,
                "summary": heuristic_summary,
                "intervention": intervention,
                "evidence": evidence,
                "simulations": simulations,
                "baseline_confidence": baseline_confidence,
                "fallback_actions": fallback_actions,
            },
        }
        llm_response = try_call_llm_dict(
            "counterfactual_analysis",
            input_payload=_sanitize_payload(llm_payload),
            logger=logger,
        )

        if isinstance(llm_response, Mapping):
            guidance: Dict[str, Any] = {"raw": llm_response}

            summary_text = str(llm_response.get("summary") or "").strip()
            if summary_text:
                report.summary = summary_text
                guidance["summary"] = summary_text

            confidence_value = _coerce_float(llm_response.get("confidence"))
            if confidence_value is not None:
                report.confidence = float(max(0.0, min(1.0, confidence_value)))
                guidance["confidence"] = report.confidence

            assumptions_payload = llm_response.get("assumptions")
            if isinstance(assumptions_payload, Iterable) and not isinstance(assumptions_payload, (str, bytes)):
                assumptions = [
                    str(item).strip()
                    for item in assumptions_payload
                    if isinstance(item, (str, bytes)) and str(item).strip()
                ]
                if assumptions:
                    report.assumptions = assumptions
                    guidance["assumptions"] = assumptions

            checks_payload = llm_response.get("checks")
            if isinstance(checks_payload, Iterable) and not isinstance(checks_payload, (str, bytes)):
                parsed_checks: List[Dict[str, Any]] = []
                for item in checks_payload:
                    if not isinstance(item, Mapping):
                        continue
                    description = str(item.get("description") or item.get("label") or "").strip()
                    if not description:
                        continue
                    check_entry: Dict[str, Any] = {"description": description}
                    goal = str(item.get("goal") or item.get("objectif") or "").strip()
                    if goal:
                        check_entry["goal"] = goal
                    priority_val = _coerce_int(item.get("priority"))
                    if priority_val is not None:
                        check_entry["priority"] = priority_val
                    parsed_checks.append(check_entry)
                if parsed_checks:
                    report.checks = parsed_checks
                    guidance["checks"] = parsed_checks

            actions_payload = llm_response.get("actions")
            if isinstance(actions_payload, Iterable) and not isinstance(actions_payload, (str, bytes)):
                parsed_actions: List[Dict[str, Any]] = []
                for item in actions_payload:
                    if not isinstance(item, Mapping):
                        continue
                    label = str(item.get("label") or item.get("action") or "").strip()
                    if not label:
                        continue
                    action_entry: Dict[str, Any] = {"label": label}
                    notes_val = str(item.get("notes") or item.get("why") or "").strip()
                    if notes_val:
                        action_entry["notes"] = notes_val
                    priority_val = _coerce_int(item.get("priority"))
                    if priority_val is not None:
                        action_entry["priority"] = priority_val
                    utility_val = _coerce_float(item.get("utility"))
                    if utility_val is not None:
                        action_entry["utility"] = float(max(0.0, min(1.0, utility_val)))
                    parsed_actions.append(action_entry)
                if parsed_actions:
                    report.actions = parsed_actions
                    guidance["actions"] = parsed_actions

            alerts_payload = llm_response.get("alerts")
            if isinstance(alerts_payload, Iterable) and not isinstance(alerts_payload, (str, bytes)):
                alerts = [
                    str(item).strip()
                    for item in alerts_payload
                    if isinstance(item, (str, bytes)) and str(item).strip()
                ]
                if alerts:
                    report.alerts = alerts
                    guidance["alerts"] = alerts

            notes_text = str(llm_response.get("notes") or "").strip()
            if notes_text:
                report.notes = notes_text
                guidance["notes"] = notes_text

            report.llm_guidance = guidance

        return report

    # ------------------------------------------------------------------
    def _summarise_counterfactual(
        self,
        cause: Optional[str],
        effect: Optional[str],
        evidence: Mapping[str, Any],
        intervention: Mapping[str, Any],
        simulations: List[Dict[str, Any]],
    ) -> str:
        cause_label = cause or "(inconnu)"
        effect_label = effect or "(inconnu)"
        supported = bool(evidence.get("supported"))
        predicted = intervention.get("predicted_effects") or []
        success_count = sum(1 for sim in simulations if isinstance(sim, Mapping) and sim.get("success"))
        total_sims = len(simulations)
        support_text = "semble soutenir" if supported else "reste incertain pour"
        summary_parts = [
            f"Intervention sur {cause_label} → {effect_label}: {support_text} la relation.",
        ]
        if predicted:
            main_effect = predicted[0]
            if isinstance(main_effect, Mapping):
                effect_name = main_effect.get("effect") or effect_label
                strength = main_effect.get("strength")
                if strength is not None:
                    summary_parts.append(
                        f"Effet principal anticipé: {effect_name} (force≈{float(strength):.2f})."
                    )
                else:
                    summary_parts.append(f"Effet principal anticipé: {effect_name}.")
        if total_sims:
            ratio = success_count / total_sims
            summary_parts.append(
                f"Simulations positives: {success_count}/{total_sims} (≈{ratio:.0%})."
            )
        return " ".join(summary_parts)

    def _fallback_actions(self, intervention: Mapping[str, Any]) -> List[Dict[str, Any]]:
        predicted = intervention.get("predicted_effects") or []
        if not predicted:
            return [
                {
                    "label": "Collecter une observation supplémentaire",
                    "priority": 1,
                    "notes": "Vérifier rapidement si un effet mesurable se manifeste.",
                }
            ]
        actions: List[Dict[str, Any]] = []
        top_effect = predicted[0]
        if isinstance(top_effect, Mapping):
            effect_name = top_effect.get("effect") or "effet clé"
            actions.append(
                {
                    "label": f"Surveiller {effect_name}",
                    "priority": 1,
                    "notes": "Confirmer ou infirmer l'effet principal prédit par le modèle causal.",
                }
            )
            if len(predicted) > 1:
                secondary = predicted[1]
                if isinstance(secondary, Mapping):
                    secondary_effect = secondary.get("effect")
                    if secondary_effect:
                        actions.append(
                            {
                                "label": f"Comparer {effect_name} vs {secondary_effect}",
                                "priority": 2,
                                "notes": "Identifier quel effet domine réellement après l'intervention.",
                            }
                        )
        return actions

    def _baseline_confidence(
        self,
        evidence: Mapping[str, Any],
        simulations: List[Dict[str, Any]],
    ) -> float:
        supported = bool(evidence.get("supported"))
        total = len(simulations)
        if total:
            success_ratio = sum(
                1.0 for sim in simulations if isinstance(sim, Mapping) and sim.get("success")
            ) / total
        else:
            success_ratio = 0.0
        base = 0.4 + 0.4 * success_ratio
        if supported:
            base += 0.1
        return float(max(0.05, min(0.95, base)))

