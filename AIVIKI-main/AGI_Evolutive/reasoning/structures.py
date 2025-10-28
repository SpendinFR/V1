"""Lightweight data structures used by the reasoning module."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Hypothesis:
    """Lightweight hypothesis used during reasoning sessions.

    Elle représente un candidat explicatif manipulé par les moteurs de
    raisonnement.  Contrairement à ``beliefs.graph.Hypothesis`` (qui n'existe
    pas) ou aux structures de croyances persistantes, ces instances sont
    éphémères et destinées à être sérialisées dans des journaux d'épisode.
    """
    content: str
    prior: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["prior"] = float(self.prior)
        return data


@dataclass
class Test:
    description: str
    cost_est: float = 0.2
    expected_information_gain: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "cost_est": float(self.cost_est),
            "expected_information_gain": float(self.expected_information_gain),
        }


@dataclass
class Evidence:
    """Qualitative evidence note kept for reasoning audit trails.

    Cette forme réduite (texte + confiance) documente les étapes d'un
    raisonnement.  Les objets plus détaillés destinés à la mémoire long
    terme vivent dans :mod:`AGI_Evolutive.beliefs.graph`.
    """
    notes: str
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {"notes": self.notes, "confidence": float(self.confidence)}


@dataclass
class Update:
    posterior: float
    decision: str
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["posterior"] = float(self.posterior)
        return payload


@dataclass
class CausalLink:
    """Represents a directed causal influence between two concepts."""

    cause: str
    effect: str
    strength: float = 0.5
    conditions: Optional[List[str]] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "cause": self.cause,
            "effect": self.effect,
            "strength": float(self.strength),
        }
        if self.conditions:
            data["conditions"] = list(self.conditions)
        if self.description:
            data["description"] = self.description
        return data


class CausalStore:
    """Minimal causal graph storage with helper queries and tests."""

    def __init__(self) -> None:
        self._by_cause: Dict[str, List[CausalLink]] = {}
        self._by_effect: Dict[str, List[CausalLink]] = {}

    def add_link(self, link: CausalLink) -> None:
        self._by_cause.setdefault(link.cause, []).append(link)
        self._by_effect.setdefault(link.effect, []).append(link)

    def register(self, cause: str, effect: str, **kwargs: Any) -> None:
        """Convenience wrapper to create and add a :class:`CausalLink`."""

        link = CausalLink(cause=cause, effect=effect, **kwargs)
        self.add_link(link)

    def get_effects(self, cause: str) -> List[CausalLink]:
        return list(self._by_cause.get(cause, []))

    def get_causes(self, effect: str) -> List[CausalLink]:
        return list(self._by_effect.get(effect, []))

    def chains(self, start: str, end: str, max_depth: int = 3) -> List[List[CausalLink]]:
        """Search for short causal chains linking *start* to *end*."""

        paths: List[List[CausalLink]] = []

        def _dfs(current: str, depth: int, acc: List[CausalLink]) -> None:
            if depth > max_depth:
                return
            for link in self._by_cause.get(current, []):
                new_acc = acc + [link]
                if link.effect == end:
                    paths.append(new_acc)
                else:
                    _dfs(link.effect, depth + 1, new_acc)

        _dfs(start, 1, [])
        return paths

    def test_relation(
        self,
        cause: str,
        effect: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Lightweight causal test: checks if a direct link exists and if conditions are
        satisfied within the provided *context*.
        """

        links = [link for link in self.get_effects(cause) if link.effect == effect]
        supported = bool(links)
        context = context or {}
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
            "links": [link.to_dict() for link in links],
            "satisfied_conditions": satisfied,
            "unsatisfied_conditions": unsatisfied,
        }

    def intervention(self, cause: str, action: Optional[str] = None) -> Dict[str, Any]:
        """Return the downstream effects of intervening on *cause*."""

        effects = self.get_effects(cause)
        predicted = [
            {
                "effect": link.effect,
                "strength": float(link.strength),
                "conditions": list(link.conditions or []),
            }
            for link in effects
        ]
        return {
            "cause": cause,
            "action": action or f"do({cause})",
            "predicted_effects": predicted,
        }


@dataclass
class SimulationResult:
    """Outcome of a concrete or mental simulation."""

    success: bool
    outcome: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "success": bool(self.success),
            "outcome": self.outcome,
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


class DomainSimulator:
    """
    Registry of small domain-specific simulators.

    Each simulator is a callable ``f(scenario: Dict[str, Any]) -> SimulationResult``.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Any] = {}

    def register(self, domain: str, simulator: Any) -> None:
        self._registry[domain] = simulator

    def has_domain(self, domain: str) -> bool:
        return domain in self._registry

    def simulate(self, domain: str, scenario: Dict[str, Any]) -> SimulationResult:
        simulator = self._registry.get(domain)
        if simulator is None:
            return SimulationResult(
                success=False,
                outcome="Aucun simulateur pour ce domaine (test mental requis).",
                details={"domain": domain, "scenario": scenario},
            )
        try:
            result = simulator(scenario)
            if isinstance(result, SimulationResult):
                return result
            if isinstance(result, dict):
                return SimulationResult(**result)
            return SimulationResult(success=True, outcome=str(result))
        except Exception as exc:  # pragma: no cover - defensive path
            return SimulationResult(
                success=False,
                outcome=f"Simulation échouée: {exc}",
                details={"domain": domain, "scenario": scenario},
            )


@dataclass
class TaskNode:
    """Node of a hierarchical plan (HTN-style)."""

    name: str
    actions: List[str] = field(default_factory=list)
    sub_tasks: List[TaskNode] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "actions": list(self.actions),
            "sub_tasks": [sub.to_dict() for sub in self.sub_tasks],
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
        }


class HTNPlanner:
    """Minimal hierarchical task network planner.

    Le module :mod:`AGI_Evolutive.planning.htn` en fournit une extension
    concrète connectée aux croyances/ontologie.  Cette implémentation-ci
    reste purement algorithmique pour éviter les dépendances croisées.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, TaskNode] = {}

    def register_template(self, goal: str, root: TaskNode) -> None:
        self._templates[goal] = root

    def has_template(self, goal: str) -> bool:
        return goal in self._templates

    def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        root = self._templates.get(goal)
        if root is None:
            return []
        context = context or {}
        steps: List[str] = []
        self._expand(node=root, context=context, steps=steps)
        return steps

    # -- internal helpers -------------------------------------------------
    def _expand(self, node: TaskNode, context: Dict[str, Any], steps: List[str]) -> None:
        missing = self._unsatisfied(node.preconditions, context)
        if missing:
            steps.append(
                f"Vérifier/obtenir préconditions pour {node.name}: {', '.join(missing)}"
            )

        for sub in node.sub_tasks:
            self._expand(sub, context, steps)

        for action in node.actions:
            steps.append(action)

        if node.postconditions:
            steps.append(
                f"Valider postconditions pour {node.name}: {', '.join(node.postconditions)}"
            )

    def _unsatisfied(
        self, conditions: Iterable[str], context: Dict[str, Any]
    ) -> List[str]:
        missing: List[str] = []
        for cond in conditions:
            cond = cond.strip()
            if not cond:
                continue
            key, has_eq, expected = cond.partition("=")
            key = key.strip()
            if not key:
                continue
            value = context.get(key)
            if has_eq:
                if str(value).lower() != expected.strip().lower():
                    missing.append(cond)
            elif not value:
                missing.append(cond)
        return missing


def now() -> float:
    return time.time()


def episode_record(
    user_msg: str,
    hypotheses: List[Hypothesis],
    chosen_index: int,
    tests: List[Test],
    evidence: Evidence,
    result_text: str,
    final_confidence: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "timestamp": now(),
        "user_msg": user_msg,
        "hypotheses": [h.to_dict() for h in hypotheses],
        "chosen_index": int(chosen_index),
        "tests": [t.to_dict() for t in tests],
        "evidence": evidence.to_dict(),
        "result_text": result_text,
        "final_confidence": float(final_confidence),
    }
    if metadata:
        record.update(metadata)
    return record


def summary_record(
    user_msg: str,
    hypotheses: List[Hypothesis],
    chosen_index: int,
    tests: List[Test],
    evidence: Optional[Evidence],
    result_text: str,
    final_confidence: float,
) -> Dict[str, Any]:
    return {
        "t": now(),
        "user_msg": user_msg,
        "hypotheses": [h.to_dict() for h in hypotheses],
        "chosen_idx": int(chosen_index),
        "tests": [t.to_dict() for t in tests],
        "evidence": evidence.to_dict() if evidence else None,
        "solution": result_text,
        "final_confidence": float(final_confidence),
    }
