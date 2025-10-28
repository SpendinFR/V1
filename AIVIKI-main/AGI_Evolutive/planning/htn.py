"""HTN planning helpers tied to the belief graph and ontology."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional

from AGI_Evolutive.reasoning.structures import HTNPlanner as _BasePlanner
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


class HTNPlanner(_BasePlanner):
    """
    Small extension of :class:`AGI_Evolutive.reasoning.structures.HTNPlanner`.

    Elle relie le planner abstrait à la couche *planning* en injectant des
    métriques issues du graphe de croyances/ontologie dans le contexte de
    planification.
    """

    def __init__(self, beliefs: Any, ontology: Any) -> None:
        super().__init__()
        self.beliefs = beliefs
        self.ontology = ontology
        self._last_llm_plan: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # LLM integration helpers

    def _context_snapshot(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                snapshot[key] = value
        if "known_beliefs" not in snapshot:
            query_fn = getattr(self.beliefs, "query", None)
            if callable(query_fn):
                try:
                    snapshot["known_beliefs"] = len(query_fn(active_only=True))
                except TypeError:
                    snapshot["known_beliefs"] = len(query_fn())
            else:
                snapshot["known_beliefs"] = 0
        return snapshot

    def _llm_payload(self, goal: str, context: Mapping[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "goal": goal,
            "context": self._context_snapshot(context),
            "templates": list(getattr(self, "_templates", {}).keys()),
        }
        return payload

    def _llm_plan(self, goal: str, context: Mapping[str, Any]) -> Optional[List[str]]:
        response = try_call_llm_dict(
            "htn_planning",
            input_payload=self._llm_payload(goal, context),
            logger=LOGGER,
        )
        if not isinstance(response, Mapping):
            return None

        steps: List[str] = []
        root = response.get("root_task")
        if isinstance(root, str) and root.strip():
            steps.append(f"Objectif: {root.strip()}")

        methods = response.get("methods")
        if isinstance(methods, Iterable):
            for method in methods:
                if not isinstance(method, Mapping):
                    continue
                name = method.get("name") or method.get("id")
                if isinstance(name, str) and name.strip():
                    steps.append(f"Méthode: {name.strip()}")
                subtasks = method.get("subtasks")
                if isinstance(subtasks, Iterable):
                    for task in subtasks:
                        if isinstance(task, str) and task.strip():
                            steps.append(f"  - {task.strip()}")
        notes = response.get("notes")
        if isinstance(notes, str) and notes.strip():
            steps.append(f"Notes: {notes.strip()}")

        self._last_llm_plan = steps or None
        return self._last_llm_plan

    def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> List[str]:  # type: ignore[override]
        context = dict(context or {})
        if "known_beliefs" not in context:
            query_fn = getattr(self.beliefs, "query", None)
            if callable(query_fn):
                try:
                    context["known_beliefs"] = len(query_fn(active_only=True))
                except TypeError:
                    context["known_beliefs"] = len(query_fn())
            else:
                context["known_beliefs"] = 0

        llm_plan = self._llm_plan(goal, context)
        if llm_plan:
            return llm_plan

        return super().plan(goal, context=context)

