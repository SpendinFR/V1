from copy import deepcopy
import time
from typing import Any, Dict, List
import pathlib
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.io.action_interface import ActionInterface
from AGI_Evolutive.cognition import planner as planner_mod


_SAMPLE_PLAN = {
    "learn_planning_bonjour": {
        "goal_id": "learn_planning_bonjour",
        "description": "Comprendre le concept 'bonjour' dans le domaine planning",
        "steps": [
            {
                "id": "s1",
                "desc": "Collecter exemples concrets pour 'Comprendre le concept 'bonjour' dans le domaine planning'",
                "status": "blocked",
                "history": [
                    {"ts": 1761035202.84399, "event": "completed", "success": False}
                ],
            },
            {
                "id": "s2",
                "desc": "Poser 2 questions ciblées sur 'Comprendre le concept 'bonjour' dans le domaine planning'",
                "status": "blocked",
                "history": [
                    {"ts": 1761035271.3573518, "event": "completed", "success": False}
                ],
            },
        ],
        "created_at": 1761035202.8353796,
    }
}


@pytest.fixture
def planner(tmp_path, monkeypatch):
    plans_path = tmp_path / "plans.json"
    monkeypatch.setattr(planner_mod, "_PLANS", str(plans_path), raising=False)
    pl = planner_mod.Planner()
    pl.state["plans"] = deepcopy(_SAMPLE_PLAN)
    pl._save()
    return pl


def _emit_step(pl: planner_mod.Planner, interface: ActionInterface) -> Dict[str, Any]:
    step = pl.pop_next_action("learn_planning_bonjour")
    assert step is not None, "expected a step to process"
    desc = step.get("desc", "").lower()
    if "poser" in desc or "question" in desc:
        action = {
            "type": "communicate",
            "text": "Peux-tu me décrire ton émotion actuelle et pourquoi ?",
            "target": "human",
        }
    elif "observer" in desc:
        action = {"type": "simulate", "what": "observe_exchange"}
    else:
        action = {"type": "simulate", "desc": desc}
    result = interface.execute(action)
    pl.mark_action_done("learn_planning_bonjour", step["id"], result=result)
    return step


def test_planner_marks_fallback_simulation_as_failure(planner, tmp_path):
    interface = ActionInterface(
        path_log=str(tmp_path / "actions.jsonl"),
        output_dir=str(tmp_path / "out"),
    )

    first_step = _emit_step(planner, interface)
    plan = planner.state["plans"]["learn_planning_bonjour"]
    step1 = plan["steps"][0]
    assert step1["id"] == first_step["id"]
    assert step1["status"] == "todo"
    assert step1.get("retry_at", 0) > time.time()
    assert step1["history"]
    last_entry = step1["history"][-1]
    assert last_entry["success"] is False
    assert last_entry["reason"] == "simulator_unavailable"
    assert last_entry.get("reported_success") is False

    second_step = _emit_step(planner, interface)
    step2 = plan["steps"][1]
    assert step2["id"] == second_step["id"]
    assert step2["status"] == "done"
    entry2 = step2["history"][-1]
    assert entry2["success"] is True
    assert entry2.get("reported_success") is not False

    # All steps are waiting on retries, so no further actions are emitted immediately.
    assert planner.pop_next_action("learn_planning_bonjour") is None


class _HappySimulator:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def run(self, query: Dict[str, Any]) -> Any:
        self.calls.append(dict(query))
        return SimpleNamespace(
            supported=True,
            evidence=[{"kind": "heuristic", "detail": "stub"}],
            intervention={"kind": "mental_rehearsal"},
            simulations=[{"success": True, "note": "stub"}],
        )


def test_planner_uses_bound_simulator_for_success(planner, tmp_path):
    interface = ActionInterface(
        path_log=str(tmp_path / "actions_success.jsonl"),
        output_dir=str(tmp_path / "out_success"),
    )
    simulator = _HappySimulator()
    interface.bind(simulator=simulator)

    first_step = _emit_step(planner, interface)
    plan = planner.state["plans"]["learn_planning_bonjour"]
    step1 = plan["steps"][0]

    assert first_step["id"] == step1["id"]
    assert step1["status"] == "done"
    assert simulator.calls, "expected simulator to be invoked"

    entry = step1["history"][-1]
    assert entry["success"] is True
    assert entry.get("reported_success") is True
