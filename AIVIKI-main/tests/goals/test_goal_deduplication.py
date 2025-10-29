import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.goals import GoalSystem
from AGI_Evolutive.goals.dag_store import DagStore, normalize_goal_signature


@pytest.fixture()
def stub_llm(monkeypatch):
    def _stub(*args, **kwargs):
        return {}

    monkeypatch.setattr("AGI_Evolutive.goals.try_call_llm_dict", _stub)
    monkeypatch.setattr("AGI_Evolutive.goals.curiosity.try_call_llm_dict", _stub)
    monkeypatch.setattr("AGI_Evolutive.goals.dag_store.try_call_llm_dict", _stub)
    yield


def _make_store(tmp_path, monkeypatch):
    monkeypatch.setattr("AGI_Evolutive.goals.dag_store.try_call_llm_dict", lambda *a, **k: {})
    persist = tmp_path / "goals.json"
    dashboard = tmp_path / "dashboard.json"
    return DagStore(str(persist), str(dashboard))


def test_store_add_goal_refuses_duplicates(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    node = store.add_goal("Diagnostiquer une capacité", criteria=["A", "B"], parent_ids=["root"])
    before = len(store.nodes)

    duplicate = store.add_goal("Diagnostiquer une capacité", criteria=["B", "A"], parent_ids=["root"])

    assert duplicate.id == node.id
    assert len(store.nodes) == before
    assert any(ev.get("event") == "duplicate_goal_attempt" for ev in duplicate.evidence)


def test_curiosity_auto_propose_deduplicates(tmp_path, stub_llm, monkeypatch):
    data_path = tmp_path / "intentions.json"
    persist_path = tmp_path / "goals.json"
    dashboard_path = tmp_path / "dashboard.json"
    system = GoalSystem(
        persist_path=str(persist_path),
        dashboard_path=str(dashboard_path),
        intention_data_path=str(data_path),
    )

    proposal = {
        "description": "Curiosity dedup test",
        "criteria": ["first", "second"],
        "parent_ids": [],
        "created_by": "curiosity",
        "value": 0.6,
        "competence": 0.5,
        "curiosity": 0.75,
        "urgency": 0.4,
    }

    monkeypatch.setattr(
        system.curiosity,
        "suggest_subgoals",
        lambda parent: [proposal, dict(proposal)],
    )

    signature = normalize_goal_signature(
        proposal["description"], proposal["criteria"], proposal["parent_ids"]
    )

    before = sum(1 for node in system.store.nodes.values() if node.description == proposal["description"])
    system._propose_curiosity_goals()
    after = sum(1 for node in system.store.nodes.values() if node.description == proposal["description"])
    assert after - before == 1
    assert system.curiosity.is_signature_known(signature)

    system._propose_curiosity_goals()
    after_twice = sum(
        1 for node in system.store.nodes.values() if node.description == proposal["description"]
    )
    assert after_twice == after


def test_default_actions_skip_redundant_reflect(goal_system_stubbed):
    system = goal_system_stubbed
    goal = system.add_goal("Réduire la redondance des actions")
    system.store.set_active(goal.id)
    first = system.get_next_action()
    assert first["type"] == "reflect"

    system.pending_actions.clear()
    second = system.get_next_action()
    assert second["type"] != "reflect"


def test_question_block_parks_and_restores(goal_system_stubbed):
    system = goal_system_stubbed
    goal = system.add_goal("Tester le blocage")
    system.store.set_active(goal.id)
    system._ensure_pending_actions()
    snapshot = list(system.pending_actions)
    assert snapshot

    system.set_question_block(True)
    assert not system.pending_actions

    system.set_question_block(False)
    assert list(system.pending_actions)[: len(snapshot)] == snapshot


def test_progress_propagation_limits_abandoned(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    parent = store.add_goal("Parent")
    active_child = store.add_goal("Active", parent_ids=[parent.id])
    abandoned_child = store.add_goal("Abandoned", parent_ids=[parent.id])

    store.update_goal(active_child.id, {"progress": 1.0})
    store.update_goal(
        abandoned_child.id,
        {
            "progress": 0.9,
            "status": "abandoned",
        },
    )

    refreshed_parent = store.get_goal(parent.id)
    assert math.isclose(refreshed_parent.progress, 0.55, rel_tol=1e-6)


def test_goal_dag_not_exported():
    import AGI_Evolutive.goals.dag_store as dag_module

    assert not hasattr(dag_module, "GoalDAG")


@pytest.fixture()
def goal_system_stubbed(tmp_path, stub_llm):
    data_path = tmp_path / "goal_intentions.json"
    persist_path = tmp_path / "goals.json"
    dashboard_path = tmp_path / "dashboard.json"
    system = GoalSystem(
        persist_path=str(persist_path),
        dashboard_path=str(dashboard_path),
        intention_data_path=str(data_path),
    )
    yield system
    for path in (data_path, persist_path, dashboard_path):
        if path.exists():
            path.unlink()
