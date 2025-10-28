from collections import deque

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.goals import GoalSystem


class MemoryStub:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add_memory(self, payload):
        self.entries.append(payload)
        return payload


class ArchRecorderStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def _record_skill(self, concept: str, **kwargs):  # noqa: D401 - simple stub
        self.calls.append((concept, kwargs))
        return {"ok": True, "concept": concept, "payload": kwargs}


@pytest.fixture()
def goal_system(tmp_path):
    data_path = tmp_path / "goal_intentions.json"
    persist_path = tmp_path / "goals.json"
    dashboard_path = tmp_path / "dashboard.json"
    system = GoalSystem(
        persist_path=str(persist_path),
        dashboard_path=str(dashboard_path),
        intention_data_path=str(data_path),
    )
    yield system
    # cleanup persisted files to avoid interference
    for path in (data_path, persist_path, dashboard_path):
        if path.exists():
            path.unlink()


@pytest.fixture()
def goal_system_with_memory(tmp_path):
    data_path = tmp_path / "goal_intentions.json"
    persist_path = tmp_path / "goals.json"
    dashboard_path = tmp_path / "dashboard.json"
    memory = MemoryStub()
    system = GoalSystem(
        persist_path=str(persist_path),
        dashboard_path=str(dashboard_path),
        intention_data_path=str(data_path),
        memory=memory,
    )
    yield system, memory
    for path in (data_path, persist_path, dashboard_path):
        if path.exists():
            path.unlink()


def _deque_types(actions: deque) -> list[str]:
    return [action["type"] for action in actions]


def test_concept_heuristic_detects_quotes(goal_system):
    goal = goal_system.add_goal("Apprendre le concept « auto-éco-régulation »")
    actions = goal_system._goal_to_actions(goal)
    types = _deque_types(actions)
    assert types[0] == "learn_concept"
    assert actions[0]["payload"]["concept"].lower() == "auto-éco-régulation"


def test_classifier_informs_actions(goal_system, tmp_path):
    goal = goal_system.add_goal("Planifier une étude sur la collaboration ouverte")
    # simulate execution success and feed back outcome
    goal_system.record_goal_outcome(
        goal.id, succeeded=True, executed_actions=[{"type": "plan"}]
    )
    # new goal with similar wording should now map to plan directly
    successor = goal_system.add_goal("Planifier une étude sur la collaboration ouverte")
    actions = goal_system._goal_to_actions(successor)
    types = _deque_types(actions)
    assert types[0] == "plan"


def test_probe_generated_on_low_confidence(goal_system):
    goal = goal_system.add_goal("Explorer un thème émergent")
    # feed conflicting labels to keep confidence low
    goal_system.intention_model.classifier.update("plan", goal.description)
    goal_system.intention_model.classifier.update("reflect", goal.description)
    goal_system.intent_confidence_threshold = 0.9
    actions = goal_system._goal_to_actions(goal)
    types = _deque_types(actions)
    assert types[0] == "probe_goal"
    assert types[1] == "reflect"


def test_structural_hierarchy_creates_depth_two_children(goal_system):
    goal_system.step()
    root = next(node for node in goal_system.store.nodes.values() if not node.parent_ids)
    grandchildren = []
    for child_id in root.child_ids:
        child = goal_system.store.nodes[child_id]
        for cid in child.child_ids:
            grandchildren.append(goal_system.store.nodes[cid].description)
    assert any(desc.startswith("Formuler une hypothèse") for desc in grandchildren)
    assert any("QuestionManager" in desc for desc in grandchildren)
    assert any("Analyser l'inbox" in desc or "Consigner l'hypothèse" in desc for desc in grandchildren)


def test_goal_completion_updates_status_and_memory(goal_system_with_memory):
    system, memory = goal_system_with_memory
    leaf = next(
        node
        for node in system.store.nodes.values()
        if "Consigner l'hypothèse" in node.description
    )
    parent = system.store.nodes[leaf.parent_ids[0]] if leaf.parent_ids else None
    system.update_goal(leaf.id, {"progress": 1.0})
    updated_leaf = system.store.get_goal(leaf.id)
    assert updated_leaf.status == "done"
    assert memory.entries
    recorded = memory.entries[-1]
    assert recorded["kind"] == "goal_completion"
    assert recorded["goal_id"] == leaf.id
    if parent:
        refreshed_parent = system.store.get_goal(parent.id)
        assert refreshed_parent.progress > 0.0


def test_manual_completion_records_skill(goal_system_with_memory):
    system, _ = goal_system_with_memory
    recorder = ArchRecorderStub()
    system.architecture = recorder

    goal = system.add_goal("Assimiler la patience profonde")
    system.update_goal(goal.id, {"progress": 1.0})

    assert recorder.calls, "La compétence devrait être enregistrée lors de la complétion manuelle"
    concept, payload = recorder.calls[-1]
    assert "patience" in concept.lower()
    assert payload["metadata"]["goal_id"] == goal.id


def test_question_manager_structural_action(goal_system):
    goal = goal_system.add_goal(
        "Solliciter le QuestionManager pour clarifier « introspection » si ma compréhension reste faible.",
        created_by="structure",
    )
    actions = goal_system._goal_to_actions(goal)
    types = _deque_types(actions)
    assert types[0] == "ask"


def test_inbox_structural_action(goal_system):
    goal = goal_system.add_goal(
        "Explorer l'inbox pour enrichir « introspection » avant de formuler l'hypothèse.",
        created_by="structure",
    )
    actions = goal_system._goal_to_actions(goal)
    types = _deque_types(actions)
    assert "scan_inbox" in types


def test_integrate_understanding_updates_progress(goal_system_with_memory):
    system, memory = goal_system_with_memory
    goal = system.add_goal("Comprendre l'empathie en profondeur.")
    system.store.set_active(goal.id)

    partial = system.integrate_understanding(
        topic="empathie",
        score=0.72,
        prediction_error=0.24,
        gaps=["manque d'exemples"],
        goal_id=goal.id,
        source="test",
    )

    assert partial is not None
    assert not partial["completed"]
    updated = system.store.get_goal(goal.id)
    assert updated.progress > 0.0
    assert any(entry.get("kind") == "goal_evidence" for entry in memory.entries)

    final = system.integrate_understanding(
        topic="empathie",
        score=0.93,
        prediction_error=0.05,
        gaps=[],
        goal_id=goal.id,
        source="test",
    )

    assert final is not None
    assert final["completed"]
    completed_goal = system.store.get_goal(goal.id)
    assert completed_goal.status == "done"
    assert completed_goal.progress == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert any(
        entry.get("kind") == "goal_completion" and entry.get("goal_id") == goal.id
        for entry in memory.entries
    )


def test_integrate_understanding_records_skill_once(goal_system_with_memory):
    system, _ = goal_system_with_memory
    recorder = ArchRecorderStub()
    system.architecture = recorder

    goal = system.add_goal("Comprendre l'empathie en profondeur.")
    system.store.set_active(goal.id)

    system.integrate_understanding(
        topic="empathie",
        score=0.72,
        prediction_error=0.24,
        gaps=["manque d'exemples"],
        goal_id=goal.id,
        source="test",
    )

    assert not recorder.calls

    system.integrate_understanding(
        topic="empathie",
        score=0.95,
        prediction_error=0.03,
        gaps=[],
        goal_id=goal.id,
        source="test",
    )

    assert len(recorder.calls) == 1
    concept, payload = recorder.calls[0]
    assert "empathie" in concept.lower()
    assert payload["metadata"]["goal_id"] == goal.id


def test_integrate_understanding_matches_goal_without_id(goal_system_with_memory):
    system, _ = goal_system_with_memory
    goal = system.add_goal("Explorer l'empathie dans la communication quotidienne")
    system.store.set_active(goal.id)

    update = system.integrate_understanding(
        topic="empathie",
        score=0.8,
        prediction_error=0.2,
        gaps=["clarifier"],
        source="test",
    )

    assert update is not None
    assert update["goal_id"] == goal.id
