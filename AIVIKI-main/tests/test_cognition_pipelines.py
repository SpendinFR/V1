import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.cognition.pipelines_registry import ActMode, MODE_SELECTORS
from AGI_Evolutive.cognition.reflection_loop import ReflectionLoop
from AGI_Evolutive.goals.dag_store import GoalNode
from AGI_Evolutive.goals.heuristics import default_heuristics


@pytest.mark.parametrize(
    ("selector_key", "context", "expected"),
    [
        ("reflex_if_immediate", {"meta": {"immediacy": 0.85}}, ActMode.REFLEX),
        ("reflex_if_immediate", {"meta": {"immediacy": 0.2}}, ActMode.DELIBERATE),
        ("habit_if_low_importance", {"meta": {"importance": 0.2}}, ActMode.HABIT),
        ("habit_if_low_importance", {"meta": {"importance": 0.6}}, ActMode.DELIBERATE),
    ],
)
def test_mode_selectors_choose_expected_mode(selector_key, context, expected):
    selector = MODE_SELECTORS[selector_key]
    assert selector(context) is expected


def test_reflection_loop_tests_multiple_hypotheses_and_counterexamples():
    class DummyAbduction:
        def __init__(self):
            self.calls = []

        def generate(self, observation):
            self.calls.append(observation)
            return [
                SimpleNamespace(label="Hypothèse A", explanation="", score=0.8, ask_next=None),
                SimpleNamespace(label="Hypothèse B", explanation="", score=0.6, ask_next="Pourquoi ?"),
                SimpleNamespace(label="Hypothèse C", explanation="", score=0.4, ask_next=None),
            ]

    abduction = DummyAbduction()
    memory = SimpleNamespace(
        get_recent_memories=lambda n=80: [
            {"text": "Hypothèse A ? Non, cela ne colle pas.", "ts": 1.0, "id": "m1"},
            {"text": "Observation neutre", "ts": 2.0, "id": "m2"},
        ]
    )
    meta = SimpleNamespace(memory=memory, architecture=SimpleNamespace(abduction=abduction))
    loop = ReflectionLoop(meta, interval_sec=60)

    result = loop.test_hypotheses({"observation": "Hypothèse A sur l'empathie"}, max_tests=3)

    assert result["tested"] == 3
    labels = [hyp["label"] for hyp in result["hypotheses"]]
    assert labels == ["Hypothèse A", "Hypothèse B", "Hypothèse C"]
    counterexamples = [hyp["counterexample"] for hyp in result["hypotheses"]]
    assert any(counterexamples), "At least one hypothesis should surface a counterexample"


@pytest.mark.parametrize(
    ("description", "expected_types"),
    [
        (
            "Diagnostiquer ma compréhension de « empathie »",
            ["reflect", "search_memory"],
        ),
        (
            "Solliciter le QuestionManager pour approfondir « empathie »",
            ["ask"],
        ),
        (
            "Explorer l'inbox pour détecter des signaux sur « empathie »",
            ["scan_inbox", "reflect"],
        ),
    ],
)
def test_goal_heuristics_provide_specialised_actions(description, expected_types):
    heuristics = default_heuristics()
    goal = GoalNode(id="g1", description=description, priority=0.6)

    actions = heuristics.match(goal)

    assert actions is not None
    action_types = [action["type"] for action in actions]
    assert action_types == expected_types
