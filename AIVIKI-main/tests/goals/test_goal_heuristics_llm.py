from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.goals import heuristics as gh


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


def test_goal_registry_prefers_llm(monkeypatch):
    registry = gh.HeuristicRegistry()
    goal = SimpleNamespace(id="g1", description="Stabiliser le service API", priority=0.8)

    manager = StubManager(
        {
            "normalized_goal": "stabiliser le service API",
            "candidate_actions": [
                {"action": "diagnostic_incident", "rationale": "erreurs 500"},
                {"action": "notifier_oncall", "rationale": "impact élevé"},
            ],
        }
    )

    monkeypatch.setattr(gh, "_llm_enabled", lambda: True)
    monkeypatch.setattr(gh, "_llm_manager", lambda: manager)

    actions = registry.match(goal)

    assert actions is not None
    assert actions[0]["type"] == "diagnostic_incident"
    assert manager.last_spec == "goal_interpreter"
