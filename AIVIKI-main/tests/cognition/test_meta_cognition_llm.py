from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.cognition import meta_cognition as mc


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


class DummyMemory:
    def get_recent_memories(self, n: int):
        now = time.time()
        return [
            {"kind": "lesson", "text": "Gestion des erreurs", "ts": now - 60},
            {"kind": "error", "text": "Erreur 500", "ts": now - 30},
        ]

    def add_memory(self, payload):
        pass


class DummyPlanner:
    def plan_for_goal(self, goal_id, desc, steps=None):
        return {"goal_id": goal_id, "description": desc, "steps": steps or []}

    def add_step(self, goal_id, desc):
        return "s1"


def test_meta_cognition_records_llm_output(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "_llm_enabled", lambda: True)
    monkeypatch.setattr(mc.MetaCognition, "_load", lambda self: None)
    monkeypatch.setattr(mc.MetaCognition, "_save", lambda self: None)

    manager = StubManager(
        {
            "assimilation_score": 0.72,
            "signals": [{"name": "prediction_error", "value": 0.2, "interpretation": "stable"}],
            "recommendation": "Focus sur les erreurs API",
            "knowledge_gaps": [{"topic": "gestion erreurs"}],
            "learning_goals": [{"goal": "Documenter la procédure", "impact": "haut"}],
        }
    )

    monkeypatch.setattr(mc, "_llm_manager", lambda: manager)

    meta = mc.MetaCognition(DummyMemory(), DummyPlanner(), SimpleNamespace(), data_dir=str(tmp_path))
    assessment = meta.assess_understanding()

    assert assessment["llm"]["assimilation_score"] == 0.72
    goals = meta.propose_learning_goals(max_goals=1)
    assert any(goal.get("source") == "llm" for goal in goals)
    assert manager.last_spec == "meta_cognition"
