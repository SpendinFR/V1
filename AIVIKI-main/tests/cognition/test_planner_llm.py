from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.cognition import planner


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


def test_planner_uses_llm_when_no_plan(monkeypatch):
    monkeypatch.setattr(planner, "_llm_enabled", lambda: True)
    monkeypatch.setattr(planner.Planner, "_load", lambda self: None)
    monkeypatch.setattr(planner.Planner, "_save", lambda self: None)
    monkeypatch.setattr(planner.Planner, "_maybe_preplan_with_rag", lambda self, frame, arch: {"rag_out": {}, "grounded_context": None, "rag_signals": {}, "rag_signal_meta": {}})

    manager = StubManager(
        {
            "plan": [
                {
                    "id": "collect_logs",
                    "description": "Collecter les logs",
                    "priority": 0.9,
                    "depends_on": [],
                }
            ],
            "risks": ["logs incomplets"],
        }
    )
    monkeypatch.setattr(planner, "_llm_manager", lambda: manager)

    pl = planner.Planner()
    frame = {"goal_id": "g1", "description": "Diagnostiquer les erreurs"}
    plan = pl.plan(frame)

    assert plan["steps"][0]["id"] == "collect_logs"
    assert manager.last_spec == "planner_support"
