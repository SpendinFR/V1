from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("numpy")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive import orchestrator


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


def test_orchestrator_cycle_recommendations(monkeypatch):
    orch = object.__new__(orchestrator.Orchestrator)
    manager = StubManager({"recommendations": [{"horizon": "immédiat", "action": "stabiliser"}]})

    monkeypatch.setattr(orchestrator, "_llm_enabled", lambda: True)
    monkeypatch.setattr(orchestrator, "_llm_manager", lambda: manager)

    result = orchestrator.Orchestrator._llm_cycle_recommendations(
        orch,
        {"mode": "travail", "urgent": False},
    )

    assert result["recommendations"][0]["action"] == "stabiliser"
    assert manager.last_spec == "orchestrator_service"
