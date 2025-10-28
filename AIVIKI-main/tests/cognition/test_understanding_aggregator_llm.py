from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.cognition import understanding_aggregator as ua


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


def test_understanding_aggregator_blends_llm(monkeypatch):
    agg = ua.UnderstandingAggregator()
    manager = StubManager(
        {
            "assimilation_score": 0.9,
            "signals": [],
            "recommendation": "Continuer", 
        }
    )

    monkeypatch.setattr(ua, "_llm_enabled", lambda: True)
    monkeypatch.setattr(ua, "_llm_manager", lambda: manager)

    result = agg.compute(
        topic="tests",
        prediction_error=0.2,
        memory_consistency=0.6,
        transfer_success=0.5,
        explanatory_adequacy=0.5,
        social_appraisal=0.5,
        clarification_penalty=0.1,
        calibration_gap=0.05,
    )

    assert "llm" in result.details
    assert manager.last_spec == "understanding_aggregator"
