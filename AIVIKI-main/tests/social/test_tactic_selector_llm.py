import types

from pathlib import Path
import sys
import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.social.tactic_selector import TacticSelector


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.calls.append((spec_key, input_payload))
        return self.payload


def _selector(monkeypatch, llm_payload):
    arch = types.SimpleNamespace()
    memory = types.SimpleNamespace()
    memory.get_recent_memories = lambda kind="", limit=0: [
        {
            "id": "rule-1",
            "tactic": {"name": "empathic_acknowledgment"},
            "ema_reward": 0.4,
            "bandit": {"type": "linucb_full", "dim": 16, "alpha": 0.6, "l2": 1.0, "forget": 1.0},
        }
    ]
    arch.memory = memory
    selector = TacticSelector(arch)
    monkeypatch.setattr("AGI_Evolutive.social.tactic_selector.is_llm_enabled", lambda: True)
    stub = StubManager(llm_payload)
    monkeypatch.setattr("AGI_Evolutive.social.tactic_selector.get_llm_manager", lambda: stub)
    return selector, stub


def test_llm_adjusts_priority(monkeypatch):
    payload = {
        "tactics": [
            {
                "rule_id": "rule-1",
                "name": "empathic_acknowledgment",
                "utility": 0.9,
                "risk": 0.1,
                "explanation": "favorise la relation",
            }
        ]
    }
    selector, stub = _selector(monkeypatch, payload)

    rule, meta = selector.pick({"dialogue_act": "question", "risk_level": "low"})

    assert stub.calls, "LLM manager should be invoked"
    assert rule is not None
    assert meta["llm"]["priority"] > 0.5


def test_llm_disabled_falls_back(monkeypatch):
    selector, _ = _selector(monkeypatch, {"tactics": []})
    monkeypatch.setattr("AGI_Evolutive.social.tactic_selector.is_llm_enabled", lambda: False)

    rule, meta = selector.pick({"dialogue_act": "question", "risk_level": "low"})

    assert rule is not None
    assert "llm" not in meta
