from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.cognition import reflection_loop as rl


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


class DummyMeta:
    def __init__(self):
        self.memory = SimpleNamespace(get_recent_memories=lambda n: [])

    def log_inner_monologue(self, text, tags=None):
        pass

    def assess_understanding(self):
        return {"domains": {}, "uncertainty": 0.5}

    def propose_learning_goals(self, max_goals=2):
        return []


def test_reflection_loop_uses_llm(monkeypatch):
    manager = StubManager(
        {
            "hypotheses": [
                {"statement": "Les erreurs viennent d'un pic", "status": "à_valider", "support": ["courbes"]}
            ],
            "follow_up_checks": [{"action": "Vérifier le trafic", "priority": 1}],
        }
    )

    monkeypatch.setattr(rl, "_llm_enabled", lambda: True)
    monkeypatch.setattr(rl, "_llm_manager", lambda: manager)

    loop = rl.ReflectionLoop(DummyMeta())
    result = loop.test_hypotheses({"observation": "Erreur 500"}, max_tests=2)

    assert result["tested"] == 1
    assert result["hypotheses"][0]["label"].startswith("Les erreurs")
    assert manager.last_spec == "reflection_loop"
