import types

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.social.interaction_miner import InteractionMiner


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.calls.append((spec_key, input_payload))
        return self.payload


def _miner(monkeypatch, llm_payload):
    arch = types.SimpleNamespace(memory=types.SimpleNamespace())
    miner = InteractionMiner(arch)
    monkeypatch.setattr(miner, "_llm_enabled", lambda: True)
    stub = StubManager(llm_payload)
    monkeypatch.setattr(miner, "_llm_manager", lambda: stub)
    return miner, stub


def test_llm_rules_are_injected(monkeypatch):
    payload = {
        "speech_act": "question",
        "confidence": 0.9,
        "suggested_rules": [
            {
                "rule": "offrir_aide",
                "expected_effect": "réduit l'anxiété",
                "confidence": 0.8,
            }
        ],
    }
    miner, stub = _miner(monkeypatch, payload)

    rules = miner.mine_text("User: Peux-tu m'aider ?\nBot: Bien sûr !", source="inbox:test")

    assert stub.calls, "LLM manager should be invoked"
    assert rules, "Rules should be returned"
    llm_rules = [r for r in rules if "llm" in (r.tags or [])]
    assert llm_rules, "LLM rules should be tagged"
    assert llm_rules[0].tactic.name == "offrir_aide"


def test_llm_failure_falls_back(monkeypatch):
    miner, _ = _miner(monkeypatch, {})
    monkeypatch.setattr(miner, "_llm_manager", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(miner, "_llm_annotate", lambda *args, **kwargs: None)
    monkeypatch.setattr(miner, "_extract_rules", lambda turns, source, ann=None: [])

    rules = miner.mine_text("User: Merci !\nBot: Avec plaisir.")

    assert rules == []
