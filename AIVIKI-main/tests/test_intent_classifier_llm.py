import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.io import intent_classifier as ic


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


def _empty_patterns():
    return {"THREAT": [], "QUESTION": [], "COMMAND": []}


def test_llm_classification_overrides_when_confident(monkeypatch):
    manager = StubManager({"intent": "command", "confidence": 0.92})
    monkeypatch.setattr(ic, "_llm_enabled", lambda: True)
    monkeypatch.setattr(ic, "_llm_manager", lambda: manager)
    monkeypatch.setattr(ic, "_get_patterns", _empty_patterns)
    monkeypatch.setattr(ic, "_load_fallback_model", lambda: None)
    monkeypatch.setattr(ic, "_classify_with_fallback", lambda original, normalized: "INFO")
    monkeypatch.setattr(ic, "log_uncertain_intent", lambda *args, **kwargs: None)
    monkeypatch.setattr(ic, "_log_llm_decision", lambda *args, **kwargs: None)

    label = ic.classify("merci de lancer backup manuel")

    assert label == "COMMAND"
    assert manager.last_spec == "intent_classification"
    assert manager.last_payload["utterance"] == "merci de lancer backup manuel"


def test_llm_rejects_low_confidence(monkeypatch):
    manager = StubManager({"intent": "info", "confidence": 0.2})
    monkeypatch.setattr(ic, "_llm_enabled", lambda: True)
    monkeypatch.setattr(ic, "_llm_manager", lambda: manager)
    monkeypatch.setattr(ic, "_get_patterns", _empty_patterns)
    monkeypatch.setattr(ic, "_load_fallback_model", lambda: None)
    monkeypatch.setattr(ic, "_classify_with_fallback", lambda original, normalized: "INFO")
    logged = {}
    monkeypatch.setattr(ic, "log_uncertain_intent", lambda *args, **kwargs: logged.setdefault("called", True))
    monkeypatch.setattr(ic, "_log_llm_decision", lambda *args, **kwargs: None)

    label = ic.classify("analyse le document joint")

    assert label == "INFO"
    assert logged.get("called") is True
