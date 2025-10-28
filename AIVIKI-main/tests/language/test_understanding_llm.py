from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.language import understanding


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


def test_llm_understanding_enriches_frame(monkeypatch):
    su = understanding.SemanticUnderstanding()

    manager = StubManager(
        {
            "intent": "request",
            "confidence": 0.84,
            "slots": {"target": "backup"},
            "canonical_query": "lancer un backup",
        }
    )

    monkeypatch.setattr(understanding, "_llm_enabled", lambda: True)
    monkeypatch.setattr(understanding, "_llm_manager", lambda: manager)
    monkeypatch.setattr(
        understanding.SemanticUnderstanding,
        "_classify_intent",
        lambda self, *args, **kwargs: ("info", 0.3),
    )

    frame = su.parse_utterance("Peux-tu lancer un backup manuel ?")

    assert frame.intent == "request"
    assert frame.meta["llm_understanding"]["canonical_query"] == "lancer un backup"
    assert frame.slots["target"] == "backup"
    assert manager.last_spec == "language_understanding"
    assert manager.last_payload["utterance"].startswith("Peux-tu")
