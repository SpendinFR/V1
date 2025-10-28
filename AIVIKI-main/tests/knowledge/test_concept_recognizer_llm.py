import types

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.knowledge.concept_recognizer import ConceptRecognizer, ItemCandidate


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.calls.append((spec_key, input_payload))
        return self.payload


def _recognizer(monkeypatch, llm_payload):
    arch = types.SimpleNamespace()
    arch.memory = types.SimpleNamespace()
    arch.planner = types.SimpleNamespace()
    recognizer = ConceptRecognizer(arch)
    stub = StubManager(llm_payload)
    monkeypatch.setattr("AGI_Evolutive.knowledge.concept_recognizer.is_llm_enabled", lambda: True)
    monkeypatch.setattr("AGI_Evolutive.knowledge.concept_recognizer.get_llm_manager", lambda: stub)
    return recognizer, stub


def test_llm_adjusts_scores(monkeypatch):
    payload = {
        "candidates": [
            {
                "candidate": "proxy inverse",
                "status": "à_apprendre",
                "justification": "utile pour incidents",
                "recommended_learning": "lire guide",
                "confidence": 0.9,
            }
        ]
    }
    recognizer, stub = _recognizer(monkeypatch, payload)

    item = ItemCandidate(
        kind="concept",
        label="proxy inverse",
        score=0.3,
        evidence={},
        features={},
        ts=0.0,
    )
    items = recognizer._apply_llm_guidance("Le proxy inverse est clé.", [item])

    assert stub.calls, "LLM should be invoked"
    assert items[0].score >= 0.5
    assert "llm" in items[0].features


def test_llm_notes_logged(monkeypatch):
    recognizer, stub = _recognizer(monkeypatch, {"notes": "aucun"})
    monkeypatch.setattr("AGI_Evolutive.knowledge.concept_recognizer.get_llm_manager", lambda: stub)
    items = recognizer.extract_candidates("Un proxy inverse permet de sécuriser l'accès.")
    assert items
