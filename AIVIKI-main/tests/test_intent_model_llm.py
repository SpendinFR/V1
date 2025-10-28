import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.models import intent as intent_module


def _intent(label: str, description: str, confidence: float = 0.7):
    item = intent_module.Intent(label=label, description=description)
    item.confidence = confidence
    item.last_seen = time.time()
    item.evidence.append(description)
    return item


def test_intent_model_llm_summary(monkeypatch, tmp_path):
    model = intent_module.IntentModel(path=str(tmp_path / "intent.json"))
    model._intents = {
        "resolve": _intent("resolve", "Résoudre l'incident", confidence=0.8),
        "learn": _intent("learn", "Apprendre du post-mortem", confidence=0.6),
    }

    captured = {}

    def _stub(spec_key, *, input_payload=None, **kwargs):
        captured["spec"] = spec_key
        captured["payload"] = input_payload
        return {
            "intent": "resolve",
            "horizon": "immédiat",
            "justification": "Priorité confirmée par l'utilisateur",
            "notes": "LLM ok",
        }

    monkeypatch.setattr(intent_module, "try_call_llm_dict", _stub)

    summary = model.llm_summary(recent_message="besoin d'aide pour l'incident")

    assert summary["intent"] == "resolve"
    assert summary["source"] == "llm"
    assert summary["notes"] == "LLM ok"
    assert captured["spec"] == "models_intent"
    assert captured["payload"]["recent_message"] == "besoin d'aide pour l'incident"
    assert captured["payload"]["top_intents"][0]["label"] == "resolve"


def test_intent_model_llm_fallback(monkeypatch, tmp_path):
    model = intent_module.IntentModel(path=str(tmp_path / "intent.json"))
    model._intents = {"resolve": _intent("resolve", "Résoudre l'incident", confidence=0.9)}

    monkeypatch.setattr(intent_module, "try_call_llm_dict", lambda *args, **kwargs: None)

    summary = model.llm_summary()

    assert summary["source"] == "fallback"
    assert summary["intent"] == "resolve"
    assert "llm_payload" not in summary
