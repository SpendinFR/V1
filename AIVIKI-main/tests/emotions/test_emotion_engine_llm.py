from pathlib import Path
import sys
import types
import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.emotions.emotion_engine import EmotionEngine


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.calls.append((spec_key, input_payload))
        return self.payload


def _engine(monkeypatch, llm_payload):
    engine = EmotionEngine(
        path_state="data/test_mood.json",
        path_dashboard="data/test_dashboard.json",
        path_log="data/test_log.jsonl",
    )
    monkeypatch.setattr("AGI_Evolutive.emotions.emotion_engine.is_llm_enabled", lambda: True)
    stub = StubManager(llm_payload)
    monkeypatch.setattr("AGI_Evolutive.emotions.emotion_engine.get_llm_manager", lambda: stub)
    monkeypatch.setattr(engine, "_collect_context", lambda: {"recent_success": 0.2})
    monkeypatch.setattr(engine, "save", lambda: None)
    monkeypatch.setattr(engine.aggregator, "step", lambda ctx, quality=None: (0.1, 0.05, 0.02, {"error": 0.5}, {"error": 1.0}))
    monkeypatch.setattr(engine._synthesizer, "observe", lambda ctx, quality=None: None)
    return engine, stub


def test_llm_enriches_step(monkeypatch):
    payload = {
        "emotions": [
            {"name": "stress", "intensity": 0.6},
            {"name": "espoir", "intensity": 0.3},
        ],
        "regulation_suggestion": "Prendre une pause",
    }
    engine, stub = _engine(monkeypatch, payload)

    engine.step(force=True, quality=0.1)

    assert stub.calls, "LLM should have been invoked"
    assert engine._last_llm_annotation is not None
    assert "regulation_suggestion" in engine.last_modulators


def test_llm_disabled(monkeypatch):
    payload = {"emotions": []}
    engine, _ = _engine(monkeypatch, payload)
    monkeypatch.setattr("AGI_Evolutive.emotions.emotion_engine.is_llm_enabled", lambda: False)

    engine.step(force=True, quality=0.0)
    assert engine._last_llm_annotation is None
