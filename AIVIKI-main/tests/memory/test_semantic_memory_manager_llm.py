import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.memory import semantic_memory_manager as smm


class DummyMemory:
    def __init__(self):
        self._now = 10.0

    def now(self):
        return self._now

    def list_items(self, _query):
        return [{"id": 1}, {"id": 2}]


class DummyConcept:
    def __init__(self):
        self.updated = False

    def decay_tick(self, _now):
        return None

    def update_from_items(self, _items):
        self.updated = True


class DummyEpisodic:
    def link(self, _items):
        return [{"link": 1}]


class DummySummarizer:
    def __init__(self):
        self.steps = 0

    def step(self, _now):
        self.steps += 1
        return {"written": 1}


@pytest.fixture
def manager(monkeypatch):
    concept = DummyConcept()
    episodic = DummyEpisodic()
    summarizer = DummySummarizer()
    store = DummyMemory()
    mgr = smm.SemanticMemoryManager(
        memory_store=store,
        concept_store=concept,
        episodic_linker=episodic,
        summarizer=summarizer,
        concept_period_s=30,
        episodic_period_s=45,
        summarize_period_s=60,
    )
    return mgr


def test_semantic_memory_manager_uses_llm_guidance(monkeypatch, manager):
    captured = {}

    def _stub(key, input_payload, **_kwargs):
        captured["key"] = key
        captured["payload"] = input_payload
        return {
            "tasks": [
                {"task": "concept", "category": "urgent"},
                {"task": "episodic", "category": "court_terme"},
                {"task": "summarize", "category": "long_terme"},
            ],
            "notes": "Ajuster les priorités",
        }

    monkeypatch.setattr(smm, "try_call_llm_dict", _stub)

    stats = manager.tick()

    assert captured["key"] == "semantic_memory_manager"
    payload = captured["payload"]
    assert payload["tasks"]
    assert stats["llm_guidance"]["notes"] == "Ajuster les priorités"
    applied = stats["llm_guidance"].get("applied_adjustments")
    assert applied
    assert "concept" in applied
    assert applied["concept"]["period"] <= applied["concept"]["previous_period"]
    assert manager.last_llm_guidance == stats["llm_guidance"]
    assert len(manager.llm_guidance_history) == 1


def test_semantic_memory_manager_llm_fallback(monkeypatch, manager):
    monkeypatch.setattr(smm, "try_call_llm_dict", lambda *_args, **_kwargs: None)
    stats = manager.tick()

    assert "llm_guidance" not in stats
    assert manager.last_llm_guidance is None
