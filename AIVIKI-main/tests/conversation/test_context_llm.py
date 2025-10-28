from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.conversation import context


class DummyMemory:
    def get_recent_memories(self, n: int):
        now = time.time()
        return [
            {"kind": "interaction", "text": "On doit préparer un rapport.", "ts": now - 60},
            {"kind": "interaction", "text": "Peux-tu accélérer les tests ?", "ts": now - 30},
        ]


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


def test_context_builder_uses_llm(monkeypatch):
    arch = SimpleNamespace(
        memory=DummyMemory(),
        consolidator=SimpleNamespace(state={"lessons": []}),
    )
    builder = context.ContextBuilder(arch)

    manager = StubManager(
        {
            "summary": "Utilisateur focalisé sur les tests.",
            "topics": [{"rank": 1, "label": "tests"}],
            "tone": "pressé",
        }
    )

    monkeypatch.setattr(context, "_llm_enabled", lambda: True)
    monkeypatch.setattr(context, "_llm_manager", lambda: manager)
    monkeypatch.setattr(context.ContextBuilder, "_persona_settings", lambda self, user_id: {})

    ctx = builder.build("Peux-tu prioriser les tests unitaires ?")

    assert ctx["llm_summary"]["tone"] == "pressé"
    assert ctx["llm_topics"][0]["label"] == "tests"
    assert manager.last_spec == "conversation_context"
