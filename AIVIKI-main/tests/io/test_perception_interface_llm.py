from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.io import perception_interface as pi


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.last_spec = None
        self.last_payload = None

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.last_spec = spec_key
        self.last_payload = input_payload
        return self.payload


class DummyMemory:
    def __init__(self):
        self.records = []

    def add_memory(self, payload):
        self.records.append(payload)


def test_perception_ingest_uses_llm(monkeypatch, tmp_path):
    manager = StubManager(
        {
            "modality": "texte",
            "salient_entities": ["rapport"],
            "requires_attention": True,
        }
    )

    monkeypatch.setattr(pi, "_llm_enabled", lambda: True)
    monkeypatch.setattr(pi, "_llm_manager", lambda: manager)

    memory = DummyMemory()
    iface = pi.PerceptionInterface(memory_store=memory, inbox_dir=tmp_path)
    captured = []
    monkeypatch.setattr(iface, "_log", lambda rec: captured.append(rec))

    iface.ingest_user_message("Priorise le rapport", meta={"channel": "chat"})

    assert captured[0]["llm"]["requires_attention"] is True
    assert memory.records[0]["llm"]["salient_entities"] == ["rapport"]
    assert manager.last_spec == "perception_preprocess"
