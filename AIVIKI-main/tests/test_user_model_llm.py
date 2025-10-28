import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.models import user as user_module


class StubCall:
    def __init__(self, payload):
        self.payload = payload
        self.spec = None
        self.input_payload = None

    def __call__(self, spec_key, *, input_payload=None, **kwargs):
        self.spec = spec_key
        self.input_payload = input_payload
        return self.payload


def _memory(content: str) -> dict:
    return {"content": content, "metadata": {"timestamp": time.time(), "channel": "chat"}}


def test_user_model_updates_with_llm(monkeypatch, tmp_path):
    model = user_module.UserModel(path=str(tmp_path / "user.json"))
    stub = StubCall(
        {
            "tone": "enthousiaste",
            "satisfaction": 0.72,
            "persona_traits": [{"trait": "curieux"}],
            "notes": "Profil ajusté",
        }
    )
    monkeypatch.setattr(user_module, "try_call_llm_dict", stub)

    count = model.ingest_memories([_memory("like:cafe")])

    assert count > 0
    assert stub.spec == "user_model"
    assert stub.input_payload["current_persona"]["tone"] == "neutral"
    assert model.state["persona"]["tone"] == "enthousiaste"
    assert pytest.approx(model.state["persona"]["satisfaction"], rel=1e-6) == 0.72
    assert model.state["llm_profiles"]["user_model"]["notes"] == "Profil ajusté"


def test_user_model_llm_fallback(monkeypatch, tmp_path):
    model = user_module.UserModel(path=str(tmp_path / "user.json"))

    calls = {"count": 0}

    def _stub(spec_key, *, input_payload=None, **kwargs):
        calls["count"] += 1
        return None

    monkeypatch.setattr(user_module, "try_call_llm_dict", _stub)

    memories = [_memory("like:tea"), _memory("did:jogging")]
    count = model.ingest_memories(memories)

    assert count >= 2
    assert calls["count"] == 1
    assert "tea" in model.state["preferences"]
    assert "user_model" not in model.state.get("llm_profiles", {})
