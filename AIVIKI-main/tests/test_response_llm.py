import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.runtime import response as resp


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.calls.append((spec_key, input_payload))
        return self.payload


def test_llm_rewrite_reasoning_used_when_no_sentinels(monkeypatch):
    manager = StubManager(
        {
            "hypothese": "Incident lié au proxy",
            "incertitude": 0.3,
            "besoins": ["logs proxy"],
            "questions": ["Dernier déploiement ?"],
            "summary": "Latence observée sur EU",
        }
    )
    monkeypatch.setattr(resp, "_llm_enabled", lambda: True)
    monkeypatch.setattr(resp, "_llm_manager", lambda: manager)

    normalized, diagnostics = resp.humanize_reasoning_block("analyse brute sans structure")

    assert "Hypothèse" in normalized
    assert diagnostics is not None
    assert diagnostics["hypothesis"] == "Incident lié au proxy"
    assert diagnostics["needs"] == ["logs proxy"]
    assert manager.calls and manager.calls[0][0] == "response_formatter"
