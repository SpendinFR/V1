import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AGI_Evolutive as agi_pkg  # noqa: E402


def test_package_overview_uses_llm_when_available(monkeypatch):
    payload = {
        "summary": "Synthèse experte du noyau AGI.",
        "capabilities": "Coordination avancée",
        "recommended_focus": ["Normaliser les intégrations"],
        "alerts": [],
        "confidence": 0.91,
        "notes": "Profil validé.",
    }
    captured = {}

    def _stub(spec_key, *, input_payload=None, **_kwargs):
        captured["spec_key"] = spec_key
        captured["payload"] = input_payload
        return payload

    monkeypatch.setattr(agi_pkg, "try_call_llm_dict", _stub)

    overview = agi_pkg.package_overview(extra_notes=["Mise à jour du référentiel"])

    assert captured["spec_key"] == "package_overview"
    assert "stats" in captured["payload"]
    assert overview["source"] == "llm"
    assert overview["summary"] == "Synthèse experte du noyau AGI."
    assert overview["capabilities"] == ["Coordination avancée"]
    assert overview["stats"] == captured["payload"]["stats"]


def test_package_overview_fallback_when_llm_disabled(monkeypatch):
    monkeypatch.setattr(agi_pkg, "try_call_llm_dict", lambda *_, **__: None)

    overview = agi_pkg.package_overview()

    assert overview["source"] == "heuristic"
    assert overview["summary"]
    assert overview["capabilities"]
    assert overview["stats"]["package"] == "AGI_Evolutive"
