import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.emotions import EmotionalState, EmotionalSystem  # noqa: E402


def _system_with_stub(monkeypatch, stub):
    monkeypatch.setattr("AGI_Evolutive.emotions.try_call_llm_dict", stub)
    return EmotionalSystem()


def test_appraise_consequences_llm_enriches(monkeypatch):
    captured = {}

    def fake_try_call(spec_key, *, input_payload=None, **_kwargs):
        assert spec_key == "emotional_system_appraisal"
        assert input_payload is not None
        captured["baseline"] = dict(input_payload["baseline_appraisal"])
        return {
            "appraisal": {
                "desirability": 0.8,
                "certainty": 0.9,
                "urgency": 0.7,
                "impact": 0.9,
                "controllability": 0.4,
            },
            "primary_emotion": "joie",
            "primary_intensity": 0.7,
            "secondary_candidates": [{"emotion": "confiance", "intensity": 0.5}],
            "confidence": 0.85,
        }

    system = _system_with_stub(monkeypatch, fake_try_call)
    result = system._appraise_consequences("Réussite urgente confirmée", {"channel": "ops"})

    assert "baseline" in captured
    baseline = captured["baseline"]
    expected_desirability = 0.4 * baseline["desirability"] + 0.6 * 0.8
    assert result["desirability"] == pytest.approx(expected_desirability)
    assert system._last_llm_appraisal is not None
    assert result["certainty"] > baseline["certainty"]


def test_appraise_consequences_llm_fallback(monkeypatch):
    system = _system_with_stub(monkeypatch, lambda *args, **kwargs: None)
    system._last_llm_appraisal = {"previous": True}
    result = system._appraise_consequences("Stimulus neutre", {})

    assert system._last_llm_appraisal is None
    assert isinstance(result["desirability"], float)


def test_primary_emotion_prefers_llm_hint(monkeypatch):
    system = _system_with_stub(monkeypatch, lambda *args, **kwargs: None)
    system._last_llm_appraisal = {
        "primary_emotion": "tristesse",
        "primary_intensity": 0.9,
        "confidence": 0.9,
    }

    consequence_appraisal = {
        "desirability": -0.8,
        "certainty": 0.9,
        "urgency": 0.3,
        "impact": 0.7,
        "controllability": 0.8,
    }

    emotion, intensity = system._determine_primary_emotion(
        consequence_appraisal,
        goal_congruence=0.2,
        coping_potential=0.6,
        norm_compatibility=0.8,
    )

    assert emotion == EmotionalState.SADNESS
    assert intensity > 0.4
