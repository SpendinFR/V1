from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - dependency optional in tests
    import numpy as _np  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("numpy is required for perception LLM tests", allow_module_level=True)

from AGI_Evolutive.perception import __init__ as perception_module


def _perception_system(monkeypatch, llm_payload):
    calls = {}

    def _stub(key, input_payload, **kwargs):
        calls["key"] = key
        calls["payload"] = input_payload
        return llm_payload

    monkeypatch.setattr(perception_module, "try_call_llm_dict", _stub)
    system = perception_module.PerceptionSystem()
    return system, calls


def test_perception_system_enriches_scene_with_llm(monkeypatch):
    llm_response = {
        "observations": [{"modality": "visuel", "issue": "faible contraste"}],
        "recommended_settings": {"sensibility": 0.2, "window_seconds": 0.3},
        "notes": "Prioriser le canal visuel",
    }
    system, calls = _perception_system(monkeypatch, llm_response)

    scene = system.process_sensory_input({
        perception_module.Modality.VISUAL: perception_module.np.zeros((2, 2)),
    })

    assert calls["key"] == "perception_module"
    payload = calls["payload"]
    assert payload["object_count"] >= 0
    assert "metrics" in payload
    assert "llm_analysis" in scene.background
    assert scene.background["llm_analysis"]["notes"] == "Prioriser le canal visuel"
    assert system._last_llm_summary == scene.background["llm_analysis"]
    assert system.perceptual_parameters["sensitivity_threshold"] == pytest.approx(0.2)
    assert system.perceptual_parameters["integration_window"] == pytest.approx(0.3)


def test_perception_system_llm_fallback(monkeypatch):
    def _stub(*_args, **_kwargs):
        return None

    monkeypatch.setattr(perception_module, "try_call_llm_dict", _stub)
    system = perception_module.PerceptionSystem()
    baseline = dict(system.perceptual_parameters)

    scene = system.process_sensory_input({
        perception_module.Modality.VISUAL: perception_module.np.zeros((1, 1)),
    })

    assert "llm_analysis" not in scene.background
    assert system._last_llm_summary is None
    assert system.perceptual_parameters["sensitivity_threshold"] == baseline["sensitivity_threshold"]
