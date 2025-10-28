import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.utils.llm_client import JSON_ONLY_DIRECTIVE  # noqa: E402
from AGI_Evolutive.utils.llm_specs import (  # noqa: E402
    AVAILABLE_MODELS,
    LLM_INTEGRATION_SPECS,
    get_spec,
)


def test_specs_have_unique_keys_and_valid_models():
    keys = [spec.key for spec in LLM_INTEGRATION_SPECS]
    assert len(keys) == len(set(keys))

    valid_models = set(AVAILABLE_MODELS.values())
    for spec in LLM_INTEGRATION_SPECS:
        assert spec.preferred_model in valid_models


def test_example_outputs_are_json_serializable_and_files_exist():
    for spec in LLM_INTEGRATION_SPECS:
        json.dumps(spec.example_output)
        module_path = ROOT / spec.module
        assert module_path.exists(), f"missing module path for {spec.key}: {spec.module}"


def test_build_prompt_includes_directive_example_and_extra():
    spec = get_spec("planner_support")
    prompt = spec.build_prompt(
        input_payload={"goal": "stabiliser"}, extra=["Considère les risques."]
    )

    assert JSON_ONLY_DIRECTIVE in prompt
    assert "Considère les risques." in prompt
    assert "Si tu n'es pas certain" in prompt
    assert '"plan"' in prompt
    assert '"goal"' in prompt


@pytest.mark.parametrize("key", [
    "intent_classification",
    "language_understanding",
    "concept_extraction",
    "planner_support",
    "rag5_controller",
    "reward_engine",
    "identity_mission",
])
def test_expected_specs_are_present(key):
    assert any(spec.key == key for spec in LLM_INTEGRATION_SPECS)
