import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.utils.llm_client import LLMResult
from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationManager,
    LLMUnavailableError,
    get_llm_manager,
    set_llm_manager,
    try_call_llm_dict,
)


class DummyClient:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def generate_json(
        self,
        model,
        task_description,
        *,
        input_data=None,
        extra_instructions=None,
        example_output=None,
        max_retries=1,
    ):
        self.calls.append(
            {
                "model": model,
                "task_description": task_description,
                "input_data": input_data,
                "extra_instructions": list(extra_instructions or []),
                "example_output": example_output,
            }
        )
        return LLMResult(parsed=self.parsed, raw="{}", response={"response": "{}"})


def test_call_disabled_raises_unavailable():
    manager = LLMIntegrationManager(client=DummyClient({}), enabled=False)
    with pytest.raises(LLMUnavailableError):
        manager.call_json("intent_classification")


def test_call_dict_returns_mapping_and_instructions():
    client = DummyClient({"intent": "QUESTION", "confidence": 0.9})
    manager = LLMIntegrationManager(client=client, enabled=True)

    payload = {"utterance": "Peux-tu m'aider ?"}
    data = manager.call_dict("intent_classification", input_payload=payload)

    assert data["intent"] == "QUESTION"
    assert client.calls, "The dummy client should have been invoked"
    instructions = client.calls[0]["extra_instructions"]
    assert any("incertitude" in instruction for instruction in instructions)
    example_output = client.calls[0]["example_output"]
    assert isinstance(example_output, dict)
    assert "class_probabilities" in example_output


def test_singleton_override_roundtrip(monkeypatch):
    original = get_llm_manager()
    try:
        dummy_manager = LLMIntegrationManager(client=DummyClient({"ok": True}), enabled=True)
        set_llm_manager(dummy_manager)
        assert get_llm_manager() is dummy_manager
    finally:
        set_llm_manager(original)


def test_try_call_llm_dict_handles_disabled(monkeypatch):
    manager = LLMIntegrationManager(client=DummyClient({"ok": True}), enabled=False)
    original = get_llm_manager()
    try:
        set_llm_manager(manager)
        assert try_call_llm_dict("intent_classification", input_payload={"x": 1}) is None
    finally:
        set_llm_manager(original)


def test_try_call_llm_dict_returns_mapping(monkeypatch):
    client = DummyClient({"intent": "INFO", "confidence": 0.51})
    manager = LLMIntegrationManager(client=client, enabled=True)
    original = get_llm_manager()
    try:
        set_llm_manager(manager)
        result = try_call_llm_dict("intent_classification", input_payload={"utterance": "test"})
        assert result == {"intent": "INFO", "confidence": 0.51}
        assert client.calls, "LLM client should be invoked via helper"
    finally:
        set_llm_manager(original)
