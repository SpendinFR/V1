import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.utils.llm_client import (  # noqa: E402
    JSON_ONLY_DIRECTIVE,
    LLMCallError,
    OllamaLLMClient,
    OllamaModelConfig,
    build_json_prompt,
)


def test_build_json_prompt_includes_directive_and_example():
    prompt = build_json_prompt(
        "Analyse les données et produis un résumé.",
        input_data={"a": 1},
        extra_instructions=["Priorise la clarté."],
        example_output={"resume": "..."},
    )

    assert "Analyse les données" in prompt
    assert JSON_ONLY_DIRECTIVE in prompt
    assert '"a": 1' in prompt
    assert '"resume"' in prompt
    assert "Priorise la clarté." in prompt


def test_generate_json_success(monkeypatch):
    captured = {}

    def fake_transport(path, data, timeout):
        captured["path"] = path
        captured["timeout"] = timeout
        payload = json.loads(data.decode("utf-8"))
        captured["payload"] = payload
        body = json.dumps({
            "response": json.dumps({"ok": True, "payload": payload["prompt"][:30]}),
            "done": True,
        }).encode("utf-8")
        return 200, body

    client = OllamaLLMClient(transport=fake_transport)
    model = OllamaModelConfig(name="qwen2.5:7b-instruct-q4_K_M", temperature=0.1, top_p=0.8)

    result = client.generate_json(
        model,
        "Fournis les actions recommandées.",
        input_data={"goal": "stabiliser"},
        example_output={"actions": []},
    )

    assert captured["path"] == "/api/generate"
    assert captured["payload"]["model"] == "qwen2.5:7b-instruct-q4_K_M"
    assert captured["payload"]["options"]["temperature"] == 0.1
    assert result.parsed["ok"] is True
    assert isinstance(result.raw, str)


def test_generate_json_raises_on_invalid_response():
    def fake_transport(path, data, timeout):
        body = json.dumps({"response": "pas du json", "done": True}).encode("utf-8")
        return 200, body

    client = OllamaLLMClient(transport=fake_transport)
    model = OllamaModelConfig(name="qwen3:8b-q4_K_M")

    with pytest.raises(LLMCallError):
        client.generate_json(
            model,
            "Teste la robustesse du parseur.",
            input_data="abc",
            example_output={"ok": False},
            max_retries=0,
        )

