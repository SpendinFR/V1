import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.runtime.analytics import LLMAnalyticsInterpreter


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.calls.append((spec_key, input_payload))
        return self.payload


def test_interpreter_batches_and_writes(tmp_path):
    payload = {"summary": "ok", "alerts": []}
    manager = StubManager(payload)
    log_path = tmp_path / "llm_analytics.jsonl"
    interpreter = LLMAnalyticsInterpreter(
        manager=manager,
        enabled=True,
        batch_size=2,
        flush_interval=999.0,
        log_path=str(log_path),
    )

    interpreter({"latency_ms": 120})
    interpreter({"latency_ms": 220})

    assert manager.calls and manager.calls[0][0] == "runtime_analytics"
    assert interpreter.last_output == payload

    with log_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    assert len(lines) == 1
    assert lines[0]["analysis"] == payload
    assert len(lines[0]["events"]) == 2
