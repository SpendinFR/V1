from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.core.structures.mai import MAI
from AGI_Evolutive.knowledge import mechanism_store as ms
from AGI_Evolutive.knowledge import ontology_facade as of


def test_entity_linker_prefers_llm_prediction(monkeypatch):
    linker = of.EntityLinker()

    def _fake_try_call(spec_key: str, *, input_payload, **kwargs):
        assert spec_key == "knowledge_entity_typing"
        assert input_payload["label"] == "Jane Doe"
        return {"type": "Person", "confidence": 0.7}

    monkeypatch.setattr(of, "try_call_llm_dict", _fake_try_call)

    assert linker._infer_type("Jane Doe") == "Person"


def test_entity_linker_falls_back_when_llm_missing(monkeypatch):
    linker = of.EntityLinker()

    monkeypatch.setattr(of, "try_call_llm_dict", lambda *_args, **_kwargs: None)

    assert linker._infer_type("code 1234") == "Identifier"


def test_mechanism_store_llm_prioritises_candidates(monkeypatch, tmp_path: Path):
    store = ms.MechanismStore(path=tmp_path / "mai.jsonl")
    mai_a = MAI(id="mai_a", title="A", summary="First")
    mai_b = MAI(id="mai_b", title="B", summary="Second")
    store.add(mai_a)
    store.add(mai_b)

    def _fake_try_call(spec_key: str, *, input_payload, **kwargs):
        assert spec_key == "knowledge_mechanism_screening"
        assert len(input_payload["candidates"]) == 2
        return {
            "decisions": [
                {"id": "mai_b", "decision": "accept", "priority": 1},
                {"id": "mai_a", "decision": "reject", "reason": "Moins pertinent"},
            ]
        }

    monkeypatch.setattr(ms, "try_call_llm_dict", _fake_try_call)

    result = store.scan_applicable({}, {})
    assert [mai.id for mai in result] == ["mai_b"]


def test_mechanism_store_llm_graceful_fallback(monkeypatch, tmp_path: Path):
    store = ms.MechanismStore(path=tmp_path / "mai.jsonl")
    mai_a = MAI(id="mai_a", title="A")
    mai_b = MAI(id="mai_b", title="B")
    store.add(mai_a)
    store.add(mai_b)

    monkeypatch.setattr(ms, "try_call_llm_dict", lambda *_args, **_kwargs: None)

    result = store.scan_applicable({}, {})
    assert [mai.id for mai in result] == ["mai_a", "mai_b"]
