import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.memory import concept_extractor as ce


class StubManager:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def call_dict(self, spec_key, *, input_payload=None, **_kwargs):
        self.calls.append((spec_key, input_payload))
        return self.payload


def test_llm_augmentation_injects_concepts(tmp_path, monkeypatch):
    manager = StubManager({
        "concepts": [
            {"term": "gestion erreurs", "score": 0.9},
            {"term": "cache redis", "score": 0.4},
        ]
    })
    monkeypatch.setattr(ce, "_llm_enabled", lambda: True)
    monkeypatch.setattr(ce, "_llm_manager", lambda: manager)

    extractor = ce.ConceptExtractor(memory_store=None, data_path=str(tmp_path / "concepts.json"))
    profile = ce.Profile("test", min_concepts=1)

    concepts = extractor._extract_concepts("Bug critique", {"metadata": {}}, profile)

    assert any("gestion erreurs" in concept for concept in concepts)
    assert manager.calls, "LLM manager should have been invoked"
    feedback = extractor.last_llm_feedback
    assert feedback is not None and feedback.get("suggestions")
