import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from AGI_Evolutive.beliefs import graph as belief_graph_module  # noqa: E402


def _build_graph(tmp_path):
    graph = belief_graph_module.BeliefGraph(
        path=str(tmp_path / "beliefs.jsonl")
    )
    graph._cache.clear()
    return graph


def test_belief_graph_summarize_uses_llm(monkeypatch, tmp_path):
    captured = {}

    def _stub(spec_key, *, input_payload=None, **_kwargs):
        captured["spec_key"] = spec_key
        captured["payload"] = input_payload
        if spec_key == "belief_graph_summary":
            return {
                "narrative": "Synthèse riche des croyances.",
                "highlights": [
                    {
                        "fact": "L'agent adore coder",
                        "support": "likes -> coding",
                        "confidence": 0.8,
                    }
                ],
                "alerts": ["Contradiction sur la disponibilité"],
                "confidence": 0.77,
                "notes": "Mettre à jour la surveillance.",
            }
        return None

    monkeypatch.setattr(belief_graph_module, "try_call_llm_dict", _stub)

    graph = _build_graph(tmp_path)

    belief = belief_graph_module.Belief.new(
        "agent:me",
        "likes",
        "coding",
        subject_label="Agent",
        value_label="Coding",
        confidence=0.85,
    )
    graph._cache[belief.id] = belief

    summary = graph.summarize()

    assert captured["spec_key"] == "belief_graph_summary"
    assert captured["payload"]["relations"]["likes"][0]["value"] == "Coding"
    assert summary["source"] == "llm"
    assert summary["narrative"] == "Synthèse riche des croyances."
    assert summary["relations"]["likes"][0]["value"] == "Coding"


def test_belief_graph_summarize_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(belief_graph_module, "try_call_llm_dict", lambda *_, **__: None)

    graph = _build_graph(tmp_path)
    belief = belief_graph_module.Belief.new(
        "agent:me",
        "prefers",
        "testing",
        subject_label="Agent",
        value_label="Testing",
        confidence=0.6,
    )
    graph._cache[belief.id] = belief

    summary = graph.summarize()

    assert summary["source"] == "heuristic"
    assert "prefers" in summary["relations"]
    assert summary["relations"]["prefers"][0]["value"] == "Testing"
