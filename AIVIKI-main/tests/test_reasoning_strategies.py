from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from AGI_Evolutive.reasoning.strategies import (
    DecompositionStrategy,
    HypothesisRankingStrategy,
    ReasoningStrategy,
    SelfCheckStrategy,
)


def test_reasoning_strategy_interface_contract():
    class DummyStrategy(ReasoningStrategy):
        name = "dummy"

        def apply(self, prompt, context, toolkit):
            return self.validate_output(
                {
                    "notes": "ok",
                    "proposals": [],
                    "questions": [],
                    "cost": 0.1,
                    "time": 0.0,
                }
            )

    dummy = DummyStrategy()
    out = dummy.apply("", {}, {})
    assert out["notes"] == "ok"
    assert set(ReasoningStrategy.REQUIRED_FIELDS).issubset(out.keys())


def test_decomposition_handles_accents_and_definition():
    strategy = DecompositionStrategy()
    toolkit = {}
    prompt = "Qu'est-ce qu'une étoile filante ?"
    result = strategy.apply(prompt, {}, toolkit)

    assert any("Définir/clarifier" in q for q in result["questions"])
    assert any("caractéristiques essentielles" in q for q in result["questions"])


def test_hypothesis_ranking_updates_with_feedback():
    strategy = HypothesisRankingStrategy()
    toolkit = {}
    support = ["Doc: Les chats dorment 16h par jour"]

    first = strategy.apply("Pourquoi les chats dorment autant ?", {"support": support}, toolkit)
    first_conf = first["proposals"][0]["confidence"]

    feedback_context = {
        "support": support,
        "hypothesis_feedback": [
            {"answer": first["proposals"][0]["answer"], "accepted": True}
        ],
    }
    second = strategy.apply("Pourquoi les chats dorment autant ?", feedback_context, toolkit)
    second_conf = second["proposals"][0]["confidence"]

    assert second_conf > first_conf


def test_selfcheck_detects_contradiction_with_accents():
    strategy = SelfCheckStrategy()
    context = {
        "last_answer": "Oui, c'est possible",
        "proposals": [
            {"answer": "Non, ce n'est pas possible", "confidence": 0.9, "support": []}
        ],
    }
    result = strategy.apply("", context, {})
    proposal = result["proposals"][0]
    assert proposal["confidence"] < 0.9
