from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from AGI_Evolutive.models.intent import TTL_CANDIDATES, IntentModel, _text_features


@pytest.fixture()
def intent_model(tmp_path: Path) -> IntentModel:
    path = tmp_path / "intent_state.json"
    return IntentModel(path=str(path))


def test_goal_pattern_handles_article_variants(intent_model: IntentModel) -> None:
    intent, confidence = intent_model.predict(
        "C'est un objectif pour moi de lancer une startup durable."
    )
    assert intent == "set_goal"
    assert confidence >= 0.7


def test_fallback_learning_handles_planification(intent_model: IntentModel) -> None:
    features = _text_features("Planification du sprint produit", base_conf=0.5)
    intent_model._update_fallback_learning("plan", features, 0.9)
    label, score = intent_model._fallback_classifier.predict(features)
    assert label == "plan"
    assert score >= 0.55


def test_ttl_assignment_uses_bandit(intent_model: IntentModel) -> None:
    findings = intent_model.observe_user_message(
        "Mon objectif est de finaliser le rapport trimestriel.",
        source="unit_test",
    )
    assert findings
    intent = findings[0]
    assert intent.ttl_days in TTL_CANDIDATES
    assert intent.ttl_choice in {str(value) for value in TTL_CANDIDATES}
