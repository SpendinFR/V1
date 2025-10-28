import random

from AGI_Evolutive.language.understanding import SemanticUnderstanding


def test_definition_pattern_handles_est_un():
    parser = SemanticUnderstanding()
    frame = parser.parse_utterance("La polymathie est une discipline où l'on excelle dans plusieurs domaines.")
    defs = frame.slots.get("definition_candidates")
    assert defs, "Une définition implicite devrait être détectée"
    assert defs[0]["term"].lower().startswith("la polymathie")
    assert "discipline" in defs[0]["description"].lower()


def test_online_classifier_fallback_for_plan_intent():
    random.seed(0)
    parser = SemanticUnderstanding()
    parser.intent_classifier.partial_fit("organise un planning pour demain", "plan")
    parser.intent_classifier.partial_fit("planifie la roadmap trimestrielle", "plan")
    parser.intent_classifier.partial_fit("merci de planifier la stratégie Q2 !", "plan")
    frame = parser.parse_utterance("Plan stratégique Q4")
    assert frame.intent == "plan"
    assert frame.confidence >= 0.5


def test_uncertainty_tracker_records_drift_and_correlation():
    parser = SemanticUnderstanding()
    parser.parse_utterance("Bonjour !")
    parser.parse_utterance("Je crois que le Xylotech est un concept étrange 😅")
    assert parser.uncertainty_tracker.drift_log, "Les variations fortes doivent être journalisées"
    assert parser.uncertainty_correlation is not None
    assert -1.0 <= parser.uncertainty_correlation <= 1.0
