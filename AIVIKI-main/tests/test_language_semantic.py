import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.language import SemanticUnderstanding


def test_regex_handles_accented_questions():
    parser = SemanticUnderstanding()
    utt = parser.parse_utterance("EST CE QUE tu peux m'indiquer où est le dossier ?")
    assert utt.frame.intent == "ask_info"
    assert utt.frame.confidence > 0.6
    assert "ou est le dossier" in utt.normalized_text


def test_classifier_fallback_learns_plan_intent():
    parser = SemanticUnderstanding()
    parser.intent_classifier.partial_fit("organise un planning pour demain", "plan")
    parser.intent_classifier.partial_fit("planifie la roadmap trimestrielle", "plan")
    parser._classifier_state_dirty = True
    utt = parser.parse_utterance("Blueprint stratégique Q3")
    assert utt.frame.intent == "plan"
    assert utt.frame.confidence >= 0.55


def test_uncertainty_adjusts_with_hedges():
    parser = SemanticUnderstanding()
    utt = parser.parse_utterance("Je crois que tu pourrais peut-être envoyer le résumé.")
    assert utt.pragmatics["uncertainty"] >= 0.4
    assert utt.frame.intent in {"send", "summarize", "inform"}
    assert utt.frame.confidence < 0.72
    assert parser.intent_conf_tracker.drift_log
