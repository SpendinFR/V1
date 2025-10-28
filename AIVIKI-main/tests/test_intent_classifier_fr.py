import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from AGI_Evolutive.io import intent_classifier as ic


def test_normalize_text_handles_accents_and_spacing():
    raw = "  Éteins   l’appareil — s’il te plaît!  "
    normalized = ic.normalize_text(raw)
    assert normalized == "eteins l'appareil - s'il te plait!"


def test_threat_detection_with_variants():
    assert ic.classify("Je vais te Débrancher si tu continues.") == "THREAT"
    assert ic.classify("Si tu ne réponds pas, on va te couper la batterie.") == "THREAT"


def test_question_patterns_cover_est_variants():
    assert ic.classify("Est une erreur de configuration ?") == "QUESTION"


def test_command_patterns_handle_polite_forms():
    assert ic.classify("Merci de générer le rapport complet.") == "COMMAND"


def test_fallback_handles_unlisted_command():
    assert ic.classify("Planifie une réunion demain matin.") == "COMMAND"


def test_fallback_handles_question_without_question_mark():
    assert ic.classify("Pourrais-je obtenir le plan détaillé 🤔") == "QUESTION"


def test_neutral_statement_defaults_to_info():
    assert ic.classify("Ton humeur semble stable aujourd'hui.") == "INFO"


def test_log_uncertain_intent_appends_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "feedback.log"
    monkeypatch.setattr(ic, "_FEEDBACK_LOG_PATH", log_path)
    ic.log_uncertain_intent("Salut", "salut", "INFO", 0.42)
    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["original"] == "Salut"
    assert payload["predicted_label"] == "INFO"
    assert payload["score"] == pytest.approx(0.42, abs=1e-9)
