import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.metacog.calibration import CalibrationMeter, NoveltyDetector


def _seed_calibration(meter: CalibrationMeter, domain: str = "fr") -> None:
    # Populate with confident successes and uncertain failures to create separable signal.
    for _ in range(25):
        eid = meter.log_prediction(domain, 0.82, {"text": "Réponse assurée", "uncertainty": 0.1})
        meter.log_outcome(eid, True)
    for _ in range(25):
        eid = meter.log_prediction(domain, 0.22, {"text": "Réponse hésitante", "uncertainty": 0.8})
        meter.log_outcome(eid, False)


def test_calibration_meter_calibrated_probability(tmp_path: Path) -> None:
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    path = calib_dir / "calibration.jsonl"
    meter = CalibrationMeter(path=str(path))
    _seed_calibration(meter, domain="fr")

    low = meter.calibrate_probability("fr", 0.2, {"text": "option faible", "uncertainty": 0.7})
    high = meter.calibrate_probability("fr", 0.85, {"text": "option forte", "uncertainty": 0.1})

    assert low < high
    assert 0.0 <= low <= 0.6
    assert 0.4 <= high <= 1.0

    threshold = meter.calibrated_threshold("fr", base=0.45)
    assert 0.2 < threshold < 0.7

    fallback = meter.calibrated_threshold("uknown", base=0.33)
    # No data for this domain → fallback to dynamic heuristic
    assert math.isclose(fallback, meter.dynamic_threshold("uknown", base=0.33), rel_tol=1e-6)


def test_novelty_detector_online_classifier() -> None:
    detector = NoveltyDetector(window=16, threshold=0.5)

    # Seed baseline window with normal utterances
    for text in ("Bonjour équipe", "Je reviens", "Voici un rapport stable", "Ceci est normal"):
        detector.assess(text, update=True)
        detector.feedback(text, novel=False)

    # Teach classifier about novel bursts rich in emoji/punctuation
    novel_text = "🚀🚀 Lancement inattendu !!! 🤖"
    for _ in range(30):
        detector.feedback(novel_text, novel=True, weight=1.5)
        detector.feedback("Message classique sans surprise", novel=False)

    novel_score, novel_flag = detector.assess(novel_text, update=False)
    normal_score, normal_flag = detector.assess("Message classique sans surprise", update=False)

    assert novel_score >= detector.threshold
    assert novel_flag
    assert normal_score < detector.threshold
    assert not normal_flag
