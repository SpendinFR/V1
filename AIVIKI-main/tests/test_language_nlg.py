import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from AGI_Evolutive.language import nlg


def _make_tracker() -> nlg.QualityTracker:
    ema_explorer = nlg.BanditExplorer(epsilon=0.0, seed=777)
    ema = nlg.AdaptiveEMA(betas=(0.5,), explorer=ema_explorer)
    return nlg.QualityTracker(evaluator=nlg.DefaultQualityEvaluator(), ema=ema)


def test_paraphrase_light_bandit_guides_choices():
    explorer = nlg.BanditExplorer(epsilon=0.0, seed=123)
    tracker = _make_tracker()
    classifier = nlg.OnlineFallbackClassifier()

    text = "C'est important pour le plan."
    result = nlg.paraphrase_light(
        text,
        prob=1.0,
        explorer=explorer,
        quality_tracker=tracker,
        classifier=classifier,
        rng=random.Random(0),
    )

    assert "important" not in result
    assert any(syn in result for syn in ("clé", "central", "majeur"))
    params = explorer.arm_parameters("important")
    assert params
    # The chosen synonym should have received additional reward mass
    chosen_alpha = max(alpha for alpha, _ in params.values())
    assert chosen_alpha > 1.0


def test_paraphrase_light_handles_accents_and_elisions():
    explorer = nlg.BanditExplorer(epsilon=0.0, seed=321)
    tracker = _make_tracker()
    classifier = nlg.OnlineFallbackClassifier()
    rng = random.Random(1)

    text = "L'idée est simple et par défaut nous gagnons."
    result = nlg.paraphrase_light(
        text,
        prob=1.0,
        explorer=explorer,
        quality_tracker=tracker,
        classifier=classifier,
        rng=rng,
    )

    # Ensure normalized entries were matched despite accents
    assert "L’" in result or "L'" in result
    assert "intuition" in result or "piste" in result
    assert "basique" in result or "élémentaire" in result


def test_paraphrase_light_fallback_classifier_after_training():
    explorer = nlg.BanditExplorer(epsilon=0.0, seed=555)
    tracker = _make_tracker()
    classifier = nlg.OnlineFallbackClassifier()
    for _ in range(8):
        classifier.update("cool", 1.0)

    text = "C'est vraiment cool!"
    result = nlg.paraphrase_light(
        text,
        prob=0.0,
        explorer=explorer,
        quality_tracker=tracker,
        classifier=classifier,
        rng=random.Random(2),
    )

    assert "cool" not in result
    assert any(syn in result for syn in ("sympa", "amusant"))


def test_quality_tracker_records_history():
    tracker = _make_tracker()
    scores = [tracker.observe("test", candidate) for candidate in ("test", "Test", "TEST!")]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert len(tracker.quality_history()) == len(scores)


def test_join_tokens_elides_vowels():
    result = nlg.join_tokens(["le", "amour", "arrive"])
    assert "l’" in result or "l'" in result

