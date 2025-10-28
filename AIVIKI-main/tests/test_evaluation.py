import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.core import evaluation


def test_unified_priority_returns_clamped_value():
    score = evaluation.unified_priority(
        impact=0.9,
        probability=0.8,
        reversibility=0.7,
        effort=0.5,
        uncertainty=0.1,
        valence=0.2,
    )
    assert 0.0 <= score <= 1.0
    token = evaluation.get_last_priority_token()
    assert token is not None


def test_record_priority_feedback_updates_model_and_bandit():
    manager = evaluation._MANAGER  # type: ignore[attr-defined]
    manager.bandit._values["glm"] = 1.0  # force GLM selection
    manager.bandit._values["heuristic"] = 0.0
    manager.bandit.exploration = 0.0

    # Ensure deterministic state for the pending queue
    manager._pending.clear()

    before_weights = dict(manager.model.weights)

    score = evaluation.unified_priority(
        impact=0.6,
        probability=0.6,
        reversibility=0.9,
        effort=0.4,
        uncertainty=0.2,
        valence=0.0,
    )
    assert 0.0 <= score <= 1.0

    token = evaluation.get_last_priority_token()
    assert token is not None

    evaluation.record_priority_feedback(token, 0.8)

    assert manager.bandit._values["glm"] > 0.0
    after_weights = manager.model.weights
    assert any(after_weights[k] != before_weights.get(k, 0.0) for k in after_weights)
