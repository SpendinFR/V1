from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple
import logging
import math
import random
from bisect import bisect_left, bisect_right

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


_LOGGER = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(value: float, *, limit: float = 1.0) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


@dataclass
class RunningMoments:
    forgetting: float
    mean: float = 0.0
    var: float = 1e-6
    initialized: bool = False

    def update(self, value: float) -> None:
        if not self.initialized:
            self.mean = value
            self.var = 1e-6
            self.initialized = True
            return
        delta = value - self.mean
        self.mean += self.forgetting * delta
        adjusted = delta * delta * self.forgetting
        self.var = max(1e-9, (1.0 - self.forgetting) * (self.var + adjusted))

    def threshold(self, *, base: float = 0.0, uncertainty: float = 0.5) -> float:
        if not self.initialized:
            return base - uncertainty
        spread = math.sqrt(max(self.var, 1e-9))
        return self.mean - uncertainty * spread


@dataclass
class AdaptiveDominanceModel:
    forgetting: float = 0.08
    learning_rate: float = 0.6
    ridge_penalty: float = 0.04
    exploration: float = 0.6
    weights: MutableMapping[str, float] = field(
        default_factory=lambda: {"acc": 2.0, "cal_ece": 1.2, "time": 0.8}
    )
    bias: float = 0.0
    score_moments: RunningMoments = field(
        default_factory=lambda: RunningMoments(forgetting=0.1)
    )
    warmup_samples: int = 5
    observations: int = 0

    def _features(
        self, champion: Mapping[str, float], challenger: Mapping[str, float]
    ) -> Dict[str, float]:
        acc_c = float(challenger.get("acc", 0.0))
        acc_p = float(champion.get("acc", 0.0))
        ece_c = float(challenger.get("cal_ece", 1.0))
        ece_p = float(champion.get("cal_ece", 1.0))
        time_c = float(challenger.get("time", 1.0))
        time_p = float(champion.get("time", 1.0))

        acc_diff = _clip(acc_c - acc_p, limit=1.0)
        ece_gain = _clip(ece_p - ece_c, limit=1.0)
        denom = max(time_p, 1e-6)
        time_gain = _clip((time_p - time_c) / denom, limit=1.0)
        return {"acc": acc_diff, "cal_ece": ece_gain, "time": time_gain}

    def _target(self, features: Mapping[str, float]) -> float:
        reward = 0.5
        reward += 0.9 * features.get("acc", 0.0)
        reward += 0.6 * features.get("cal_ece", 0.0)
        reward += 0.4 * features.get("time", 0.0)
        return min(1.0, max(0.0, reward))

    def _decay(self) -> float:
        return max(0.0, 1.0 - self.forgetting * self.ridge_penalty)

    def _score(self, features: Mapping[str, float]) -> float:
        score = self.bias
        for key, value in features.items():
            score += self.weights.get(key, 0.0) * value
        return score

    def _update(self, *, features: Mapping[str, float], target: float, score: float) -> None:
        prob = _sigmoid(score)
        error = target - prob
        decay = self._decay()
        self.bias = decay * self.bias + self.learning_rate * error
        for key, value in features.items():
            updated = decay * self.weights.get(key, 0.0) + self.learning_rate * error * value
            self.weights[key] = updated

    def should_accept(
        self, champion: Mapping[str, float], challenger: Mapping[str, float]
    ) -> bool:
        features = self._features(champion, challenger)
        score = self._score(features)
        threshold = self.score_moments.threshold(base=self.bias, uncertainty=self.exploration)
        accepted = score >= threshold
        target = self._target(features)
        self._update(features=features, target=target, score=score)
        self.score_moments.update(score)
        self.observations += 1
        return accepted


_ADAPTIVE_MODEL = AdaptiveDominanceModel()


def _llm_dominance_decision(
    champion: Mapping[str, float], challenger: Mapping[str, float]
) -> bool | None:
    payload = {
        "champion": dict(champion),
        "challenger": dict(challenger),
    }
    response = try_call_llm_dict(
        "self_improver_dominance",
        input_payload=payload,
        logger=_LOGGER,
        max_retries=2,
    )
    if not response:
        return None

    decision_raw = str(response.get("decision", "")).strip().lower()
    if not decision_raw:
        return None

    confidence = response.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else 1.0
    except (TypeError, ValueError):
        confidence_value = 0.0

    if decision_raw in {"accept", "approve", "go", "promote", "oui"}:
        if confidence_value >= 0.5:
            return True
    elif decision_raw in {"reject", "refuse", "stop", "non", "no"}:
        if confidence_value >= 0.5:
            return False
    return None


def _iter_metric_keys(samples: Iterable[Dict[str, Any]]) -> List[str]:
    keys = set()
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def aggregate_metrics(samples: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metric samples by averaging shared numeric keys."""
    if not samples:
        return {"acc": 0.0, "cal_ece": 1.0, "time": 0.0}

    keys = _iter_metric_keys(samples)
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [float(sample[key]) for sample in samples if key in sample]
        if values:
            aggregated[key] = float(mean(values))
    return aggregated


def _dominates_static(
    champion: Mapping[str, float],
    challenger: Mapping[str, float],
    eps_acc: float,
    max_ece_worsen: float,
    max_time_increase: float,
) -> bool:
    acc_c = challenger.get("acc", 0.0)
    acc_p = champion.get("acc", 0.0)
    ece_c = challenger.get("cal_ece", 1.0)
    ece_p = champion.get("cal_ece", 1.0)
    t_c = challenger.get("time", 1.0)
    t_p = champion.get("time", 1.0)

    if acc_c < acc_p + eps_acc:
        return False
    if (ece_c - ece_p) > max_ece_worsen:
        return False
    if (t_c - t_p) > max_time_increase * max(1e-6, t_p):
        return False
    return True


def dominates(
    champion: Dict[str, float],
    challenger: Dict[str, float],
    eps_acc: float = 0.01,
    max_ece_worsen: float = 0.02,
    max_time_increase: float = 0.15,
    *,
    use_adaptive: bool = True,
    model: AdaptiveDominanceModel | None = None,
) -> bool:
    """Return True if the challenger dominates the champion.

    When ``use_adaptive`` is True (default), a contextual multi-objective model with
    online-learned weights decides on dominance. The historical threshold logic is
    retained as a safety net and is still used whenever the adaptive model rejects a
    challenger, ensuring backwards-compatible behaviour when little data is
    available.
    """

    if not use_adaptive:
        return _dominates_static(
            champion,
            challenger,
            eps_acc,
            max_ece_worsen,
            max_time_increase,
        )

    comparator = model or _ADAPTIVE_MODEL

    llm_decision = _llm_dominance_decision(champion, challenger)
    if llm_decision is True:
        return True

    adaptive_accept = comparator.should_accept(champion, challenger)
    if adaptive_accept:
        warmup = getattr(comparator, "warmup_samples", 0)
        seen = getattr(comparator, "observations", 0)
        if seen >= max(0, warmup):
            return True
    return _dominates_static(
        champion,
        challenger,
        eps_acc,
        max_ece_worsen,
        max_time_increase,
    )


def _pairwise_win_counts(
    champion_scores: Iterable[float], challenger_scores: Iterable[float]
) -> Tuple[int, int, int]:
    sorted_champion = sorted(float(score) for score in champion_scores)
    wins = losses = draws = 0
    n = len(sorted_champion)
    for challenger_score in challenger_scores:
        challenger_score = float(challenger_score)
        left = bisect_left(sorted_champion, challenger_score)
        right = bisect_right(sorted_champion, challenger_score)
        wins += left
        draws += right - left
        losses += n - right
    return wins, losses, draws


def bootstrap_superiority(
    champion_scores: List[float],
    challenger_scores: List[float],
    trials: int = 1000,
    *,
    exploration: float = 0.05,
) -> float:
    """Approximate one-sided p-value via Thompson Sampling over superiority.

    The posterior distribution of ``P(challenger > champion)`` is represented with a
    Beta distribution whose parameters are estimated from pairwise comparisons. Each
    trial samples from this posterior, occasionally forcing random exploration to
    keep uncertainty high when data is scarce.
    """
    if not champion_scores or not challenger_scores:
        return 1.0

    wins, losses, draws = _pairwise_win_counts(champion_scores, challenger_scores)
    alpha = 1.0 + wins + 0.5 * draws
    beta_param = 1.0 + losses + 0.5 * draws

    successes = 0
    total_trials = max(1, trials)
    for _ in range(total_trials):
        if random.random() < exploration:
            sample = random.random()
        else:
            sample = random.betavariate(alpha, beta_param)
        if sample > 0.5:
            successes += 1
    return 1.0 - (successes / total_trials)
