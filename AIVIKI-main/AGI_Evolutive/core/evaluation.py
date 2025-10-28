"""Adaptive prioritisation utilities.

This module keeps the legacy ``unified_priority`` interface while extending it
with instrumentation, online learning and light-weight exploration between the
hand-crafted heuristic and a learnt model.  Decisions are logged to a JSONL file
(`data/priority_log.jsonl` by default) so that downstream analytics can inspect
how priorities were produced and how feedback updates the model.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import logging

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)

__all__ = [
    "unified_priority",
    "get_last_priority_token",
    "record_priority_feedback",
    "PriorityContext",
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class PriorityContext:
    """Container for priority inputs with convenience sanitisation helpers."""

    impact: float
    probability: float
    reversibility: float
    effort: float
    uncertainty: float = 0.0
    valence: float = 0.0

    def sanitised(self) -> "PriorityContext":
        return PriorityContext(
            impact=_clamp(self.impact),
            probability=_clamp(self.probability),
            reversibility=_clamp(self.reversibility),
            effort=max(0.2, float(self.effort)),
            uncertainty=_clamp(self.uncertainty),
            valence=max(-1.0, min(1.0, float(self.valence))),
        )

    def to_features(self) -> Dict[str, float]:
        ctx = self.sanitised()
        features = {
            "impact": ctx.impact,
            "probability": ctx.probability,
            "reversibility": ctx.reversibility,
            "effort": ctx.effort,
            "uncertainty": ctx.uncertainty,
            "valence": ctx.valence,
        }
        # Quadratic terms for slight non-linearity
        features["impact_sq"] = ctx.impact ** 2
        features["probability_sq"] = ctx.probability ** 2
        features["reversibility_sq"] = ctx.reversibility ** 2
        features["effort_sq"] = ctx.effort ** 2
        features["uncertainty_sq"] = ctx.uncertainty ** 2
        features["valence_sq"] = ctx.valence ** 2
        # Interaction terms capture simple dependencies between dimensions.
        features["impact_probability"] = ctx.impact * ctx.probability
        features["impact_reversibility"] = ctx.impact * ctx.reversibility
        features["probability_reversibility"] = ctx.probability * ctx.reversibility
        features["impact_effort"] = ctx.impact * ctx.effort
        features["probability_effort"] = ctx.probability * ctx.effort
        features["uncertainty_effort"] = ctx.uncertainty * ctx.effort
        features["valence_impact"] = ctx.valence * ctx.impact
        features["valence_probability"] = ctx.valence * ctx.probability
        # Helpful derived features.
        features["effort_inv"] = 1.0 / ctx.effort
        features["effort_log"] = math.log(ctx.effort)
        return features

    def to_dict(self) -> Dict[str, float]:
        return asdict(self.sanitised())


def _heuristic_priority(ctx: PriorityContext) -> float:
    ctx = ctx.sanitised()
    base = (ctx.impact * ctx.probability * ctx.reversibility) / ctx.effort
    mod = (1.0 - 0.5 * ctx.uncertainty) * (1.0 + 0.3 * ctx.valence)
    return _clamp(base * mod)


class OnlinePriorityModel:
    """Simple online GLM (logistic regression) with optional regularisation."""

    def __init__(
        self,
        feature_names: Iterable[str],
        lr: float = 0.05,
        l2: float = 1e-3,
        bias: float = 0.0,
        init_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.lr = lr
        self.l2 = l2
        self.bias = bias
        self.weights: Dict[str, float] = {name: 0.0 for name in self.feature_names}
        if init_weights:
            for name, value in init_weights.items():
                if name in self.weights:
                    self.weights[name] = float(value)

    def _linear(self, features: Mapping[str, float], weights: Mapping[str, float]) -> float:
        z = self.bias
        for name in self.feature_names:
            z += weights.get(name, 0.0) * float(features.get(name, 0.0))
        return z

    @staticmethod
    def _sigmoid(z: float) -> float:
        # Guard against overflow while keeping differentiability.
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict(self, features: Mapping[str, float]) -> float:
        return self._sigmoid(self._linear(features, self.weights))

    def predict_with_weights(
        self, features: Mapping[str, float], weights: Mapping[str, float]
    ) -> float:
        return self._sigmoid(self._linear(features, weights))

    def update(self, features: Mapping[str, float], target: float) -> float:
        target = _clamp(target)
        pred = self.predict(features)
        error = pred - target
        for name in self.feature_names:
            value = float(features.get(name, 0.0))
            self.weights[name] -= self.lr * ((error * value) + self.l2 * self.weights[name])
        self.bias -= self.lr * (error + self.l2 * self.bias)
        return pred


class PriorityBandit:
    """Epsilon-greedy bandit over scoring strategies."""

    def __init__(self, arms: Iterable[str], exploration: float = 0.05) -> None:
        self.arms = list(arms)
        self.exploration = max(0.0, min(1.0, float(exploration)))
        self._values: Dict[str, float] = {name: 0.0 for name in self.arms}
        self._counts: Dict[str, int] = {name: 0 for name in self.arms}
        self._rng = random.Random()

    def choose(self, estimates: Mapping[str, float]) -> str:
        estimates = dict(estimates)
        for arm in self.arms:
            estimates.setdefault(arm, 0.0)  # ensure keys exist
        if self._rng.random() < self.exploration:
            return self._rng.choice(self.arms)
        return max(self.arms, key=lambda arm: self._values.get(arm, 0.0))

    def update(self, arm: str, reward: float) -> None:
        if arm not in self._values:
            return
        reward = _clamp(reward)
        self._counts[arm] += 1
        count = self._counts[arm]
        value = self._values[arm]
        self._values[arm] = value + (reward - value) / float(count)


class PriorityLogger:
    """JSONL logger to keep track of decisions and feedback."""

    def __init__(self, path: Optional[Path] = None, buffer_size: int = 16) -> None:
        default_path = Path(__file__).resolve().parents[2] / "data" / "priority_log.jsonl"
        env_path = os.environ.get("AGI_PRIORITY_LOG")
        self.path = Path(env_path) if env_path else default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = max(1, int(buffer_size))
        self._buffer = []
        self._lock = threading.Lock()
        atexit.register(self.flush)

    def log(self, record: Mapping[str, object]) -> None:
        with self._lock:
            self._buffer.append(dict(record))
            if len(self._buffer) >= self.buffer_size:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            for entry in self._buffer:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._buffer.clear()


class PriorityManager:
    def __init__(self) -> None:
        feature_names = [
            "impact",
            "probability",
            "reversibility",
            "effort",
            "uncertainty",
            "valence",
            "impact_sq",
            "probability_sq",
            "reversibility_sq",
            "effort_sq",
            "uncertainty_sq",
            "valence_sq",
            "impact_probability",
            "impact_reversibility",
            "probability_reversibility",
            "impact_effort",
            "probability_effort",
            "uncertainty_effort",
            "valence_impact",
            "valence_probability",
            "effort_inv",
            "effort_log",
        ]
        init_weights = {
            "impact": 1.1,
            "probability": 0.9,
            "reversibility": 0.6,
            "effort": -0.7,
            "uncertainty": -0.4,
            "valence": 0.3,
            "impact_probability": 0.4,
            "impact_reversibility": 0.3,
            "probability_reversibility": 0.2,
            "impact_effort": -0.2,
            "probability_effort": -0.2,
            "uncertainty_effort": -0.3,
            "valence_impact": 0.1,
            "valence_probability": 0.1,
            "effort_inv": 0.5,
            "effort_log": -0.1,
        }
        self.model = OnlinePriorityModel(feature_names, init_weights=init_weights)
        self.bandit = PriorityBandit(["heuristic", "glm"])
        self.logger = PriorityLogger()
        self._pending: Dict[str, Tuple[PriorityContext, Dict[str, float], str]] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._local = threading.local()

    def evaluate(self, ctx: PriorityContext) -> float:
        features = ctx.to_features()
        baseline = _heuristic_priority(ctx)
        model_score = self.model.predict(features)
        choice = self.bandit.choose({"heuristic": baseline, "glm": model_score})
        score = baseline if choice == "heuristic" else model_score
        llm_payload = self._call_llm_priority(ctx, baseline, model_score, strategy=choice)
        if llm_payload is not None:
            priority = llm_payload.get("priority")
            try:
                score = _clamp(float(priority))
            except (TypeError, ValueError):
                pass
        token = self._register(
            ctx,
            features,
            choice,
            baseline,
            model_score,
            score,
            llm_payload=llm_payload,
        )
        self._local.last_token = token
        return score

    def last_token(self) -> Optional[str]:
        return getattr(self._local, "last_token", None)

    def record_feedback(self, token: Optional[str], reward: float) -> None:
        if not token:
            return
        with self._lock:
            payload = self._pending.pop(token, None)
        if not payload:
            return
        ctx, features, strategy = payload
        reward = _clamp(reward)
        self.bandit.update(strategy, reward)
        self.model.update(features, reward)
        self.logger.log(
            {
                "kind": "feedback",
                "ts": time.time(),
                "token": token,
                "reward": reward,
                "strategy": strategy,
                "inputs": ctx.to_dict(),
            }
        )

    def _register(
        self,
        ctx: PriorityContext,
        features: Dict[str, float],
        strategy: str,
        baseline: float,
        model_score: float,
        score: float,
        *,
        llm_payload: Optional[Mapping[str, Any]] = None,
    ) -> str:
        with self._lock:
            self._counter += 1
            token = f"prio-{self._counter}"
            self._pending[token] = (ctx, features, strategy)
        self.logger.log(
            {
                "kind": "evaluation",
                "ts": time.time(),
                "token": token,
                "strategy": strategy,
                "baseline": baseline,
                "glm": model_score,
                "score": score,
                "inputs": ctx.to_dict(),
                "features": {k: float(features.get(k, 0.0)) for k in features},
                "llm": dict(llm_payload) if llm_payload else None,
            }
        )
        return token

    def _call_llm_priority(
        self,
        ctx: PriorityContext,
        baseline: float,
        model_score: float,
        *,
        strategy: str,
    ) -> Optional[Dict[str, Any]]:
        recent_records = []
        buffer = getattr(self.logger, "_buffer", None)
        if isinstance(buffer, list):
            recent_records = buffer[-5:]
        payload = {
            "context": ctx.to_dict(),
            "heuristics": {"baseline": baseline, "glm": model_score, "strategy": strategy},
            "history": recent_records,
        }
        response = try_call_llm_dict(
            "unified_priority",
            input_payload=payload,
            logger=LOGGER,
            max_retries=2,
        )
        if not response:
            return None
        return dict(response)


_MANAGER = PriorityManager()


def unified_priority(
    impact: float,
    probability: float,
    reversibility: float,
    effort: float,
    uncertainty: float = 0.0,
    valence: float = 0.0,
) -> float:
    ctx = PriorityContext(
        impact=impact,
        probability=probability,
        reversibility=reversibility,
        effort=effort,
        uncertainty=uncertainty,
        valence=valence,
    )
    return _MANAGER.evaluate(ctx)


def get_last_priority_token() -> Optional[str]:
    """Return the token associated with the most recent ``unified_priority`` call."""

    return _MANAGER.last_token()


def record_priority_feedback(token: Optional[str], outcome: float) -> None:
    """Record feedback for a previous priority decision.

    Parameters
    ----------
    token:
        The token returned by :func:`get_last_priority_token` (or stored elsewhere)
        identifying the decision to update.
    outcome:
        Observed reward in ``[0, 1]`` used to update both the bandit and the
        online GLM.
    """

    _MANAGER.record_feedback(token, outcome)
