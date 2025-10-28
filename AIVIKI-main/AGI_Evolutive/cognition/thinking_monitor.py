from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Mapping, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


def _safe_clip(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """Small helper keeping scores in-range while staying tolerant to NaNs."""

    if value != value:  # NaN guard
        return min_value
    return max(min_value, min(max_value, value))


@dataclass
class ThinkingSnapshot:
    thinking_time: float
    hypotheses: int
    depth: int
    thinking_score: float
    thinking_flag: bool
    normalized_time: float = 0.0
    normalized_hypotheses: float = 0.0
    normalized_depth: float = 0.0
    surprise: float = 0.0


class ThinkingMonitor:
    """Lightweight metacognitive tracker for (Reflect/Reason) activity."""

    def __init__(
        self,
        t_reflect_weight: float = 1.0,
        t_reason_weight: float = 1.2,
        history_window: int = 50,
        learning_rate: float = 0.15,
    ) -> None:
        self.t_reflect_weight = t_reflect_weight
        self.t_reason_weight = t_reason_weight
        self._history_window = max(5, history_window)
        self._learning_rate = max(0.0, learning_rate)
        self._weights: Dict[str, float] = {
            "bias": 0.0,
            "time": 0.6,
            "hypotheses": 0.25,
            "depth": 0.15,
            "time_hyp": 0.05,
            "time_depth": 0.05,
            "hyp_depth": 0.05,
        }
        self._history: Deque[Mapping[str, float]] = deque(maxlen=self._history_window)
        self._last_features: Mapping[str, float] | None = None
        self.begin_cycle()

    def begin_cycle(self) -> None:
        self._t_reflect = 0.0
        self._t_reason = 0.0
        self._t_reflect_start: float | None = None
        self._t_reason_start: float | None = None
        self._hypotheses = 0
        self._depth = 0

    # ---- hooks ----
    def on_reflect_start(self) -> None:
        if self._t_reflect_start is None:
            self._t_reflect_start = time.time()

    def on_reflect_end(self) -> None:
        if self._t_reflect_start is not None:
            self._t_reflect += time.time() - self._t_reflect_start
            self._t_reflect_start = None

    def on_reason_start(self) -> None:
        if self._t_reason_start is None:
            self._t_reason_start = time.time()

    def on_reason_end(self) -> None:
        if self._t_reason_start is not None:
            self._t_reason += time.time() - self._t_reason_start
            self._t_reason_start = None

    def on_hypothesis_tested(self, n: int = 1) -> None:
        self._hypotheses += max(0, int(n))

    def set_depth(self, n_steps_before_act: int) -> None:
        self._depth = max(0, int(n_steps_before_act))

    # ---- compute ----
    def snapshot(self) -> ThinkingSnapshot:
        # ensure to close any open spans (best-effort)
        self.on_reflect_end()
        self.on_reason_end()
        weighted_time = (
            self.t_reflect_weight * self._t_reflect
            + self.t_reason_weight * self._t_reason
        )

        normalized = self._normalize_metrics(
            weighted_time, self._hypotheses, self._depth
        )
        features = self._build_features(normalized)
        score = self._compute_score(features)
        surprise = self._compute_surprise(score)
        flag = self._compute_flag(weighted_time, normalized, score, surprise)

        llm_feedback = self._llm_review(weighted_time, normalized, score)
        if llm_feedback:
            try:
                proposed = llm_feedback.get("score")
                if proposed is not None:
                    score = _safe_clip(float(proposed))
            except (TypeError, ValueError):
                logger.debug("Invalid LLM score override: %r", llm_feedback.get("score"))
            flag = flag or score < 0.35

        snapshot_features = {
            "weighted_time": weighted_time,
            "hypotheses": self._hypotheses,
            "depth": self._depth,
            "score": score,
            "surprise": surprise,
            **{f"feature_{k}": v for k, v in features.items()},
        }
        if llm_feedback:
            snapshot_features["llm"] = llm_feedback
        self._history.append(snapshot_features)
        self._last_features = {"score": score, **features}
        return ThinkingSnapshot(
            thinking_time=weighted_time,
            hypotheses=self._hypotheses,
            depth=self._depth,
            thinking_score=score,
            thinking_flag=flag,
            normalized_time=normalized["time"],
            normalized_hypotheses=normalized["hypotheses"],
            normalized_depth=normalized["depth"],
            surprise=surprise,
        )

    # ---- adaptive learning ----
    def update_feedback(self, reward: float) -> None:
        """Update the weight vector from an external feedback signal.

        Args:
            reward: Target score in [0, 1] describing how satisfactory
                the latest thinking cycle outcome was.
        """

        if self._last_features is None or self._learning_rate <= 0.0:
            return

        reward = _safe_clip(reward)
        prediction = _safe_clip(self._last_features.get("score", 0.0))
        error = reward - prediction
        if abs(error) < 1e-6:
            return

        for key, value in self._last_features.items():
            if key == "score":
                continue
            self._weights[key] = self._weights.get(key, 0.0) + self._learning_rate * error * value

        # keep weights within a moderate range
        for key in tuple(self._weights.keys()):
            if key == "bias":
                continue
            self._weights[key] = _safe_clip(self._weights[key], -1.0, 1.5)

    def reset_learning(self) -> None:
        """Reset adaptive components while keeping the hook interface intact."""

        self._weights.update({
            "bias": 0.0,
            "time": 0.6,
            "hypotheses": 0.25,
            "depth": 0.15,
            "time_hyp": 0.05,
            "time_depth": 0.05,
            "hyp_depth": 0.05,
        })
        self._history.clear()
        self._last_features = None

    def _llm_review(
        self,
        weighted_time: float,
        normalized: Mapping[str, float],
        score: float,
    ) -> Optional[Dict[str, Any]]:
        payload = {
            "thinking_time": weighted_time,
            "normalized": dict(normalized),
            "score": score,
            "history": list(self._history),
        }
        response = try_call_llm_dict(
            "thinking_monitor",
            input_payload=payload,
            logger=logger,
        )
        return dict(response) if response else None

    # ---- internals ----
    def _normalize_metrics(
        self, weighted_time: float, hypotheses: int, depth: int
    ) -> Mapping[str, float]:
        window = list(self._history)
        if not window:
            return {
                "time": _safe_clip(weighted_time / 2.0),
                "hypotheses": _safe_clip(hypotheses / 5.0),
                "depth": _safe_clip(depth / 4.0),
            }

        def _collect(key: str) -> Iterable[float]:
            return (float(item.get(key, 0.0)) for item in window)

        times = list(_collect("weighted_time"))
        hyps = list(_collect("hypotheses"))
        depths = list(_collect("depth"))

        def _adaptive_norm(value: float, series: Iterable[float]) -> float:
            series_list = list(series)
            if not series_list:
                return _safe_clip(value)
            min_v = min(series_list)
            max_v = max(series_list)
            if max_v - min_v < 1e-6:
                return 0.5
            return _safe_clip((value - min_v) / (max_v - min_v))

        return {
            "time": _adaptive_norm(weighted_time, times),
            "hypotheses": _adaptive_norm(float(hypotheses), hyps),
            "depth": _adaptive_norm(float(depth), depths),
        }

    def _build_features(self, normalized: Mapping[str, float]) -> Dict[str, float]:
        time_norm = normalized["time"]
        hyp_norm = normalized["hypotheses"]
        depth_norm = normalized["depth"]
        features = {
            "bias": 1.0,
            "time": time_norm,
            "hypotheses": hyp_norm,
            "depth": depth_norm,
            "time_hyp": time_norm * hyp_norm,
            "time_depth": time_norm * depth_norm,
            "hyp_depth": hyp_norm * depth_norm,
        }
        return features

    def _compute_score(self, features: Mapping[str, float]) -> float:
        raw = 0.0
        for key, value in features.items():
            weight = self._weights.get(key, 0.0)
            raw += weight * value
        return _safe_clip(raw)

    def _compute_surprise(self, score: float) -> float:
        if not self._history:
            return 0.0
        mean_score = sum(item.get("score", 0.0) for item in self._history) / len(
            self._history
        )
        surprise = abs(score - mean_score)
        return _safe_clip(surprise, 0.0, 1.0)

    def _compute_flag(
        self,
        weighted_time: float,
        normalized: Mapping[str, float],
        score: float,
        surprise: float,
    ) -> bool:
        base_flag = (weighted_time > 0.25) or (self._hypotheses >= 1)
        if (
            normalized.get("time", 0.0) > 0.7
            or normalized.get("hypotheses", 0.0) > 0.6
            or normalized.get("depth", 0.0) > 0.6
        ):
            base_flag = True
        dynamic_threshold = 0.4
        if self._history:
            avg_score = sum(item.get("score", 0.0) for item in self._history) / len(
                self._history
            )
            dynamic_threshold = max(0.35, avg_score + 0.1)
        return base_flag or score > dynamic_threshold or surprise > 0.2
