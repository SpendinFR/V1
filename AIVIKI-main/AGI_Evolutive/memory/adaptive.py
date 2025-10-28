"""Adaptive controllers for the memory subsystem.

These helpers keep the legacy dictionary-based API intact while introducing
online learning and Thompson Sampling controllers for critical parameters.
"""
from __future__ import annotations

import logging
import math
import random
from collections.abc import MutableMapping
from typing import Dict, Iterable, Mapping, Optional

try:  # pragma: no cover - optional numpy dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - lightweight fallback for tests
    class _FallbackArray(list):
        """Minimal list-backed array supporting scalar arithmetic."""

        def __mul__(self, other):  # type: ignore[override]
            if isinstance(other, (int, float)):
                return _FallbackArray(float(v) * float(other) for v in self)
            return NotImplemented

        def __rmul__(self, other):
            return self.__mul__(other)

        def __imul__(self, other):  # type: ignore[override]
            if not isinstance(other, (int, float)):
                return NotImplemented
            scale = float(other)
            for idx, value in enumerate(self):
                self[idx] = float(value) * scale
            return self

        def __add__(self, other):  # type: ignore[override]
            if isinstance(other, (list, tuple, _FallbackArray)):
                return _FallbackArray(float(a) + float(b) for a, b in zip(self, other))
            return NotImplemented

        def __iadd__(self, other):  # type: ignore[override]
            if not isinstance(other, (list, tuple, _FallbackArray)):
                return NotImplemented
            for idx, value in enumerate(other):
                if idx < len(self):
                    self[idx] = float(self[idx]) + float(value)
                else:
                    self.append(float(value))
            return self

    class _FallbackNumpy:
        inf = float("inf")

        @staticmethod
        def zeros(length: int, dtype=float):
            return _FallbackArray(dtype() for _ in range(length))

        @staticmethod
        def append(array, value):
            return _FallbackArray(list(array) + [value])

        @staticmethod
        def dot(a, b):
            return sum(float(x) * float(y) for x, y in zip(a, b))

        @staticmethod
        def clip(value, low, high):
            try:
                numeric = float(value)
            except Exception:
                numeric = 0.0
            return max(low, min(high, numeric))

        @staticmethod
        def copy(array):
            return _FallbackArray(array)

    np = _FallbackNumpy()  # type: ignore

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class OnlineLinearParameter:
    """Simple online logistic regressor with bounded output."""

    def __init__(
        self,
        baseline: float,
        *,
        bounds: tuple[float, float] = (0.0, 1.0),
        learning_rate: float = 0.05,
        l2: float = 0.0,
    ) -> None:
        self.bounds = (min(bounds[0], bounds[1]), max(bounds[0], bounds[1]))
        self.learning_rate = learning_rate
        self.l2 = l2
        self._feature_index: Dict[str, int] = {}
        self._weights = np.zeros(0, dtype=float)
        baseline_unit = self._scale_to_unit(baseline)
        self._bias = _logit(baseline_unit)
        self._last_value = float(np.clip(baseline, *self.bounds))

    def _scale_to_unit(self, value: float) -> float:
        low, high = self.bounds
        if math.isclose(high, low):
            return 0.5
        clipped = min(max(value, low), high)
        return (clipped - low) / (high - low)

    def _unit_to_value(self, unit: float) -> float:
        low, high = self.bounds
        unit = min(max(unit, 0.0), 1.0)
        return low + unit * (high - low)

    def _ensure_features(self, features: Mapping[str, float]) -> np.ndarray:
        for name in features:
            if name not in self._feature_index:
                self._feature_index[name] = len(self._feature_index)
                self._weights = np.append(self._weights, 0.0)
        x = np.zeros(len(self._feature_index), dtype=float)
        for name, value in features.items():
            idx = self._feature_index[name]
            x[idx] = float(value)
        return x

    def predict(self, features: Mapping[str, float]) -> float:
        if not features:
            return self._last_value
        x = self._ensure_features(features)
        raw = float(np.dot(self._weights, x) + self._bias)
        unit = _sigmoid(raw)
        value = self._unit_to_value(unit)
        self._last_value = value
        return value

    def update(self, features: Mapping[str, float], target: float) -> float:
        if not features:
            return self.current_value
        x = self._ensure_features(features)
        target_unit = self._scale_to_unit(target)
        raw = float(np.dot(self._weights, x) + self._bias)
        pred_unit = _sigmoid(raw)
        error = target_unit - pred_unit
        grad = self.learning_rate * error
        if self.l2 > 0:
            self._weights *= (1.0 - self.learning_rate * self.l2)
        self._weights += grad * x
        self._bias += grad
        new_unit = _sigmoid(float(np.dot(self._weights, x) + self._bias))
        new_value = self._unit_to_value(new_unit)
        self._last_value = new_value
        return new_value

    def force_value(self, value: float) -> None:
        value = float(np.clip(value, *self.bounds))
        unit = self._scale_to_unit(value)
        self._bias = _logit(unit)
        self._last_value = value

    @property
    def current_value(self) -> float:
        return self._last_value


class ThompsonBetaScheduler:
    """Discrete Thompson Sampling controller for EMA decay."""

    def __init__(
        self,
        candidates: Iterable[float],
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        candidates = list(sorted(float(c) for c in candidates))
        if not candidates:
            raise ValueError("ThompsonBetaScheduler requires at least one candidate")
        self._candidates = candidates
        self._posterior: Dict[float, list[float]] = {
            c: [float(prior_alpha), float(prior_beta)] for c in candidates
        }
        self._current: Optional[float] = None

    def sample(self) -> float:
        draw = {
            candidate: random.betavariate(params[0], params[1])
            for candidate, params in self._posterior.items()
        }
        self._current = max(draw, key=draw.get)
        return self._current

    def update(self, reward: float) -> float:
        reward = float(np.clip(reward, 0.0, 1.0))
        if self._current is None:
            self.sample()
        assert self._current is not None
        alpha, beta_val = self._posterior[self._current]
        self._posterior[self._current] = [alpha + reward, beta_val + (1.0 - reward)]
        # Resample to allow quick adaptation
        self._current = None
        return self.current_value

    @property
    def current_value(self) -> float:
        if self._current is None:
            self.sample()
        assert self._current is not None
        return self._current


class AdaptiveMemoryParameters(MutableMapping):
    """Dictionary-like container with adaptive controllers."""

    def __init__(
        self,
        base: Mapping[str, float],
        *,
        adaptive_config: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> None:
        self._values: Dict[str, float] = {k: float(v) for k, v in base.items()}
        self._adaptive: Dict[str, OnlineLinearParameter] = {}
        self._config: Dict[str, Mapping[str, object]] = {}
        adaptive_config = adaptive_config or {}
        for name, cfg in adaptive_config.items():
            if name not in self._values:
                continue
            bounds = cfg.get("bounds", (0.0, 1.0))  # type: ignore[assignment]
            lr = float(cfg.get("lr", 0.05))  # type: ignore[arg-type]
            l2 = float(cfg.get("l2", 0.0))  # type: ignore[arg-type]
            controller = OnlineLinearParameter(
                self._values[name], bounds=bounds, learning_rate=lr, l2=l2
            )
            self._adaptive[name] = controller
            self._values[name] = controller.current_value
            self._config[name] = dict(cfg)

    def __getitem__(self, key: str) -> float:
        return self._values[key]

    def __setitem__(self, key: str, value: float) -> None:
        self._values[key] = float(value)
        if key in self._adaptive:
            self._adaptive[key].force_value(float(value))

    def __delitem__(self, key: str) -> None:  # pragma: no cover - not used today
        del self._values[key]
        self._adaptive.pop(key, None)
        self._config.pop(key, None)

    def __iter__(self):  # type: ignore[override]
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self._values.get(key, default)

    def update_from_feedback(self, context: Mapping[str, float], reward: float) -> None:
        reward = float(np.clip(reward, 0.0, 1.0))
        for name, controller in self._adaptive.items():
            cfg = self._config.get(name, {})
            feature_keys = cfg.get("feature_keys")
            if feature_keys:
                features = {key: float(context.get(key, 0.0)) for key in feature_keys}  # type: ignore[arg-type]
            else:
                features = {k: float(v) for k, v in context.items()}
            if not any(features.values()):
                continue
            target_key = cfg.get("target_key")
            target = float(context.get(target_key, reward) if target_key else reward)
            previous = controller.current_value
            new_value = controller.update(features, target)
            max_step = float(cfg.get("max_step", np.inf))
            delta = new_value - previous
            if abs(delta) > max_step:
                capped = previous + float(np.clip(delta, -max_step, max_step))
                controller.force_value(capped)
            self._values[name] = controller.current_value

        payload_context = {key: float(value) for key, value in context.items()}
        payload_parameters = {
            name: float(value) for name, value in self._values.items()
        }
        llm_payload = {
            "reward": reward,
            "context": payload_context,
            "parameters": payload_parameters,
        }
        response = try_call_llm_dict(
            "memory_adaptive_guidance",
            input_payload=llm_payload,
            logger=LOGGER,
        )
        if not response:
            return
        updates = response.get("parameter_updates", [])
        if not isinstance(updates, Iterable):
            return
        for update in updates:
            if not isinstance(update, Mapping):
                continue
            name = update.get("name")
            if not isinstance(name, str) or name not in self._values:
                continue
            suggested = update.get("suggested_value")
            if not isinstance(suggested, (int, float)):
                continue
            controller = self._adaptive.get(name)
            if controller is None:
                self._values[name] = float(suggested)
                continue
            blend = update.get("confidence")
            try:
                blend_f = max(0.0, min(1.0, float(blend)))
            except Exception:
                blend_f = 0.5
            current = controller.current_value
            target = float(suggested)
            new_value = current + (target - current) * blend_f
            controller.force_value(new_value)
            self._values[name] = controller.current_value

    def snapshot(self) -> Dict[str, float]:
        return dict(self._values)

    def copy(self) -> "AdaptiveMemoryParameters":
        clone = AdaptiveMemoryParameters(self._values, adaptive_config=self._config)
        for name, controller in self._adaptive.items():
            clone._adaptive[name]._weights = np.copy(controller._weights)
            clone._adaptive[name]._bias = controller._bias
            clone._adaptive[name]._feature_index = dict(controller._feature_index)
            clone._adaptive[name]._last_value = controller.current_value
        return clone
