from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


DEFAULTS: Dict[str, float] = {
    "style.hedging": 0.3,
    "learning.self_assess.threshold": 0.90,
    "abduction.tie_gap": 0.12,
    "abduction.weights.prior": 0.5,
    "abduction.weights.boost": 1.0,
    "abduction.weights.match": 1.0,
}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _base_amplitude(key: str) -> float:
    return 0.05 if "threshold" in key else 0.1


@dataclass
class _KeyStats:
    """Light-weight tracker for mutation behaviour on a single key."""

    attempts: int = 0
    ewma_score: float = 0.0
    volatility: float = 1.0
    history: deque[float] = field(default_factory=lambda: deque(maxlen=32))

    def register_mutation(self, value: float) -> None:
        self.attempts += 1
        self.history.append(value)
        if len(self.history) > 1:
            # Update a crude volatility proxy based on recent spread.
            span = max(self.history) - min(self.history)
            self.volatility = 0.5 * self.volatility + 0.5 * (span + 1e-6)

    def update_score(self, score: float, alpha: float = 0.3) -> None:
        if self.attempts == 0:
            return
        if self.ewma_score == 0.0:
            self.ewma_score = score
        else:
            self.ewma_score = (1 - alpha) * self.ewma_score + alpha * score
        target = max(score - 0.5, -0.5)
        adjust = 0.85 if target > 0 else 1.15
        self.volatility = _clip(self.volatility * adjust, 0.05, 2.5)


class _MutationMemory:
    """Maintain mutation statistics to drive adaptive exploration."""

    def __init__(self) -> None:
        self._keys: Dict[str, _KeyStats] = {}
        self._temperature: float = 1.0
        self._leaderboard: deque[tuple[float, Dict[str, Any]]] = deque(maxlen=10)
        self._last_base: Dict[str, Any] = dict(DEFAULTS)

    def update_keys(self, keys: Iterable[str]) -> None:
        for key in keys:
            self._keys.setdefault(key, _KeyStats())

    def _key_weight(self, key: str) -> float:
        stats = self._keys.get(key)
        if not stats:
            return 1.0
        novelty = 1.0 / (1.0 + stats.attempts)
        reward = max(stats.ewma_score - 0.5, 0.0)
        return 0.2 + novelty + reward + 0.1 * self._temperature

    def select_keys(self, keys: Sequence[str]) -> List[str]:
        if not keys:
            return []
        max_keys = min(len(keys), 3)
        cold_keys = [k for k in keys if self._keys.get(k, _KeyStats()).attempts == 0]
        if cold_keys:
            target = min(len(cold_keys), max_keys)
        else:
            explore_bias = _clip(self._temperature, 0.5, 2.0)
            target = 1 + int(random.random() < 0.35 * explore_bias)
            target = min(target, max_keys)
        selected: List[str] = []
        pool = list(keys)
        while pool and len(selected) < target:
            weights = [self._key_weight(k) for k in pool]
            total = sum(weights)
            pivot = random.random() * total if total else 0.0
            acc = 0.0
            chosen = pool[0]
            for idx, weight in enumerate(weights):
                acc += weight
                if acc >= pivot:
                    chosen = pool.pop(idx)
                    break
            selected.append(chosen)
        return selected or [random.choice(keys)]

    def scale_amplitude(self, key: str, base_amp: float) -> float:
        stats = self._keys.get(key)
        if not stats or stats.attempts == 0:
            return base_amp * 1.25
        novelty = 1.0 + 0.5 / math.sqrt(stats.attempts)
        volatility = _clip(stats.volatility, 0.05, 2.5)
        return base_amp * _clip(novelty * volatility, 0.5, 3.0)

    def register_candidate(self, candidate: MutableMapping[str, Any], mutated_keys: Sequence[str]) -> None:
        for key in mutated_keys:
            stats = self._keys.setdefault(key, _KeyStats())
            value = candidate.get(key)
            if isinstance(value, (int, float)):
                stats.register_mutation(float(value))

    def record_feedback(
        self,
        candidate: MutableMapping[str, Any],
        score: float,
        reference: MutableMapping[str, Any] | None = None,
    ) -> None:
        base = dict(self._last_base)
        if reference:
            base.update(reference)
        deltas = {
            key: candidate[key]
            for key in candidate
            if key in self._keys
            and isinstance(candidate[key], (int, float))
            and not math.isclose(
                candidate[key],
                base.get(key, candidate[key]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        }
        if not deltas:
            return
        leaderboard_entry = (score, dict(deltas))
        self._leaderboard.append(leaderboard_entry)
        for key in deltas:
            self._keys[key].update_score(score)
        self._temperature = _clip(self._temperature * (0.9 if score > 0.5 else 1.1), 0.2, 3.0)

    @property
    def leaderboard(self) -> Sequence[tuple[float, Dict[str, Any]]]:
        return tuple(self._leaderboard)


_MEMORY = _MutationMemory()


_LOGGER = logging.getLogger(__name__)


def _apply_llm_guidance(
    base: Mapping[str, Any],
    candidate: MutableMapping[str, Any],
    mutated_keys: Sequence[str],
) -> List[str]:
    payload = {
        "base": dict(base),
        "candidate": dict(candidate),
        "mutated_keys": list(mutated_keys),
        "leaderboard": [
            {"score": score, "deltas": deltas}
            for score, deltas in _MEMORY.leaderboard
        ],
    }
    response = try_call_llm_dict(
        "self_improver_mutation_plan",
        input_payload=payload,
        logger=_LOGGER,
        max_retries=2,
    )
    if not response:
        return list(mutated_keys)

    updated_keys = set(mutated_keys)
    suggested = response.get("suggested_updates", {})
    if isinstance(suggested, Mapping):
        for key, value in suggested.items():
            try:
                candidate[key] = float(value)
            except (TypeError, ValueError):
                continue
            updated_keys.add(str(key))

    llm_keys = response.get("mutated_keys")
    if isinstance(llm_keys, Sequence) and not isinstance(llm_keys, (str, bytes)):
        updated_keys.update(str(key) for key in llm_keys)

    return list(updated_keys)


def _probability_like(key: str, value: float) -> bool:
    if "threshold" in key:
        return True
    default = DEFAULTS.get(key)
    if default is not None and 0.0 <= default <= 1.0:
        return True
    return 0.0 <= value <= 1.0


def _mutate_value(key: str, value: float) -> float:
    base_amp = _base_amplitude(key)
    sigma = _MEMORY.scale_amplitude(key, base_amp)
    mutated = random.gauss(value, sigma)
    if _probability_like(key, value):
        return _clip(mutated)
    return mutated


def generate_overrides(base: Dict[str, Any], n: int = 4) -> List[Dict[str, Any]]:
    """Generate adaptive override candidates via evolutionary-style mutations."""

    merged: Dict[str, Any] = dict(DEFAULTS, **(base or {}))
    keys = [key for key, value in merged.items() if isinstance(value, (int, float))]
    candidates: List[Dict[str, Any]] = []

    if not keys:
        return [dict(DEFAULTS) for _ in range(max(1, n))]

    _MEMORY.update_keys(keys)
    _MEMORY._last_base = dict(merged)

    for _ in range(max(1, n)):
        candidate = dict(merged)
        mutated_keys = _MEMORY.select_keys(keys)
        for key in mutated_keys:
            original = candidate[key]
            candidate[key] = _mutate_value(key, float(original))
        mutated_keys = _apply_llm_guidance(merged, candidate, mutated_keys)
        _MEMORY.register_candidate(candidate, mutated_keys)
        candidates.append(candidate)

    return candidates


def record_feedback(candidate: Dict[str, Any], score: float, reference: Dict[str, Any] | None = None) -> None:
    """Feed evaluation results back into the adaptive memory.

    Parameters
    ----------
    candidate:
        The mutated configuration that has been evaluated.
    score:
        Performance score normalised in ``[0, 1]`` where ``0.5`` represents a
        neutral baseline. Higher scores increase exploitation, lower scores
        encourage exploration.
    reference:
        Optional configuration to compare against when inferring which keys were
        effectively mutated. When omitted, ``DEFAULTS`` combined with the last
        ``base`` passed to :func:`generate_overrides` is used as the reference.
    """

    if not isinstance(candidate, dict):
        raise TypeError("candidate must be a mapping produced by generate_overrides")
    _MEMORY.record_feedback(candidate, score, reference)


def current_leaderboard() -> Sequence[tuple[float, Dict[str, Any]]]:
    """Expose the best performing mutations recorded so far."""

    return _MEMORY.leaderboard
