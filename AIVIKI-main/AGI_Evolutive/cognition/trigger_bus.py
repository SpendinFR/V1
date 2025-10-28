from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json
import math
import random
import time

from AGI_Evolutive.core.trigger_types import Trigger, TriggerType
from AGI_Evolutive.core.evaluation import (
    get_last_priority_token,
    record_priority_feedback,
    unified_priority,
)
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


Collector = Callable[[], List[Trigger]]


@dataclass
class ScoredTrigger:
    trigger: Trigger
    priority: float
    feedback_token: Optional[str] = None
    llm_meta: Optional[Dict[str, Any]] = None


class OnlineLinear:
    """Very small online GLM with sigmoid link and bounded weights."""

    def __init__(
        self,
        feature_names: Iterable[str],
        lr: float = 0.05,
        bounds: Tuple[float, float] = (0.0, 1.0),
        l2: float = 0.0,
        init_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.lr = lr
        self.bounds = bounds
        self.l2 = l2
        self.intercept = 0.0
        self.weights: Dict[str, float] = {name: 0.0 for name in feature_names}
        if init_weights:
            for name, value in init_weights.items():
                self.weights[name] = float(value)

    def _clip(self, value: float) -> float:
        lo, hi = self.bounds
        if lo is None and hi is None:
            return value
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return value

    def predict(self, features: Dict[str, float]) -> float:
        z = self.intercept
        for name, val in features.items():
            z += self.weights.get(name, 0.0) * val
        return 1.0 / (1.0 + math.exp(-z))

    def predict_with_weights(self, features: Dict[str, float], weights: Dict[str, float]) -> float:
        z = self.intercept
        for name, val in features.items():
            z += weights.get(name, self.weights.get(name, 0.0)) * val
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, features: Dict[str, float], target: float) -> None:
        target = self._clip(target)
        prediction = self.predict(features)
        error = target - prediction
        self.intercept += self.lr * error
        for name, value in features.items():
            weight = self.weights.get(name, 0.0)
            gradient = error * value - self.l2 * weight
            weight += self.lr * gradient
            self.weights[name] = self._clip(weight)


class DiscreteThompsonSampler:
    """Simple Thompson Sampling over a discrete set of weight dictionaries."""

    def __init__(self, candidates: List[Dict[str, float]]) -> None:
        self.candidates = candidates
        self._alphas: List[float] = [1.0] * len(candidates)
        self._betas: List[float] = [1.0] * len(candidates)

    def sample(self) -> Tuple[int, Dict[str, float]]:
        scores = [random.betavariate(a, b) for a, b in zip(self._alphas, self._betas)]
        idx = max(range(len(scores)), key=lambda i: scores[i])
        return idx, self.candidates[idx]

    def update(self, idx: int, reward: float) -> None:
        reward = max(0.0, min(1.0, reward))
        self._alphas[idx] += reward
        self._betas[idx] += 1.0 - reward


class TriggerBus:
    """Collects triggers from various sources, normalizes meta, scores, arbitrates."""

    def __init__(self):
        self.collectors: List[Collector] = []
        self.cooldown_cache: Dict[str, float] = {}  # simple dedup/cooldown
        self._habit_strength_source: Optional[Any] = None
        self._adaptive_model = OnlineLinear(
            feature_names=[
                "importance",
                "probability",
                "reversibility",
                "effort",
                "uncertainty",
                "immediacy",
                "habit_strength",
                "importance_sq",
                "probability_sq",
                "immediacy_sq",
            ],
            lr=0.05,
            bounds=(0.0, 1.0),
            l2=1e-3,
            init_weights={
                "importance": 0.9,
                "probability": 0.7,
                "reversibility": 0.2,
                "effort": -0.4,
                "uncertainty": -0.3,
                "immediacy": 0.6,
                "habit_strength": 0.4,
                "importance_sq": 0.2,
                "probability_sq": 0.1,
                "immediacy_sq": 0.15,
            },
        )
        self._ts_sampler = DiscreteThompsonSampler(
            [
                {
                    "importance": 0.9,
                    "probability": 0.7,
                    "reversibility": 0.2,
                    "effort": -0.5,
                    "uncertainty": -0.4,
                    "immediacy": 0.6,
                    "habit_strength": 0.4,
                    "importance_sq": 0.1,
                    "probability_sq": 0.1,
                    "immediacy_sq": 0.05,
                },
                {
                    "importance": 1.1,
                    "probability": 0.9,
                    "reversibility": 0.1,
                    "effort": -0.2,
                    "uncertainty": -0.2,
                    "immediacy": 0.8,
                    "habit_strength": 0.2,
                    "importance_sq": 0.05,
                    "probability_sq": 0.15,
                    "immediacy_sq": 0.25,
                },
                {
                    "importance": 0.7,
                    "probability": 0.6,
                    "reversibility": 0.4,
                    "effort": -0.6,
                    "uncertainty": -0.5,
                    "immediacy": 0.5,
                    "habit_strength": 0.6,
                    "importance_sq": 0.3,
                    "probability_sq": 0.0,
                    "immediacy_sq": 0.1,
                },
            ]
        )
        self._ts_blend = 0.35
        self._adaptive_blend = 0.5
        self._pending_feedback_limit = 1024
        self._pending_feedback: Dict[
            str,
            Tuple[
                Dict[str, float],
                Optional[int],
                Optional[str],
                Optional[str],
                Optional[str],
            ],
        ] = {}
        self._logger = None

    def register(self, fn: Collector):
        self.collectors.append(fn)

    def set_habit_strength_source(self, source: Any) -> None:
        """Alimente la force d'habitude via EvolutionManager ou un callable."""

        self._habit_strength_source = source

    def _payload_fingerprint(self, payload: Any) -> str:
        try:
            serialized = json.dumps(payload, sort_keys=True, default=str)
        except (TypeError, ValueError):
            serialized = repr(payload)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _key(self, t: Trigger) -> Optional[str]:
        # build a stable key per (type, salient meta/payload)
        src = t.meta.get("source", "unknown")
        base_info: Any = t.meta.get("hash")
        if not base_info and isinstance(t.payload, dict):
            base_info = t.payload.get("id")
        if base_info:
            base = str(base_info)
        elif t.payload is not None:
            base = self._payload_fingerprint(t.payload)
        else:
            return None
        return f"{src}:{t.type.name}:{base}:{t.meta.get('immediacy','')}:{t.meta.get('importance','')}"

    def _normalize(self, t: Trigger, valence: float = 0.0) -> Trigger:
        m = t.meta
        # defaults
        m.setdefault("importance", 0.5)
        m.setdefault("probability", 0.6)
        m.setdefault("reversibility", 1.0)
        m.setdefault("effort", 0.5)
        m.setdefault("uncertainty", 0.2)
        m.setdefault("immediacy", 0.2)
        m.setdefault("habit_strength", 0.0)
        m.setdefault("source", "system")
        habit_key = m.get("habit_key") or m.get("key")
        if not habit_key and t.payload and isinstance(t.payload, dict):
            habit_key = t.payload.get("habit_key") or t.payload.get("id")
        strength = self._lookup_habit_strength(habit_key) if habit_key else 0.0
        if strength:
            m["habit_strength"] = strength
        # emotion influence is applied in scoring (not stored)
        return t

    def _build_features(self, t: Trigger) -> Dict[str, float]:
        features = {
            "importance": float(t.meta.get("importance", 0.5)),
            "probability": float(t.meta.get("probability", 0.6)),
            "reversibility": float(t.meta.get("reversibility", 1.0)),
            "effort": float(t.meta.get("effort", 0.5)),
            "uncertainty": float(t.meta.get("uncertainty", 0.2)),
            "immediacy": float(t.meta.get("immediacy", 0.2)),
            "habit_strength": float(t.meta.get("habit_strength", 0.0)),
        }
        features["importance_sq"] = features["importance"] ** 2
        features["probability_sq"] = features["probability"] ** 2
        features["immediacy_sq"] = features["immediacy"] ** 2
        return features

    def _compute_adaptive_priority(
        self,
        t: Trigger,
        base_priority: float,
        sampled_weights: Optional[Tuple[int, Dict[str, float]]],
    ) -> Tuple[float, Dict[str, float], Optional[int]]:
        features = self._build_features(t)
        adaptive_score = self._adaptive_model.predict(features)
        blended = self._adaptive_blend * adaptive_score + (1.0 - self._adaptive_blend) * base_priority
        if sampled_weights:
            idx, weights = sampled_weights
            ts_score = self._adaptive_model.predict_with_weights(features, weights)
            blended = (1.0 - self._ts_blend) * blended + self._ts_blend * ts_score
        else:
            idx = None
        return blended, features, idx

    def register_feedback(self, token_or_trigger: Any, reward: float) -> None:
        """Update adaptive scorers based on downstream feedback in [0,1]."""

        if isinstance(token_or_trigger, Trigger):
            token = token_or_trigger.meta.get("_feedback_token") or self._key(token_or_trigger)
        else:
            token = str(token_or_trigger) if token_or_trigger is not None else None
        if not token:
            return
        payload = self._pending_feedback.pop(token, None)
        if not payload:
            return
        features, ts_idx, alias, origin, priority_token = payload
        reward = max(0.0, min(1.0, float(reward)))
        self._adaptive_model.update(features, reward)
        if ts_idx is not None:
            self._ts_sampler.update(ts_idx, reward)
        if alias is not None and alias != token:
            self._pending_feedback.pop(alias, None)
        if origin and origin != token:
            self._pending_feedback.pop(origin, None)
        if priority_token:
            record_priority_feedback(priority_token, reward)

    def _lookup_habit_strength(self, key: Any) -> float:
        if key is None or self._habit_strength_source is None:
            return 0.0
        source = self._habit_strength_source
        try:
            if callable(source):
                return float(source(key))
            if isinstance(source, dict):
                return float(source.get(key, 0.0))
            getter = getattr(source, "get", None)
            if callable(getter):
                return float(getter(key, 0.0))
        except Exception:
            return 0.0
        return 0.0

    def _llm_adjust_priorities(self, scored: List[ScoredTrigger]) -> List[ScoredTrigger]:
        if not scored:
            return scored
        payload = {
            "total": len(scored),
            "top": [
                {
                    "token": item.feedback_token,
                    "type": item.trigger.type.name,
                    "source": item.trigger.meta.get("source"),
                    "priority": item.priority,
                    "meta": {
                        "importance": item.trigger.meta.get("importance"),
                        "probability": item.trigger.meta.get("probability"),
                        "immediacy": item.trigger.meta.get("immediacy"),
                        "habit_strength": item.trigger.meta.get("habit_strength"),
                    },
                }
                for item in scored[:6]
            ],
        }
        response = try_call_llm_dict(
            "cognition_trigger_bus",
            input_payload=payload,
            logger=getattr(self, "_logger", None),
        )
        if not isinstance(response, dict):
            return scored

        updates: Dict[str, Any] = {}
        if isinstance(response.get("priorities"), dict):
            updates = {str(k): v for k, v in response["priorities"].items()}
        elif isinstance(response.get("priorities"), list):
            for entry in response["priorities"]:
                if isinstance(entry, dict) and entry.get("token") is not None:
                    updates[str(entry["token"])] = entry.get("priority")

        for item in scored:
            token = item.feedback_token
            if token and token in updates:
                try:
                    item.priority = max(0.0, min(1.0, float(updates[token])))
                except (TypeError, ValueError):
                    continue

        if isinstance(response.get("order"), list):
            order_map = {str(tok): idx for idx, tok in enumerate(response["order"])}
            scored.sort(key=lambda it: (order_map.get(it.feedback_token, len(order_map)), -it.priority))

        meta: Dict[str, Any] = {"llm": response}
        if isinstance(response.get("notes"), str) and response["notes"].strip():
            meta["notes"] = response["notes"].strip()
        if "confidence" in response:
            try:
                meta["confidence"] = max(0.0, min(1.0, float(response["confidence"])))
            except (TypeError, ValueError):
                pass
        if meta:
            for item in scored[: len(scored)]:
                existing = item.llm_meta or {}
                merged = dict(existing)
                merged.update(meta)
                item.llm_meta = merged

        return scored

    def collect_and_score(self, valence: float = 0.0) -> List[ScoredTrigger]:
        now = time.time()
        scored: List[ScoredTrigger] = []
        sampled_weights = self._ts_sampler.sample() if self._ts_sampler else None
        for fn in self.collectors:
            try:
                for t in fn() or []:
                    t = self._normalize(t, valence=valence)
                    # hard overrides
                    if t.type is TriggerType.THREAT and t.meta.get("immediacy", 0.0) >= 0.8:
                        pr = 1.0
                        features: Dict[str, float] = self._build_features(t)
                        ts_idx: Optional[int] = None
                        priority_token: Optional[str] = None
                    else:
                        base_priority = unified_priority(
                            impact=t.meta["importance"],
                            probability=t.meta["probability"],
                            reversibility=t.meta["reversibility"],
                            effort=t.meta["effort"],
                            uncertainty=t.meta["uncertainty"],
                            valence=valence,
                        )
                        priority_token = get_last_priority_token()
                        adaptive_priority, features, ts_idx = self._compute_adaptive_priority(
                            t, base_priority, sampled_weights
                        )
                        pr = adaptive_priority
                        habit = float(t.meta.get("habit_strength", 0.0))
                        if habit:
                            pr = 0.85 * pr + 0.15 * habit
                    key = self._key(t)
                    # cooldown 1.5s to avoid storms
                    if key is not None:
                        if self.cooldown_cache.get(key, 0) + 1.5 > now:
                            continue
                        self.cooldown_cache[key] = now
                    token = (
                        f"{key}:{int(now * 1000)}" if key is not None else f"anon:{id(t)}:{int(now * 1000)}"
                    )
                    t.meta["_feedback_token"] = token
                    alias = key if key is not None else None
                    self._pending_feedback[token] = (
                        features,
                        ts_idx,
                        alias,
                        token,
                        priority_token,
                    )
                    if alias is not None:
                        self._pending_feedback[alias] = (
                            features,
                            ts_idx,
                            alias,
                            token,
                            priority_token,
                        )
                    if len(self._pending_feedback) > self._pending_feedback_limit:
                        # prune the oldest inserted token (approximate)
                        oldest = next(iter(self._pending_feedback))
                        if oldest != token:
                            self._pending_feedback.pop(oldest, None)
                    scored.append(ScoredTrigger(trigger=t, priority=pr, feedback_token=token))
            except Exception:
                continue
        # preemption rules
        scored.sort(key=lambda s: s.priority, reverse=True)
        return self._llm_adjust_priorities(scored)
