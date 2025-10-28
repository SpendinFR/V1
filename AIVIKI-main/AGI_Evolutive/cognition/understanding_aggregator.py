from __future__ import annotations

import random
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Mapping

from AGI_Evolutive.cognition.meta_cognition import OnlineLinear
from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)


LOGGER = logging.getLogger(__name__)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class _ThompsonBandit:
    """Simple discrete Thompson Sampling helper for adaptive smoothing."""

    def __init__(self, candidates: Iterable[float]) -> None:
        self._candidates = tuple(sorted({_clamp(c, 0.01, 0.99) for c in candidates})) or (0.25, 0.5, 0.75)
        self._state: Dict[float, Dict[str, float]] = {
            c: {"success": 1.0, "failure": 1.0} for c in self._candidates
        }

    def sample(self) -> float:
        draw = {
            c: random.betavariate(state["success"], state["failure"]) for c, state in self._state.items()
        }
        return max(draw, key=draw.get)

    def update(self, choice: float, reward: float) -> None:
        if choice not in self._state:
            return
        reward = _clamp(reward)
        state = self._state[choice]
        state["success"] += reward
        state["failure"] += (1.0 - reward)


@dataclass
class UnderstandingResult:
    U_topic: float
    U_global: float
    details: Dict[str, float]
    alpha_topic: float = 0.0
    alpha_global: float = 0.0
    uncertainty: float = 0.0


class UnderstandingAggregator:
    """Aggregates signals into understanding scores with adaptive smoothing."""

    def __init__(
        self,
        alpha_topic: float = 0.25,
        alpha_global: float = 0.15,
        alpha_options: Optional[Iterable[float]] = None,
        max_update: float = 0.18,
    ) -> None:
        self.alpha_topic = _clamp(alpha_topic, 0.01, 0.99)
        self.alpha_global = _clamp(alpha_global, 0.01, 0.99)
        base_options = tuple(alpha_options) if alpha_options else (0.2, 0.4, 0.6, 0.8)
        option_set = set(base_options)
        option_set.add(self.alpha_topic)
        option_set.add(self.alpha_global)
        self.alpha_options = tuple(sorted(_clamp(o, 0.01, 0.99) for o in option_set))
        self.max_update = max(0.01, min(0.5, float(max_update)))
        self._topic_ema: Dict[str, float] = {}
        self._topic_bandits: Dict[str, _ThompsonBandit] = {}
        self._global_ema: float = 0.5
        self._global_bandit = _ThompsonBandit(self.alpha_options)
        base_features = (
            "one_minus_prediction_error",
            "prediction_error",
            "memory_consistency",
            "transfer_success",
            "explanatory_adequacy",
            "social_appraisal",
            "penalty",
            "penalty_squared",
            "consistency_gap",
        )
        self._model = OnlineLinear(
            base_features,
            bounds=(-0.75, 0.75),
            lr=0.045,
            l2=0.002,
            max_grad=0.28,
            warmup=24,
            init_weight=0.0,
        )
        self._last_llm_summary: Optional[Mapping[str, Any]] = None

    @staticmethod
    def _ema(prev: float, x: float, alpha: float) -> float:
        return (1 - alpha) * prev + alpha * x

    def _bandit(self, key: str) -> _ThompsonBandit:
        if key not in self._topic_bandits:
            self._topic_bandits[key] = _ThompsonBandit(self.alpha_options)
        return self._topic_bandits[key]

    def _build_features(
        self,
        topic: Optional[str],
        pe: float,
        mc: float,
        tr: float,
        ex: float,
        sa: float,
        cp: float,
        cg: float,
    ) -> Dict[str, float]:
        penalty = _clamp(cp + cg)
        features: Dict[str, float] = {
            "one_minus_prediction_error": 1.0 - pe,
            "prediction_error": pe,
            "memory_consistency": mc,
            "transfer_success": tr,
            "explanatory_adequacy": ex,
            "social_appraisal": sa,
            "penalty": penalty,
            "penalty_squared": penalty * penalty,
            "consistency_gap": _clamp(abs(mc - tr) * 0.5 + abs(mc - ex) * 0.5),
        }
        if topic:
            features[f"topic::{topic}"] = 1.0
        return features

    def _static_baseline(
        self,
        pe: float,
        mc: float,
        tr: float,
        ex: float,
        sa: float,
        cp: float,
        cg: float,
    ) -> float:
        U_inst = 0.40 * (1.0 - pe)
        U_inst += 0.15 * mc
        U_inst += 0.15 * tr
        U_inst += 0.15 * ex
        U_inst += 0.10 * sa
        U_inst -= 0.05 * cp
        U_inst -= 0.05 * cg
        return _clamp(U_inst)

    def _llm_assimilate(self, payload: Dict[str, Any]) -> Optional[Mapping[str, Any]]:
        if not _llm_enabled():
            return None

        try:
            response = _llm_manager().call_dict(
                "understanding_aggregator",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM understanding aggregator unavailable", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        self._last_llm_summary = dict(response)
        return self._last_llm_summary

    def compute(
        self,
        topic: Optional[str],
        prediction_error: float,
        memory_consistency: float = 0.5,
        transfer_success: float = 0.5,
        explanatory_adequacy: float = 0.5,
        social_appraisal: float = 0.5,
        clarification_penalty: float = 0.0,
        calibration_gap: float = 0.0,
    ) -> UnderstandingResult:
        pe = _clamp(prediction_error)
        mc = _clamp(memory_consistency)
        tr = _clamp(transfer_success)
        ex = _clamp(explanatory_adequacy)
        sa = _clamp(social_appraisal)
        cp = _clamp(clarification_penalty)
        cg = _clamp(calibration_gap)

        baseline = self._static_baseline(pe, mc, tr, ex, sa, cp, cg)
        features = self._build_features(topic, pe, mc, tr, ex, sa, cp, cg)
        model_pred = self._model.predict(features)
        confidence = _clamp(self._model.confidence())
        llm_payload = {
            "topic": topic,
            "prediction_error": pe,
            "memory_consistency": mc,
            "transfer_success": tr,
            "explanatory_adequacy": ex,
            "social_appraisal": sa,
            "clarification_penalty": cp,
            "calibration_gap": cg,
        }
        llm_summary = self._llm_assimilate(llm_payload)
        llm_weight = 0.0
        blended = baseline * (1.0 - confidence) + model_pred * confidence
        if llm_summary:
            try:
                llm_score = _clamp(float(llm_summary.get("assimilation_score", blended)))
            except (TypeError, ValueError):
                llm_score = blended
            llm_weight = 0.3
            total_weight = (1.0 - confidence) + confidence + llm_weight
            if total_weight > 0:
                blended = (
                    baseline * (1.0 - confidence)
                    + model_pred * confidence
                    + llm_score * llm_weight
                ) / total_weight
            else:
                blended = llm_score

        reward = _clamp(1.0 - pe - 0.5 * cp - 0.5 * cg)

        if topic:
            prev_topic = self._topic_ema.get(topic, 0.5)
            bandit = self._bandit(topic)
            alpha_topic = bandit.sample()
            raw_topic = self._ema(prev_topic, blended, alpha_topic)
            delta = max(-self.max_update, min(self.max_update, raw_topic - prev_topic))
            U_topic = _clamp(prev_topic + delta)
            bandit.update(alpha_topic, reward)
        else:
            alpha_topic = self.alpha_topic
            U_topic = blended
            prev_topic = U_topic

        prev_global = self._global_ema
        alpha_global = self._global_bandit.sample()
        raw_global = self._ema(prev_global, U_topic if topic else blended, alpha_global)
        delta_global = max(-self.max_update, min(self.max_update, raw_global - prev_global))
        self._global_ema = _clamp(prev_global + delta_global)
        self._global_bandit.update(alpha_global, reward)

        if topic:
            self._topic_ema[topic] = U_topic
        else:
            self._topic_ema["__global__"] = blended

        self._model.update(features, reward)

        details = {
            "prediction_error": pe,
            "memory_consistency": mc,
            "transfer_success": tr,
            "explanatory_adequacy": ex,
            "social_appraisal": sa,
            "clarification_penalty": cp,
            "calibration_gap": cg,
            "baseline": baseline,
            "model_prediction": model_pred,
            "blended": blended,
            "reward_signal": reward,
            "topic_drift": abs(U_topic - prev_topic) if topic else 0.0,
            "global_drift": abs(self._global_ema - prev_global),
        }
        if llm_summary:
            details["llm"] = llm_summary
            details["llm_weight"] = llm_weight

        uncertainty = _clamp(1.0 - confidence)

        return UnderstandingResult(
            U_topic=U_topic,
            U_global=self._global_ema,
            details=details,
            alpha_topic=alpha_topic,
            alpha_global=alpha_global,
            uncertainty=uncertainty,
        )
