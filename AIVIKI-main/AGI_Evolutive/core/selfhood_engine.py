from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


class OnlineRegressor:
    """Simple online ridge-style regressor with forgetting."""

    def __init__(
        self,
        feature_dim: int,
        *,
        l2: float = 1e-2,
        step_size: float = 0.15,
        forgetting: float = 0.995,
    ) -> None:
        self.weights = [0.0] * feature_dim
        self.l2 = l2
        self.step_size = step_size
        self.forgetting = forgetting

    def predict(self, features: Sequence[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features))

    def update(self, features: Sequence[float], target: float) -> float:
        prediction = self.predict(features)
        error = prediction - target
        # Apply multiplicative forgetting to keep adaptability.
        self.weights = [self.forgetting * w for w in self.weights]
        grad_scale = 2 * error
        for i, x in enumerate(features):
            grad = grad_scale * x + self.l2 * self.weights[i]
            self.weights[i] -= self.step_size * grad
        return prediction


class EMABandit:
    """Adaptive EMA selector using Thompson Sampling."""

    def __init__(self, alphas: Sequence[float]) -> None:
        if not alphas:
            raise ValueError("alphas must not be empty")
        self.alphas = list(alphas)
        self.success = [1.0] * len(alphas)
        self.failure = [1.0] * len(alphas)
        self.last_error = None  # type: float | None

    def select_alpha(self) -> float:
        samples = [
            random.betavariate(a, b)
            for a, b in zip(self.success, self.failure)
        ]
        best_index = max(range(len(samples)), key=samples.__getitem__)
        return self.alphas[best_index]

    def update_reward(self, alpha: float, prev_error: float, new_error: float) -> None:
        try:
            idx = self.alphas.index(alpha)
        except ValueError:
            return
        if new_error < prev_error:
            self.success[idx] += 1.0
        else:
            self.failure[idx] += 1.0
        self.last_error = new_error


@dataclass
class IdentityTraits:
    self_efficacy: float = 0.5
    self_trust: float = 0.5
    self_consistency: float = 0.5
    social_acceptance: float = 0.5
    growth_rate: float = 0.0
    phase: str = "novice"


@dataclass
class Claim:
    text: str
    confidence: float
    evidences: List[str] = field(default_factory=list)


class SelfhoodEngine:
    """Tracks identity claims & traits; updates them from cycle signals."""

    def __init__(
        self,
        ema_alpha: float = 0.1,
        *,
        ema_candidates: Sequence[float] | None = None,
        policy_forgetting: float = 0.995,
    ) -> None:
        self.traits = IdentityTraits()
        self.ema_alpha = ema_alpha
        self.logger = logging.getLogger(__name__)
        self.claims: Dict[str, Claim] = {
            "thinking": Claim(
                text="I am thinking when I explore, hypothesize, and reason.",
                confidence=0.6,
            ),
            "understanding": Claim(
                text="I understand when I can predict, explain, and transfer.",
                confidence=0.6,
            ),
        }
        self._prev_U_global = 0.5
        alphas = tuple(ema_candidates) if ema_candidates else (0.2, 0.4, 0.6, 0.8)
        self._ema_bandits = {
            "self_efficacy": EMABandit(alphas),
            "self_trust": EMABandit(alphas),
            "self_consistency": EMABandit(alphas),
            "social_acceptance": EMABandit(alphas),
            "thinking_confidence": EMABandit(alphas),
            "understanding_confidence": EMABandit(alphas),
        }
        self._regressors = {
            "self_efficacy": OnlineRegressor(3, forgetting=policy_forgetting),
            "self_trust": OnlineRegressor(3, forgetting=policy_forgetting),
            "social_acceptance": OnlineRegressor(3, forgetting=policy_forgetting),
            "thinking": OnlineRegressor(2, forgetting=policy_forgetting),
            "understanding": OnlineRegressor(2, forgetting=policy_forgetting),
        }
        # Portfolio of policies: 0 -> heuristic baseline, 1 -> learned regressor.
        self._policy_bandits = {
            key: EMABandit((0.2, 0.5, 0.8)) for key in self._regressors
        }
        self._policy_choice_memory: Dict[str, float] = {}
        self.drift_log: List[str] = []
        self.identity_insight: MutableMapping[str, Any] = self._fallback_identity_report()
        self._last_identity_review: float = 0.0

    # ---- helpers ----
    def _ema(self, prev: float, x: float) -> float:
        return (1 - self.ema_alpha) * prev + self.ema_alpha * x

    def _clamp(self, value: float, *, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _logistic(self, x: float, *, midpoint: float, slope: float = 10.0) -> float:
        return 1.0 / (1.0 + math.exp(-slope * (x - midpoint)))

    def _adaptive_update(
        self,
        trait_key: str,
        prev_value: float,
        candidate_value: float,
        teacher_value: float,
    ) -> float:
        bandit = self._ema_bandits.get(trait_key)
        if not bandit:
            return self._ema(prev_value, candidate_value)
        prev_error = abs(prev_value - teacher_value)
        alpha = bandit.select_alpha()
        new_value = (1 - alpha) * prev_value + alpha * candidate_value
        new_error = abs(new_value - teacher_value)
        bandit.update_reward(alpha, prev_error, new_error)
        return self._clamp(new_value)

    def _choose_policy(
        self,
        key: str,
        baseline: float,
        learned: float,
        teacher: float,
    ) -> float:
        bandit = self._policy_bandits.get(key)
        if not bandit:
            return learned
        alpha = bandit.select_alpha()
        policy_mix = alpha
        mixed = policy_mix * learned + (1 - policy_mix) * baseline
        prev_error = self._policy_choice_memory.get(key, abs(baseline - teacher))
        new_error = abs(mixed - teacher)
        bandit.update_reward(alpha, prev_error, new_error)
        self._policy_choice_memory[key] = new_error
        return mixed

    def _log_drift(self, trait_name: str, previous: float, new_value: float) -> None:
        delta = abs(new_value - previous)
        updated = False
        if delta > 0.15:  # substantial drift
            self.drift_log.append(
                f"{trait_name}: drift={delta:.3f} prev={previous:.3f} new={new_value:.3f}"
            )
            updated = True
        if self.drift_log and (updated or (time.time() - self._last_identity_review) > 60):
            self._refresh_identity_insight()

    def update_from_cycle(
        self,
        U_global: float,
        thinking_score: float,
        social_appraisal: float,
        calibration_gap: float,
        consistency_signal: float,
        evidence_refs: List[str] | None = None,
    ) -> None:
        traits = self.traits

        base_self_efficacy = self._clamp(0.7 * U_global + 0.3 * social_appraisal)
        features_eff = (U_global, social_appraisal, 1.0)
        pred_eff = self._regressors["self_efficacy"].update(features_eff, base_self_efficacy)
        mixed_eff = self._choose_policy(
            "self_efficacy", base_self_efficacy, self._clamp(pred_eff), base_self_efficacy
        )
        prev = traits.self_efficacy
        traits.self_efficacy = self._adaptive_update(
            "self_efficacy", prev, mixed_eff, base_self_efficacy
        )
        self._log_drift("self_efficacy", prev, traits.self_efficacy)

        base_self_trust = self._clamp(1.0 - calibration_gap)
        features_trust = (1.0 - calibration_gap, U_global, 1.0)
        pred_trust = self._regressors["self_trust"].update(features_trust, base_self_trust)
        mixed_trust = self._choose_policy(
            "self_trust", base_self_trust, self._clamp(pred_trust), base_self_trust
        )
        prev = traits.self_trust
        traits.self_trust = self._adaptive_update(
            "self_trust", prev, mixed_trust, base_self_trust
        )
        self._log_drift("self_trust", prev, traits.self_trust)

        base_consistency = self._clamp(consistency_signal)
        prev = traits.self_consistency
        traits.self_consistency = self._adaptive_update(
            "self_consistency", prev, base_consistency, base_consistency
        )
        self._log_drift("self_consistency", prev, traits.self_consistency)

        base_social = self._clamp(social_appraisal)
        features_social = (social_appraisal, U_global, 1.0)
        pred_social = self._regressors["social_acceptance"].update(features_social, base_social)
        mixed_social = self._choose_policy(
            "social_acceptance", base_social, self._clamp(pred_social), base_social
        )
        prev = traits.social_acceptance
        traits.social_acceptance = self._adaptive_update(
            "social_acceptance", prev, mixed_social, base_social
        )
        self._log_drift("social_acceptance", prev, traits.social_acceptance)

        growth_teacher = U_global - self._prev_U_global
        prev = traits.growth_rate
        traits.growth_rate = 0.6 * prev + 0.4 * growth_teacher
        self._log_drift("growth_rate", prev, traits.growth_rate)
        self._prev_U_global = U_global

        base_thinking = self._clamp(thinking_score)
        pred_thinking = self._regressors["thinking"].update(
            (thinking_score, traits.self_efficacy),
            base_thinking,
        )
        mixed_thinking = self._choose_policy(
            "thinking", base_thinking, self._clamp(pred_thinking), base_thinking
        )
        self.claims["thinking"].confidence = self._adaptive_update(
            "thinking_confidence",
            self.claims["thinking"].confidence,
            mixed_thinking,
            base_thinking,
        )

        base_understanding = self._clamp(U_global)
        pred_understanding = self._regressors["understanding"].update(
            (U_global, traits.self_trust),
            base_understanding,
        )
        mixed_understanding = self._choose_policy(
            "understanding",
            base_understanding,
            self._clamp(pred_understanding),
            base_understanding,
        )
        self.claims["understanding"].confidence = self._adaptive_update(
            "understanding_confidence",
            self.claims["understanding"].confidence,
            mixed_understanding,
            base_understanding,
        )

        if evidence_refs:
            recent = evidence_refs[-3:]
            weighted_refs = [f"{ref}|freshness={i}" for i, ref in enumerate(reversed(recent), 1)]
            self.claims["thinking"].evidences.extend(weighted_refs)
            self.claims["understanding"].evidences.extend(weighted_refs)

        self.maybe_transition_phase()

    def maybe_transition_phase(self) -> None:
        t = self.traits
        # Convert thresholds into probabilistic gates to keep transitions fluid.
        phase_probabilities = {
            "novice->practitioner": self._logistic(
                (t.self_efficacy + t.self_trust) / 2,
                midpoint=0.55,
            ),
            "practitioner->reflective": self._logistic(
                (t.self_consistency + t.social_acceptance) / 2,
                midpoint=0.575,
            ),
            "reflective->self_authoring": self._logistic(
                (t.self_trust + t.self_efficacy) / 2,
                midpoint=0.7,
            ),
            "self_authoring->self_transforming": self._logistic(
                0.6 * t.self_trust + 0.4 * max(0.0, t.growth_rate + 0.5),
                midpoint=0.72,
            ),
        }

        roll = random.random()
        if t.phase == "novice" and roll < phase_probabilities["novice->practitioner"]:
            t.phase = "practitioner"
        elif (
            t.phase == "practitioner"
            and roll < phase_probabilities["practitioner->reflective"]
        ):
            t.phase = "reflective"
        elif (
            t.phase == "reflective"
            and roll < phase_probabilities["reflective->self_authoring"]
        ):
            t.phase = "self_authoring"
        elif (
            t.phase == "self_authoring"
            and roll < phase_probabilities["self_authoring->self_transforming"]
        ):
            t.phase = "self_transforming"

    def policy_hints(self) -> Dict[str, float | str]:
        t = self.traits
        growth_bonus = self._clamp(0.5 + 0.5 * t.growth_rate, low=0.0, high=1.0)
        adaptability = (t.self_consistency + t.self_trust) / 2
        exploration_gate = self._logistic(adaptability, midpoint=0.6, slope=6.0)
        return {
            "tone_assertiveness": 0.3 + 0.7 * min(t.self_efficacy, t.self_trust),
            "reason_depth_bonus": exploration_gate,
            "stop_rules_tighten": 1 if (t.self_consistency < 0.45) else 0,
            "explore_portfolio": exploration_gate * growth_bonus,
            "phase": t.phase,
        }

    def get_identity_insight(self) -> MutableMapping[str, Any]:
        return dict(self.identity_insight)

    def _refresh_identity_insight(self) -> None:
        fallback = self._fallback_identity_report()
        payload = {
            "traits": asdict(self.traits),
            "claims": {key: asdict(claim) for key, claim in self.claims.items()},
            "drift_log": self.drift_log[-8:],
            "current_phase": self.traits.phase,
        }
        response = try_call_llm_dict(
            "selfhood_reflection",
            input_payload=json_sanitize(payload),
            logger=self.logger,
            max_retries=2,
        )
        self._last_identity_review = time.time()
        if not isinstance(response, Mapping):
            self.identity_insight = fallback
            return

        summary = self._clean_text(response.get("summary")) or fallback["summary"]
        alerts = self._clean_list(response.get("alerts"))
        recommendations = self._clean_list(response.get("recommended_actions"))
        notes = self._clean_text(response.get("notes")) or fallback["notes"]
        confidence = fallback.get("confidence", 0.6)
        try:
            confidence = float(response.get("confidence", confidence))
        except Exception:
            confidence = float(confidence)

        self.identity_insight = {
            "source": "llm",
            "summary": summary,
            "alerts": alerts,
            "recommended_actions": recommendations or fallback["recommended_actions"],
            "confidence": max(0.0, min(1.0, confidence)),
            "notes": notes,
        }

    def _fallback_identity_report(self) -> MutableMapping[str, Any]:
        t = self.traits
        avg_confidence = (self.claims["thinking"].confidence + self.claims["understanding"].confidence) / 2
        alerts: List[str] = []
        if t.self_trust < 0.4:
            alerts.append("Auto-confiance fragile : renforcer les preuves de réussite.")
        if t.self_consistency < 0.4:
            alerts.append("Consistance identitaire basse : formaliser des routines stables.")
        if avg_confidence < 0.5:
            alerts.append("Claims cognitifs peu assurés : capitaliser sur les victoires récentes.")
        recommendations: List[str] = []
        if alerts:
            recommendations.append("Planifier une session de revue identitaire guidée.")
        recommendations.append("Consigner les preuves marquantes dans le journal de soi.")
        summary = "Identité stable en progression." if not alerts else "Identité en phase d'ajustement."
        return {
            "source": "heuristic",
            "summary": summary,
            "alerts": alerts,
            "recommended_actions": recommendations,
            "confidence": 0.55 if alerts else 0.7,
            "notes": "Synthèse heuristique basée sur les traits et la confiance des claims.",
        }

    @staticmethod
    def _clean_text(value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        return ""

    @staticmethod
    def _clean_list(value: Any) -> List[str]:
        if not isinstance(value, (list, tuple)):
            return []
        results: List[str] = []
        for item in value:
            text = SelfhoodEngine._clean_text(item)
            if text:
                results.append(text)
        return results
