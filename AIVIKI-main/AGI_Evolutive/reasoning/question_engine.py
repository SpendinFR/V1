"""Information gain oriented question engine for abductive reasoning."""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from AGI_Evolutive.cognition.meta_cognition import OnlineLinear
except Exception:  # pragma: no cover - graceful fallback
    OnlineLinear = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .abduction import Hypothesis

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)

class QuestionEngine:
    """Rank potential clarification questions by expected information gain."""

    def __init__(self, beliefs: Any, user_model: Any) -> None:
        self.beliefs = beliefs
        self.user_model = user_model
        feature_names = [
            "bias",
            "score",
            "prior",
            "rank",
            "causal_support",
            "simulations",
            "entropy",
            "user_prior",
        ]
        self._response_model: Optional[OnlineLinear]
        if OnlineLinear is not None:
            self._response_model = OnlineLinear(
                feature_names,
                bounds=(0.0, 1.0),
                lr=0.04,
                l2=0.001,
                max_grad=0.2,
                warmup=18,
                init_weight=0.05,
            )
        else:  # pragma: no cover - fallback for environments without the model
            self._response_model = None
        self._pending_questions: Dict[str, Dict[str, Any]] = {}
        self._ts_options: List[Dict[str, float]] = [
            {
                "confirm_target": 1.35,
                "confirm_others": 0.9,
                "neutral_target": 1.08,
                "neutral_others": 0.97,
                "reject_target": 0.65,
                "reject_others": 1.1,
                "success": 1.0,
                "failure": 1.0,
            },
            {
                "confirm_target": 1.22,
                "confirm_others": 0.93,
                "neutral_target": 1.05,
                "neutral_others": 0.99,
                "reject_target": 0.7,
                "reject_others": 1.08,
                "success": 1.0,
                "failure": 1.0,
            },
            {
                "confirm_target": 1.15,
                "confirm_others": 0.96,
                "neutral_target": 1.02,
                "neutral_others": 1.0,
                "reject_target": 0.78,
                "reject_others": 1.05,
                "success": 1.0,
                "failure": 1.0,
            },
        ]

    # ------------------------------------------------------------------
    def best_question(
        self, hypotheses: Iterable["Hypothesis"], observation: str
    ) -> Optional[str]:
        hyps = list(hypotheses)
        if len(hyps) < 2:
            return None
        self.set_hypotheses(hyps)
        scores = [max(1e-6, min(0.999, h.score)) for h in hyps]
        base_entropy = self._entropy(scores)
        candidates: List[Dict[str, Any]] = []
        for hyp in hyps[:3]:
            text = self._craft_question(hyp, observation)
            if not text:
                continue
            posterior, meta = self._posterior_scenarios(hyp, scores, base_entropy)
            expected = sum(p * self._entropy(dist) for p, dist in posterior)
            gain = max(0.0, base_entropy - expected)
            candidates.append(
                {
                    "text": text,
                    "gain": gain,
                    "score": hyp.score,
                    "meta": meta,
                }
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item["gain"], item["score"]), reverse=True)
        winner = self._llm_rank_candidates(candidates, observation) or candidates[0]
        self._register_question(winner)
        return winner["text"]

    # ------------------------------------------------------------------
    def _posterior_scenarios(
        self, hyp: "Hypothesis", scores: List[float], base_entropy: float
    ) -> Tuple[List[tuple[float, List[float]]], Dict[str, Any]]:
        params, ts_index = self._select_parameter_set()
        features = self._feature_vector(hyp, scores, base_entropy)
        confirm_prob, neutral_prob, reject_prob = self._predict_outcome_probabilities(features)
        scenarios = [
            (
                confirm_prob,
                self._scenario_distribution(scores, hyp, params, "confirm"),
            ),
            (
                neutral_prob,
                self._scenario_distribution(scores, hyp, params, "neutral"),
            ),
            (
                reject_prob,
                self._scenario_distribution(scores, hyp, params, "reject"),
            ),
        ]
        return (
            scenarios,
            {
                "hypothesis": hyp.label,
                "features": features,
                "ts_index": ts_index,
                "outcome_probs": (
                    float(confirm_prob),
                    float(neutral_prob),
                    float(reject_prob),
                ),
            },
        )

    def _select_parameter_set(self) -> Tuple[Dict[str, float], int]:
        best_idx = 0
        best_sample = float("-inf")
        for idx, option in enumerate(self._ts_options):
            alpha = max(1e-3, float(option.get("success", 1.0)))
            beta = max(1e-3, float(option.get("failure", 1.0)))
            sample = random.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_idx = idx
        return self._ts_options[best_idx], best_idx

    def _feature_vector(
        self, hyp: "Hypothesis", scores: List[float], base_entropy: float
    ) -> Dict[str, float]:
        idx = self._hyp_list.index(hyp) if hyp in self._hyp_list else -1
        rank = 1.0 / float(idx + 1) if idx >= 0 else 0.0
        user_prior = self._user_prior(hyp.label)
        features: Dict[str, float] = {
            "bias": 1.0,
            "score": float(max(0.0, min(1.0, hyp.score))),
            "prior": float(hyp.priors.get("pri", hyp.score) if getattr(hyp, "priors", None) else hyp.score),
            "rank": rank,
            "causal_support": float(bool(getattr(hyp, "causal_support", None))),
            "simulations": float(len(getattr(hyp, "simulations", []) or [])),
            "entropy": float(base_entropy),
            "user_prior": user_prior,
        }
        return features

    def _user_prior(self, label: str) -> float:
        if self.user_model is None:
            return 0.5
        prior_fn = getattr(self.user_model, "prior", None)
        if callable(prior_fn):
            try:
                return float(max(0.0, min(1.0, prior_fn(label))))
            except Exception:
                return 0.5
        return 0.5

    def _predict_outcome_probabilities(
        self, features: Dict[str, float]
    ) -> Tuple[float, float, float]:
        neutral_floor = 0.08
        neutral_span = 0.22
        model_confidence = 0.0
        if self._response_model is None:
            predicted = 0.5
        else:
            predicted = float(self._response_model.predict(features))
            model_confidence = float(self._response_model.confidence())
        user_prior = features.get("user_prior", 0.5)
        blended = max(1e-3, min(1.0 - 1e-3, 0.5 * (predicted + user_prior)))
        neutral = neutral_floor + neutral_span * (1.0 - model_confidence)
        neutral = max(0.05, min(0.4, neutral))
        confirm = blended * (1.0 - neutral)
        confirm = max(1e-3, min(1.0 - neutral - 1e-3, confirm))
        reject = max(1e-3, 1.0 - neutral - confirm)
        total = confirm + neutral + reject
        return confirm / total, neutral / total, reject / total

    def _scenario_distribution(
        self,
        scores: List[float],
        hyp: "Hypothesis",
        params: Dict[str, float],
        key: str,
    ) -> List[float]:
        target_key = f"{key}_target"
        other_key = f"{key}_others"
        target_factor = float(params.get(target_key, 1.0))
        other_factor = float(params.get(other_key, 1.0))
        adjusted = []
        for score, candidate in zip(scores, self._hyp_list):
            factor = target_factor if candidate is hyp else other_factor
            adjusted.append(min(0.99, max(1e-6, score * factor)))
        return self._normalise(adjusted)

    def _register_question(self, winner: Dict[str, Any]) -> None:
        meta = winner.get("meta") or {}
        text = winner.get("text")
        if not text:
            return
        self._pending_questions[text] = {
            "hypothesis": meta.get("hypothesis"),
            "features": meta.get("features"),
            "ts_index": meta.get("ts_index"),
            "outcome_probs": meta.get("outcome_probs"),
        }

    def record_feedback(
        self,
        question: str,
        outcome: str,
        confidence: float = 1.0,
    ) -> None:
        payload = self._pending_questions.pop(question, None)
        if not payload:
            return
        confidence = float(max(0.0, min(1.0, confidence)))
        features = payload.get("features")
        if features and self._response_model is not None:
            target = 0.5
            if outcome == "confirm":
                target = 1.0
            elif outcome == "reject":
                target = 0.0
            self._response_model.update(features, target)
        ts_index = payload.get("ts_index")
        if isinstance(ts_index, int) and 0 <= ts_index < len(self._ts_options):
            option = self._ts_options[ts_index]
            if outcome == "confirm":
                option["success"] = option.get("success", 1.0) + confidence
            elif outcome == "reject":
                option["failure"] = option.get("failure", 1.0) + confidence
            else:
                option["success"] = option.get("success", 1.0) + 0.3 * confidence
                option["failure"] = option.get("failure", 1.0) + 0.3 * confidence
        self._update_user_model(payload.get("hypothesis"), outcome, confidence)

    def _update_user_model(
        self, hypothesis_label: Optional[str], outcome: str, confidence: float
    ) -> None:
        if not hypothesis_label or self.user_model is None:
            return
        observer = getattr(self.user_model, "observe_preference", None)
        if not callable(observer):
            return
        weight = max(0.05, confidence)
        if outcome == "confirm":
            observer(hypothesis_label, True, weight=weight)
        elif outcome == "reject":
            observer(hypothesis_label, False, weight=weight)

    def _craft_question(self, hyp: "Hypothesis", observation: str) -> Optional[str]:
        cue = hyp.label or "cette hypothèse"
        if hyp.causal_support:
            cue += f" (indices: {hyp.causal_support[0]})"
        return f"Quels éléments concrets confirmeraient {cue} dans « {observation[:80]} » ?"

    def _llm_rank_candidates(
        self, candidates: List[Dict[str, Any]], observation: str
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None

        payload = {
            "observation": observation,
            "candidates": [
                {
                    "text": cand.get("text"),
                    "gain": float(cand.get("gain", 0.0)),
                    "score": float(cand.get("score", 0.0)),
                    "meta": cand.get("meta", {}),
                }
                for cand in candidates
            ],
        }

        response = try_call_llm_dict(
            "question_engine",
            input_payload=payload,
            logger=LOGGER,
        )
        if not isinstance(response, Mapping):
            return None

        selected_text = response.get("selected_question") or response.get("question")
        if not isinstance(selected_text, str):
            return None

        for cand in candidates:
            if cand.get("text") == selected_text:
                enriched = dict(cand)
                enriched.setdefault("llm_guidance", response)
                return enriched
        return None

    # ------------------------------------------------------------------
    def _entropy(self, scores: Iterable[float]) -> float:
        probs = self._normalise(scores)
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _normalise(self, scores: Iterable[float]) -> List[float]:
        scores = list(scores)
        total = sum(scores) or 1.0
        return [max(1e-6, s / total) for s in scores]

    @property
    def _hyp_list(self) -> List["Hypothesis"]:
        # Helper for typing convenience; updated externally.
        return getattr(self, "_cached_hypotheses", [])

    def set_hypotheses(self, hyps: List["Hypothesis"]) -> None:
        self._cached_hypotheses = hyps

