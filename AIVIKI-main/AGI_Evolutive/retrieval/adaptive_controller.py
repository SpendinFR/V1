import json
import logging
import os
import random
import time
from collections import deque
from copy import deepcopy
from typing import Any, Deque, Dict, Iterable, List, Optional

from AGI_Evolutive.cognition.meta_cognition import OnlineLinear
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


class DiscreteThompsonSampler:
    """Lightweight Thompson Sampling helper persisted as a dict."""

    def __init__(self, options: Iterable[float], state: Optional[Dict[str, Any]] = None):
        self.options: List[float] = list(dict.fromkeys(float(o) for o in options))
        if not self.options:
            raise ValueError("DiscreteThompsonSampler requires at least one option")
        self.state: Dict[str, Any] = state if isinstance(state, dict) else {}
        self.state.setdefault("_meta", {})
        for opt in self.options:
            key = self._key(opt)
            self.state.setdefault(key, {"alpha": 1.0, "beta": 1.0})
        last = self.state["_meta"].get("last_choice")
        if last is not None and last not in self.options:
            last = None
        self._last_choice: Optional[float] = float(last) if last is not None else None

    @staticmethod
    def _key(value: float) -> str:
        return f"{float(value):.6f}"

    def ensure_choice(self, default: float) -> float:
        if self._last_choice is None:
            return self.sample(default)
        return float(self._last_choice)

    def sample(self, default: float) -> float:
        best_choice: Optional[float] = None
        best_draw = -1.0
        for opt in self.options:
            entry = self.state[self._key(opt)]
            draw = random.betavariate(float(entry.get("alpha", 1.0)), float(entry.get("beta", 1.0)))
            if draw > best_draw:
                best_draw = draw
                best_choice = opt
        if best_choice is None:
            best_choice = float(default)
        self._mark_choice(best_choice)
        return float(best_choice)

    def update(self, reward: float) -> None:
        if self._last_choice is None:
            return
        key = self._key(self._last_choice)
        entry = self.state.setdefault(key, {"alpha": 1.0, "beta": 1.0})
        reward = _clamp(reward)
        entry["alpha"] = float(entry.get("alpha", 1.0)) + reward
        entry["beta"] = float(entry.get("beta", 1.0)) + (1.0 - reward)

    def _mark_choice(self, choice: float) -> None:
        self._last_choice = float(choice)
        self.state.setdefault("_meta", {})["last_choice"] = float(choice)

    def to_state(self) -> Dict[str, Any]:
        return deepcopy(self.state)


class AdaptiveEMA:
    """EMA whose decay factor is chosen adaptively via Thompson Sampling."""

    def __init__(self, sampler: DiscreteThompsonSampler, default_beta: float = 0.6, initial: float = 0.5):
        self.sampler = sampler
        self.default_beta = float(default_beta)
        self.value = float(initial)
        self._last_beta: Optional[float] = None

    def update(self, signal: float) -> float:
        beta = self.sampler.ensure_choice(self.default_beta)
        beta = _clamp(beta, 0.05, 0.95)
        self.value = beta * self.value + (1.0 - beta) * float(signal)
        self._last_beta = beta
        return self.value

    def reinforce(self, reward: float) -> None:
        if self._last_beta is None:
            return
        self.sampler.update(_clamp(reward))
        self._last_beta = None


class RAGAdaptiveController:
    """Learned controller for RAG hyper-parameters with lightweight persistence."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        state_path: str = "data/rag/adaptive_state.json",
    ) -> None:
        self.base_config = deepcopy(base_config)
        self.state_path = state_path
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self.state = self._load_state()
        weight_state = self.state.get("weight_model")
        self.weight_model = OnlineLinear.from_state(
            weight_state,
            bounds=(0.05, 0.95),
            lr=0.04,
            l2=0.002,
            max_grad=0.25,
            warmup=18,
            init_weight=0.4,
        )
        threshold_state = self.state.get("threshold_model")
        self.threshold_model = OnlineLinear.from_state(
            threshold_state,
            bounds=(0.1, 0.9),
            lr=0.03,
            l2=0.001,
            max_grad=0.2,
            warmup=20,
            init_weight=0.2,
        )
        half_life_state = self.state.get("half_life_bandit")
        support_state = self.state.get("support_bandit")
        top1_state = self.state.get("top1_bandit")
        beta_state = self.state.get("ema_bandit")

        self.half_life_sampler = DiscreteThompsonSampler((3, 7, 14, 30), half_life_state)
        self.support_sampler = DiscreteThompsonSampler((0.12, 0.15, 0.18, 0.2), support_state)
        self.top1_sampler = DiscreteThompsonSampler((0.18, 0.22, 0.25, 0.28), top1_state)
        self.activation_sampler = DiscreteThompsonSampler((0.2, 0.4, 0.6, 0.8), beta_state)

        self.global_activation_tracker = AdaptiveEMA(self.activation_sampler, default_beta=0.6, initial=0.5)
        self.pending_interactions: Deque[Dict[str, Any]] = deque(maxlen=32)
        self.last_config: Dict[str, Any] = self._make_config({}, {})
        self._dirty = False

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self) -> None:
        if not self._dirty:
            return
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "weight_model": self.weight_model.to_state(),
                        "threshold_model": self.threshold_model.to_state(),
                        "half_life_bandit": self.half_life_sampler.to_state(),
                        "support_bandit": self.support_sampler.to_state(),
                        "top1_bandit": self.top1_sampler.to_state(),
                        "ema_bandit": self.activation_sampler.to_state(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self._dirty = False
        except Exception:
            pass

    def _make_config(self, retrieval_overrides: Dict[str, float], guard_overrides: Dict[str, float]) -> Dict[str, Any]:
        cfg = deepcopy(self.base_config)
        retr = cfg.setdefault("retrieval", {})
        retr.update(retrieval_overrides)
        guards = cfg.setdefault("guards", {})
        guards.update(guard_overrides)
        retr.setdefault("recency_half_life_days", float(self.base_config.get("retrieval", {}).get("recency_half_life_days", 14.0)))
        return cfg

    def prepare_query(self, question: str) -> Dict[str, Any]:
        norm_len = _clamp(len(question) / 256.0)
        token_count = len(question.split())
        punctuation = sum(1 for c in question if c in {"?", "!"})
        features = {
            "bias": 1.0,
            "len_norm": norm_len,
            "token_norm": _clamp(token_count / 80.0),
            "punctuation": _clamp(punctuation / 3.0),
            "digits": _clamp(sum(ch.isdigit() for ch in question) / max(1.0, len(question))),
        }
        alpha_dense = _clamp(self.weight_model.predict(features), 0.05, 0.95)
        beta_sparse = _clamp(1.0 - alpha_dense, 0.05, 0.95)

        threshold_features = {
            "bias": 1.0,
            "len_norm": norm_len,
            "tokens": _clamp(token_count / 60.0),
        }
        min_support = _clamp(self.threshold_model.predict(threshold_features), 0.05, 0.6)

        detail_level: Optional[str] = None
        llm_analysis: Optional[Dict[str, Any]] = None
        if question.strip():
            try:
                llm_analysis = try_call_llm_dict(
                    "rag_adaptive_controller",
                    input_payload={
                        "question": question,
                        "features": features,
                        "threshold_features": threshold_features,
                        "defaults": {
                            "alpha_dense": alpha_dense,
                            "beta_sparse": beta_sparse,
                            "min_support": min_support,
                            "activation": getattr(self.global_activation_tracker, "value", 0.5),
                        },
                    },
                    logger=logger,
                )
            except Exception:
                llm_analysis = None
        if llm_analysis:
            weights = llm_analysis.get("weights")
            if isinstance(weights, dict):
                dense = weights.get("dense")
                sparse = weights.get("sparse")
                try:
                    if dense is not None:
                        alpha_dense = _clamp(float(dense), 0.05, 0.95)
                    if sparse is not None:
                        beta_sparse = _clamp(float(sparse), 0.05, 0.95)
                    if dense is None and sparse is not None:
                        alpha_dense = _clamp(1.0 - beta_sparse, 0.05, 0.95)
                    elif sparse is None and dense is not None:
                        beta_sparse = _clamp(1.0 - alpha_dense, 0.05, 0.95)
                except (TypeError, ValueError):
                    logger.debug("LLM rag weights invalid: %r", weights)
            detail = llm_analysis.get("detail_level")
            if isinstance(detail, str) and detail.strip():
                detail_level = detail.strip().lower()

        half_life = float(
            self.half_life_sampler.ensure_choice(
                self.base_config.get("retrieval", {}).get("recency_half_life_days", 14.0)
            )
        )
        min_support_score = float(self.support_sampler.ensure_choice(min_support))
        min_top1_score = float(self.top1_sampler.ensure_choice(0.25))

        if detail_level:
            if detail_level in {"technique", "haut", "detaillé", "detaillé"}:
                min_support_score = _clamp(max(min_support_score, min_support + 0.05), 0.05, 0.95)
                min_top1_score = _clamp(min_top1_score + 0.05, 0.05, 0.95)
            elif detail_level in {"synthese", "bas", "résumé", "resume"}:
                min_support_score = _clamp(min_support_score * 0.85, 0.05, 0.95)
                min_top1_score = _clamp(min_top1_score * 0.9, 0.05, 0.95)

        retrieval_overrides = {
            "alpha_dense": alpha_dense,
            "beta_sparse": beta_sparse,
            "recency_half_life_days": half_life,
        }
        guard_overrides = {
            "min_support_score": min_support_score,
            "min_top1_score": min_top1_score,
        }
        cfg = self._make_config(retrieval_overrides, guard_overrides)
        self.last_config = deepcopy(cfg)
        context = {
            "timestamp": time.time(),
            "features": features,
            "threshold_features": threshold_features,
            "alpha_dense": alpha_dense,
            "beta_sparse": beta_sparse,
            "min_support_score": min_support_score,
            "min_top1_score": min_top1_score,
            "recency_half_life_days": half_life,
            "config": cfg,
        }
        if detail_level:
            context["llm_detail_level"] = detail_level
        if llm_analysis and llm_analysis.get("justification"):
            context["llm_justification"] = str(llm_analysis["justification"])
        return context

    def observe_outcome(self, context: Dict[str, Any], rag_out: Dict[str, Any]) -> None:
        diagnostics = rag_out.get("diagnostics", {}) if isinstance(rag_out, dict) else {}
        citations = rag_out.get("citations") or []
        top_scores = diagnostics.get("top_scores") or []
        top1 = float(top_scores[0]) if top_scores else 0.0
        coverage = _clamp(len(citations) / float(diagnostics.get("fused_hits", max(1, len(citations)))))
        status = rag_out.get("status")
        auto_reward = 0.1
        if status == "ok":
            auto_reward = 0.4 + 0.4 * _clamp(top1)
            auto_reward += 0.2 * coverage
        elif status == "refused":
            auto_reward = 0.05
        interaction = {
            "context": context,
            "rag_out": rag_out,
            "diagnostics": diagnostics,
            "auto_reward": _clamp(auto_reward),
            "pending": True,
        }
        self.pending_interactions.append(interaction)
        self._apply_online_updates(context, auto_reward)
        self._dirty = True
        self._save_state()

    def _apply_online_updates(self, context: Dict[str, Any], reward: float) -> None:
        reward = _clamp(reward)
        self.weight_model.update(context.get("features", {}), reward)
        self.threshold_model.update(context.get("threshold_features", {}), reward)
        self.half_life_sampler.update(reward)
        self.support_sampler.update(reward)
        self.top1_sampler.update(reward)

    def apply_feedback(self, reward: float, horizon: float = 45.0) -> None:
        if not self.pending_interactions:
            return
        now = time.time()
        reward = _clamp(reward)
        for interaction in reversed(self.pending_interactions):
            context = interaction.get("context", {})
            ts = float(context.get("timestamp", 0.0))
            if not interaction.get("pending"):
                continue
            if now - ts > horizon:
                interaction["pending"] = False
                continue
            combined = 0.6 * float(interaction.get("auto_reward", 0.3)) + 0.4 * reward
            self._apply_online_updates(context, combined)
            interaction["pending"] = False
            self._dirty = True
            break
        self._save_state()

    def current_config(self) -> Dict[str, Any]:
        return deepcopy(self.last_config)

    def update_global_activation(self, signal: float, reinforce: Optional[float] = None) -> float:
        value = self.global_activation_tracker.update(signal)
        if reinforce is not None:
            self.global_activation_tracker.reinforce(reinforce)
            self._dirty = True
            self._save_state()
        return value
