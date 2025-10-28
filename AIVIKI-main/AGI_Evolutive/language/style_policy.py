"""Language style policy definitions used by dialogue modules."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


def _scaled_reward(value: float) -> float:
    """Scale rewards from [-1, 1] to [0, 1] for bandit updates."""

    value = max(-1.0, min(1.0, float(value)))
    return (value + 1.0) * 0.5


@dataclass
class OnlineLinear:
    """Very small linear learner with bounded online updates."""

    step_size: float = 0.05
    max_step: float = 0.2
    weight_bounds: Tuple[float, float] = (-2.0, 2.0)
    weights: Dict[str, float] = field(default_factory=dict)

    def predict(self, features: Dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in features.items())

    def update(self, features: Dict[str, float], target: float) -> None:
        prediction = self.predict(features)
        error = target - prediction
        step = max(-self.max_step, min(self.max_step, self.step_size * error))
        for name, value in features.items():
            if value == 0:
                continue
            delta = step * value
            new_weight = self.weights.get(name, 0.0) + delta
            lo, hi = self.weight_bounds
            new_weight = max(lo, min(hi, new_weight))
            self.weights[name] = new_weight


@dataclass
class ThompsonBandit:
    """Discrete Thompson Sampling helper."""

    arms: Tuple[str, ...]
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    rng: random.Random = field(default_factory=random.Random)
    _stats: Dict[str, List[float]] = field(init=False)

    def __post_init__(self) -> None:
        self._stats = {arm: [self.prior_alpha, self.prior_beta] for arm in self.arms}

    def sample(self) -> str:
        draws = {
            arm: self.rng.betavariate(*self._stats[arm]) for arm in self.arms
        }
        choice = max(draws, key=draws.get)
        logger.debug("ThompsonBandit.sample", extra={"draws": draws, "choice": choice})
        return choice

    def update(self, arm: str, reward: float) -> None:
        if arm not in self._stats:
            return
        scaled = _scaled_reward(reward)
        stats = self._stats[arm]
        stats[0] += scaled
        stats[1] += 1.0 - scaled
        logger.debug(
            "ThompsonBandit.update",
            extra={"arm": arm, "scaled_reward": scaled, "stats": tuple(stats)},
        )


DEFAULT_MODES: Tuple[str, ...] = ("brief", "pedagogique", "audit")
DEFAULT_TTLS: Tuple[str, ...] = ("3", "7", "14", "30")


@dataclass
class StylePolicy:
    """Politique de style adaptable avec modes explicites."""

    params: Dict[str, float] = field(
        default_factory=lambda: {
            "politeness": 0.6,
            "directness": 0.5,
            "asking_rate": 0.4,
            "concretude_bias": 0.6,
            "hedging": 0.3,
            "warmth": 0.5,
            "verbosity": 0.6,
            "structure": 0.5,
        }
    )
    decay: float = 0.85
    current_mode: str = "pedagogique"
    persona_tone: str = "neutre"
    max_param_step: float = 0.25
    recent_rewards: List[Tuple[float, float]] = field(default_factory=list)
    param_drift_log: List[Tuple[str, float, float]] = field(default_factory=list)
    adaptive_models: Dict[str, OnlineLinear] = field(init=False)
    mode_bandit: ThompsonBandit = field(
        default_factory=lambda: ThompsonBandit(arms=DEFAULT_MODES)
    )
    ttl_bandit: ThompsonBandit = field(
        default_factory=lambda: ThompsonBandit(arms=DEFAULT_TTLS)
    )
    current_ttl: int = field(init=False)

    MODE_PRESETS: ClassVar[Dict[str, Dict[str, float]]] = {
        "brief": {
            "directness": 0.8,
            "verbosity": 0.3,
            "structure": 0.6,
            "asking_rate": 0.3,
        },
        "pedagogique": {
            "warmth": 0.7,
            "verbosity": 0.8,
            "structure": 0.75,
            "hedging": 0.35,
            "asking_rate": 0.5,
        },
        "audit": {
            "directness": 0.75,
            "hedging": 0.2,
            "asking_rate": 0.65,
            "structure": 0.8,
            "concretude_bias": 0.7,
        },
    }

    PERSONA_TONE_BIASES: ClassVar[Dict[str, Dict[str, float]]] = {
        "neutre": {},
        "chaleureux": {"warmth": 0.75, "politeness": 0.7},
        "professionnel": {"structure": 0.8, "directness": 0.6},
        "coach": {"warmth": 0.65, "asking_rate": 0.55},
    }

    MODE_KEYWORDS: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "brief": ("mode bref", "mode synthétique", "sois bref", "fais court", "résume"),
        "pedagogique": ("mode pédagogique", "explique", "détaillé", "prends le temps"),
        "audit": ("mode audit", "challenge-moi", "questionne", "fais un audit"),
    }

    MACRO_DELTAS: ClassVar[Dict[str, Dict[str, float]]] = {
        "taquin": {"warmth": 0.12, "politeness": -0.05, "hedging": -0.05},
        "coach": {"warmth": 0.15, "directness": 0.08, "verbosity": 0.1},
        "sobre": {"emoji": -0.2, "warmth": -0.08, "structure": 0.05},
        "deadpan": {"hedging": -0.08, "politeness": -0.04},
    }
    STYLE_MACROS: ClassVar[Dict[str, Dict[str, float]]] = {
        "taquin": {"warmth": +0.10, "directness": +0.10, "hedging": -0.10},
        "coach": {"warmth": +0.10, "asking_rate": +0.15},
        "sobre": {"structure": +0.10, "hedging": -0.05},
        "deadpan": {"warmth": -0.15, "structure": +0.10},
    }

    def __post_init__(self) -> None:
        self.adaptive_models = {}
        for param, coeff in self._default_reward_coeffs().items():
            model = OnlineLinear()
            model.weights["bias"] = coeff
            self.adaptive_models[param] = model
        self._apply_presets()
        self.current_ttl = int(self.ttl_bandit.sample())
        self.decay = self._ttl_to_decay(self.current_ttl)
        logger.debug(
            "StylePolicy.init",
            extra={
                "mode": self.current_mode,
                "persona": self.persona_tone,
                "decay": self.decay,
                "ttl": self.current_ttl,
            },
        )

    def set_mode(self, mode: str, persona_tone: Optional[str] = None) -> None:
        mode = (mode or "").lower()
        if mode not in self.MODE_PRESETS:
            return
        if persona_tone:
            self.persona_tone = persona_tone
        self.current_mode = mode
        self._apply_presets()
        logger.debug(
            "StylePolicy.set_mode",
            extra={"mode": self.current_mode, "persona": self.persona_tone},
        )

    def update_persona_tone(self, tone: str) -> None:
        if tone:
            self.persona_tone = tone.lower()
            self._apply_presets()
            logger.debug(
                "StylePolicy.update_persona_tone",
                extra={"tone": self.persona_tone, "mode": self.current_mode},
            )

    def apply_macro(self, macro: str) -> None:
        key = (macro or "").lower()
        deltas = self.MACRO_DELTAS.get(key)
        if not deltas:
            deltas = self.STYLE_MACROS.get(key, {})
        if not deltas:
            return
        for param, delta in deltas.items():
            base = self.params.get(param, 0.5)
            target = base + delta
            self._bounded_update(param, target, apply_decay=True)
        logger.debug("StylePolicy.apply_macro", extra={"macro": key, "deltas": deltas})

    def adapt_from_instruction(self, text: str) -> Dict[str, float]:
        """Infère des deltas de style depuis des instructions libres."""

        t = (text or "").lower()
        hints: Dict[str, float] = {}

        llm_response = try_call_llm_dict(
            "style_policy",
            input_payload={
                "instruction": text,
                "mode": self.current_mode,
                "persona_tone": self.persona_tone,
                "current": {
                    "chaleur": float(self.params.get("warmth", 0.5)),
                    "directivite": float(self.params.get("directness", 0.5)),
                    "questionnement": float(self.params.get("asking_rate", 0.4)),
                },
            },
            logger=logger,
        )
        if llm_response and isinstance(llm_response.get("directives"), dict):
            directives = llm_response["directives"]
            chaleur = directives.get("chaleur")
            directivite = directives.get("directivite")
            questionnement = directives.get("questionnement")
            try:
                if chaleur is not None:
                    hints["warmth"] = self._clip(float(chaleur))
                if directivite is not None:
                    hints["directness"] = self._clip(float(directivite))
                if questionnement is not None:
                    hints["asking_rate"] = self._clip(float(questionnement))
            except (TypeError, ValueError):
                logger.debug("StylePolicy LLM directives invalid: %r", directives)

        maybe_mode = self._detect_mode_keyword(t)
        if maybe_mode:
            self.set_mode(maybe_mode, persona_tone=self.persona_tone)

        warm_cues = ["bienveill", "empath", "chaleur", "gentil", "compréhens", "soigneux", "attentionné"]
        if any(c in t for c in warm_cues):
            hints["warmth"] = self._clip(self.params.get("warmth", 0.5) + 0.2)

        if any(k in t for k in ["prudent", "nuancé", "mesuré", "avec précaution"]):
            hints["hedging"] = self._clip(self.params.get("hedging", 0.3) + 0.2)
        if any(k in t for k in ["direct", "cash", "franc", "sans détour", "tranché"]):
            hints["hedging"] = self._clip(self.params.get("hedging", 0.3) - 0.2)

        if any(k in t for k in ["concret", "exemples", "pratique", "spécifique"]):
            hints["concretude_bias"] = self._clip(self.params.get("concretude_bias", 0.6) + 0.2)
        if any(k in t for k in ["haut niveau", "vue d'ensemble", "général"]):
            hints["concretude_bias"] = self._clip(self.params.get("concretude_bias", 0.6) - 0.2)

        if any(k in t for k in ["pose des questions", "questionne", "clarifie"]):
            hints["asking_rate"] = self._clip(self.params.get("asking_rate", 0.4) + 0.2)

        if any(k in t for k in ["structure", "plan", "étapes"]):
            hints["structure"] = self._clip(self.params.get("structure", 0.5) + 0.15)

        if hints:
            logger.debug(
                "StylePolicy.adapt_from_instruction",
                extra={"text": text, "hints": hints, "llm": bool(llm_response)},
            )

        return hints

    def as_dict(self) -> Dict[str, float]:
        payload = dict(self.params)
        payload["mode"] = self.current_mode
        payload["persona_tone"] = self.persona_tone
        return payload

    def update_from_reward(self, reward: float) -> None:
        reward = float(max(-1.0, min(1.0, reward)))
        timestamp = time.time()
        self.recent_rewards.append((timestamp, reward))
        if len(self.recent_rewards) > 200:
            self.recent_rewards.pop(0)

        features = self._build_context_features()

        for param, model in self.adaptive_models.items():
            coeff = model.predict(features)
            current_value = self.params.get(param, 0.5)
            delta = coeff * reward
            target = current_value + delta
            self._bounded_update(param, target, apply_decay=True)
            applied_delta = self.params.get(param, 0.5) - current_value
            if abs(reward) >= 1e-6:
                target_coeff = applied_delta / reward
                model.update(features, target_coeff)

        self.mode_bandit.update(self.current_mode, reward)
        self.ttl_bandit.update(str(self.current_ttl), reward)
        self.current_ttl = int(self.ttl_bandit.sample())
        self.decay = self._ttl_to_decay(self.current_ttl)

        logger.debug(
            "StylePolicy.update_from_reward",
            extra={
                "reward": reward,
                "mode": self.current_mode,
                "ttl": self.current_ttl,
                "decay": self.decay,
            },
        )

    def detect_mode_command(self, text: str) -> Optional[str]:
        """Détecte une commande explicite de changement de mode."""

        t = (text or "").strip().lower()
        if not t:
            return None
        if t.startswith("/mode"):
            parts = t.split()
            if len(parts) >= 2:
                candidate = parts[1]
                return candidate if candidate in self.MODE_PRESETS else None
        return self._detect_mode_keyword(t)

    def _detect_mode_keyword(self, text: str) -> Optional[str]:
        for mode, keywords in self.MODE_KEYWORDS.items():
            if any(k in text for k in keywords):
                return mode
        return None

    def _apply_presets(self, use_decay: bool = False) -> None:
        base = {
            "politeness": 0.6,
            "directness": 0.5,
            "asking_rate": 0.4,
            "concretude_bias": 0.6,
            "hedging": 0.3,
            "warmth": 0.5,
            "verbosity": 0.6,
            "structure": 0.5,
        }
        mode_overrides = self.MODE_PRESETS.get(self.current_mode, {})
        persona_overrides = self.PERSONA_TONE_BIASES.get(self.persona_tone, {})
        for key, value in base.items():
            if use_decay:
                self._bounded_update(key, value, apply_decay=True)
            else:
                old_value = self.params.get(key, value)
                self.params[key] = value
                if abs(old_value - value) > 1e-6:
                    self._log_drift(key, old_value, value)
        for overrides in (persona_overrides, mode_overrides):
            for key, value in overrides.items():
                target = self._clip(value)
                if use_decay:
                    self._bounded_update(key, target, apply_decay=True)
                else:
                    old_value = self.params.get(key, target)
                    self.params[key] = target
                    if abs(old_value - target) > 1e-6:
                        self._log_drift(key, old_value, target)

    def recommend_mode(self) -> str:
        """Propose un mode en se basant sur le bandit discret."""

        recommendation = self.mode_bandit.sample()
        logger.debug(
            "StylePolicy.recommend_mode",
            extra={"recommended": recommendation, "current": self.current_mode},
        )
        return recommendation

    @staticmethod
    def _clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, val))

    def _bounded_update(self, param: str, target: float, apply_decay: bool) -> None:
        current = self.params.get(param, 0.5)
        if apply_decay:
            target = current * self.decay + (1.0 - self.decay) * target
        delta = target - current
        if abs(delta) > self.max_param_step:
            target = current + math.copysign(self.max_param_step, delta)
        new_value = self._clip(target)
        if abs(new_value - current) <= 1e-6:
            return
        self.params[param] = new_value
        self._log_drift(param, current, new_value)

    def _log_drift(self, param: str, old_value: float, new_value: float) -> None:
        entry = (param, old_value, new_value)
        self.param_drift_log.append(entry)
        if len(self.param_drift_log) > 200:
            self.param_drift_log.pop(0)
        logger.debug(
            "StylePolicy.param_update",
            extra={"param": param, "old": old_value, "new": new_value},
        )

    def _build_context_features(self) -> Dict[str, float]:
        features: Dict[str, float] = {"bias": 1.0}
        features[f"mode::{self.current_mode}"] = 1.0
        features[f"persona::{self.persona_tone}"] = 1.0
        for key, value in self.params.items():
            features[f"param::{key}"] = value
        return features

    @staticmethod
    def _ttl_to_decay(ttl: int) -> float:
        ttl = max(1, int(ttl))
        return math.pow(0.5, 1.0 / ttl)

    @staticmethod
    def _default_reward_coeffs() -> Dict[str, float]:
        return {
            "concretude_bias": 0.1,
            "hedging": -0.05,
            "warmth": 0.08,
            "politeness": 0.06,
            "directness": 0.05,
        }
