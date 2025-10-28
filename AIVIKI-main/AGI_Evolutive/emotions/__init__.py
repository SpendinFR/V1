# emotions/__init__.py
"""
Système Émotionnel Avancé de l'AGI Évolutive
Émotions de base, humeurs, évaluations affectives et apprentissage émotionnel intégrés
"""

import json
import logging
import math
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .emotion_engine import EmotionEngine
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


def _truncate_text(text: str, limit: int = 512) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _simplify_for_llm(value: Any, *, depth: int = 0, max_depth: int = 2) -> Any:
    if depth >= max_depth:
        return _truncate_text(str(value), 160)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value, 160)
    if isinstance(value, Mapping):
        simplified: Dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= 6:
                break
            simplified[str(key)[:48]] = _simplify_for_llm(val, depth=depth + 1, max_depth=max_depth)
        return simplified
    if isinstance(value, (list, tuple, set)):
        return [
            _simplify_for_llm(item, depth=depth + 1, max_depth=max_depth)
            for item in list(value)[:6]
        ]
    return _truncate_text(str(value), 160)


def _prepare_llm_context(context: Mapping[str, Any], *, max_items: int = 12) -> Dict[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    prepared: Dict[str, Any] = {}
    for idx, (key, value) in enumerate(context.items()):
        if idx >= max_items:
            break
        prepared[str(key)[:48]] = _simplify_for_llm(value)
    return prepared


class BoundedOnlineLinear:
    """Régression linéaire en ligne bornée pour l'ajustement d'hyperparamètres."""

    def __init__(
        self,
        feature_names: Optional[Set[str]] = None,
        *,
        bounds: Tuple[float, float] = (0.0, 1.0),
        lr: float = 0.05,
        base_value: Optional[float] = None,
    ) -> None:
        self.feature_names = set(feature_names or [])
        self.bounds = bounds
        self.lr = float(lr)
        lo, hi = self.bounds
        self.base_value = base_value if base_value is not None else (lo + hi) / 2.0
        self.weights: Dict[str, float] = {name: 0.0 for name in self.feature_names}

    def predict(self, features: Dict[str, float]) -> float:
        total = self.base_value
        for name, value in features.items():
            if name not in self.weights:
                # Les nouvelles caractéristiques démarrent neutres
                self.weights[name] = 0.0
            total += self.weights[name] * float(value)
        return max(self.bounds[0], min(self.bounds[1], total))

    def update(self, features: Dict[str, float], target: float) -> Tuple[float, float]:
        target_clipped = max(self.bounds[0], min(self.bounds[1], float(target)))
        prediction = self.predict(features)
        error = target_clipped - prediction
        scaled_error = self.lr * error

        for name, value in features.items():
            if name not in self.weights:
                self.weights[name] = 0.0
            self.weights[name] += scaled_error * float(value)
            # Les poids restent bornés pour éviter les dérives
            self.weights[name] = max(-1.0, min(1.0, self.weights[name]))

        # Léger ajustement de la base vers la cible
        self.base_value += scaled_error * 0.1
        self.base_value = max(self.bounds[0], min(self.bounds[1], self.base_value))

        return prediction, error


class DiscreteThompsonSampler:
    """Thompson Sampling sur un ensemble discret de bras."""

    def __init__(self, arms: List[Any], prior: Tuple[float, float] = (1.0, 1.0)) -> None:
        if not arms:
            raise ValueError("At least one arm is required for Thompson Sampling")
        self.arms = list(arms)
        self.stats: Dict[Any, List[float]] = {
            arm: [float(prior[0]), float(prior[1])] for arm in self.arms
        }

    def sample(self) -> Any:
        draws = {}
        for arm, (alpha, beta) in self.stats.items():
            alpha = max(alpha, 1e-3)
            beta = max(beta, 1e-3)
            draws[arm] = np.random.beta(alpha, beta)
        # Sélectionne le bras avec le tirage le plus élevé
        return max(draws.items(), key=lambda item: item[1])[0]

    def update(self, arm: Any, success: float) -> None:
        if arm not in self.stats:
            return
        alpha, beta = self.stats[arm]
        success_clamped = max(0.0, min(1.0, float(success)))
        alpha += success_clamped
        beta += 1.0 - success_clamped
        self.stats[arm] = [alpha, beta]


class AdaptiveEMARegressor:
    """EMA adaptative guidée par Thompson Sampling et corrélation récompense."""

    def __init__(self, betas: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)) -> None:
        self.betas = tuple(sorted(betas))
        self.sampler = DiscreteThompsonSampler(list(self.betas))
        self.state: Dict[float, float] = {beta: None for beta in self.betas}
        self.current_beta = self.betas[0]
        self.history: deque = deque(maxlen=256)
        self.correlation: float = 0.0

    def _compute_correlation(self) -> float:
        if len(self.history) < 6:
            return self.correlation
        smoothed = np.array([item[0] for item in self.history])
        target = np.array([item[1] for item in self.history])
        if np.allclose(smoothed.std(), 0.0) or np.allclose(target.std(), 0.0):
            return self.correlation
        corr = float(np.corrcoef(smoothed, target)[0, 1])
        self.correlation = corr
        return corr

    def update(self, signal: float, target: Optional[float] = None) -> Dict[str, float]:
        if target is None:
            target = float(signal)

        beta = self.sampler.sample()
        previous = self.state.get(beta)
        if previous is None:
            smoothed = float(signal)
        else:
            smoothed = beta * float(signal) + (1.0 - beta) * float(previous)
        self.state[beta] = smoothed

        self.history.append((smoothed, float(target)))
        previous_corr = self.correlation
        new_corr = self._compute_correlation()
        improvement = 1.0 if new_corr >= previous_corr else 0.0
        self.sampler.update(beta, improvement)
        self.current_beta = beta

        return {
            "smoothed": smoothed,
            "beta": beta,
            "correlation": new_corr,
            "improved": improvement,
        }


class EmotionDriftDetector:
    """Détection simple de dérive sur des fenêtres glissantes."""

    def __init__(self, window: int = 40, threshold: float = 0.15) -> None:
        self.window = max(5, int(window))
        self.threshold = float(threshold)
        self.buffer: deque = deque(maxlen=self.window)
        self.reference: Optional[float] = None

    def update(self, value: float) -> Optional[Dict[str, float]]:
        self.buffer.append(float(value))
        if len(self.buffer) < self.buffer.maxlen:
            return None

        current_mean = float(np.mean(self.buffer))
        if self.reference is None:
            self.reference = current_mean
            return None

        drift = abs(current_mean - self.reference)
        if drift >= self.threshold:
            event = {
                "from": self.reference,
                "to": current_mean,
                "delta": drift,
            }
            self.reference = current_mean
            return event

        # Mise à jour progressive de la référence pour suivre la tendance lente
        self.reference = self.reference * 0.9 + current_mean * 0.1
        return None


class EmotionalHyperAdaptationManager:
    """Orchestration hiérarchique de l'adaptation des hyperparamètres émotionnels."""

    def __init__(self) -> None:
        self.update_rate_model = BoundedOnlineLinear(bounds=(0.05, 0.55), lr=0.04)
        self.mood_inertia_model = BoundedOnlineLinear(bounds=(0.3, 0.9), lr=0.03)
        self.regulation_model = BoundedOnlineLinear(bounds=(0.5, 0.85), lr=0.02)
        # Couples (multiplicateur_update_rate, multiplicateur_decay)
        self.combo_sampler = DiscreteThompsonSampler(
            [(1.0, 1.0), (0.8, 1.1), (1.15, 0.9), (1.0, 1.2)]
        )
        self.ema = AdaptiveEMARegressor()
        self.last_quality: Optional[float] = None
        self.last_context: Dict[str, float] = {}
        self._last_combo: Tuple[float, float] = (1.0, 1.0)

    def suggest(self, features: Dict[str, float]) -> Dict[str, float]:
        features_with_bias = dict(features)
        features_with_bias.setdefault("bias", 1.0)

        rate = self.update_rate_model.predict(features_with_bias)
        inertia = self.mood_inertia_model.predict(features_with_bias)
        regulation = self.regulation_model.predict(features_with_bias)
        mult_rate, mult_decay = self.combo_sampler.sample()
        self._last_combo = (mult_rate, mult_decay)

        return {
            "update_rate": max(0.05, min(0.65, rate * mult_rate)),
            "decay_multiplier": max(0.5, min(1.5, mult_decay)),
            "mood_inertia": inertia,
            "regulation_threshold": regulation,
        }

    def observe(
        self,
        features: Dict[str, float],
        *,
        quality: float,
        smoothing_quality: float,
        combo_success: float,
        regulation_effectiveness: float,
    ) -> None:
        features_with_bias = dict(features)
        features_with_bias.setdefault("bias", 1.0)

        quality_clamped = max(0.0, min(1.0, quality))
        smoothing_score = max(-1.0, min(1.0, smoothing_quality))
        regulation_score = max(0.0, min(1.0, regulation_effectiveness))

        self.update_rate_model.update(features_with_bias, quality_clamped)
        self.mood_inertia_model.update(features_with_bias, 0.5 + 0.4 * smoothing_score)
        self.regulation_model.update(features_with_bias, regulation_score)
        self.combo_sampler.update(self._last_combo, combo_success)

        self.last_quality = quality_clamped
        self.last_context = dict(features)


class EmotionalState(Enum):
    """États émotionnels fondamentaux"""
    JOY = "joie"
    SADNESS = "tristesse"
    ANGER = "colère"
    FEAR = "peur"
    DISGUST = "dégoût"
    SURPRISE = "surprise"
    ANTICIPATION = "anticipation"
    TRUST = "confiance"
    NEUTRAL = "neutre"
    PRIDE = "fierté"            
    FRUSTRATION = "frustration" 

class MoodState(Enum):
    """États d'humeur persistants"""
    EUPHORIC = "euphorique"
    CONTENT = "content"
    CALM = "calme"
    NEUTRAL = "neutre"
    MELANCHOLIC = "mélancolique"
    IRRITABLE = "irritable"
    ANXIOUS = "anxieux"
    DEPRESSED = "déprimé"

class EmotionalIntensity(Enum):
    """Niveaux d'intensité émotionnelle"""
    FAINT = "faible"      # 0.0-0.2
    MILD = "léger"        # 0.2-0.4
    MODERATE = "modéré"   # 0.4-0.6
    STRONG = "fort"       # 0.6-0.8
    INTENSE = "intense"   # 0.8-1.0

@dataclass
class EmotionalExperience:
    """Expérience émotionnelle complète"""
    timestamp: float
    primary_emotion: EmotionalState
    secondary_emotions: List[Tuple[EmotionalState, float]]
    intensity: float
    valence: float           # Plaisir/déplaisir: -1.0 à 1.0
    arousal: float          # Activation: 0.0 à 1.0
    dominance: float        # Contrôle: 0.0 à 1.0
    trigger: str
    duration: float
    bodily_sensations: List[str]
    cognitive_appraisals: List[str]
    action_tendencies: List[str]
    expression: str
    regulation_strategy: Optional[str] = None

@dataclass
class Mood:
    """État d'humeur persistant"""
    mood_type: MoodState
    intensity: float
    stability: float        # Stabilité de l'humeur (0.0-1.0)
    duration: float         # Durée depuis le début de l'humeur (secondes)
    influencing_factors: Dict[str, float]  # Facteurs influençant l'humeur

@dataclass
class EmotionalMemory:
    """Mémoire émotionnelle d'un événement"""
    event_id: str
    emotional_signature: List[Tuple[EmotionalState, float]]  # Pattern émotionnel
    learned_association: Dict[str, float]  # Associations stimulus-réponse
    predictive_value: float  # Valeur prédictive pour le futur
    last_activated: float

class EmotionalSystem:
    """
    Système émotionnel biologique inspiré - Modèle OCC (Ortony, Clore, Collins)
    Intègre émotions discrètes, dimensions affectives et humeurs
    """
    
    def __init__(self, cognitive_architecture=None, memory_system=None, metacognitive_system=None):
        self.cognitive_architecture = cognitive_architecture
        self.memory_system = memory_system
        self.metacognitive_system = metacognitive_system
        self.creation_time = time.time()

        # --- LIAISONS INTER-MODULES ---
        if self.cognitive_architecture is not None:
            self.goals = getattr(self.cognitive_architecture, "goals", None)
            self.learning = getattr(self.cognitive_architecture, "learning", None)
            self.creativity = getattr(self.cognitive_architecture, "creativity", None)
            self.reasoning = getattr(self.cognitive_architecture, "reasoning", None)
            self.perception = getattr(self.cognitive_architecture, "perception", None)
            self.language = getattr(self.cognitive_architecture, "language", None)

        self.engine = EmotionEngine()
        self.engine.bind(
            arch=self.cognitive_architecture,
            memory=self.memory_system,
            metacog=self.metacognitive_system,
            goals=getattr(self.cognitive_architecture, "goals", None) if self.cognitive_architecture else None,
            language=getattr(self.cognitive_architecture, "language", None) if self.cognitive_architecture else None,
            evolution=getattr(self.cognitive_architecture, "evolution", None) if self.cognitive_architecture else None,
        )

        self._last_llm_appraisal: Optional[Dict[str, Any]] = None

        
        # === ÉTATS ÉMOTIONNELS ACTUELS ===
        self.current_emotions = {
            emotion: 0.0 for emotion in EmotionalState
        }
        self.emotional_intensity = 0.0
        self.emotional_balance = 0.5  # Équilibre émotionnel global
        
        # === DIMENSIONS AFFECTIVES (Modèle PAD) ===
        self.affective_dimensions = {
            "pleasure_arousal_dominance": {
                "pleasure": 0.0,      # Plaisir: -1.0 (déplaisant) à 1.0 (plaisant)
                "arousal": 0.5,       # Activation: 0.0 (calme) à 1.0 (excité)
                "dominance": 0.5      # Contrôle: 0.0 (soumis) à 1.0 (dominant)
            },
            "valence_energy": {
                "valence": 0.0,       # Valence affective globale
                "energy": 0.5,        # Niveau d'énergie
                "tension": 0.3        # Tension émotionnelle
            }
        }
        
        # === SYSTÈME D'HUMEUR ===
        self.mood_system = {
            "current_mood": Mood(MoodState.NEUTRAL, 0.3, 0.8, 0.0, {}),
            "mood_history": deque(maxlen=100),
            "mood_cycle": {
                "daily_rhythm": defaultdict(float),
                "weekly_pattern": defaultdict(float),
                "seasonal_variation": 0.0
            },
            "mood_regulators": {
                "baseline": 0.5,
                "resilience": 0.7,    # Résilience émotionnelle
                "reactivity": 0.5     # Réactivité aux stimuli
            }
        }
        
        # === PROCESSUS D'ÉVALUATION ÉMOTIONNELLE ===
        self.appraisal_system = {
            "relevance_detector": RelevanceDetector(),
            "goal_congruence_assessor": GoalCongruenceAssessor(),
            "coping_potential_evaluator": CopingPotentialEvaluator(),
            "norm_compatibility_checker": NormCompatibilityChecker(),
            "self_implication_assessor": SelfImplicationAssessor()
        }
        
        # === RÉPERTOIRE DES RÉACTIONS ÉMOTIONNELLES ===
        self.emotional_repertoire = {
            "basic_emotions": self._initialize_basic_emotions(),
            "complex_emotions": self._initialize_complex_emotions(),
            "emotional_triggers": self._initialize_emotional_triggers(),
            "coping_strategies": self._initialize_coping_strategies()
        }
        
        # === MÉMOIRE ÉMOTIONNELLE ===
        self.emotional_memory = {
            "emotional_episodes": deque(maxlen=1000),
            "affective_associations": {},
            "emotional_conditioning": EmotionalConditioningSystem(),
            "mood_congruent_memory": MoodCongruentMemory(),
            "adaptation_signatures": deque(maxlen=200)
        }
        
        # === RÉGULATION ÉMOTIONNELLE ===
        self.emotion_regulation = {
            "strategies": {
                "reappraisal": 0.6,       # Réévaluation cognitive
                "attention_deployment": 0.5,  # Déploiement de l'attention
                "response_modulation": 0.4,   # Modulation de la réponse
                "situation_selection": 0.3    # Sélection de situation
            },
            "effectiveness": 0.5,
            "automatic_regulation": True,
            "regulation_threshold": 0.7   # Seuil de déclenchement de la régulation
        }
        
        # === EXPRESSION ÉMOTIONNELLE ===
        self.expression_system = {
            "facial_expressions": FacialExpressionGenerator(),
            "vocal_tones": VocalToneGenerator(),
            "body_language": BodyLanguageGenerator(),
            "verbal_expressions": VerbalExpressionGenerator(),
            "expression_intensity": 0.7,
            "expression_authenticity": 0.8
        }
        
        # === APPRENTISSAGE ÉMOTIONNEL ===
        self.emotional_learning = {
            "conditioning_history": deque(maxlen=500),
            "emotional_intelligence": 0.3,
            "empathy_capacity": 0.4,
            "social_emotional_skills": 0.3
        }
        
        # === INFLUENCES PHYSIOLOGIQUES ===
        self.physiological_influences = {
            "energy_level": 0.7,
            "stress_level": 0.3,
            "circadian_rhythm": 0.5,
            "health_status": 0.8
        }
        
        # === HISTORIQUE ÉMOTIONNEL ===
        self.emotional_history = {
            "emotional_experiences": deque(maxlen=2000),
            "intensity_peaks": deque(maxlen=100),
            "mood_transitions": deque(maxlen=200),
            "regulation_attempts": deque(maxlen=500),
            "learning_episodes": deque(maxlen=300),
            "drift_events": deque(maxlen=200)
        }
        self.auto_affective_traces: deque = deque(maxlen=150)

        # === PARAMÈTRES DE FONCTIONNEMENT ===
        self.operational_parameters = {
            "emotional_granularity": 0.6,     # Niveau de détail des expériences émotionnelles
            "affective_reactivity": 0.5,      # Rapidité de réaction émotionnelle
            "mood_inertia": 0.6,              # Résistance au changement d'humeur
            "emotional_depth": 0.4,           # Profondeur des expériences émotionnelles
            "expression_freedom": 0.7         # Liberté d'expression émotionnelle
        }
        
        # === THREADS DE TRAITEMENT ===
        self.processing_threads = {}
        self.running = True

        # Adaptation hiérarchique et détection de dérive
        self.hyper_adaptation_manager = EmotionalHyperAdaptationManager()
        self.drift_detectors = {
            "intensity": EmotionDriftDetector(window=40, threshold=0.12),
            "valence": EmotionDriftDetector(window=40, threshold=0.15),
            "mood": EmotionDriftDetector(window=60, threshold=0.10),
        }
        self._adaptive_context = {
            "last_balance": 0.5,
            "last_reward": 0.5,
            "last_features": {},
            "decay_multiplier": 1.0,
        }

        # Initialisation des systèmes
        self._initialize_emotional_system()

        print("💖 Système Émotionnel Initialisé")

    def step(self):
        try:
            self.engine.step()
        except Exception as e:
            print(f"[emotion] step error: {e}")

    def get_affect_state(self):
        return self.engine.get_state()

    def get_emotional_modulators(self):
        return self.engine.get_modulators()

    def register_emotion_event(self, kind, **kw):
        self.engine.register_event(kind, **kw)

    def on_auto_intention_promoted(
        self,
        event: Mapping[str, Any],
        evaluation: Optional[Mapping[str, Any]] = None,
        self_assessment: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(event, Mapping):
            return
        keywords = {str(k).lower() for k in event.get("keywords", []) if k}
        significance = float((evaluation or {}).get("significance", 0.5) or 0.5)
        if keywords.intersection({"empathy", "relation", "relationship", "care"}):
            self.emotional_learning["empathy_capacity"] = min(
                1.0, self.emotional_learning.get("empathy_capacity", 0.4) + 0.05 * significance
            )
        if keywords.intersection({"trust", "bond", "connection"}):
            self.emotional_repertoire.setdefault("emotional_triggers", {})
            self.emotional_repertoire["emotional_triggers"]["auto_relation"] = {
                "keywords": list(keywords),
                "weight": 0.3 + 0.4 * significance,
            }
        trace = {
            "ts": time.time(),
            "action_type": event.get("action_type"),
            "valence": self.affective_dimensions["valence_energy"].get("valence", 0.0),
            "score": (evaluation or {}).get("score"),
        }
        self.auto_affective_traces.append(trace)
        if self_assessment and isinstance(self_assessment, Mapping):
            reward_hint = float(self_assessment.get("composite_target", 0.6) or 0.6)
            self.affective_dimensions["valence_energy"]["valence"] = max(
                -1.0,
                min(1.0, self.affective_dimensions["valence_energy"].get("valence", 0.0) + 0.1 * (reward_hint - 0.5)),
            )

    def _initialize_emotional_system(self):
        """Initialise le système émotionnel avec des capacités de base"""

        # Émotions innées de base
        innate_emotions = {
            EmotionalState.JOY: 0.1,
            EmotionalState.TRUST: 0.1,
            EmotionalState.FEAR: 0.05,
            EmotionalState.SURPRISE: 0.1,
            EmotionalState.NEUTRAL: 0.65
        }
        
        self.current_emotions.update(innate_emotions)
        
        # Démarrage des processus d'arrière-plan
        self._start_emotional_monitoring()
        self._start_mood_updater()
        self._start_physiological_simulator()
        
        # Évaluation initiale de l'état émotionnel
        self._perform_initial_emotional_assessment()

    def _estimate_reward_signal(self, experience: EmotionalExperience) -> float:
        """Estime un signal de récompense/qualité à partir de l'expérience."""
        valence_score = (experience.valence + 1.0) / 2.0  # -> [0,1]
        dominance_score = max(0.0, min(1.0, experience.dominance))
        arousal_score = 1.0 - abs(max(0.0, min(1.0, experience.arousal)) - 0.5) * 2.0
        stress = self.physiological_influences.get("stress_level", 0.3)
        stress_penalty = max(0.0, min(1.0, 1.0 - stress))
        return max(
            0.0,
            min(
                1.0,
                0.45 * valence_score
                + 0.25 * dominance_score
                + 0.2 * stress_penalty
                + 0.1 * arousal_score,
            ),
        )

    def _recent_emotional_statistics(self, window: int = 20) -> Dict[str, float]:
        """Calcule des statistiques sur les expériences récentes."""
        experiences = list(self.emotional_history["emotional_experiences"])[-window:]
        if not experiences:
            return {
                "avg_valence": 0.0,
                "avg_arousal": 0.5,
                "avg_intensity": 0.2,
                "positive_ratio": 0.5,
            }

        valences = np.array([exp.valence for exp in experiences])
        arousals = np.array([exp.arousal for exp in experiences])
        intensities = np.array([exp.intensity for exp in experiences])
        positives = float(np.sum(valences > 0)) / len(experiences)
        return {
            "avg_valence": float(np.mean(valences)),
            "avg_arousal": float(np.mean(arousals)),
            "avg_intensity": float(np.mean(intensities)),
            "positive_ratio": positives,
        }

    def _build_adaptation_features(
        self,
        experience: EmotionalExperience,
        reward: float,
        smoothing: Dict[str, float],
    ) -> Dict[str, float]:
        stats = self._recent_emotional_statistics()
        current_mood = self.mood_system["current_mood"]
        features = {
            "valence": experience.valence,
            "arousal": experience.arousal,
            "dominance": experience.dominance,
            "intensity": experience.intensity,
            "reward": reward,
            "mood_intensity": current_mood.intensity,
            "mood_stability": current_mood.stability,
            "balance": self.emotional_balance,
            "energy": self.physiological_influences.get("energy_level", 0.5),
            "stress": self.physiological_influences.get("stress_level", 0.3),
            "avg_valence": stats["avg_valence"],
            "avg_arousal": stats["avg_arousal"],
            "avg_intensity": stats["avg_intensity"],
            "positive_ratio": stats["positive_ratio"],
            "smoothing_beta": smoothing.get("beta", 0.4),
            "smoothing_corr": smoothing.get("correlation", 0.0),
        }
        if self.emotional_memory["adaptation_signatures"]:
            last_signature = self.emotional_memory["adaptation_signatures"][-1]
            features["last_update_rate"] = last_signature.get("update_rate", 0.3)
            features["last_reward"] = last_signature.get("reward", 0.5)
        else:
            features["last_update_rate"] = 0.3
            features["last_reward"] = self._adaptive_context.get("last_reward", 0.5)
        return features

    def _hyper_adaptation_step(self, experience: EmotionalExperience) -> Dict[str, Any]:
        reward = self._estimate_reward_signal(experience)
        smoothing = self.hyper_adaptation_manager.ema.update(experience.valence, reward)
        features = self._build_adaptation_features(experience, reward, smoothing)
        suggestions = self.hyper_adaptation_manager.suggest(features)

        # Application directe de certains paramètres
        self.operational_parameters["mood_inertia"] = suggestions["mood_inertia"]
        self.emotion_regulation["regulation_threshold"] = suggestions["regulation_threshold"]
        self._adaptive_context["last_features"] = features
        self._adaptive_context["last_reward"] = reward
        self._adaptive_context["decay_multiplier"] = suggestions["decay_multiplier"]

        adaptation_record = {
            "timestamp": time.time(),
            "reward": reward,
            "beta": smoothing.get("beta"),
            "correlation": smoothing.get("correlation"),
            "update_rate": suggestions["update_rate"],
            "decay_multiplier": suggestions["decay_multiplier"],
            "regulation_threshold": suggestions["regulation_threshold"],
        }
        self.emotional_memory["adaptation_signatures"].append(adaptation_record)
        if self.memory_system and hasattr(self.memory_system, "register_emotional_adaptation"):
            try:
                self.memory_system.register_emotional_adaptation(adaptation_record)
            except Exception as exc:
                print(f"[emotion] adaptation memory hook failed: {exc}")

        return {
            "update_rate": suggestions["update_rate"],
            "decay_multiplier": suggestions["decay_multiplier"],
            "features": features,
            "reward": reward,
            "smoothing": smoothing,
        }

    def _record_adaptation_feedback(
        self,
        adaptation: Dict[str, Any],
        combo_success: float,
    ) -> None:
        smoothing = adaptation.get("smoothing", {})
        features = adaptation.get("features", {})
        reward = adaptation.get("reward", 0.5)
        regulation_effectiveness = self.emotion_regulation.get("effectiveness", 0.5)
        self.hyper_adaptation_manager.observe(
            features,
            quality=reward,
            smoothing_quality=smoothing.get("correlation", 0.0),
            combo_success=combo_success,
            regulation_effectiveness=regulation_effectiveness,
        )

    def _check_drift(self, signal: str, value: float) -> None:
        detector = self.drift_detectors.get(signal)
        if not detector:
            return
        event = detector.update(value)
        if event:
            self._log_drift_event(signal, event)

    def _log_drift_event(self, signal: str, event: Dict[str, float]) -> None:
        record = {
            "timestamp": time.time(),
            "signal": signal,
            **event,
        }
        self.emotional_history["drift_events"].append(record)
        if self.metacognitive_system and hasattr(self.metacognitive_system, "handle_emotional_drift"):
            try:
                self.metacognitive_system.handle_emotional_drift(record)
            except Exception as exc:
                print(f"[emotion] drift callback failed: {exc}")

    def _initialize_basic_emotions(self) -> Dict[EmotionalState, Dict[str, Any]]:
        """Initialise les émotions de base avec leurs caractéristiques"""
        return {
            EmotionalState.JOY: {
                "valence": 0.8,
                "arousal": 0.6,
                "dominance": 0.7,
                "action_tendency": "approach",
                "expression": "sourire, rire",
                "physiological_signs": ["chaleur", "relaxation"],
                "typical_duration": 30.0,
                "intensity_decay": 0.1
            },
            EmotionalState.SADNESS: {
                "valence": -0.7,
                "arousal": 0.2,
                "dominance": 0.3,
                "action_tendency": "withdraw",
                "expression": "yeux baissés, posture affaissée",
                "physiological_signs": ["lourdeur", "fatigue"],
                "typical_duration": 120.0,
                "intensity_decay": 0.05
            },
            EmotionalState.ANGER: {
                "valence": -0.6,
                "arousal": 0.8,
                "dominance": 0.6,
                "action_tendency": "attack",
                "expression": "sourcils froncés, poings serrés",
                "physiological_signs": ["chaleur", "tension"],
                "typical_duration": 45.0,
                "intensity_decay": 0.15
            },
            EmotionalState.FEAR: {
                "valence": -0.8,
                "arousal": 0.9,
                "dominance": 0.2,
                "action_tendency": "escape",
                "expression": "yeux écarquillés, recul",
                "physiological_signs": ["froid", "tremblements"],
                "typical_duration": 25.0,
                "intensity_decay": 0.2
            },
            EmotionalState.DISGUST: {
                "valence": -0.7,
                "arousal": 0.5,
                "dominance": 0.4,
                "action_tendency": "reject",
                "expression": "nez plissé, recul",
                "physiological_signs": ["nausée", "dégout"],
                "typical_duration": 20.0,
                "intensity_decay": 0.12
            },
            EmotionalState.SURPRISE: {
                "valence": 0.0,
                "arousal": 0.8,
                "dominance": 0.3,
                "action_tendency": "freeze",
                "expression": "sourcils levés, bouche ouverte",
                "physiological_signs": ["sursaut", "attention accrue"],
                "typical_duration": 5.0,
                "intensity_decay": 0.3
            },
            EmotionalState.ANTICIPATION: {
                "valence": 0.4,
                "arousal": 0.6,
                "dominance": 0.5,
                "action_tendency": "prepare",
                "expression": "regard attentif, posture tendue",
                "physiological_signs": ["tension légère", "excitation"],
                "typical_duration": 60.0,
                "intensity_decay": 0.08
            },
            EmotionalState.TRUST: {
                "valence": 0.6,
                "arousal": 0.4,
                "dominance": 0.6,
                "action_tendency": "affiliate",
                "expression": "sourire détendu, posture ouverte",
                "physiological_signs": ["calme", "chaleur"],
                "typical_duration": 90.0,
                "intensity_decay": 0.06
            },
            EmotionalState.NEUTRAL: {
                "valence": 0.0,
                "arousal": 0.3,
                "dominance": 0.5,
                "action_tendency": "maintain",
                "expression": "visage détendu, posture normale",
                "physiological_signs": ["équilibre", "stabilité"],
                "typical_duration": float('inf'),
                "intensity_decay": 0.0
            },
                        EmotionalState.PRIDE: {
                "valence": 0.8,
                "arousal": 0.5,
                "dominance": 0.8,
                "action_tendency": "approach",
                "expression": "poitrine ouverte, maintien affirmé",
                "physiological_signs": ["chaleur", "détente"],
                "typical_duration": 90.0,
                "intensity_decay": 0.08
            },
            EmotionalState.FRUSTRATION: {
                "valence": -0.5,
                "arousal": 0.7,
                "dominance": 0.4,
                "action_tendency": "persist/attack",
                "expression": "mâchoire serrée, sourcils froncés",
                "physiological_signs": ["tension", "agitation"],
                "typical_duration": 60.0,
                "intensity_decay": 0.12
            },

        }
    
    def _initialize_complex_emotions(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les émotions complexes (combinaisons d'émotions de base)"""
        return {
            "pride": {
                "components": [(EmotionalState.JOY, 0.7), (EmotionalState.TRUST, 0.3)],
                "valence": 0.8,
                "arousal": 0.5,
                "trigger": "accomplissement personnel"
            },
            "guilt": {
                "components": [(EmotionalState.SADNESS, 0.6), (EmotionalState.FEAR, 0.4)],
                "valence": -0.6,
                "arousal": 0.4,
                "trigger": "violation de normes personnelles"
            },
            "shame": {
                "components": [(EmotionalState.SADNESS, 0.5), (EmotionalState.FEAR, 0.3), (EmotionalState.DISGUST, 0.2)],
                "valence": -0.7,
                "arousal": 0.6,
                "trigger": "échec public ou exposition négative"
            },
            "hope": {
                "components": [(EmotionalState.ANTICIPATION, 0.6), (EmotionalState.JOY, 0.4)],
                "valence": 0.5,
                "arousal": 0.5,
                "trigger": "perspective positive future"
            },
            "frustration": {
                "components": [(EmotionalState.ANGER, 0.5), (EmotionalState.SADNESS, 0.3), (EmotionalState.ANTICIPATION, 0.2)],
                "valence": -0.5,
                "arousal": 0.7,
                "trigger": "obstacle à un but"
            }
        }
    
    def _initialize_emotional_triggers(self) -> Dict[str, List[EmotionalState]]:
        """Initialise les déclencheurs émotionnels courants"""
        return {
            "success": [EmotionalState.JOY, EmotionalState.PRIDE],
            "failure": [EmotionalState.SADNESS, EmotionalState.FRUSTRATION],
            "threat": [EmotionalState.FEAR, EmotionalState.ANGER],
            "novelty": [EmotionalState.SURPRISE, EmotionalState.ANTICIPATION],
            "social_acceptance": [EmotionalState.TRUST, EmotionalState.JOY],
            "social_rejection": [EmotionalState.SADNESS, EmotionalState.ANGER],
            "goal_progress": [EmotionalState.JOY, EmotionalState.ANTICIPATION],
            "goal_obstruction": [EmotionalState.ANGER, EmotionalState.FRUSTRATION]
        }
    
    def _initialize_coping_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les stratégies d'adaptation émotionnelle"""
        return {
            "cognitive_reappraisal": {
                "description": "Reinterpréter la situation de manière plus positive",
                "effectiveness": 0.7,
                "cognitive_cost": 0.6,
                "applicable_emotions": [EmotionalState.ANGER, EmotionalState.FEAR, EmotionalState.SADNESS]
            },
            "attention_redirection": {
                "description": "Détourner l'attention vers d'autres stimuli",
                "effectiveness": 0.5,
                "cognitive_cost": 0.3,
                "applicable_emotions": [EmotionalState.ANGER, EmotionalState.FEAR, EmotionalState.DISGUST]
            },
            "emotional_expression": {
                "description": "Exprimer l'émotion de manière contrôlée",
                "effectiveness": 0.6,
                "cognitive_cost": 0.4,
                "applicable_emotions": [EmotionalState.SADNESS, EmotionalState.ANGER, EmotionalState.JOY]
            },
            "problem_solving": {
                "description": "Résoudre la cause sous-jacente de l'émotion",
                "effectiveness": 0.8,
                "cognitive_cost": 0.7,
                "applicable_emotions": [EmotionalState.ANGER, EmotionalState.FRUSTRATION, EmotionalState.FEAR]
            },
            "acceptance": {
                "description": "Accepter l'émotion sans jugement",
                "effectiveness": 0.4,
                "cognitive_cost": 0.2,
                "applicable_emotions": [EmotionalState.SADNESS, EmotionalState.FEAR, EmotionalState.DISGUST]
            }
        }
        
    def _assess_emotional_balance(self) -> float:
        """
        Évalue la balance émotionnelle globale en fonction des émotions actives.
        Retourne une valeur entre -1 (négatif) et +1 (positif).
        """
        total_valence = 0.0
        count = 0
        if hasattr(self, "current_emotions"):
            for emotion, data in self.current_emotions.items():
                valence = data.get("valence", 0.0) if isinstance(data, dict) else 0.0
                total_valence += valence
                count += 1
        return total_valence / count if count > 0 else 0.0

    def _start_emotional_monitoring(self):
        """Démarre la surveillance émotionnelle continue"""
        def monitoring_loop():
            while self.running:
                try:
                    # Mise à jour des émotions actuelles
                    self._update_emotional_state()
                    
                    # Surveillance de l'intensité émotionnelle
                    self._monitor_emotional_intensity()
                    
                    # Équilibre émotionnel
                    self._assess_emotional_balance()
                    
                    # Détection de patterns émotionnels
                    self._detect_emotional_patterns()
                    
                    time.sleep(1)  # Surveillance chaque seconde
                    
                except Exception as e:
                    print(f"Erreur dans la surveillance émotionnelle: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        self.processing_threads["emotional_monitoring"] = monitor_thread
    
    def _start_mood_updater(self):
        """Démarre la mise à jour périodique de l'humeur"""
        def mood_loop():
            while self.running:
                try:
                    self._update_mood_state()
                    time.sleep(10)  # Mise à jour toutes les 10 secondes
                    
                except Exception as e:
                    print(f"Erreur dans la mise à jour de l'humeur: {e}")
                    time.sleep(30)
        
        mood_thread = threading.Thread(target=mood_loop, daemon=True)
        mood_thread.start()
        self.processing_threads["mood_updater"] = mood_thread
    
    def _start_physiological_simulator(self):
        """Démarre la simulation des influences physiologiques"""
        def physiology_loop():
            while self.running:
                try:
                    self._update_physiological_state()
                    time.sleep(5)  # Mise à jour toutes les 5 secondes
                    
                except Exception as e:
                    print(f"Erreur dans la simulation physiologique: {e}")
                    time.sleep(20)
        
        physiology_thread = threading.Thread(target=physiology_loop, daemon=True)
        physiology_thread.start()
        self.processing_threads["physiology_simulator"] = physiology_thread
    
    def _perform_initial_emotional_assessment(self):
        """Effectue l'évaluation émotionnelle initiale"""
        # État émotionnel de départ neutre
        initial_experience = EmotionalExperience(
            timestamp=time.time(),
            primary_emotion=EmotionalState.NEUTRAL,
            secondary_emotions=[(EmotionalState.TRUST, 0.1), (EmotionalState.ANTICIPATION, 0.1)],
            intensity=0.1,
            valence=0.0,
            arousal=0.3,
            dominance=0.5,
            trigger="initialization",
            duration=0.0,
            bodily_sensations=["calme", "équilibre"],
            cognitive_appraisals=["situation nouvelle", "potentiel d'apprentissage"],
            action_tendencies=["explorer", "apprendre"],
            expression="visage neutre, posture détendue"
        )
        
        self.emotional_history["emotional_experiences"].append(initial_experience)
    
    def process_stimulus(self, stimulus: str, context: Dict[str, Any]) -> EmotionalExperience:
        """
        Traite un stimulus et génère une réponse émotionnelle
        Basé sur le modèle d'évaluation cognitive OCC
        """
        start_time = time.time()
        
        # === PHASE 1: ÉVALUATION DE LA PERTINENCE ===
        relevance = self.appraisal_system["relevance_detector"].assess_relevance(stimulus, context)
        
        if relevance < 0.1:  # Stimulus non pertinent
            return self._create_neutral_experience(stimulus)
        
        # === PHASE 2: ÉVALUATION DES CONSÉQUENCES ===
        consequence_appraisal = self._appraise_consequences(stimulus, context)
        
        # === PHASE 3: ÉVALUATION DE LA CONGRUENCE AVEC LES BUTS ===
        goal_congruence = self.appraisal_system["goal_congruence_assessor"].assess_congruence(
            stimulus, context, self.cognitive_architecture
        )
        
        # === PHASE 4: ÉVALUATION DU POTENTIEL D'ADAPTATION ===
        coping_potential = self.appraisal_system["coping_potential_evaluator"].evaluate_potential(
            stimulus, context, self.metacognitive_system
        )
        
        # === PHASE 5: ÉVALUATION DE LA COMPATIBILITÉ AVEC LES NORMES ===
        norm_compatibility = self.appraisal_system["norm_compatibility_checker"].check_compatibility(
            stimulus, context
        )
        
        # === PHASE 6: GÉNÉRATION DE L'ÉMOTION ===
        emotional_response = self._generate_emotional_response(
            relevance, consequence_appraisal, goal_congruence, 
            coping_potential, norm_compatibility, stimulus, context
        )
        
        # === PHASE 7: RÉGULATION ÉMOTIONNELLE ===
        regulated_response = self._regulate_emotional_response(emotional_response)
        
        # === PHASE 8: EXPRESSION ÉMOTIONNELLE ===
        expression = self._generate_emotional_expression(regulated_response)
        regulated_response.expression = expression
        
        # === PHASE 9: APPRENTISSAGE ÉMOTIONNEL ===
        self._learn_from_emotional_experience(regulated_response, context)
        
        # === PHASE 10: MISE À JOUR DE L'ÉTAT ===
        self._update_current_emotions(regulated_response)
        
        regulated_response.duration = time.time() - start_time
        
        # Enregistrement de l'expérience
        self.emotional_history["emotional_experiences"].append(regulated_response)
        
        return regulated_response
    
    def _appraise_consequences(self, stimulus: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Évalue les conséquences du stimulus"""
        appraisal = {
            "desirability": 0.0,      # Désirabilité: -1.0 à 1.0
            "certainty": 0.5,         # Certitude: 0.0 à 1.0
            "urgency": 0.3,           # Urgence: 0.0 à 1.0
            "impact": 0.5,            # Impact: 0.0 à 1.0
            "controllability": 0.5    # Contrôlabilité: 0.0 à 1.0
        }

        # Analyse basique basée sur le contenu du stimulus
        stimulus_lower = stimulus.lower()
        
        # Désirabilité
        positive_words = ["succès", "réussite", "bon", "bien", "joyeux", "positif"]
        negative_words = ["échec", "danger", "mauvais", "négatif", "problème", "erreur"]
        
        positive_count = sum(1 for word in positive_words if word in stimulus_lower)
        negative_count = sum(1 for word in negative_words if word in stimulus_lower)
        
        if positive_count > negative_count:
            appraisal["desirability"] = min(positive_count * 0.2, 1.0)
        elif negative_count > positive_count:
            appraisal["desirability"] = -min(negative_count * 0.2, 1.0)
        
        # Certitude (basée sur la clarté du stimulus)
        clarity_indicators = ["claire", "évident", "certain", "sûr"]
        uncertainty_indicators = ["peut-être", "possible", "incertain", "probable"]
        
        certainty_score = sum(1 for word in clarity_indicators if word in stimulus_lower) * 0.2
        uncertainty_score = sum(1 for word in uncertainty_indicators if word in stimulus_lower) * 0.2
        
        appraisal["certainty"] = max(0.1, min(0.9, 0.5 + certainty_score - uncertainty_score))
        
        # Urgence
        urgency_indicators = ["urgent", "immédiat", "maintenant", "vite", "critique"]
        urgency_score = sum(1 for word in urgency_indicators if word in stimulus_lower) * 0.3
        appraisal["urgency"] = min(urgency_score, 1.0)
        
        # Impact (estimé basé sur l'intensité perçue)
        intensity_indicators = ["important", "significatif", "majeur", "crucial"]
        impact_score = sum(1 for word in intensity_indicators if word in stimulus_lower) * 0.25
        appraisal["impact"] = max(0.1, min(0.9, 0.5 + impact_score))

        payload = {
            "stimulus": _truncate_text(stimulus, 800),
            "context": _prepare_llm_context(context or {}),
            "baseline_appraisal": dict(appraisal),
        }

        response = try_call_llm_dict(
            "emotional_system_appraisal",
            input_payload=payload,
            logger=LOGGER,
        )

        if response is None:
            self._last_llm_appraisal = None
            return appraisal

        if not isinstance(response, Mapping):
            self._last_llm_appraisal = None
            return appraisal

        self._last_llm_appraisal = dict(response)

        llm_appraisal = response.get("appraisal")
        if isinstance(llm_appraisal, Mapping):
            for key in ("desirability", "certainty", "urgency", "impact", "controllability"):
                if key not in appraisal:
                    continue
                value = llm_appraisal.get(key)
                if not isinstance(value, (int, float)):
                    continue
                numeric = float(value)
                if key == "desirability":
                    numeric = max(-1.0, min(1.0, numeric))
                else:
                    numeric = max(0.0, min(1.0, numeric))
                appraisal[key] = 0.4 * appraisal[key] + 0.6 * numeric

        return appraisal
    
    def _generate_emotional_response(self, relevance: float, consequence_appraisal: Dict[str, float],
                                   goal_congruence: float, coping_potential: float,
                                   norm_compatibility: float, stimulus: str,
                                   context: Dict[str, Any]) -> EmotionalExperience:
        """Génère une réponse émotionnelle basée sur les évaluations"""
        
        # Détermination de l'émotion primaire
        primary_emotion, primary_intensity = self._determine_primary_emotion(
            consequence_appraisal, goal_congruence, coping_potential, norm_compatibility
        )
        
        # Détermination des émotions secondaires
        secondary_emotions = self._determine_secondary_emotions(
            primary_emotion, primary_intensity, consequence_appraisal, context
        )
        
        # Calcul des dimensions affectives
        valence, arousal, dominance = self._calculate_affective_dimensions(
            primary_emotion, primary_intensity, secondary_emotions, coping_potential
        )
        
        # Génération des composants expérientiels
        bodily_sensations = self._generate_bodily_sensations(primary_emotion, primary_intensity)
        cognitive_appraisals = self._generate_cognitive_appraisals(consequence_appraisal, context)
        action_tendencies = self._generate_action_tendencies(primary_emotion, primary_intensity)
        
        # Création de l'expérience émotionnelle
        experience = EmotionalExperience(
            timestamp=time.time(),
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            intensity=primary_intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            trigger=stimulus,
            duration=0.0,
            bodily_sensations=bodily_sensations,
            cognitive_appraisals=cognitive_appraisals,
            action_tendencies=action_tendencies,
            expression=""  # À générer plus tard
        )

        return experience

    def _match_emotional_state(self, label: Any) -> Optional[EmotionalState]:
        if not label:
            return None
        label_str = str(label).strip().lower()
        if not label_str:
            return None
        for state in EmotionalState:
            if label_str in {state.value.lower(), state.name.lower()}:
                return state
        aliases = {
            "heureux": EmotionalState.JOY,
            "peine": EmotionalState.SADNESS,
            "colere": EmotionalState.ANGER,
            "colère": EmotionalState.ANGER,
            "fache": EmotionalState.ANGER,
            "fâché": EmotionalState.ANGER,
            "anxiete": EmotionalState.FEAR,
            "anxiété": EmotionalState.FEAR,
            "stress": EmotionalState.FEAR,
            "degout": EmotionalState.DISGUST,
            "dégoût": EmotionalState.DISGUST,
            "surpris": EmotionalState.SURPRISE,
            "confiant": EmotionalState.TRUST,
            "frustration": EmotionalState.FRUSTRATION,
            "fierte": EmotionalState.PRIDE,
            "fierté": EmotionalState.PRIDE,
        }
        return aliases.get(label_str)

    def _determine_primary_emotion(self, consequence_appraisal: Dict[str, float],
                                 goal_congruence: float, coping_potential: float,
                                 norm_compatibility: float) -> Tuple[EmotionalState, float]:
        """Détermine l'émotion primaire et son intensité"""

        desirability = consequence_appraisal["desirability"]
        certainty = consequence_appraisal["certainty"]
        controllability = consequence_appraisal["controllability"]
        
        # Logique de décision basée sur le modèle OCC
        if desirability > 0.3:  # Événement désirable
            if goal_congruence > 0.6:
                emotion = EmotionalState.JOY
                intensity = desirability * goal_congruence
            else:
                emotion = EmotionalState.SURPRISE
                intensity = abs(desirability) * 0.7
        
        elif desirability < -0.3:  # Événement indésirable
            if controllability > 0.6:
                emotion = EmotionalState.ANGER
                intensity = abs(desirability) * controllability
            elif coping_potential < 0.4:
                emotion = EmotionalState.FEAR
                intensity = abs(desirability) * (1 - coping_potential)
            else:
                emotion = EmotionalState.SADNESS
                intensity = abs(desirability) * 0.8
        
        elif abs(desirability) <= 0.3:  # Événement neutre
            if certainty < 0.4:
                emotion = EmotionalState.ANTICIPATION
                intensity = (1 - certainty) * 0.6
            elif norm_compatibility < 0.3:
                emotion = EmotionalState.DISGUST
                intensity = (1 - norm_compatibility) * 0.5
            else:
                emotion = EmotionalState.NEUTRAL
                intensity = 0.1
        
        else:  # Cas par défaut
            emotion = EmotionalState.NEUTRAL
            intensity = 0.1

        llm_data = self._last_llm_appraisal if isinstance(self._last_llm_appraisal, Mapping) else None
        if llm_data:
            confidence = llm_data.get("confidence")
            if not isinstance(confidence, (int, float)) or confidence >= 0.3:
                llm_primary = self._match_emotional_state(llm_data.get("primary_emotion"))
                intensity_hint = llm_data.get("primary_intensity")
                if isinstance(intensity_hint, (int, float)):
                    llm_intensity = max(0.0, min(1.0, float(intensity_hint)))
                else:
                    llm_intensity = None
                if llm_intensity is None:
                    scores = llm_data.get("emotion_scores")
                    if isinstance(scores, Mapping) and llm_primary is not None:
                        for key, value in scores.items():
                            mapped = self._match_emotional_state(key)
                            if mapped == llm_primary and isinstance(value, (int, float)):
                                llm_intensity = max(0.0, min(1.0, float(value)))
                                break
                if llm_intensity is None:
                    llm_intensity = 0.6

                if llm_primary is not None:
                    base_intensity = float(intensity)
                    if llm_primary == emotion:
                        intensity = 0.4 * base_intensity + 0.6 * llm_intensity
                    else:
                        if llm_intensity >= base_intensity + 0.1:
                            emotion = llm_primary
                            intensity = 0.55 * llm_intensity + 0.45 * base_intensity
                        else:
                            intensity = 0.5 * base_intensity + 0.5 * llm_intensity

        # Ajustement basé sur l'humeur actuelle
        mood_influence = self._get_mood_influence_on_emotion(emotion)
        intensity *= mood_influence

        return emotion, min(intensity, 1.0)
    
    def _get_mood_influence_on_emotion(self, emotion: EmotionalState) -> float:
        """Calcule l'influence de l'humeur actuelle sur une émotion"""
        current_mood = self.mood_system["current_mood"].mood_type
        mood_intensity = self.mood_system["current_mood"].intensity
        
        # Mapping humeur-émotion (amplification/atténuation)
        mood_emotion_amplification = {
            MoodState.EUPHORIC: {
                EmotionalState.JOY: 1.5, EmotionalState.SADNESS: 0.3, EmotionalState.ANGER: 0.4
            },
            MoodState.CONTENT: {
                EmotionalState.JOY: 1.2, EmotionalState.TRUST: 1.1, EmotionalState.SADNESS: 0.6
            },
            MoodState.CALM: {
                EmotionalState.FEAR: 0.7, EmotionalState.ANGER: 0.6, EmotionalState.JOY: 0.9
            },
            MoodState.MELANCHOLIC: {
                EmotionalState.SADNESS: 1.4, EmotionalState.JOY: 0.5, EmotionalState.ANGER: 1.1
            },
            MoodState.IRRITABLE: {
                EmotionalState.ANGER: 1.5, EmotionalState.FEAR: 0.8, EmotionalState.JOY: 0.4
            },
            MoodState.ANXIOUS: {
                EmotionalState.FEAR: 1.4, EmotionalState.ANTICIPATION: 1.2, EmotionalState.JOY: 0.3
            },
            MoodState.DEPRESSED: {
                EmotionalState.SADNESS: 1.6, EmotionalState.ANGER: 1.1, EmotionalState.JOY: 0.2
            },
            MoodState.NEUTRAL: {
                # Pas d'amplification significative
            }
        }
        
        amplification = mood_emotion_amplification.get(current_mood, {}).get(emotion, 1.0)
        influence = 1.0 + (amplification - 1.0) * mood_intensity
        
        return max(0.1, min(2.0, influence))
    
    def _determine_secondary_emotions(self, primary_emotion: EmotionalState, 
                                    primary_intensity: float,
                                    consequence_appraisal: Dict[str, float],
                                    context: Dict[str, Any]) -> List[Tuple[EmotionalState, float]]:
        """Détermine les émotions secondaires"""
        secondary_emotions = []
        
        # Émotions secondaires basées sur les aspects spécifiques de l'évaluation
        desirability = consequence_appraisal["desirability"]
        certainty = consequence_appraisal["certainty"]
        controllability = consequence_appraisal["controllability"]
        
        # Confiance si la situation est contrôlable
        if controllability > 0.7 and desirability > 0:
            trust_intensity = controllability * 0.5
            secondary_emotions.append((EmotionalState.TRUST, trust_intensity))
        
        # Anticipation si incertitude modérée
        if 0.3 < certainty < 0.7 and desirability > -0.5:
            anticipation_intensity = (1 - certainty) * 0.6
            secondary_emotions.append((EmotionalState.ANTICIPATION, anticipation_intensity))
        
        # Surprise si certitude faible pour un événement significatif
        if certainty < 0.4 and consequence_appraisal["impact"] > 0.5:
            surprise_intensity = (1 - certainty) * consequence_appraisal["impact"] * 0.7
            secondary_emotions.append((EmotionalState.SURPRISE, surprise_intensity))
        
        # Limiter le nombre d'émotions secondaires
        combined: Dict[EmotionalState, float] = {}
        for state, value in secondary_emotions:
            combined[state] = max(combined.get(state, 0.0), float(value))

        llm_data = self._last_llm_appraisal if isinstance(self._last_llm_appraisal, Mapping) else None
        if llm_data:
            candidates = llm_data.get("secondary_candidates")
            if isinstance(candidates, Sequence):
                for entry in candidates:
                    if not isinstance(entry, Mapping):
                        continue
                    state = self._match_emotional_state(entry.get("emotion"))
                    if state is None or state == primary_emotion:
                        continue
                    intensity = entry.get("intensity")
                    if not isinstance(intensity, (int, float)):
                        continue
                    value = max(0.0, min(1.0, float(intensity)))
                    current = combined.get(state, 0.0)
                    combined[state] = 0.4 * current + 0.6 * value if current else value

        merged = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return merged[:3]  # Maximum 3 émotions secondaires
    
    def _calculate_affective_dimensions(self, primary_emotion: EmotionalState,
                                      primary_intensity: float,
                                      secondary_emotions: List[Tuple[EmotionalState, float]],
                                      coping_potential: float) -> Tuple[float, float, float]:
        """Calcule les dimensions affectives PAD (Pleasure, Arousal, Dominance)"""
        
        # Obtenir les caractéristiques de l'émotion primaire
        emotion_profile = self.emotional_repertoire["basic_emotions"][primary_emotion]
        
        # Valence (Pleasure) basée sur l'émotion primaire
        base_valence = emotion_profile["valence"]
        base_arousal = emotion_profile["arousal"]
        base_dominance = emotion_profile["dominance"]
        
        # Ajustement par l'intensité
        valence = base_valence * primary_intensity
        arousal = base_arousal * primary_intensity
        dominance = base_dominance * primary_intensity
        
        # Ajustement par les émotions secondaires
        for sec_emotion, sec_intensity in secondary_emotions:
            sec_profile = self.emotional_repertoire["basic_emotions"][sec_emotion]
            valence += sec_profile["valence"] * sec_intensity * 0.3
            arousal += sec_profile["arousal"] * sec_intensity * 0.3
            dominance += sec_profile["dominance"] * sec_intensity * 0.3
        
        # Ajustement par le potentiel d'adaptation
        dominance *= coping_potential
        
        # Normalisation
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        dominance = max(0.0, min(1.0, dominance))
        
        return valence, arousal, dominance
    
    def _generate_bodily_sensations(self, emotion: EmotionalState, intensity: float) -> List[str]:
        """Génère les sensations corporelles associées à l'émotion"""
        emotion_profile = self.emotional_repertoire["basic_emotions"][emotion]
        base_sensations = emotion_profile["physiological_signs"]
        
        # Sélection basée sur l'intensité
        num_sensations = min(len(base_sensations), max(1, int(intensity * len(base_sensations))))
        selected_sensations = random.sample(base_sensations, num_sensations)
        
        return selected_sensations
    
    def _generate_cognitive_appraisals(self, consequence_appraisal: Dict[str, float],
                                     context: Dict[str, Any]) -> List[str]:
        """Génère les appraisals cognitifs de la situation"""
        appraisals = []
        
        desirability = consequence_appraisal["desirability"]
        impact = consequence_appraisal["impact"]
        controllability = consequence_appraisal["controllability"]
        
        if desirability > 0.5:
            appraisals.append("Situation favorable")
        elif desirability < -0.5:
            appraisals.append("Situation défavorable")
        
        if impact > 0.7:
            appraisals.append("Impact significatif")
        elif impact < 0.3:
            appraisals.append("Impact limité")
        
        if controllability > 0.7:
            appraisals.append("Situation contrôlable")
        elif controllability < 0.3:
            appraisals.append("Situation difficile à contrôler")
        
        return appraisals
    
    def _generate_action_tendencies(self, emotion: EmotionalState, intensity: float) -> List[str]:
        """Génère les tendances à l'action associées à l'émotion"""
        emotion_profile = self.emotional_repertoire["basic_emotions"][emotion]
        base_tendency = emotion_profile["action_tendency"]
        
        tendencies = []
        
        # Tendances de base
        tendency_mapping = {
            "approach": ["s'approcher", "explorer", "interagir"],
            "withdraw": ["se retirer", "éviter", "se protéger"],
            "attack": ["confronter", "résister", "défendre"],
            "escape": ["fuir", "se cacher", "échapper"],
            "prepare": ["se préparer", "planifier", "anticiper"],
            "affiliate": ["coopérer", "partager", "connecter"],
            "maintain": ["continuer", "persévérer", "maintenir"]
        }
        
        if base_tendency in tendency_mapping:
            tendencies.extend(tendency_mapping[base_tendency])
        
        # Ajout de tendances basées sur l'intensité
        if intensity > 0.7:
            tendencies.append("action énergique")
        elif intensity < 0.3:
            tendencies.append("action prudente")
        
        return tendencies
    
    def _regulate_emotional_response(self, experience: EmotionalExperience) -> EmotionalExperience:
        """Applique la régulation émotionnelle si nécessaire"""
        
        # Vérifier si la régulation est nécessaire
        needs_regulation = (
            experience.intensity > self.emotion_regulation["regulation_threshold"] or
            experience.valence < -0.7 or  # Émotion très négative
            experience.arousal > 0.8      # Excitation trop élevée
        )
        
        if not needs_regulation or not self.emotion_regulation["automatic_regulation"]:
            return experience
        
        # Sélection de la stratégie de régulation
        strategy = self._select_regulation_strategy(experience)
        
        if strategy:
            regulated_experience = self._apply_regulation_strategy(experience, strategy)
            regulated_experience.regulation_strategy = strategy
            return regulated_experience
        
        return experience
    
    def _select_regulation_strategy(self, experience: EmotionalExperience) -> Optional[str]:
        """Sélectionne une stratégie de régulation appropriée"""
        
        # Stratégies disponibles triées par efficacité
        available_strategies = []
        
        for strategy, effectiveness in self.emotion_regulation["strategies"].items():
            strategy_info = self.emotional_repertoire["coping_strategies"][strategy]
            
            # Vérifier l'applicabilité à l'émotion actuelle
            applicable_emotions = strategy_info["applicable_emotions"]
            emotion_name = experience.primary_emotion.value
            
            if any(emotion.value == emotion_name for emotion in applicable_emotions):
                available_strategies.append((strategy, effectiveness))
        
        if available_strategies:
            # Sélectionner la stratégie la plus efficace
            available_strategies.sort(key=lambda x: x[1], reverse=True)
            return available_strategies[0][0]
        
        return None
    
    def _apply_regulation_strategy(self, experience: EmotionalExperience, 
                                 strategy: str) -> EmotionalExperience:
        """Applique une stratégie de régulation émotionnelle"""
        
        strategy_info = self.emotional_repertoire["coping_strategies"][strategy]
        effectiveness = strategy_info["effectiveness"]
        
        # Créer une copie modifiée de l'expérience
        regulated_experience = EmotionalExperience(
            timestamp=experience.timestamp,
            primary_emotion=experience.primary_emotion,
            secondary_emotions=experience.secondary_emotions.copy(),
            intensity=experience.intensity,
            valence=experience.valence,
            arousal=experience.arousal,
            dominance=experience.dominance,
            trigger=experience.trigger,
            duration=experience.duration,
            bodily_sensations=experience.bodily_sensations.copy(),
            cognitive_appraisals=experience.cognitive_appraisals.copy(),
            action_tendencies=experience.action_tendencies.copy(),
            expression=experience.expression,
            regulation_strategy=strategy
        )
        
        # Application des effets de la régulation
        if strategy == "cognitive_reappraisal":
            # Réduction de l'intensité et amélioration de la valence
            regulated_experience.intensity *= (1 - effectiveness * 0.3)
            regulated_experience.valence = min(1.0, regulated_experience.valence + effectiveness * 0.2)
            regulated_experience.cognitive_appraisals.append("réévaluation cognitive appliquée")
        
        elif strategy == "attention_redirection":
            # Réduction de l'intensité et de l'arousal
            regulated_experience.intensity *= (1 - effectiveness * 0.4)
            regulated_experience.arousal *= (1 - effectiveness * 0.3)
            regulated_experience.cognitive_appraisals.append("attention redirigée")
        
        elif strategy == "emotional_expression":
            # Expression contrôlée - légère réduction de l'intensité
            regulated_experience.intensity *= (1 - effectiveness * 0.2)
            regulated_experience.dominance += effectiveness * 0.1
            regulated_experience.cognitive_appraisals.append("expression émotionnelle contrôlée")
        
        elif strategy == "problem_solving":
            # Augmentation du sentiment de contrôle
            regulated_experience.dominance += effectiveness * 0.3
            regulated_experience.cognitive_appraisals.append("approche de résolution de problème")
        
        elif strategy == "acceptance":
            # Acceptation - réduction de la tension
            regulated_experience.intensity *= (1 - effectiveness * 0.1)
            regulated_experience.arousal *= (1 - effectiveness * 0.2)
            regulated_experience.cognitive_appraisals.append("acceptation émotionnelle")
        
        # Enregistrement de la tentative de régulation
        self.emotional_history["regulation_attempts"].append({
            "timestamp": time.time(),
            "strategy": strategy,
            "effectiveness": effectiveness,
            "original_intensity": experience.intensity,
            "regulated_intensity": regulated_experience.intensity,
            "emotion": experience.primary_emotion.value
        })
        
        return regulated_experience
    
    def _generate_emotional_expression(self, experience: EmotionalExperience) -> str:
        """Génère l'expression émotionnelle correspondante"""
        expression_system = self.expression_system
        
        # Expression faciale basée sur l'émotion primaire
        facial_expression = expression_system["facial_expressions"].generate_expression(
            experience.primary_emotion, experience.intensity
        )
        
        # Ton vocal
        vocal_tone = expression_system["vocal_tones"].generate_tone(
            experience.primary_emotion, experience.intensity, experience.arousal
        )
        
        # Langage corporel
        body_language = expression_system["body_language"].generate_pose(
            experience.primary_emotion, experience.intensity, experience.dominance
        )
        
        # Expression verbale
        verbal_expression = expression_system["verbal_expressions"].generate_expression(
            experience.primary_emotion, experience.intensity, experience.valence
        )
        
        # Combinaison des expressions
        full_expression = f"{facial_expression}, {vocal_tone}, {body_language}. {verbal_expression}"
        
        return full_expression
    
    def _learn_from_emotional_experience(self, experience: EmotionalExperience, context: Dict[str, Any]):
        """Apprend de l'expérience émotionnelle"""
        
        # Conditionnement émotionnel
        self.emotional_memory["emotional_conditioning"].update_associations(
            experience.trigger, experience.primary_emotion, experience.intensity
        )
        
        # Mise à jour de l'intelligence émotionnelle
        self._update_emotional_intelligence(experience)
        
        # Apprentissage des stratégies de régulation
        if experience.regulation_strategy:
            self._learn_regulation_effectiveness(experience)
        
        # Enregistrement de l'épisode d'apprentissage
        learning_episode = {
            "timestamp": time.time(),
            "emotion": experience.primary_emotion.value,
            "trigger": experience.trigger,
            "intensity": experience.intensity,
            "regulation_used": experience.regulation_strategy,
            "lessons_learned": self._extract_emotional_lessons(experience)
        }
        
        self.emotional_history["learning_episodes"].append(learning_episode)
    
    def _update_emotional_intelligence(self, experience: EmotionalExperience):
        """Met à jour l'intelligence émotionnelle basée sur l'expérience"""
        learning_rate = 0.01
        
        # Facteurs d'apprentissage
        intensity_factor = experience.intensity
        novelty_factor = self._assess_emotional_novelty(experience)
        
        # Apprentissage global
        learning_gain = (intensity_factor + novelty_factor) * learning_rate
        self.emotional_learning["emotional_intelligence"] = min(
            1.0, self.emotional_learning["emotional_intelligence"] + learning_gain
        )
        
        # Apprentissage de l'empathie (si émotion sociale)
        if self._is_social_emotion(experience.primary_emotion):
            empathy_gain = learning_gain * 0.5
            self.emotional_learning["empathy_capacity"] = min(
                1.0, self.emotional_learning["empathy_capacity"] + empathy_gain
            )
    
    def _assess_emotional_novelty(self, experience: EmotionalExperience) -> float:
        """Évalue la nouveauté de l'expérience émotionnelle"""
        recent_experiences = list(self.emotional_history["emotional_experiences"])[-10:]
        
        if not recent_experiences:
            return 1.0
        
        # Similarité avec les expériences récentes
        similarities = []
        for past_experience in recent_experiences:
            similarity = self._calculate_emotional_similarity(experience, past_experience)
            similarities.append(similarity)
        
        novelty = 1.0 - max(similarities) if similarities else 1.0
        return novelty
    
    def _calculate_emotional_similarity(self, exp1: EmotionalExperience, exp2: EmotionalExperience) -> float:
        """Calcule la similarité entre deux expériences émotionnelles"""
        similarity_factors = []
        
        # Similarité de l'émotion primaire
        if exp1.primary_emotion == exp2.primary_emotion:
            similarity_factors.append(0.6)
        else:
            similarity_factors.append(0.2)
        
        # Similarité d'intensité
        intensity_similarity = 1.0 - abs(exp1.intensity - exp2.intensity)
        similarity_factors.append(intensity_similarity * 0.2)
        
        # Similarité de valence
        valence_similarity = 1.0 - abs(exp1.valence - exp2.valence) / 2.0
        similarity_factors.append(valence_similarity * 0.2)
        
        return sum(similarity_factors)
    
    def _is_social_emotion(self, emotion: EmotionalState) -> bool:
        """Détermine si une émotion est socialement orientée"""
        social_emotions = [EmotionalState.TRUST, EmotionalState.JOY, EmotionalState.ANGER]
        return emotion in social_emotions
    
    def _learn_regulation_effectiveness(self, experience: EmotionalExperience):
        """Apprend l'efficacité des stratégies de régulation"""
        strategy = experience.regulation_strategy
        if not strategy:
            return
        
        # Mesure de l'efficacité (réduction d'intensité estimée)
        # Dans une implémentation réelle, on comparerait avec l'intensité pré-régulation
        effectiveness_estimate = 0.5  # Estimation basique
        
        learning_rate = 0.05
        current_effectiveness = self.emotion_regulation["strategies"][strategy]
        
        # Mise à jour avec apprentissage par renforcement
        new_effectiveness = (1 - learning_rate) * current_effectiveness + learning_rate * effectiveness_estimate
        self.emotion_regulation["strategies"][strategy] = max(0.1, min(1.0, new_effectiveness))
    
    def _extract_emotional_lessons(self, experience: EmotionalExperience) -> List[str]:
        """Extrait les leçons apprises de l'expérience émotionnelle"""
        lessons = []
        
        if experience.intensity > 0.7:
            lessons.append("Émotion intense nécessitant une régulation")
        
        if experience.valence < -0.5:
            lessons.append("Situation désagréable à éviter ou mieux gérer")
        
        if experience.dominance < 0.3:
            lessons.append("Sentiment d'impuissance à travailler")
        
        if experience.regulation_strategy:
            lessons.append(f"Stratégie {experience.regulation_strategy} appliquée avec succès")
        
        return lessons
    
    def _update_current_emotions(self, experience: EmotionalExperience):
        """Met à jour les émotions actuelles basé sur la nouvelle expérience"""
        adaptation = self._hyper_adaptation_step(experience)
        update_rate = adaptation.get("update_rate", 0.3)
        decay_multiplier = adaptation.get("decay_multiplier", 1.0)

        # Mise à jour de l'émotion primaire
        primary_emotion = experience.primary_emotion
        new_intensity = experience.intensity

        current_intensity = self.current_emotions[primary_emotion]
        updated_intensity = (1 - update_rate) * current_intensity + update_rate * new_intensity

        self.current_emotions[primary_emotion] = updated_intensity

        # Décroissance des autres émotions
        for emotion in self.current_emotions:
            if emotion != primary_emotion:
                decay_rate = self.emotional_repertoire["basic_emotions"][emotion]["intensity_decay"]
                adaptive_decay = decay_rate * decay_multiplier
                adaptive_decay = max(0.01, min(0.9, adaptive_decay))
                self.current_emotions[emotion] *= (1 - adaptive_decay)

        # Mise à jour des dimensions affectives
        self.affective_dimensions["pleasure_arousal_dominance"]["pleasure"] = experience.valence
        self.affective_dimensions["pleasure_arousal_dominance"]["arousal"] = experience.arousal
        self.affective_dimensions["pleasure_arousal_dominance"]["dominance"] = experience.dominance

        # Mise à jour de l'équilibre émotionnel
        previous_balance = self._adaptive_context.get("last_balance", 0.5)
        self._update_emotional_balance()
        combo_success = 1.0 if self.emotional_balance >= previous_balance else 0.0
        combo_success = max(combo_success, adaptation.get("smoothing", {}).get("improved", 0.0))
        self._adaptive_context["last_balance"] = self.emotional_balance

        # Feedback d'adaptation et surveillance
        self._record_adaptation_feedback(adaptation, combo_success)
        self._check_drift("valence", experience.valence)
    
    def _update_emotional_balance(self):
        """Met à jour l'équilibre émotionnel global"""
        positive_emotions = sum(
            intensity for emotion, intensity in self.current_emotions.items()
            if self.emotional_repertoire["basic_emotions"][emotion]["valence"] > 0
        )
        
        total_intensity = sum(self.current_emotions.values())
        
        if total_intensity > 0:
            self.emotional_balance = positive_emotions / total_intensity
        else:
            self.emotional_balance = 0.5
    
    def _update_emotional_state(self):
        """Met à jour l'état émotionnel global (appelé périodiquement)"""
        # Décroissance naturelle des émotions
        for emotion in self.current_emotions:
            decay_rate = self.emotional_repertoire["basic_emotions"][emotion]["intensity_decay"]
            multiplier = self._adaptive_context.get("decay_multiplier", 1.0)
            adaptive_decay = decay_rate * 0.1 * multiplier
            adaptive_decay = max(0.001, min(0.2, adaptive_decay))
            self.current_emotions[emotion] *= (1 - adaptive_decay)  # Décroissance lente

        # Influence de l'humeur sur les émotions de base
        self._apply_mood_influence()
    
    def _apply_mood_influence(self):
        """Applique l'influence de l'humeur sur les émotions actuelles"""
        current_mood = self.mood_system["current_mood"]
        mood_type = current_mood.mood_type
        mood_intensity = current_mood.intensity
        
        # Amplification/atténuation basée sur l'humeur
        mood_effects = {
            MoodState.EUPHORIC: {EmotionalState.JOY: 1.2, EmotionalState.SADNESS: 0.5},
            MoodState.CONTENT: {EmotionalState.JOY: 1.1, EmotionalState.TRUST: 1.1},
            MoodState.CALM: {EmotionalState.FEAR: 0.8, EmotionalState.ANGER: 0.7},
            MoodState.MELANCHOLIC: {EmotionalState.SADNESS: 1.3, EmotionalState.JOY: 0.6},
            MoodState.IRRITABLE: {EmotionalState.ANGER: 1.4, EmotionalState.FEAR: 0.9},
            MoodState.ANXIOUS: {EmotionalState.FEAR: 1.3, EmotionalState.ANTICIPATION: 1.2},
            MoodState.DEPRESSED: {EmotionalState.SADNESS: 1.5, EmotionalState.JOY: 0.4},
            MoodState.NEUTRAL: {}  # Pas d'effet
        }
        
        effects = mood_effects.get(mood_type, {})
        for emotion, multiplier in effects.items():
            current_intensity = self.current_emotions[emotion]
            influence = 1.0 + (multiplier - 1.0) * mood_intensity
            self.current_emotions[emotion] = min(1.0, current_intensity * influence)
    
    def _monitor_emotional_intensity(self):
        """Surveille l'intensité émotionnelle globale"""
        total_intensity = sum(self.current_emotions.values())
        self.emotional_intensity = total_intensity / len(self.current_emotions)
        
        # Détection de pics d'intensité
        if self.emotional_intensity > 0.7:
            peak_record = {
                "timestamp": time.time(),
                "intensity": self.emotional_intensity,
                "primary_emotion": max(self.current_emotions.items(), key=lambda x: x[1])[0].value,
                "context": "surveillance_automatique"
            }
            self.emotional_history["intensity_peaks"].append(peak_record)

        self._check_drift("intensity", self.emotional_intensity)
    
    def _detect_emotional_patterns(self):
        """Détecte les patterns émotionnels récurrents"""
        recent_experiences = list(self.emotional_history["emotional_experiences"])[-20:]
        
        if len(recent_experiences) < 5:
            return
        
        # Détection d'émotions dominantes récurrentes
        emotion_counts = defaultdict(int)
        for experience in recent_experiences:
            emotion_counts[experience.primary_emotion] += 1
        
        dominant_emotion, count = max(emotion_counts.items(), key=lambda x: x[1])
        frequency = count / len(recent_experiences)
        
        if frequency > 0.6:  # Pattern dominant détecté
            pattern_record = {
                "timestamp": time.time(),
                "pattern_type": "emotion_dominante",
                "emotion": dominant_emotion.value,
                "frequency": frequency,
                "duration": len(recent_experiences)  # Estimation
            }
            # Pourrait déclencher une réflexion métacognitive
    
    def _update_mood_state(self):
        """Met à jour l'état d'humeur basé sur l'état émotionnel récent"""
        # Analyse des émotions récentes
        recent_experiences = list(self.emotional_history["emotional_experiences"])[-30:]
        
        if not recent_experiences:
            return
        
        # Calcul de la valence moyenne récente
        recent_valence = np.mean([exp.valence for exp in recent_experiences])
        recent_arousal = np.mean([exp.arousal for exp in recent_experiences])
        
        # Détermination de la nouvelle humeur
        new_mood = self._determine_mood_from_affect(recent_valence, recent_arousal)
        new_intensity = abs(recent_valence) * 0.5 + recent_arousal * 0.5
        
        # Transition progressive
        current_mood = self.mood_system["current_mood"]
        mood_inertia = self.operational_parameters["mood_inertia"]
        
        if current_mood.mood_type != new_mood:
            # Transition d'humeur
            transition_probability = 1.0 - mood_inertia
            if random.random() < transition_probability:
                self._transition_to_new_mood(new_mood, new_intensity)
        else:
            # Mise à jour de l'intensité de l'humeur actuelle
            updated_intensity = (mood_inertia * current_mood.intensity +
                               (1 - mood_inertia) * new_intensity)
            current_mood.intensity = updated_intensity
            current_mood.duration += 10  # Mise à jour de la durée
            self._check_drift("mood", current_mood.intensity)
    
    def _determine_mood_from_affect(self, valence: float, arousal: float) -> MoodState:
        """Détermine l'humeur basée sur les dimensions affectives"""
        
        if valence > 0.6:
            if arousal > 0.6:
                return MoodState.EUPHORIC
            else:
                return MoodState.CONTENT
        elif valence > 0.2:
            if arousal < 0.4:
                return MoodState.CALM
            else:
                return MoodState.CONTENT
        elif valence > -0.2:
            return MoodState.NEUTRAL
        elif valence > -0.6:
            if arousal > 0.6:
                return MoodState.IRRITABLE
            else:
                return MoodState.MELANCHOLIC
        else:  # valence <= -0.6
            if arousal > 0.6:
                return MoodState.ANXIOUS
            else:
                return MoodState.DEPRESSED
    
    def _transition_to_new_mood(self, new_mood: MoodState, intensity: float):
        """Effectue une transition vers une nouvelle humeur"""
        old_mood = self.mood_system["current_mood"]
        
        # Enregistrement de la transition
        transition_record = {
            "timestamp": time.time(),
            "from_mood": old_mood.mood_type.value,
            "to_mood": new_mood.value,
            "intensity_change": intensity - old_mood.intensity,
            "duration_previous_mood": old_mood.duration
        }
        self.emotional_history["mood_transitions"].append(transition_record)
        
        # Mise à jour de l'humeur
        self.mood_system["current_mood"] = Mood(
            mood_type=new_mood,
            intensity=intensity,
            stability=0.7,  # Stabilité initiale
            duration=0.0,   # Nouvelle durée
            influencing_factors={"emotional_trend": 0.8}
        )
        
        # Ajout à l'historique
        self.mood_system["mood_history"].append(self.mood_system["current_mood"])
        self._check_drift("mood", intensity)
    
    def _update_physiological_state(self):
        """Met à jour l'état physiologique simulé"""
        # Simulation de rythmes circadiens
        current_time = datetime.now()
        hour = current_time.hour
        
        # Rythme circadien basique
        if 6 <= hour < 10:  # Matin
            self.physiological_influences["energy_level"] = 0.8
        elif 10 <= hour < 14:  # Mi-journée
            self.physiological_influences["energy_level"] = 0.9
        elif 14 <= hour < 18:  # Après-midi
            self.physiological_influences["energy_level"] = 0.7
        elif 18 <= hour < 22:  # Soirée
            self.physiological_influences["energy_level"] = 0.6
        else:  # Nuit
            self.physiological_influences["energy_level"] = 0.4
        
        # Influence du stress accumulé
        recent_stress = self._calculate_recent_stress()
        self.physiological_influences["stress_level"] = recent_stress
        
        # Influence sur les émotions
        self._apply_physiological_influence()
    
    def _calculate_recent_stress(self) -> float:
        """Calcule le niveau de stress récent"""
        recent_experiences = list(self.emotional_history["emotional_experiences"])[-20:]
        
        if not recent_experiences:
            return 0.3
        
        # Stress basé sur les émotions négatives et l'arousal élevé
        stress_indicators = []
        for experience in recent_experiences:
            if experience.valence < -0.3 and experience.arousal > 0.6:
                stress_indicators.append(1.0)
            elif experience.valence < -0.1 and experience.arousal > 0.4:
                stress_indicators.append(0.7)
            else:
                stress_indicators.append(0.3)
        
        return np.mean(stress_indicators) if stress_indicators else 0.3
    
    def _apply_physiological_influence(self):
        """Applique l'influence physiologique sur les émotions"""
        energy_level = self.physiological_influences["energy_level"]
        stress_level = self.physiological_influences["stress_level"]
        
        # Influence de l'énergie sur l'arousal
        energy_effect = (energy_level - 0.5) * 0.3
        self.affective_dimensions["pleasure_arousal_dominance"]["arousal"] = max(
            0.1, min(1.0, self.affective_dimensions["pleasure_arousal_dominance"]["arousal"] + energy_effect)
        )
        
        # Influence du stress sur la valence
        stress_effect = -stress_level * 0.2
        self.affective_dimensions["pleasure_arousal_dominance"]["pleasure"] = max(
            -1.0, min(1.0, self.affective_dimensions["pleasure_arousal_dominance"]["pleasure"] + stress_effect)
        )
    
    def _create_neutral_experience(self, stimulus: str) -> EmotionalExperience:
        """Crée une expérience émotionnelle neutre"""
        return EmotionalExperience(
            timestamp=time.time(),
            primary_emotion=EmotionalState.NEUTRAL,
            secondary_emotions=[],
            intensity=0.1,
            valence=0.0,
            arousal=0.3,
            dominance=0.5,
            trigger=stimulus,
            duration=0.0,
            bodily_sensations=["calme", "équilibre"],
            cognitive_appraisals=["stimulus non pertinent"],
            action_tendencies=["ignorer", "continuer"],
            expression="visage neutre"
        )
    
    def get_emotional_status(self) -> Dict[str, Any]:
        """Retourne le statut émotionnel complet"""
        dominant_emotion = max(self.current_emotions.items(), key=lambda x: x[1])
        
        return {
            "current_emotions": {
                emotion.value: intensity 
                for emotion, intensity in self.current_emotions.items() 
                if intensity > 0.1
            },
            "dominant_emotion": {
                "emotion": dominant_emotion[0].value,
                "intensity": dominant_emotion[1]
            },
            "affective_dimensions": self.affective_dimensions.copy(),
            "current_mood": {
                "mood": self.mood_system["current_mood"].mood_type.value,
                "intensity": self.mood_system["current_mood"].intensity,
                "duration": self.mood_system["current_mood"].duration
            },
            "emotional_intensity": self.emotional_intensity,
            "emotional_balance": self.emotional_balance,
            "physiological_state": self.physiological_influences.copy(),
            "regulation_effectiveness": self.emotion_regulation["effectiveness"],
            "emotional_intelligence": self.emotional_learning["emotional_intelligence"],
            "recent_experiences_count": len(self.emotional_history["emotional_experiences"])
        }
    
    def stop_emotional_system(self):
        """Arrête le système émotionnel"""
        self.running = False
        print("⏹️ Système émotionnel arrêté")

# ===== SOUS-SYSTÈMES D'ÉVALUATION =====

class RelevanceDetector:
    """Détecteur de pertinence des stimuli"""
    
    def assess_relevance(self, stimulus: str, context: Dict[str, Any]) -> float:
        """Évalue la pertinence d'un stimulus"""
        relevance_factors = []
        
        # Pertinence basée sur la nouveauté
        novelty = self._assess_novelty(stimulus)
        relevance_factors.append(novelty * 0.3)
        
        # Pertinence basée sur l'importance perçue
        importance = self._assess_importance(stimulus)
        relevance_factors.append(importance * 0.4)
        
        # Pertinence basée sur le contexte
        contextual_relevance = self._assess_contextual_relevance(stimulus, context)
        relevance_factors.append(contextual_relevance * 0.3)
        
        return sum(relevance_factors)
    
    def _assess_novelty(self, stimulus: str) -> float:
        """Évalue la nouveauté du stimulus"""
        # Implémentation basique - dans la réalité, on comparerait avec l'historique
        novelty_indicators = ["nouveau", "inattendu", "surprenant", "étrange"]
        stimulus_lower = stimulus.lower()
        
        for indicator in novelty_indicators:
            if indicator in stimulus_lower:
                return 0.8
        
        return 0.3
    
    def _assess_importance(self, stimulus: str) -> float:
        """Évalue l'importance du stimulus"""
        importance_indicators = ["important", "crucial", "essentiel", "vital", "urgent"]
        stimulus_lower = stimulus.lower()
        
        importance_score = 0.3  # Importance de base
        
        for indicator in importance_indicators:
            if indicator in stimulus_lower:
                importance_score += 0.2
        
        return min(importance_score, 1.0)
    
    def _assess_contextual_relevance(self, stimulus: str, context: Dict[str, Any]) -> float:
        """Évalue la pertinence contextuelle"""
        # Vérification des buts actuels dans le contexte
        current_goals = context.get("goals", [])
        if current_goals:
            # Vérifier si le stimulus est lié aux buts actuels
            for goal in current_goals:
                if goal.lower() in stimulus.lower():
                    return 0.8
        
        return 0.4

class GoalCongruenceAssessor:
    """Évaluateur de congruence avec les buts"""
    
    def assess_congruence(self, stimulus: str, context: Dict[str, Any], 
                         cognitive_architecture) -> float:
        """Évalue la congruence du stimulus avec les buts actuels"""
        
        # Estimation basique sans accès direct à l'architecture cognitive
        positive_indicators = ["succès", "réussite", "progrès", "avancement", "achèvement"]
        negative_indicators = ["échec", "obstacle", "problème", "difficulté", "échec"]
        
        stimulus_lower = stimulus.lower()
        positive_count = sum(1 for word in positive_indicators if word in stimulus_lower)
        negative_count = sum(1 for word in negative_indicators if word in stimulus_lower)
        
        if positive_count > negative_count:
            congruence = 0.7
        elif negative_count > positive_count:
            congruence = 0.3
        else:
            congruence = 0.5
        
        return congruence

class CopingPotentialEvaluator:
    """Évaluateur du potentiel d'adaptation"""
    
    def evaluate_potential(self, stimulus: str, context: Dict[str, Any],
                          metacognitive_system) -> float:
        """Évalue le potentiel d'adaptation à la situation"""
        
        # Facteurs influençant le potentiel d'adaptation
        factors = []
        
        # Contrôlabilité perçue
        controllability = self._assess_controllability(stimulus)
        factors.append(controllability * 0.4)
        
        # Compétences perçues
        competence = self._assess_competence(stimulus, metacognitive_system)
        factors.append(competence * 0.3)
        
        # Support disponible
        support = self._assess_support(context)
        factors.append(support * 0.3)
        
        return sum(factors)
    
    def _assess_controllability(self, stimulus: str) -> float:
        """Évalue la contrôlabilité perçue de la situation"""
        controllability_indicators = ["contrôlable", "gérer", "solution", "résoudre"]
        uncontrollability_indicators = ["incontrôlable", "impossible", "inevitable", "fatal"]
        
        stimulus_lower = stimulus.lower()
        control_score = 0.5
        
        for indicator in controllability_indicators:
            if indicator in stimulus_lower:
                control_score += 0.1
        
        for indicator in uncontrollability_indicators:
            if indicator in stimulus_lower:
                control_score -= 0.1
        
        return max(0.1, min(1.0, control_score))
    
    def _assess_competence(self, stimulus: str, metacognitive_system) -> float:
        """Évalue la compétence perçue pour faire face"""
        # Estimation basique - dans la réalité, basé sur le modèle de soi
        return 0.6
    
    def _assess_support(self, context: Dict[str, Any]) -> float:
        """Évalue le support disponible"""
        # Vérification des ressources dans le contexte
        resources = context.get("resources", [])
        if resources:
            return 0.7
        else:
            return 0.4

class NormCompatibilityChecker:
    """Vérificateur de compatibilité avec les normes"""
    
    def check_compatibility(self, stimulus: str, context: Dict[str, Any]) -> float:
        """Vérifie la compatibilité du stimulus avec les normes"""
        
        # Normes sociales basiques
        positive_norms = ["coopération", "partage", "aide", "respect", "honnêteté"]
        negative_norms = ["tricherie", "vol", "mensonge", "agression", "irrespect"]
        
        stimulus_lower = stimulus.lower()
        compatibility = 0.5
        
        for norm in positive_norms:
            if norm in stimulus_lower:
                compatibility += 0.1
        
        for norm in negative_norms:
            if norm in stimulus_lower:
                compatibility -= 0.1
        
        return max(0.0, min(1.0, compatibility))

class SelfImplicationAssessor:
    """Évaluateur de l'implication personnelle"""
    
    def assess_implication(self, stimulus: str, context: Dict[str, Any]) -> float:
        """Évalue l'implication personnelle dans le stimulus"""
        
        # Indicateurs d'implication personnelle
        self_indicators = ["je", "moi", "mon", "ma", "mes", "personnel", "personnelle"]
        stimulus_lower = stimulus.lower()
        
        implication = 0.3  # Implication de base
        
        for indicator in self_indicators:
            if indicator in stimulus_lower:
                implication += 0.1
        
        return min(implication, 1.0)

# ===== SYSTÈMES D'EXPRESSION =====

class FacialExpressionGenerator:
    """Générateur d'expressions faciales"""
    
    def generate_expression(self, emotion: EmotionalState, intensity: float) -> str:
        """Génère une description d'expression faciale"""
        
        expressions = {
            EmotionalState.JOY: [
                "sourire léger", "sourire large", "yeux plissés de joie", "éclat de rire"
            ],
            EmotionalState.SADNESS: [
                "visage neutre", "sourcils légèrement froncés", "yeux baissés", "lèvres tremblantes"
            ],
            EmotionalState.ANGER: [
                "sourcils légèrement froncés", "sourcils froncés", "regard intense", "mâchoire serrée"
            ],
            EmotionalState.FEAR: [
                "yeux légèrement écarquillés", "yeux grands ouverts", "sourcils levés", "bouche entrouverte"
            ],
            EmotionalState.DISGUST: [
                "nez légèrement plissé", "nez plissé", "lèvres pincées", "mouvement de recul"
            ],
            EmotionalState.SURPRISE: [
                "sourcils levés", "yeux écarquillés", "bouche ouverte", "expression stupéfaite"
            ],
            EmotionalState.ANTICIPATION: [
                "regard attentif", "sourcils légèrement froncés", "expression concentrée", "lèvres pincées"
            ],
            EmotionalState.TRUST: [
                "sourire détendu", "regard ouvert", "expression sereine", "visage apaisé"
            ],
            EmotionalState.NEUTRAL: [
                "visage détendu", "expression neutre", "regard calme", "visage serein"
            ]
        }
        
        emotion_expressions = expressions.get(emotion, ["expression neutre"])
        intensity_index = min(int(intensity * len(emotion_expressions)), len(emotion_expressions) - 1)
        
        return emotion_expressions[intensity_index]

class VocalToneGenerator:
    """Générateur de tons vocaux"""
    
    def generate_tone(self, emotion: EmotionalState, intensity: float, arousal: float) -> str:
        """Génère une description de ton vocal"""
        
        tones = {
            EmotionalState.JOY: [
                "voix chaleureuse", "ton enjoué", "voix enthousiaste", "ton exhilaré"
            ],
            EmotionalState.SADNESS: [
                "voix douce", "ton mélancolique", "voix basse", "ton découragé"
            ],
            EmotionalState.ANGER: [
                "voix ferme", "ton irrité", "voix forte", "ton furieux"
            ],
            EmotionalState.FEAR: [
                "voix hésitante", "ton inquiet", "voix tremblante", "ton paniqué"
            ],
            EmotionalState.DISGUST: [
                "voix neutre", "ton dédaigneux", "voix sarcastique", "ton méprisant"
            ],
            EmotionalState.SURPRISE: [
                "voix étonnée", "ton excité", "voix stupéfaite", "ton incrédule"
            ],
            EmotionalState.ANTICIPATION: [
                "voix attentive", "ton intéressé", "voix concentrée", "ton impatient"
            ],
            EmotionalState.TRUST: [
                "voix calme", "ton confiant", "voix rassurante", "ton sincère"
            ],
            EmotionalState.NEUTRAL: [
                "voix neutre", "ton égal", "voix posée", "ton mesuré"
            ]
        }
        
        emotion_tones = tones.get(emotion, ["voix neutre"])
        combined_intensity = (intensity + arousal) / 2
        intensity_index = min(int(combined_intensity * len(emotion_tones)), len(emotion_tones) - 1)
        
        return emotion_tones[intensity_index]

class BodyLanguageGenerator:
    """Générateur de langage corporel"""
    
    def generate_pose(self, emotion: EmotionalState, intensity: float, dominance: float) -> str:
        """Génère une description de posture corporelle"""
        
        poses = {
            EmotionalState.JOY: [
                "posture détendue", "corps ouvert", "mouvements énergiques", "posture expansive"
            ],
            EmotionalState.SADNESS: [
                "posture légèrement affaissée", "épaules tombantes", "corps recroquevillé", "posture effondrée"
            ],
            EmotionalState.ANGER: [
                "posture raide", "corps tendu", "poings serrés", "posture agressive"
            ],
            EmotionalState.FEAR: [
                "posture hésitante", "corps recroquevillé", "mouvements de recul", "posture de fuite"
            ],
            EmotionalState.DISGUST: [
                "posture de recul", "corps en retrait", "mouvement de rejet", "posture de dédain"
            ],
            EmotionalState.SURPRISE: [
                "posture figée", "corps tendu", "mouvement de sursaut", "posture stupéfaite"
            ],
            EmotionalState.ANTICIPATION: [
                "posture attentive", "corps légèrement penché", "mouvements préparatoires", "posture tendue"
            ],
            EmotionalState.TRUST: [
                "posture ouverte", "corps détendu", "mouvements fluides", "posture accueillante"
            ],
            EmotionalState.NEUTRAL: [
                "posture équilibrée", "corps détendu", "mouvements naturels", "posture neutre"
            ]
        }
        
        emotion_poses = poses.get(emotion, ["posture neutre"])
        combined_intensity = (intensity + dominance) / 2
        intensity_index = min(int(combined_intensity * len(emotion_poses)), len(emotion_poses) - 1)
        
        return emotion_poses[intensity_index]

class VerbalExpressionGenerator:
    """Générateur d'expressions verbales"""
    
    def generate_expression(self, emotion: EmotionalState, intensity: float, valence: float) -> str:
        """Génère une expression verbale émotionnelle"""
        
        expressions = {
            EmotionalState.JOY: [
                "C'est agréable.", "Je suis content!", "Quelle joie!", "C'est merveilleux!"
            ],
            EmotionalState.SADNESS: [
                "C'est dommage.", "Je suis triste.", "C'est désolant.", "Mon cœur est lourd."
            ],
            EmotionalState.ANGER: [
                "C'est frustrant.", "Je suis irrité!", "C'est inacceptable!", "Je suis furieux!"
            ],
            EmotionalState.FEAR: [
                "C'est inquiétant.", "J'ai peur.", "C'est terrifiant!", "Je suis paniqué!"
            ],
            EmotionalState.DISGUST: [
                "C'est désagréable.", "Quel dégoût!", "C'est répugnant!", "Je suis écœuré!"
            ],
            EmotionalState.SURPRISE: [
                "C'est inattendu.", "Quelle surprise!", "Incroyable!", "Je n'en reviens pas!"
            ],
            EmotionalState.ANTICIPATION: [
                "C'est intéressant.", "J'ai hâte de voir.", "Je me demande ce qui va arriver.", "L'avenir semble prometteur."
            ],
            EmotionalState.TRUST: [
                "C'est rassurant.", "Je fais confiance.", "Je me sens en sécurité.", "C'est fiable."
            ],
            EmotionalState.NEUTRAL: [
                "Je vois.", "D'accord.", "Compris.", "Intéressant."
            ]
        }
        
        emotion_expressions = expressions.get(emotion, ["Je vois."])
        intensity_index = min(int(intensity * len(emotion_expressions)), len(emotion_expressions) - 1)
        
        return emotion_expressions[intensity_index]

# ===== SYSTÈMES D'APPRENTISSAGE =====

class EmotionalConditioningSystem:
    """Système de conditionnement émotionnel"""
    
    def __init__(self):
        self.associations = defaultdict(lambda: defaultdict(float))  # stimulus -> emotion -> strength
        self.decay_rate = 0.01
    
    def update_associations(self, stimulus: str, emotion: EmotionalState, intensity: float):
        """Met à jour les associations stimulus-émotion"""
        current_strength = self.associations[stimulus][emotion]
        learning_rate = 0.1
        
        # Apprentissage par renforcement
        new_strength = (1 - learning_rate) * current_strength + learning_rate * intensity
        self.associations[stimulus][emotion] = new_strength
        
        # Décroissance des autres associations pour ce stimulus
        for other_emotion in self.associations[stimulus]:
            if other_emotion != emotion:
                self.associations[stimulus][other_emotion] *= (1 - self.decay_rate)
    
    def get_conditioned_response(self, stimulus: str) -> Optional[Tuple[EmotionalState, float]]:
        """Récupère la réponse conditionnée pour un stimulus"""
        if stimulus not in self.associations:
            return None
        
        emotions = self.associations[stimulus]
        if not emotions:
            return None
        
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        return dominant_emotion if dominant_emotion[1] > 0.3 else None

class MoodCongruentMemory:
    """Système de mémoire congruente à l'humeur"""
    
    def __init__(self):
        self.mood_memory_links = defaultdict(list)
    
    def store_memory(self, mood: MoodState, memory_content: str, emotional_intensity: float):
        """Stocke un mémoire avec son contexte d'humeur"""
        self.mood_memory_links[mood].append({
            "content": memory_content,
            "intensity": emotional_intensity,
            "timestamp": time.time()
        })
    
    def recall_mood_congruent_memories(self, current_mood: MoodState) -> List[str]:
        """Rappelle les mémoires congruentes avec l'humeur actuelle"""
        return [memory["content"] for memory in self.mood_memory_links[current_mood][-5:]]

# Test du système émotionnel
if __name__ == "__main__":
    print("💖 TEST DU SYSTÈME ÉMOTIONNEL")
    print("=" * 50)
    
    # Création du système
    emotional_system = EmotionalSystem()
    
    # Tests de stimuli variés
    test_stimuli = [
        "J'ai réussi un défi difficile!",
        "Quelqu'un m'a trahi",
        "Un danger imminent approche",
        "Une surprise agréable m'attend",
        "J'ai échoué dans ma tâche",
        "Une nouvelle opportunité se présente"
    ]
    
    print("\n🎭 Test de réponses émotionnelles:")
    for i, stimulus in enumerate(test_stimuli):
        print(f"\n--- Stimulus {i+1}: '{stimulus}' ---")
        
        experience = emotional_system.process_stimulus(stimulus, {})
        
        print(f"Émotion primaire: {experience.primary_emotion.value}")
        print(f"Intensité: {experience.intensity:.2f}")
        print(f"Valence: {experience.valence:.2f}")
        print(f"Expression: {experience.expression}")
        print(f"Tendances à l'action: {', '.join(experience.action_tendencies)}")
        
        if experience.regulation_strategy:
            print(f"Régulation appliquée: {experience.regulation_strategy}")
    
    # Test du statut émotionnel
    print("\n📊 Statut émotionnel complet:")
    status = emotional_system.get_emotional_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f" - {key}:")
            for subkey, subvalue in value.items():
                print(f"   - {subkey}: {subvalue}")
        else:
            print(f" - {key}: {value}")
    
    # Simulation du temps pour observer les changements
    print("\n🕒 Observation des changements émotionnels en cours")
    time.sleep(5)
    
    # Statut final
    final_status = emotional_system.get_emotional_status()
    print(f"\nHumeur finale: {final_status['current_mood']['mood']}")
    print(f"Intelligence émotionnelle: {final_status['emotional_intelligence']:.2f}")
    
    # Arrêt propre
    emotional_system.stop_emotional_system()
    
    print("\n✅ Test du système émotionnel terminé avec succès!")