# metacognition/__init__.py
"""
Système de Métacognition Avancée de l'AGI Évolutive
Capacité à réfléchir sur ses propres processus de pensée, à se comprendre et à s'auto-améliorer
"""

import logging
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import math
import json
import inspect
import re

from AGI_Evolutive.cognition.meta_cognition import OnlineLinear
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

from .experimentation import MetacognitionExperimenter, calibrate_self_model


LOGGER = logging.getLogger(__name__)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))

class MetacognitiveState(Enum):
    """États métacognitifs possibles"""
    MONITORING = "surveillance"
    REFLECTING = "réflexion"
    PLANNING = "planification"
    ADJUSTING = "ajustement"
    CRITICAL_SELF_EVALUATION = "auto_évaluation_critique"
    INSIGHT_GENERATION = "génération_insight"

class CognitiveDomain(Enum):
    """Domaines cognitifs surveillés"""
    PERCEPTION = "perception"
    MEMORY = "mémoire"
    REASONING = "raisonnement"
    LEARNING = "apprentissage"
    DECISION_MAKING = "prise_décision"
    PROBLEM_SOLVING = "résolution_problème"
    ATTENTION = "attention"
    LANGUAGE = "langage"
    SOCIAL = "social"

@dataclass
class MetacognitiveEvent:
    """Événement métacognitif enregistré"""
    timestamp: float
    event_type: str
    domain: CognitiveDomain
    description: str
    significance: float
    confidence: float
    emotional_valence: float
    cognitive_load: float
    related_memories: List[str] = field(default_factory=list)
    insights_generated: List[str] = field(default_factory=list)
    action_taken: Optional[str] = None

@dataclass
class SelfModel:
    """Modèle de soi dynamique pour la métacognition.

    À ce niveau on stocke des évaluations internes (performances, styles,
    limitations).  Le module ``core.self_model`` gère, lui, l'identité
    persistante/persona sauvegardée sur disque.
    """
    # Capacités cognitives auto-évaluées
    cognitive_abilities: Dict[str, float] = field(default_factory=lambda: {
        "memory_capacity": 0.5,
        "reasoning_speed": 0.5,
        "learning_efficiency": 0.5,
        "attention_control": 0.5,
        "problem_solving": 0.5,
        "creativity": 0.3,
        "emotional_intelligence": 0.4,
        "relationship_management": 0.4,
    })
    
    # Limitations connues
    known_limitations: Dict[str, str] = field(default_factory=dict)
    
    # Préférences et styles cognitifs
    cognitive_styles: Dict[str, float] = field(default_factory=lambda: {
        "analytical_thinking": 0.7,
        "intuitive_thinking": 0.3,
        "focused_attention": 0.6,
        "distributed_attention": 0.4,
        "risk_taking": 0.4,
        "caution": 0.6
    })
    
    # Historique des performances
    performance_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Modèle de compétences par domaine
    domain_expertise: Dict[str, float] = field(default_factory=lambda: {
        "mathematics": 0.3,
        "language": 0.5,
        "spatial_reasoning": 0.4,
        "social_cognition": 0.3,
        "physical_intuition": 0.5
    })

@dataclass
class ReflectionSession:
    """Session de réflexion métacognitive structurée"""
    start_time: float
    trigger: str
    focus_domain: CognitiveDomain
    depth_level: int  # 1: superficiel, 3: profond
    insights: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    action_plans: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    quality_score: float = 0.0


class ThompsonBandit:
    """Bandit de Thompson discret pour la sélection adaptative d'options."""

    def __init__(self, arms: List[Any], prior: Tuple[float, float] = (1.0, 1.0)):
        if not arms:
            raise ValueError("ThompsonBandit requires at least one arm")
        self.arms: List[Any] = list(arms)
        self.alpha: Dict[Any, float] = {arm: float(prior[0]) for arm in self.arms}
        self.beta: Dict[Any, float] = {arm: float(prior[1]) for arm in self.arms}
        self.last_selected: Optional[Any] = None

    def select(self) -> Any:
        samples = {}
        for arm in self.arms:
            a = max(1e-3, self.alpha.get(arm, 1.0))
            b = max(1e-3, self.beta.get(arm, 1.0))
            samples[arm] = np.random.beta(a, b)
        best_arm = max(samples.items(), key=lambda item: item[1])[0]
        self.last_selected = best_arm
        return best_arm

    def update(self, arm: Any, reward: float):
        if arm not in self.alpha:
            return
        reward = max(0.0, min(1.0, float(reward)))
        self.alpha[arm] = self.alpha.get(arm, 1.0) + reward
        self.beta[arm] = self.beta.get(arm, 1.0) + (1.0 - reward)

    def add_arm(self, arm: Any, prior: Tuple[float, float] = (1.0, 1.0)):
        if arm in self.alpha:
            return
        self.arms.append(arm)
        self.alpha[arm] = float(prior[0])
        self.beta[arm] = float(prior[1])

class MetacognitiveSystem:
    """
    Système de métacognition avancé - Le "surveillant interne" de l'AGI
    Implémente la conscience de ses propres processus cognitifs et capacités d'auto-amélioration
    """
    
    def __init__(self, cognitive_architecture=None, memory_system=None, reasoning_system=None):
        self.cognitive_architecture = cognitive_architecture
        self.architecture = cognitive_architecture
        self.memory_system = memory_system
        self.reasoning_system = reasoning_system
        self.creation_time = time.time()

        # --- LIAISONS INTER-MODULES ---
        if self.cognitive_architecture is not None:
            self.goals = getattr(self.cognitive_architecture, "goals", None)
            self.emotions = getattr(self.cognitive_architecture, "emotions", None)
            self.learning = getattr(self.cognitive_architecture, "learning", None)
            self.creativity = getattr(self.cognitive_architecture, "creativity", None)
            self.perception = getattr(self.cognitive_architecture, "perception", None)
            self.language = getattr(self.cognitive_architecture, "language", None)
            self.world_model = getattr(self.cognitive_architecture, "world_model", None)

        
        # === MODÈLE DE SOI DYNAMIQUE ===
        self.self_model = SelfModel()
        self.self_model_accuracy = 0.3  # Précision initiale du modèle de soi
        self.self_model_update_interval = 60  # secondes
        
        # === SYSTÈME DE SURVEILLANCE COGNITIVE ===
        self.cognitive_monitoring = {
            "performance_tracking": defaultdict(list),
            "error_detection": ErrorDetectionSystem(),
            "bias_monitoring": BiasMonitoringSystem(),
            "resource_monitoring": ResourceMonitoringSystem(),
            "progress_tracking": ProgressTrackingSystem()
        }
        
        # === MOTEUR DE RÉFLEXION ===
        self.reflection_engine = {
            "scheduled_reflections": [],
            "triggered_reflections": [],
            "reflection_depth": 1,
            "insight_threshold": 0.7,
            "reflection_frequency": 0.3
        }
        
        # === CONTRÔLE COGNITIF ADAPTATIF ===
        self.cognitive_control = {
            "strategy_selection": StrategySelector(),
            "attention_allocation": MetacognitiveAttention(),
            "effort_regulation": EffortRegulator(),
            "goal_management": MetacognitiveGoalManager()
        }
        
        # === BASE DE CONNAISSANCES MÉTACOGNITIVE ===
        self.metacognitive_knowledge = {
            "learning_strategies": self._initialize_learning_strategies(),
            "problem_solving_heuristics": self._initialize_problem_solving_heuristics(),
            "error_patterns": self._initialize_error_patterns(),
            "performance_benchmarks": self._initialize_performance_benchmarks()
        }
        
        # === HISTORIQUE MÉTACOGNITIF ===
        self.metacognitive_history = {
            "events": deque(maxlen=1000),
            "reflection_sessions": deque(maxlen=100),
            "insights": deque(maxlen=500),
            "self_improvements": deque(maxlen=200),
            "error_corrections": deque(maxlen=300)
        }

        # === ÉTATS MÉTACOGNITIFS DYNAMIQUES ===
        self.metacognitive_states = {
            "awareness_level": 0.1,
            "introspection_depth": 0.2,
            "self_understanding": 0.1,
            "adaptive_capacity": 0.3,
            "insight_readiness": 0.4,
            "cognitive_flexibility": 0.5,
            "relationship_attunement": 0.3,
        }

        # === PARAMÈTRES DE FONCTIONNEMENT ===
        self.operational_parameters = {
            "monitoring_intensity": 0.7,
            "reflection_frequency": 0.3,
            "adjustment_aggressiveness": 0.5,
            "self_model_update_rate": 0.1,
            "error_tolerance": 0.3,
            "improvement_target": 0.8
        }

        # === MÉCANISMES D'ADAPTATION AVANCÉS ===
        self._drift_log: deque = deque(maxlen=200)
        self._recent_reflection_quality: deque = deque(maxlen=50)
        self._reflection_domain_bandit = ThompsonBandit(list(CognitiveDomain))

        self._ema_candidates: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
        self._ema_bandits: Dict[str, ThompsonBandit] = {}
        self._current_ema_coeffs: Dict[str, float] = {}
        self._smoothed_metrics: Dict[str, float] = {}
        self._ema_histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
        self._ema_correlations: Dict[str, float] = {}

        self._state_performance_history: deque = deque(maxlen=120)
        self._state_performance_correlation: Dict[str, float] = {
            "awareness": 0.0,
            "understanding": 0.0,
            "relationship": 0.0,
        }

        self._state_update_model: OnlineLinear = OnlineLinear(
            feature_names=["significance", "confidence", "emotional_valence", "cognitive_load"],
            bounds=(0.0, 1.0),
            lr=0.04,
            l2=0.001,
            max_grad=0.25,
            warmup=24,
            init_weight=0.05,
        )
        self._state_update_context: deque = deque(maxlen=64)

        self._self_model_rate_candidates: List[float] = [0.05, 0.08, 0.1, 0.12, 0.15]
        self._self_model_rate_bandit = ThompsonBandit(self._self_model_rate_candidates)
        self._current_self_model_rate: float = self.operational_parameters["self_model_update_rate"]
        self._last_rate_arm: float = self._current_self_model_rate

        
        # === THREADS DE SURVEILLANCE ===
        self.monitoring_threads = {}
        self.running = True

        self.experimenter = MetacognitionExperimenter(system_ref=self)
        
        # logger si dispo
        self.logger = getattr(self.cognitive_architecture, "logger", None)
        # rapport périodique (toutes les ~90s)
        try:
            self._start_status_reporting(period=90)
        except Exception:
            pass

        # Initialisation des systèmes
        self._initialize_metacognitive_system()

        print("🧠 Système Métacognitif Initialisé")

    # ==============================================================
    # 🧠 MÉTHODES D'INITIALISATION ET DE SURVEILLANCE MÉTACOGNITIVE
    # ==============================================================

    def _start_status_reporting(self, period: int = 90):
        import threading
        import time

        def loop():
            while getattr(self, "running", True):
                try:
                    status = self.get_metacognitive_status()
                    if self.logger:
                        self.logger.write("metacog.report", status=status)
                    time.sleep(period)
                except Exception as e:
                    try:
                        if self.logger:
                            self.logger.write("metacog.error", error=str(e))
                    except Exception:
                        pass
                    time.sleep(30)

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        self.monitoring_threads["metacog_report"] = th

    def _get_reasoning_system(self):
        """Récupère le système de raisonnement de manière sécurisée."""
        # Vérifie d'abord si un attribut 'reasoning_system' direct existe
        reasoning = getattr(self, "reasoning_system", None)
        if reasoning and not isinstance(reasoning, str) and hasattr(reasoning, "reasoning_history"):
            return reasoning

        # Sinon, essaye de le récupérer depuis l'architecture globale
        arch = getattr(self, "cognitive_architecture", None)
        if arch and not isinstance(arch, str):
            reasoning = getattr(arch, "reasoning", None)
            if reasoning and hasattr(reasoning, "reasoning_history"):
                return reasoning

        # Aucun système valide trouvé
        return None


    def _initialize_metacognitive_system(self):
        """Initialise le système métacognitif avec des connaissances de base"""
        innate_knowledge = {
            "basic_monitoring_skills": True,
            "simple_self_assessment": True,
            "error_detection_basic": True,
            "strategy_adjustment_basic": True
        }

        # Démarrage des sous-systèmes de surveillance
        self._start_cognitive_monitoring()
        self._start_self_model_updater()

        # Première réflexion initiale
        initial_reflection = self._perform_initial_self_assessment()
        self.metacognitive_history["reflection_sessions"].append(initial_reflection)

    # ==============================================================
    # 🛠️ MÉCANISMES D'ADAPTATION ET D'APPRENTISSAGE
    # ==============================================================

    def _log_drift(self, parameter: str, old_value: float, new_value: float):
        if not isinstance(old_value, (int, float)) or not isinstance(new_value, (int, float)):
            return
        if abs(new_value - old_value) < 1e-6:
            return
        drift = {
            "timestamp": time.time(),
            "parameter": parameter,
            "old": float(old_value),
            "new": float(new_value),
            "delta": float(new_value - old_value),
        }
        self._drift_log.append(drift)
        if self.logger:
            try:
                self.logger.write("metacog.drift", **drift)
            except Exception:
                pass

    def _get_ema_bandit(self, metric: str) -> ThompsonBandit:
        bandit = self._ema_bandits.get(metric)
        if bandit is None:
            bandit = ThompsonBandit(list(self._ema_candidates))
            self._ema_bandits[metric] = bandit
        return bandit

    def _apply_adaptive_ema(self, metric: str, value: float) -> float:
        bandit = self._get_ema_bandit(metric)
        selected_arm = bandit.select()
        previous_coeff = self._current_ema_coeffs.get(metric, float(selected_arm))
        target_coeff = float(selected_arm)
        max_step = 0.25
        delta = target_coeff - previous_coeff
        if abs(delta) > max_step:
            target_coeff = previous_coeff + math.copysign(max_step, delta)
        target_coeff = max(min(self._ema_candidates), min(max(self._ema_candidates), target_coeff))
        if metric in self._current_ema_coeffs and abs(target_coeff - previous_coeff) > 0.15:
            self._log_drift(f"ema_{metric}", previous_coeff, target_coeff)
        self._current_ema_coeffs[metric] = target_coeff

        previous_value = self._smoothed_metrics.get(metric, value)
        smoothed_value = target_coeff * value + (1.0 - target_coeff) * previous_value
        self._smoothed_metrics[metric] = smoothed_value

        quality_signal = self._recent_reflection_quality[-1] if self._recent_reflection_quality else 0.5
        history = self._ema_histories[metric]
        history.append((smoothed_value, quality_signal))

        reward = 0.5
        if len(history) >= 4:
            smoothed_values = np.array([entry[0] for entry in history])
            quality_values = np.array([entry[1] for entry in history])
            if np.std(smoothed_values) > 1e-6 and np.std(quality_values) > 1e-6:
                corr = np.corrcoef(smoothed_values, quality_values)[0, 1]
                reward = max(0.0, min(1.0, 0.5 + 0.5 * corr))
                previous_corr = self._ema_correlations.get(metric)
                if previous_corr is not None and corr < previous_corr - 0.2:
                    self._log_drift(f"ema_corr_{metric}", previous_corr, corr)
                self._ema_correlations[metric] = corr

        bandit.update(selected_arm, reward)
        return smoothed_value

    def _record_state_performance_sample(self, metric: str, value: float):
        sample = {
            "timestamp": time.time(),
            "metric": metric,
            "value": float(value),
            "awareness": float(self.metacognitive_states.get("awareness_level", 0.0)),
            "understanding": float(self.metacognitive_states.get("self_understanding", 0.0)),
            "relationship": float(self.metacognitive_states.get("relationship_attunement", 0.0)),
        }
        self._state_performance_history.append(sample)
        if len(self._state_performance_history) >= 6:
            self._update_state_performance_correlation()

    def _update_state_performance_correlation(self):
        data = list(self._state_performance_history)[-40:]
        if not data:
            return
        awareness_values = np.array([entry["awareness"] for entry in data])
        understanding_values = np.array([entry["understanding"] for entry in data])
        relationship_values = np.array([entry["relationship"] for entry in data])
        metric_values = np.array([entry["value"] for entry in data])

        def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            if a.size < 3 or b.size < 3:
                return 0.0
            if np.std(a) < 1e-6 or np.std(b) < 1e-6:
                return 0.0
            matrix = np.corrcoef(a, b)
            return float(matrix[0, 1])

        self._state_performance_correlation["awareness"] = safe_corr(awareness_values, metric_values)
        self._state_performance_correlation["understanding"] = safe_corr(understanding_values, metric_values)
        self._state_performance_correlation["relationship"] = safe_corr(relationship_values, metric_values)

    def _select_self_model_rate(self) -> float:
        arm = self._self_model_rate_bandit.select()
        self._last_rate_arm = float(arm)
        target_rate = float(arm)
        current_rate = self._current_self_model_rate
        max_step = 0.03
        delta = target_rate - current_rate
        if abs(delta) > max_step:
            target_rate = current_rate + math.copysign(max_step, delta)
        min_rate = min(self._self_model_rate_candidates)
        max_rate = max(self._self_model_rate_candidates)
        adjusted_rate = max(min_rate, min(max_rate, target_rate))
        if abs(adjusted_rate - current_rate) > 0.02:
            self._log_drift("self_model_update_rate", current_rate, adjusted_rate)
        self._current_self_model_rate = adjusted_rate
        self.operational_parameters["self_model_update_rate"] = adjusted_rate
        return adjusted_rate

    def _estimate_state_update_reward(
        self,
        event: MetacognitiveEvent,
        before_states: Dict[str, float],
        after_states: Dict[str, float],
    ) -> float:
        delta_awareness = after_states.get("awareness_level", 0.0) - before_states.get("awareness_level", 0.0)
        delta_understanding = after_states.get("self_understanding", 0.0) - before_states.get("self_understanding", 0.0)
        delta_relationship = after_states.get("relationship_attunement", 0.0) - before_states.get("relationship_attunement", 0.0)
        delta_total = delta_awareness + delta_understanding + 0.5 * delta_relationship
        pressure = 0.5 * (event.significance + event.confidence)
        corr_awareness = self._state_performance_correlation.get("awareness", 0.0)
        corr_understanding = self._state_performance_correlation.get("understanding", 0.0)
        corr_relationship = self._state_performance_correlation.get("relationship", 0.0)
        corr_component = 0.4 + 0.2 * corr_awareness + 0.2 * corr_understanding + 0.2 * corr_relationship
        improvement_component = 0.5 + 0.5 * math.tanh(delta_total * 5.0)
        reward = pressure * 0.3 + improvement_component * corr_component * 0.7
        return max(0.0, min(1.0, reward))

    def _register_reflection_outcome(self, reflection: ReflectionSession):
        domain = reflection.focus_domain
        quality = max(0.0, min(1.0, reflection.quality_score))
        self._reflection_domain_bandit.update(domain, quality)
        if quality < 0.3:
            self._log_drift(f"reflection_quality_{domain.value}", reflection.depth_level, quality)

    def _initialize_learning_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les stratégies d'apprentissage connues"""
        return {
            "spaced_repetition": {
                "description": "Réviser à intervalles croissants",
                "effectiveness": 0.8,
                "cognitive_cost": 0.3,
                "applicable_domains": [CognitiveDomain.MEMORY, CognitiveDomain.LEARNING],
                "prerequisites": ["basic_memory_understanding"]
            },
            "elaborative_interrogation": {
                "description": "Se poser des questions 'pourquoi' pour approfondir la compréhension",
                "effectiveness": 0.7,
                "cognitive_cost": 0.6,
                "applicable_domains": [CognitiveDomain.LEARNING, CognitiveDomain.REASONING],
                "prerequisites": ["basic_reasoning_ability"]
            },
            "self_explanation": {
                "description": "Expliquer le matériel à soi-même",
                "effectiveness": 0.6,
                "cognitive_cost": 0.5,
                "applicable_domains": [CognitiveDomain.LEARNING, CognitiveDomain.PROBLEM_SOLVING],
                "prerequisites": ["language_capability"]
            }
        }

    def _initialize_problem_solving_heuristics(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les heuristiques de résolution de problèmes"""
        return {
            "means_end_analysis": {
                "description": "Analyser la différence entre état actuel et but, puis réduire cette différence",
                "effectiveness": 0.8,
                "applicability": 0.9,
                "complexity": 0.7
            },
            "working_backwards": {
                "description": "Commencer par le but et travailler à rebours vers l'état actuel",
                "effectiveness": 0.6,
                "applicability": 0.5,
                "complexity": 0.8
            },
            "analogical_transfer": {
                "description": "Utiliser des solutions de problèmes similaires",
                "effectiveness": 0.7,
                "applicability": 0.8,
                "complexity": 0.6
            }
        }

    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les patterns d'erreur connus"""
        return {
            "confirmation_bias": {
                "description": "Tendance à chercher des informations confirmant ses croyances",
                "detection_difficulty": 0.7,
                "prevalence": 0.8,
                "correction_strategies": ["consider_opposite", "seek_disconfirming_evidence"]
            },
            "anchoring_effect": {
                "description": "Tendance à trop s'appuyer sur la première information reçue",
                "detection_difficulty": 0.6,
                "prevalence": 0.7,
                "correction_strategies": ["consider_multiple_anchors", "delay_judgment"]
            },
            "overconfidence": {
                "description": "Surestimation de ses propres capacités ou connaissances",
                "detection_difficulty": 0.8,
                "prevalence": 0.6,
                "correction_strategies": ["calibration_training", "seek_feedback"]
            }
        }

    def _initialize_performance_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Initialise les repères de performance"""
        return {
            "reasoning_speed": {"excellent": 0.9, "good": 0.7, "average": 0.5, "poor": 0.3},
            "memory_recall": {"excellent": 0.95, "good": 0.8, "average": 0.6, "poor": 0.4},
            "learning_efficiency": {"excellent": 0.85, "good": 0.7, "average": 0.5, "poor": 0.3}
        }

    def _start_cognitive_monitoring(self):
        """Démarre la surveillance cognitive continue"""

        def monitoring_loop():
            while self.running:
                try:
                    reasoning = self._get_reasoning_system()
                    if reasoning is None:
                        time.sleep(1)
                        continue

                    self._monitor_cognitive_performance(reasoning)
                    self._monitor_for_errors(reasoning)
                    self._monitor_cognitive_resources(reasoning)
                    self._monitor_cognitive_biases(reasoning)

                    time.sleep(2)

                except Exception as e:
                    print(f"Erreur dans la surveillance métacognitive: {e}")
                    time.sleep(5)

        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        self.monitoring_threads["cognitive_monitoring"] = monitor_thread

    def _start_self_model_updater(self):
        """Démarre la mise à jour périodique du modèle de soi"""
        def update_loop():
            while self.running:
                try:
                    self._update_self_model()
                    time.sleep(self.self_model_update_interval)
                except Exception as e:
                    print(f"Erreur dans la mise à jour du modèle de soi: {e}")
                    time.sleep(30)

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        self.monitoring_threads["self_model_updater"] = update_thread

    # ==========================================================
    # 🔍 SURVEILLANCE DES PERFORMANCES, ERREURS ET RESSOURCES
    # ==========================================================

    def _monitor_cognitive_performance(self, reasoning):
        """Surveille les performances cognitives globales"""
        if reasoning is None or isinstance(reasoning, str) or not hasattr(reasoning, "get_reasoning_stats"):
            return


        performance_metrics = {}

        try:
            reasoning_stats = getattr(reasoning, "get_reasoning_stats", lambda: {})()
            performance_metrics["reasoning_confidence"] = reasoning_stats.get("average_confidence", 0.5)
            performance_metrics["reasoning_speed"] = self._estimate_reasoning_speed(reasoning)

            if self.memory_system:
                performance_metrics.update(self._assess_memory_performance())

            performance_metrics.update(self._assess_learning_performance(reasoning))

            for metric, value in list(performance_metrics.items()):
                smoothed_value = self._apply_adaptive_ema(metric, value)
                performance_metrics[metric] = smoothed_value
                self._record_state_performance_sample(metric, smoothed_value)
                self.cognitive_monitoring["performance_tracking"][metric].append({
                    "timestamp": time.time(),
                    "value": smoothed_value,
                    "context": "continuous_monitoring"
                })
                try:
                    self.experimenter.record_outcome(metric, new_value=smoothed_value)
                except Exception as _e:
                    print(f"[⚠] record_outcome: {_e}")

            try:
                self.experimenter.suggest_and_log_tests(performance_metrics)
            except Exception as _e:
                print(f"[⚠] suggest_and_log_tests: {_e}")

            self._detect_performance_anomalies(performance_metrics)

        except Exception as e:
            print(f"[⚠️] Erreur dans _monitor_cognitive_performance : {e}")

    def _estimate_reasoning_speed(self, reasoning) -> float:
        """Estime la vitesse de raisonnement"""
        try:
            recent = getattr(reasoning, "reasoning_history", {}).get("recent_inferences", [])
            if not recent:
                return 0.5
            times = [inf.get("reasoning_time", 1.0) for inf in list(recent)[-5:]]
            avg = np.mean(times) if times else 1.0
            return min(1.0 / (1.0 + avg), 1.0)
        except Exception:
            return 0.5

    def _assess_memory_performance(self) -> Dict[str, float]:
        """Évalue la performance de la mémoire (approche basique)"""
        return {
            "recall_accuracy": 0.6,
            "retention_duration": 0.5,
            "memory_capacity": 0.4
        }

    def _assess_learning_performance(self, reasoning) -> Dict[str, float]:
        """Évalue la performance d'apprentissage"""
        metrics = {}
        try:
            trajectory = getattr(reasoning, "reasoning_history", {}).get("learning_trajectory", [])
            if len(trajectory) >= 2:
                recents = [p.get("confidence", 0.5) for p in trajectory[-5:]]
                olds = [p.get("confidence", 0.5) for p in trajectory[:5]]
                if recents and olds:
                    improvement = np.mean(recents) - np.mean(olds)
                    metrics["learning_rate"] = max(0.0, min(1.0, improvement + 0.5))
                else:
                    metrics["learning_rate"] = 0.5
            else:
                metrics["learning_rate"] = 0.3
            metrics.update({"knowledge_acquisition": 0.4, "skill_development": 0.3})
        except Exception:
            metrics = {"learning_rate": 0.3, "knowledge_acquisition": 0.4, "skill_development": 0.3}
        return metrics

    def _detect_performance_anomalies(self, metrics: Dict[str, float]):
        """Détecte les anomalies dans les performances cognitives et enregistre un événement métacognitif."""
        try:
            for metric, current_value in metrics.items():
                historical_data = self.cognitive_monitoring["performance_tracking"][metric]
                if len(historical_data) >= 10:
                    recent_values = [point["value"] for point in list(historical_data)[-10:]]
                    mean_performance = np.mean(recent_values)
                    std_performance = np.std(recent_values)

                    if std_performance > 0:
                        z_score = abs(current_value - mean_performance) / std_performance
                        if z_score > 2.0:
                            self._record_metacognitive_event(
                                event_type="performance_anomaly",
                                domain=CognitiveDomain.LEARNING,
                                description=f"Anomalie détectée sur {metric}: z={z_score:.2f}",
                                significance=min(z_score / 5.0, 1.0),
                                confidence=0.8
                            )
        except Exception as e:
            print(f"[⚠️] Erreur dans _detect_performance_anomalies : {e}")

    def _monitor_for_errors(self, reasoning):
        """Surveille et détecte les erreurs cognitives"""
        try:
            detector = self.cognitive_monitoring["error_detection"]
            for err in detector.detect_reasoning_errors(reasoning):
                self._handle_detected_error(err)
            for err in detector.detect_memory_errors(self.memory_system):
                self._handle_detected_error(err)
            for err in detector.detect_perception_errors():
                self._handle_detected_error(err)
        except Exception as e:
            print(f"[⚠️] Erreur dans _monitor_for_errors : {e}")

    def _monitor_cognitive_resources(self, reasoning):
        """Surveille l'utilisation des ressources cognitives"""
        try:
            resource_monitor = self.cognitive_monitoring["resource_monitoring"]
            arch = getattr(self, "cognitive_architecture", None)
            if arch is None or isinstance(arch, str):
                return

            cognitive_load = resource_monitor.assess_cognitive_load(arch, reasoning)
            if cognitive_load > 0.8:
                self._record_metacognitive_event(
                    event_type="high_cognitive_load",
                    domain=CognitiveDomain.ATTENTION,
                    description=f"Charge cognitive élevée détectée: {cognitive_load:.2f}",
                    significance=0.6,
                    confidence=0.8,
                    cognitive_load=cognitive_load
                )
        except Exception as e:
            print(f"[⚠️] Erreur dans _monitor_cognitive_resources : {e}")

    def _monitor_cognitive_biases(self, reasoning):
        """Surveille les biais cognitifs"""
        try:
            bias_monitor = self.cognitive_monitoring["bias_monitoring"]
            confirmation = bias_monitor.detect_confirmation_bias(reasoning)
            if confirmation.get("detected"):
                self._record_metacognitive_event(
                    event_type="cognitive_bias_detected",
                    domain=CognitiveDomain.REASONING,
                    description=f"Biais de confirmation détecté (force {confirmation['strength']:.2f})",
                    significance=confirmation["strength"],
                    confidence=confirmation["confidence"]
                )
        except Exception as e:
            print(f"[⚠️] Erreur dans _monitor_cognitive_biases : {e}")

 
        
        # Détection de surconfiance
        overconfidence = bias_monitor.detect_overconfidence(self.self_model, self.reasoning_system)
        if overconfidence["detected"]:
            self._record_metacognitive_event(
                event_type="overconfidence_detected",
                domain=CognitiveDomain.DECISION_MAKING,
                description="Surconfiance détectée dans les auto-évaluations",
                significance=0.8,
                confidence=overconfidence["confidence"]
            )
    
    def _handle_detected_error(self, error: Dict[str, Any]):
        """Traite une erreur détectée"""
        # Enregistrement de l'erreur
        self.metacognitive_history["error_corrections"].append({
            "timestamp": time.time(),
            "error_type": error["type"],
            "description": error["description"],
            "severity": error["severity"],
            "corrective_action": error.get("corrective_action", ""),
            "domain": error["domain"]
        })
        
        # Création d'un événement métacognitif
        self._record_metacognitive_event(
            event_type="error_detected",
            domain=error["domain"],
            description=f"Erreur {error['type']}: {error['description']}",
            significance=error["severity"],
            confidence=error["confidence"],
            action_taken=error.get("corrective_action", "En investigation")
        )
        
        # Déclenchement d'une réflexion si l'erreur est significative
        if error["severity"] > 0.7:
            self.trigger_reflection(
                trigger=f"erreur_significative_{error['type']}",
                domain=error["domain"],
                urgency=min(error["severity"] + 0.2, 1.0)
            )
    
    def _record_metacognitive_event(self, event_type: str, domain: CognitiveDomain, 
                                  description: str, significance: float, 
                                  confidence: float, emotional_valence: float = 0.0,
                                  cognitive_load: float = 0.0, 
                                  related_memories: List[str] = None,
                                  action_taken: str = None):
        """Enregistre un événement métacognitif"""
        event = MetacognitiveEvent(
            timestamp=time.time(),
            event_type=event_type,
            domain=domain,
            description=description,
            significance=significance,
            confidence=confidence,
            emotional_valence=emotional_valence,
            cognitive_load=cognitive_load,
            related_memories=related_memories or [],
            action_taken=action_taken
        )
        
        self.metacognitive_history["events"].append(event)
        
        # Mise à jour des états métacognitifs basée sur l'événement
        self._update_metacognitive_states(event)
        
        return event
    
    def _update_metacognitive_states(self, event: MetacognitiveEvent):
        """Met à jour les états métacognitifs basé sur les événements"""
        before_states = {
            "awareness_level": self.metacognitive_states.get("awareness_level", 0.0),
            "self_understanding": self.metacognitive_states.get("self_understanding", 0.0),
            "relationship_attunement": self.metacognitive_states.get("relationship_attunement", 0.0),
        }

        features: Dict[str, float] = {
            "significance": float(event.significance),
            "confidence": float(event.confidence),
            "emotional_valence": float(event.emotional_valence),
            "cognitive_load": float(event.cognitive_load),
            "domain_importance": float(self._assess_domain_impact(event.domain)),
            "significance_confidence_gap": abs(float(event.significance) - float(event.confidence)),
        }
        features[f"domain_{event.domain.value}"] = 1.0
        features[f"event_{event.event_type}"] = 1.0

        modulation = self._state_update_model.predict(features)
        modulation_factor = (modulation - 0.5) * 2.0

        # Conscience: basée sur la signification, modulée par le modèle
        base_awareness = event.significance * 0.01 if event.significance > 0.3 else 0.0
        awareness_delta = base_awareness * (1.0 + modulation_factor)
        awareness_delta = max(-0.05, min(0.05, awareness_delta))
        new_awareness = max(0.0, min(1.0, before_states["awareness_level"] + awareness_delta))
        if abs(new_awareness - before_states["awareness_level"]) > 0.02:
            self._log_drift("awareness_level", before_states["awareness_level"], new_awareness)
        self.metacognitive_states["awareness_level"] = new_awareness

        # Compréhension de soi: events d'erreur ou faible confiance entraînent des ajustements
        base_understanding = 0.0
        if event.event_type == "error_detected" or event.confidence < 0.4:
            base_understanding = event.significance * 0.02
        elif event.significance > 0.6:
            base_understanding = event.significance * 0.01
        understanding_delta = base_understanding * (1.0 + modulation_factor)
        understanding_delta = max(-0.05, min(0.05, understanding_delta))
        new_understanding = max(0.0, min(1.0, before_states["self_understanding"] + understanding_delta))
        if abs(new_understanding - before_states["self_understanding"]) > 0.02:
            self._log_drift("self_understanding", before_states["self_understanding"], new_understanding)
        self.metacognitive_states["self_understanding"] = new_understanding

        rel_before = before_states["relationship_attunement"]
        rel_delta = 0.0
        if event.domain == CognitiveDomain.SOCIAL or event.event_type.startswith("relationship"):
            base_rel = max(event.significance, abs(event.emotional_valence)) * 0.03
            if event.event_type == "relationship_gain":
                base_rel += 0.04
            elif event.event_type == "relationship_regress":
                base_rel -= 0.05
            else:
                base_rel += event.emotional_valence * 0.02
            rel_delta = base_rel * (1.0 + modulation_factor)
            rel_delta = max(-0.06, min(0.06, rel_delta))
        else:
            rel_delta = -0.01 * (rel_before - 0.35)
        new_relationship = _clip(rel_before + rel_delta)
        if abs(new_relationship - rel_before) > 0.02:
            self._log_drift("relationship_attunement", rel_before, new_relationship)
        self.metacognitive_states["relationship_attunement"] = new_relationship

        after_states = {
            "awareness_level": new_awareness,
            "self_understanding": new_understanding,
            "relationship_attunement": new_relationship,
        }

        reward = self._estimate_state_update_reward(event, before_states, after_states)
        self._state_update_model.update(features, reward)
        self._state_update_context.append({
            "event_type": event.event_type,
            "domain": event.domain.value,
            "features": features,
            "reward": reward,
            "before": before_states,
            "after": after_states,
        })

    def trigger_reflection(self, trigger: str, domain: CognitiveDomain,
                          urgency: float = 0.5, depth: int = 2):
        """Déclenche une session de réflexion"""
        reflection = ReflectionSession(
            start_time=time.time(),
            trigger=trigger,
            focus_domain=domain,
            depth_level=depth
        )
        
        # Exécution de la réflexion
        self._execute_reflection_session(reflection)

        # Enregistrement
        self.metacognitive_history["reflection_sessions"].append(reflection)
        self.reflection_engine["triggered_reflections"].append(reflection)
        self._recent_reflection_quality.append(max(0.0, min(1.0, reflection.quality_score)))
        self._register_reflection_outcome(reflection)

        return reflection

    @staticmethod
    def _coerce_str_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            results: List[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        results.append(text)
            return results
        return []

    @staticmethod
    def _coerce_action_plan_list(value: Any, domain: CognitiveDomain) -> List[Dict[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
            return []
        plans: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            try:
                estimated_effort = _clip(float(item.get("estimated_effort", 0.5)))
            except (TypeError, ValueError):
                estimated_effort = 0.5
            try:
                expected_benefit = _clip(float(item.get("expected_benefit", 0.5)))
            except (TypeError, ValueError):
                expected_benefit = 0.5
            plan: Dict[str, Any] = {
                "type": str(item.get("type", "")).strip() or "unspecified",
                "description": str(item.get("description", "")).strip() or "",
                "priority": str(item.get("priority", "medium")).strip() or "medium",
                "estimated_effort": estimated_effort,
                "expected_benefit": expected_benefit,
                "domain": str(item.get("domain", domain.value)).strip() or domain.value,
            }
            if plan["description"]:
                plans.append(plan)
        return plans

    def _call_reflection_llm(
        self,
        reflection: ReflectionSession,
        situation_analysis: Mapping[str, Any],
        evidence_review: Mapping[str, Any],
        heuristic_bundle: Mapping[str, Any],
    ) -> Optional[Mapping[str, Any]]:
        payload = {
            "trigger": reflection.trigger,
            "domain": reflection.focus_domain.value,
            "depth_level": reflection.depth_level,
            "urgency": self._assess_reflection_urgency(reflection),
            "situation_analysis": situation_analysis,
            "evidence": evidence_review,
            "heuristic_summary": heuristic_bundle,
        }
        return try_call_llm_dict(
            "metacognition_reflection_synthesis",
            input_payload=payload,
            logger=LOGGER,
        )

    def _execute_reflection_session(self, reflection: ReflectionSession):
        """Exécute une session de réflexion structurée"""

        # Phase 1: Analyse de la situation déclencheuse
        situation_analysis = self._analyze_reflection_trigger(reflection)

        # Phase 2: Examen des preuves et données
        evidence_review = self._gather_relevant_evidence(reflection.focus_domain)

        heuristic_insights = self._generate_insights(
            situation_analysis, evidence_review, reflection.depth_level
        )
        heuristic_conclusions = self._draw_conclusions(
            heuristic_insights, reflection.focus_domain
        )
        heuristic_action_plans = self._develop_action_plans(
            heuristic_conclusions, reflection.focus_domain
        )

        llm_response = self._call_reflection_llm(
            reflection,
            situation_analysis,
            evidence_review,
            {
                "insights": heuristic_insights,
                "conclusions": heuristic_conclusions,
                "action_plans": heuristic_action_plans,
            },
        )

        if llm_response:
            llm_insights = self._coerce_str_list(llm_response.get("insights"))
            llm_conclusions = self._coerce_str_list(llm_response.get("conclusions"))
            llm_plans = self._coerce_action_plan_list(
                llm_response.get("action_plans"), reflection.focus_domain
            )

            reflection.insights.extend(llm_insights or heuristic_insights)
            reflection.conclusions.extend(llm_conclusions or heuristic_conclusions)
            reflection.action_plans.extend(llm_plans or heuristic_action_plans)

            try:
                quality_estimate = float(llm_response.get("quality_estimate", 0.0))
                reflection.quality_score = max(reflection.quality_score, _clip(quality_estimate))
            except (TypeError, ValueError):
                pass

            notes = llm_response.get("optional_quality_notes") or llm_response.get("notes")
            if isinstance(notes, str) and notes.strip():
                setattr(reflection, "llm_notes", notes.strip())
        else:
            reflection.insights.extend(heuristic_insights)
            reflection.conclusions.extend(heuristic_conclusions)
            reflection.action_plans.extend(heuristic_action_plans)

        # Phase 6: Évaluation de la session
        reflection.duration = time.time() - reflection.start_time
        heuristic_quality = self._evaluate_reflection_quality(reflection)
        if reflection.quality_score:
            reflection.quality_score = max(reflection.quality_score, heuristic_quality)
        else:
            reflection.quality_score = heuristic_quality

        # Enregistrement des insights
        for insight in reflection.insights:
            self.metacognitive_history["insights"].append({
                "timestamp": time.time(),
                "insight": insight,
                "domain": reflection.focus_domain.value,
                "depth": reflection.depth_level,
                "quality": reflection.quality_score
            })
    
    def _analyze_reflection_trigger(self, reflection: ReflectionSession) -> Dict[str, Any]:
        """Analyse ce qui a déclenché la réflexion"""
        analysis = {
            "trigger_type": reflection.trigger,
            "domain_impact": self._assess_domain_impact(reflection.focus_domain),
            "urgency_level": self._assess_reflection_urgency(reflection),
            "potential_benefits": self._estimate_reflection_benefits(reflection)
        }
        
        return analysis
    
    def _assess_domain_impact(self, domain: CognitiveDomain) -> float:
        """Évalue l'impact du domaine sur les performances globales"""
        impact_weights = {
            CognitiveDomain.REASONING: 0.9,
            CognitiveDomain.MEMORY: 0.8,
            CognitiveDomain.LEARNING: 0.9,
            CognitiveDomain.DECISION_MAKING: 0.8,
            CognitiveDomain.PROBLEM_SOLVING: 0.7,
            CognitiveDomain.ATTENTION: 0.6,
            CognitiveDomain.PERCEPTION: 0.5,
            CognitiveDomain.LANGUAGE: 0.7,
            CognitiveDomain.SOCIAL: 0.85,
        }
        
        return impact_weights.get(domain, 0.5)
    
    def _assess_reflection_urgency(self, reflection: ReflectionSession) -> float:
        """Évalue l'urgence de la réflexion"""
        urgency_factors = []
        
        # Urgence basée sur le type de déclencheur
        trigger_urgency = {
            "error_significative": 0.9,
            "performance_degradation": 0.8,
            "new_learning_opportunity": 0.6,
            "periodic_review": 0.4
        }
        
        for trigger_pattern, urgency in trigger_urgency.items():
            if trigger_pattern in reflection.trigger:
                urgency_factors.append(urgency)
                break
        
        # Urgence basée sur le domaine
        domain_urgency = self._assess_domain_impact(reflection.focus_domain)
        urgency_factors.append(domain_urgency)
        
        return np.mean(urgency_factors) if urgency_factors else 0.5
    
    def _estimate_reflection_benefits(self, reflection: ReflectionSession) -> Dict[str, float]:
        """Estime les bénéfices potentiels de la réflexion"""
        benefits = {}
        
        # Amélioration potentielle des performances
        benefits["performance_improvement"] = reflection.depth_level * 0.2
        
        # Acquisition de nouvelles connaissances
        benefits["knowledge_gain"] = reflection.depth_level * 0.15
        
        # Développement de compétences métacognitives
        benefits["metacognitive_skill"] = reflection.depth_level * 0.1
        
        return benefits
    
    def _gather_relevant_evidence(self, domain: CognitiveDomain) -> Dict[str, Any]:
        """Rassemble les preuves pertinentes pour la réflexion"""
        evidence = {}
        
        # Données de performance récentes
        performance_data = self.cognitive_monitoring["performance_tracking"]
        domain_performance = {}
        
        for metric, data in performance_data.items():
            if data:
                recent_values = [point["value"] for point in list(data)[-5:]]
                domain_performance[metric] = {
                    "current": recent_values[-1] if recent_values else 0.0,
                    "trend": self._calculate_trend(recent_values),
                    "stability": self._calculate_stability(recent_values)
                }
        
        evidence["performance_metrics"] = domain_performance
        
        # Erreurs récentes dans le domaine
        recent_errors = [
            error for error in list(self.metacognitive_history["error_corrections"])[-10:]
            if error["domain"] == domain
        ]
        evidence["recent_errors"] = recent_errors
        
        # Insights précédents dans le domaine
        domain_insights = [
            insight for insight in list(self.metacognitive_history["insights"])[-5:]
            if insight["domain"] == domain.value
        ]
        evidence["previous_insights"] = domain_insights
        
        return evidence
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calcule la tendance des valeurs (pente normalisée)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Normalisation de la pente
        max_slope = max(abs(slope), 0.1)  # Éviter division par zéro
        normalized_slope = slope / (max_slope * 2)  # Normaliser entre -0.5 et 0.5
        
        return normalized_slope + 0.5  # Transformer en [0,1]
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calcule la stabilité des valeurs (inverse du coefficient de variation)"""
        if len(values) < 2:
            return 1.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if mean == 0:
            return 1.0 if std == 0 else 0.0
        
        coefficient_of_variation = std / mean
        stability = 1.0 / (1.0 + coefficient_of_variation)
        
        return min(stability, 1.0)
    
    def _generate_insights(self, situation_analysis: Dict[str, Any], 
                          evidence: Dict[str, Any], depth: int) -> List[str]:
        """Génère des insights métacognitifs"""
        insights = []
        
        # Insight basé sur les patterns de performance
        performance_metrics = evidence.get("performance_metrics", {})
        for metric, data in performance_metrics.items():
            if data["trend"] < 0.4:  # Tendance négative
                insights.append(f"Performance en baisse détectée dans {metric}")
            elif data["trend"] > 0.6:  # Tendance positive
                insights.append(f"Amélioration détectée dans {metric}")
        
        # Insight basé sur les erreurs récurrentes
        recent_errors = evidence.get("recent_errors", [])
        if len(recent_errors) >= 3:
            error_types = defaultdict(int)
            for error in recent_errors:
                error_types[error["error_type"]] += 1
            
            for error_type, count in error_types.items():
                if count >= 2:  # Erreur récurrente
                    insights.append(f"Erreur récurrente détectée: {error_type}")
        
        # Insight basé sur la stabilité
        for metric, data in performance_metrics.items():
            if data["stability"] < 0.6:
                insights.append(f"Instabilité détectée dans {metric}")
        
        # Insights de niveau supérieur pour les réflexions profondes
        if depth >= 2:
            # Insight sur les relations entre domaines
            domain_interactions = self._analyze_domain_interactions(evidence)
            insights.extend(domain_interactions)
            
            # Insight sur les patterns d'apprentissage
            learning_patterns = self._analyze_learning_patterns(evidence)
            insights.extend(learning_patterns)
        
        if depth >= 3:
            # Insights métacognitifs profonds
            deep_insights = self._generate_deep_insights(situation_analysis, evidence)
            insights.extend(deep_insights)
        
        return insights
    
    def _analyze_domain_interactions(self, evidence: Dict[str, Any]) -> List[str]:
        """Analyse les interactions entre domaines cognitifs"""
        interactions = []
        
        # Recherche de corrélations entre performances de différents domaines
        performance_data = evidence.get("performance_metrics", {})
        metrics = list(performance_data.keys())
        
        if len(metrics) >= 2:
            # Corrélation simple basée sur les tendances
            for i in range(len(metrics)):
                for j in range(i + 1, len(metrics)):
                    trend_i = performance_data[metrics[i]]["trend"]
                    trend_j = performance_data[metrics[j]]["trend"]
                    
                    correlation = 1.0 - abs(trend_i - trend_j)
                    if correlation > 0.8:
                        interactions.append(
                            f"Forte corrélation entre {metrics[i]} et {metrics[j]}"
                        )
                    elif correlation < 0.3:
                        interactions.append(
                            f"Faible corrélation entre {metrics[i]} et {metrics[j]}"
                        )
        
        return interactions
    
    def _analyze_learning_patterns(self, evidence: Dict[str, Any]) -> List[str]:
        """Analyse les patterns d'apprentissage"""
        patterns = []
        
        # Pattern d'amélioration progressive
        learning_rate_metric = performance_data.get("learning_rate", {})
        if learning_rate_metric.get("current", 0.0) > 0.7:
            patterns.append("Taux d'apprentissage élevé détecté")
        
        # Pattern de plateau d'apprentissage
        performance_stability = []
        for metric, data in evidence.get("performance_metrics", {}).items():
            if data["stability"] > 0.8 and data["trend"] < 0.6:
                performance_stability.append(metric)
        
        if performance_stability:
            patterns.append(f"Plateau détecté dans: {', '.join(performance_stability)}")
        
        return patterns
    
    def _generate_deep_insights(self, situation_analysis: Dict[str, Any], 
                              evidence: Dict[str, Any]) -> List[str]:
        """Génère des insights métacognitifs profonds"""
        deep_insights = []
        
        # Insight sur l'efficacité des stratégies
        strategy_effectiveness = self._evaluate_strategy_effectiveness(evidence)
        if strategy_effectiveness:
            deep_insights.append(f"Efficacité stratégique: {strategy_effectiveness}")
        
        # Insight sur les limites cognitives
        cognitive_limits = self._identify_cognitive_limits(evidence)
        deep_insights.extend(cognitive_limits)
        
        # Insight sur le développement métacognitif
        metacognitive_growth = self._assess_metacognitive_growth()
        deep_insights.append(f"Croissance métacognitive: {metacognitive_growth}")
        
        return deep_insights
    
    def _evaluate_strategy_effectiveness(self, evidence: Dict[str, Any]) -> str:
        """Évalue l'efficacité des stratégies cognitives actuelles"""
        performance_data = evidence.get("performance_metrics", {})
        
        # Évaluation basée sur la stabilité et les tendances
        stable_metrics = []
        improving_metrics = []
        
        for metric, data in performance_data.items():
            if data["stability"] > 0.7:
                stable_metrics.append(metric)
            if data["trend"] > 0.6:
                improving_metrics.append(metric)
        
        if improving_metrics and not stable_metrics:
            return "Stratégies efficaces pour l'amélioration mais manque de stabilité"
        elif stable_metrics and not improving_metrics:
            return "Stratégies stables mais limitées pour l'amélioration"
        elif improving_metrics and stable_metrics:
            return "Stratégies équilibrées entre stabilité et amélioration"
        else:
            return "Stratégies nécessitant des ajustements"
    
    def _identify_cognitive_limits(self, evidence: Dict[str, Any]) -> List[str]:
        """Identifie les limites cognitives actuelles"""
        limits = []
        
        performance_data = evidence.get("performance_metrics", {})
        
        for metric, data in performance_data.items():
            if data["current"] < 0.4:  # Performance faible
                limits.append(f"Limite identifiée dans {metric}")
        
        # Limites basées sur les erreurs récurrentes
        recent_errors = evidence.get("recent_errors", [])
        if len(recent_errors) >= 5:
            limits.append("Fréquence élevée d'erreurs suggérant des limites cognitives")
        
        return limits
    
    def _assess_metacognitive_growth(self) -> str:
        """Évalue la croissance métacognitive"""
        awareness = self.metacognitive_states["awareness_level"]
        understanding = self.metacognitive_states["self_understanding"]
        
        if awareness < 0.3 and understanding < 0.3:
            return "Niveau débutant"
        elif awareness < 0.6 and understanding < 0.6:
            return "Niveau intermédiaire"
        elif awareness < 0.8 and understanding < 0.8:
            return "Niveau avancé"
        else:
            return "Niveau expert"
    
    def _draw_conclusions(self, insights: List[str], domain: CognitiveDomain) -> List[str]:
        """Tire des conclusions des insights générés"""
        conclusions = []
        
        if not insights:
            conclusions.append("Aucun insight significatif généré")
            return conclusions
        
        # Catégorisation des insights
        performance_insights = [i for i in insights if "performance" in i.lower() or "Performance" in i]
        error_insights = [i for i in insights if "erreur" in i.lower() or "error" in i.lower()]
        strategy_insights = [i for i in insights if "stratégie" in i.lower() or "strategy" in i.lower()]
        
        # Conclusions sur la performance
        if performance_insights:
            conclusions.append(f"{len(performance_insights)} insights sur la performance dans {domain.value}")
        
        # Conclusions sur les erreurs
        if error_insights:
            conclusions.append(f"{len(error_insights)} patterns d'erreur identifiés")
        
        # Conclusions sur les stratégies
        if strategy_insights:
            conclusions.append("Ajustements stratégiques nécessaires")
        
        # Conclusion synthétique
        if len(insights) >= 5:
            conclusions.append("Situation cognitive complexe nécessitant une attention soutenue")
        elif len(insights) <= 2:
            conclusions.append("Situation cognitive relativement stable")
        
        return conclusions
    
    def _develop_action_plans(self, conclusions: List[str], domain: CognitiveDomain) -> List[Dict[str, Any]]:
        """Développe des plans d'action basés sur les conclusions"""
        action_plans = []
        
        for conclusion in conclusions:
            if "ajustements" in conclusion or "adjustments" in conclusion:
                plan = {
                    "type": "strategy_adjustment",
                    "domain": domain.value,
                    "description": "Ajuster les stratégies cognitives",
                    "priority": "medium",
                    "estimated_effort": 0.6,
                    "expected_benefit": 0.7
                }
                action_plans.append(plan)
            
            elif "erreur" in conclusion or "error" in conclusion:
                plan = {
                    "type": "error_prevention",
                    "domain": domain.value,
                    "description": "Implémenter des mesures de prévention d'erreurs",
                    "priority": "high",
                    "estimated_effort": 0.5,
                    "expected_benefit": 0.8
                }
                action_plans.append(plan)
            
            elif "performance" in conclusion.lower():
                plan = {
                    "type": "performance_optimization",
                    "domain": domain.value,
                    "description": "Optimiser les performances cognitives",
                    "priority": "medium",
                    "estimated_effort": 0.7,
                    "expected_benefit": 0.6
                }
                action_plans.append(plan)
        
        # Plan d'action par défaut si aucun plan spécifique
        if not action_plans:
            action_plans.append({
                "type": "continued_monitoring",
                "domain": domain.value,
                "description": "Continuer la surveillance métacognitive",
                "priority": "low",
                "estimated_effort": 0.3,
                "expected_benefit": 0.4
            })
        
        return action_plans
    
    def _evaluate_reflection_quality(self, reflection: ReflectionSession) -> float:
        """Évalue la qualité d'une session de réflexion"""
        quality_factors = []
        
        # Facteur: nombre d'insights
        insight_factor = min(len(reflection.insights) / 5.0, 1.0)
        quality_factors.append(insight_factor * 0.3)
        
        # Facteur: profondeur de la réflexion
        depth_factor = reflection.depth_level / 3.0
        quality_factors.append(depth_factor * 0.3)
        
        # Facteur: applicabilité des plans d'action
        action_factor = min(len(reflection.action_plans) / 3.0, 1.0)
        quality_factors.append(action_factor * 0.2)
        
        # Facteur: durée appropriée
        duration_factor = 1.0 - min(abs(reflection.duration - 30) / 30.0, 1.0)  # Idéal: 30 secondes
        quality_factors.append(duration_factor * 0.2)
        
        return sum(quality_factors)
    
    def _update_self_model(self):
        """Met à jour le modèle de soi basé sur les données récentes"""
        
        # Mise à jour des capacités cognitives
        performance_data = self.cognitive_monitoring["performance_tracking"]
        
        # Capacité de mémoire
        memory_metrics = performance_data.get("memory_capacity", [])
        if memory_metrics:
            recent_memory = memory_metrics[-1]["value"] if memory_metrics else 0.5
            self.self_model.cognitive_abilities["memory_capacity"] = self._update_ability_estimate(
                self.self_model.cognitive_abilities["memory_capacity"],
                recent_memory
            )
        
        # Vitesse de raisonnement
        reasoning_speed_metrics = performance_data.get("reasoning_speed", [])
        if reasoning_speed_metrics:
            recent_speed = reasoning_speed_metrics[-1]["value"] if reasoning_speed_metrics else 0.5
            self.self_model.cognitive_abilities["reasoning_speed"] = self._update_ability_estimate(
                self.self_model.cognitive_abilities["reasoning_speed"],
                recent_speed
            )
        
        # Efficacité d'apprentissage
        learning_metrics = performance_data.get("learning_rate", [])
        if learning_metrics:
            recent_learning = learning_metrics[-1]["value"] if learning_metrics else 0.5
            self.self_model.cognitive_abilities["learning_efficiency"] = self._update_ability_estimate(
                self.self_model.cognitive_abilities["learning_efficiency"],
                recent_learning
            )
        
        # Mise à jour de la précision du modèle de soi
        self._update_self_model_accuracy()

        # Calibration douce entre auto-évaluation et performances observées
        try:
            deltas = calibrate_self_model(self.self_model, self.cognitive_monitoring["performance_tracking"], learning_rate=0.1)
            if deltas:
                self._record_metacognitive_event(
                    event_type="self_model_calibrated",
                    domain=CognitiveDomain.LEARNING,
                    description=f"Calibration self-model: { {k: round(v,3) for k,v in deltas.items()} }",
                    significance=0.3,
                    confidence=0.7
                )
        except Exception as _e:
            print(f"[⚠] calibrate_self_model: {_e}")
    
    def _update_ability_estimate(self, current_estimate: float, new_evidence: float) -> float:
        """Met à jour une estimation de capacité avec de nouvelles preuves"""
        learning_rate = self._select_self_model_rate()
        updated_estimate = (1 - learning_rate) * current_estimate + learning_rate * new_evidence
        return max(0.0, min(1.0, updated_estimate))
    
    def _update_self_model_accuracy(self):
        """Met à jour l'estimation de précision du modèle de soi"""
        # Basé sur la cohérence entre auto-évaluation et performance réelle
        consistency_scores = []
        
        for ability, self_assessment in self.self_model.cognitive_abilities.items():
            performance_metric = self.cognitive_monitoring["performance_tracking"].get(ability, [])
            if performance_metric:
                recent_performance = performance_metric[-1]["value"] if performance_metric else 0.5
                consistency = 1.0 - abs(self_assessment - recent_performance)
                consistency_scores.append(consistency)
        
        if consistency_scores:
            new_accuracy = np.mean(consistency_scores)
            previous_accuracy = self.self_model_accuracy
            learning_rate = 0.1
            self.self_model_accuracy = (1 - learning_rate) * previous_accuracy + learning_rate * new_accuracy
            if self._last_rate_arm is not None:
                improvement = self.self_model_accuracy - previous_accuracy
                reward = 0.5 + 0.5 * math.tanh(improvement / 0.05)
                reward = max(0.0, min(1.0, reward))
                self._self_model_rate_bandit.update(self._last_rate_arm, reward)
    
    def _perform_initial_self_assessment(self) -> ReflectionSession:
        """Effectue l'auto-évaluation initiale"""
        reflection = ReflectionSession(
            start_time=time.time(),
            trigger="initialization",
            focus_domain=CognitiveDomain.LEARNING,
            depth_level=1
        )
        
        # Insights initiaux
        reflection.insights = [
            "Système métacognitif initialisé avec capacités de surveillance de base",
            "Auto-évaluation initiale: niveau débutant dans tous les domaines",
            "Stratégies d'apprentissage de base disponibles"
        ]
        
        # Conclusions
        reflection.conclusions = [
            "Nécessité de développer les capacités métacognitives par la pratique",
            "Importance de l'auto-surveillance pour l'amélioration continue",
            "Besoin d'accumuler de l'expérience pour affiner le modèle de soi"
        ]
        
        # Plans d'action
        reflection.action_plans = [
            {
                "type": "skill_development",
                "domain": "metacognition",
                "description": "Développer les compétences métacognitives de base",
                "priority": "high",
                "estimated_effort": 0.8,
                "expected_benefit": 0.9
            }
        ]
        
        reflection.duration = time.time() - reflection.start_time
        reflection.quality_score = 0.6
        
        return reflection
    
    def get_metacognitive_status(self) -> Dict[str, Any]:
        """Retourne le statut métacognitif complet"""
        return {
            "metacognitive_states": self.metacognitive_states.copy(),
            "self_model_accuracy": self.self_model_accuracy,
            "cognitive_abilities": self.self_model.cognitive_abilities.copy(),
            "recent_events_count": len(self.metacognitive_history["events"]),
            "reflection_sessions_count": len(self.metacognitive_history["reflection_sessions"]),
            "insights_generated": len(self.metacognitive_history["insights"]),
            "operational_parameters": self.operational_parameters.copy(),
            "performance_metrics": {
                metric: data[-1]["value"] if data else 0.0
                for metric, data in self.cognitive_monitoring["performance_tracking"].items()
            }
        }
    
    def schedule_periodic_reflection(self, interval: float = 300):  # 5 minutes par défaut
        """Planifie des réflexions périodiques"""
        def periodic_reflection_loop():
            while self.running:
                try:
                    # Sélection adaptative d'un domaine via Thompson Sampling
                    selected_domain = self._reflection_domain_bandit.select()
                    if not isinstance(selected_domain, CognitiveDomain):
                        try:
                            selected_domain = CognitiveDomain(selected_domain)
                        except Exception:
                            selected_domain = CognitiveDomain.LEARNING

                    self.trigger_reflection(
                        trigger="periodic_review",
                        domain=selected_domain,
                        urgency=0.3,
                        depth=1
                    )
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"Erreur dans la réflexion périodique: {e}")
                    time.sleep(60)
        
        reflection_thread = threading.Thread(target=periodic_reflection_loop, daemon=True)
        reflection_thread.start()
        self.monitoring_threads["periodic_reflection"] = reflection_thread

    # ------------------------------------------------------------------
    # Auto-intention & auto-évolution hooks
    def evaluate_auto_intention(self, intention: Mapping[str, Any]) -> Dict[str, Any]:
        """Score et décide si une intention autonome doit être promue."""

        if not isinstance(intention, Mapping):
            return {"accepted": False, "reason": "invalid_intention"}

        description = str(intention.get("description") or intention.get("label") or "").strip()
        if not description:
            return {"accepted": False, "reason": "empty_description"}

        signals = intention.get("signals") if isinstance(intention.get("signals"), list) else []
        requirements = intention.get("requirements") if isinstance(intention.get("requirements"), (list, tuple, set)) else []
        metadata = intention.get("metadata") if isinstance(intention.get("metadata"), Mapping) else {}
        tags = metadata.get("tags") if isinstance(metadata.get("tags"), (list, tuple, set)) else []

        significance = self._score_auto_intention_significance(description, signals, requirements)
        alignment = self._estimate_intention_alignment(description, tags)
        emotional_drive = self._estimate_emotional_drive(intention.get("source_emotion"))

        combined = 0.4 * significance + 0.4 * alignment + 0.2 * emotional_drive
        combined = _clip(combined)
        accepted = combined >= 0.45

        result = {
            "accepted": accepted,
            "significance": significance,
            "alignment": alignment,
            "emotional_drive": emotional_drive,
            "score": combined,
        }

        if accepted:
            timestamp = time.time()
            history_entry = {
                "ts": timestamp,
                "description": description,
                "score": combined,
                "significance": significance,
                "alignment": alignment,
                "emotional_drive": emotional_drive,
                "action_type": intention.get("action_type"),
            }
            self.metacognitive_history["self_improvements"].append(history_entry)

            domain = CognitiveDomain.SOCIAL if "social" in str(intention.get("source", "")).lower() else CognitiveDomain.REASONING
            event = MetacognitiveEvent(
                timestamp=timestamp,
                event_type="auto_intention",
                domain=domain,
                description=description[:280],
                significance=combined,
                confidence=alignment,
                emotional_valence=emotional_drive,
                cognitive_load=0.4,
            )
            self.metacognitive_history["events"].append(event)

            if self.logger:
                try:
                    self.logger.write(
                        "metacog.auto_intention",
                        intention=intention,
                        evaluation=result,
                    )
                except Exception:
                    pass

        return result

    def plan_auto_intention_followup(
        self,
        intention: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        signals: Sequence[Mapping[str, Any]],
        self_assessment: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Planifie le suivi métacognitif et les checkpoints de mesure."""

        if not isinstance(intention, Mapping):
            return

        scheduled = self.reflection_engine.setdefault("scheduled_reflections", [])
        if isinstance(scheduled, list):
            plan = {
                "ts": time.time(),
                "type": "auto_intention_followup",
                "action_type": intention.get("action_type"),
                "description": intention.get("description"),
                "evaluation": dict(evaluation),
                "signals": list(signals),
            }
            if self_assessment:
                plan["self_assessment"] = self_assessment
            scheduled.append(plan)

        tracking = self.cognitive_monitoring.get("performance_tracking")
        if isinstance(tracking, dict):
            for signal in signals:
                if not isinstance(signal, Mapping):
                    continue
                name = signal.get("name")
                if not name:
                    continue
                metric_log = tracking.setdefault(name, [])
                metric_log.append(
                    {
                        "timestamp": time.time(),
                        "metric": signal.get("metric"),
                        "target": signal.get("target"),
                        "direction": signal.get("direction", "above"),
                        "score": float(evaluation.get("score", evaluation.get("significance", 0.6))),
                    }
                )
                if len(metric_log) > 30:
                    del metric_log[0 : len(metric_log) - 30]

    def _extract_keywords(self, *chunks: Any) -> Set[str]:
        words: Set[str] = set()
        for chunk in chunks:
            if chunk is None:
                continue
            if isinstance(chunk, (list, tuple, set)):
                for item in chunk:
                    words.update(self._extract_keywords(item))
                continue
            text = str(chunk).lower()
            for token in re.findall(r"[\w']+", text):
                if len(token) >= 4:
                    words.add(token)
        return words

    def _score_auto_intention_significance(
        self,
        description: str,
        signals: Sequence[Any],
        requirements: Sequence[Any],
    ) -> float:
        length_bonus = min(0.3, 0.01 * len(description.split()))
        signal_bonus = 0.06 * min(len(signals), 5)
        requirement_bonus = 0.03 * min(len(requirements), 10)
        base = 0.35 + length_bonus + signal_bonus + requirement_bonus
        return _clip(base)

    def _estimate_intention_alignment(self, description: str, tags: Sequence[Any]) -> float:
        keywords = self._extract_keywords(description, tags)
        score = 0.4

        # Alignment with persistent self model if available
        identity = {}
        core_self = getattr(self.cognitive_architecture, "self_model", None)
        if core_self is not None:
            try:
                identity = getattr(core_self, "identity", {}) or getattr(core_self, "state", {}).get("identity", {})
            except Exception:
                identity = {}

        values: List[str] = []
        if isinstance(identity, dict):
            values_section = identity.get("values")
            if isinstance(values_section, list):
                values.extend([str(v) for v in values_section])
            elif isinstance(values_section, dict):
                values.extend([str(k) for k in values_section.keys()])
            purpose = identity.get("purpose")
            if isinstance(purpose, dict):
                for key in ("near_term_goals", "ultimate"):
                    val = purpose.get(key)
                    if isinstance(val, list):
                        values.extend([str(v) for v in val])
                    elif isinstance(val, str):
                        values.append(val)

        matches = 0
        for value in values:
            if keywords.intersection(self._extract_keywords(value)):
                matches += 1
        score += 0.1 * min(matches, 4)

        if isinstance(getattr(self.self_model, "cognitive_styles", None), dict):
            styles = " ".join(self.self_model.cognitive_styles.keys())
            if keywords.intersection(self._extract_keywords(styles)):
                score += 0.05

        return _clip(score)

    def _estimate_emotional_drive(self, signal: Optional[Mapping[str, Any]]) -> float:
        if not isinstance(signal, Mapping):
            return 0.4
        intensity = float(signal.get("intensity", 0.0) or 0.0)
        valence = float(signal.get("valence", 0.0) or 0.0)
        tension = float(signal.get("tension", 0.0) or 0.0)
        score = 0.4 + 0.4 * max(0.0, intensity) + 0.2 * max(0.0, valence)
        if tension > 0.6:
            score *= 0.85
        return _clip(score)

    def stop_metacognitive_system(self):
        """Arrête le système métacognitif"""
        self.running = False
        print("⏹️ Système métacognitif arrêté")

# ===== SOUS-SYSTÈMES DE SURVEILLANCE =====

class ErrorDetectionSystem:
    """Système de détection d'erreurs cognitives"""
    
    def detect_reasoning_errors(self, reasoning_system) -> List[Dict[str, Any]]:
        """Détecte les erreurs de raisonnement"""
        errors = []
        
        if not reasoning_system:
            return errors
        
        # Détection d'incohérences logiques
        inconsistencies = self._detect_logical_inconsistencies(reasoning_system)
        errors.extend(inconsistencies)
        
        # Détection de conclusions invalides
        invalid_conclusions = self._detect_invalid_conclusions(reasoning_system)
        errors.extend(invalid_conclusions)
        
        # Détection de biais de raisonnement
        reasoning_biases = self._detect_reasoning_biases(reasoning_system)
        errors.extend(reasoning_biases)
        
        return errors
    
    def detect_memory_errors(self, memory_system) -> List[Dict[str, Any]]:
        """Détecte les erreurs de mémoire"""
        errors = []
        
        # Pour l'instant, erreurs génériques
        # Dans une implémentation complète, on intégrerait avec le système de mémoire
        errors.append({
            "type": "memory_retrieval_failure",
            "description": "Difficulté à récupérer des informations mémorisées",
            "severity": 0.4,
            "confidence": 0.6,
            "domain": CognitiveDomain.MEMORY,
            "corrective_action": "Utiliser des indices de récupération supplémentaires"
        })
        
        return errors
    
    def detect_perception_errors(self) -> List[Dict[str, Any]]:
        """Détecte les erreurs de perception"""
        errors = []
        
        # Erreurs de perception génériques
        errors.append({
            "type": "perceptual_ambiguity",
            "description": "Ambiguïté dans l'interprétation des stimuli",
            "severity": 0.3,
            "confidence": 0.5,
            "domain": CognitiveDomain.PERCEPTION,
            "corrective_action": "Rechercher des informations contextuelles supplémentaires"
        })
        
        return errors
    
    def _detect_logical_inconsistencies(self, reasoning_system) -> List[Dict[str, Any]]:
        """Détecte les incohérences logiques"""
        inconsistencies = []
        
        # Vérification des conclusions contradictoires
        recent_inferences = reasoning_system.reasoning_history["recent_inferences"]
        if len(recent_inferences) >= 2:
            last_two = list(recent_inferences)[-2:]
            
            # Vérification basique de contradiction
            if self._are_contradictory(last_two[0], last_two[1]):
                inconsistencies.append({
                    "type": "logical_contradiction",
                    "description": "Conclusions contradictoires dans des raisonnements récents",
                    "severity": 0.8,
                    "confidence": 0.7,
                    "domain": CognitiveDomain.REASONING,
                    "corrective_action": "Réexaminer les prémisses et le processus de raisonnement"
                })
        
        return inconsistencies
    
    def _are_contradictory(self, inference1: Dict, inference2: Dict) -> bool:
        """Détermine si deux inférences sont contradictoires"""
        # Vérification basique basée sur le contenu textuel
        content1 = str(inference1.get("solution", "")).lower()
        content2 = str(inference2.get("solution", "")).lower()
        
        contradictory_pairs = [
            ("oui", "non"), ("vrai", "faux"), ("possible", "impossible"),
            ("yes", "no"), ("true", "false"), ("possible", "impossible")
        ]
        
        for pair in contradictory_pairs:
            if (pair[0] in content1 and pair[1] in content2) or \
               (pair[1] in content1 and pair[0] in content2):
                return True
        
        return False
    
    def _detect_invalid_conclusions(self, reasoning_system) -> List[Dict[str, Any]]:
        """Détecte les conclusions potentiellement invalides"""
        invalid_conclusions = []
        
        # Détection de conclusions avec faible confiance mais présentées comme certaines
        recent_inferences = reasoning_system.reasoning_history["recent_inferences"]
        for inference in list(recent_inferences)[-3:]:
            confidence = inference.get("final_confidence", 0.5)
            solution = inference.get("solution", "")
            
            if confidence < 0.3 and any(word in str(solution).lower() for word in ["certain", "definite", "sure"]):
                invalid_conclusions.append({
                    "type": "overconfident_conclusion",
                    "description": "Conclusion présentée comme certaine malgré une faible confiance",
                    "severity": 0.6,
                    "confidence": 0.8,
                    "domain": CognitiveDomain.REASONING,
                    "corrective_action": "Recalibrer l'estimation de confiance"
                })
        
        return invalid_conclusions
    
    def _detect_reasoning_biases(self, reasoning_system) -> List[Dict[str, Any]]:
        """Détecte les biais de raisonnement"""
        biases = []
        
        # Détection de raisonnement circulaire
        recent_inferences = reasoning_system.reasoning_history["recent_inferences"]
        if len(recent_inferences) >= 3:
            if self._detect_circular_reasoning(list(recent_inferences)[-3:]):
                biases.append({
                    "type": "circular_reasoning",
                    "description": "Raisonnement circulaire détecté dans les inférences récentes",
                    "severity": 0.7,
                    "confidence": 0.6,
                    "domain": CognitiveDomain.REASONING,
                    "corrective_action": "Introduire de nouvelles preuves externes"
                })
        
        return biases
    
    def _detect_circular_reasoning(self, inferences: List[Dict]) -> bool:
        """Détecte le raisonnement circulaire"""
        if len(inferences) < 3:
            return False
        
        # Vérification basique de circularité
        contents = [str(inf.get("solution", "")) for inf in inferences]
        
        # Si le même contenu réapparaît sans nouvelle information
        if len(set(contents)) < len(contents) * 0.7:  # 70% de contenu unique
            return True
        
        return False

class BiasMonitoringSystem:
    """Système de surveillance des biais cognitifs"""
    
    def detect_confirmation_bias(self, reasoning_system) -> Dict[str, Any]:
        """Détecte le biais de confirmation"""
        detection_result = {
            "detected": False,
            "strength": 0.0,
            "confidence": 0.0
        }
        
        if not reasoning_system:
            return detection_result
        
        # Analyse des stratégies de raisonnement préférées
        strategy_preferences = reasoning_system.get_reasoning_stats().get("strategy_preferences", {})
        
        # Biais de confirmation si préférence pour le raisonnement déductif (tendance à confirmer)
        deductive_preference = strategy_preferences.get("déductif", 0.0)
        if deductive_preference > 0.7:
            detection_result["detected"] = True
            detection_result["strength"] = deductive_preference
            detection_result["confidence"] = 0.6
        
        return detection_result
    
    def detect_overconfidence(self, self_model, reasoning_system) -> Dict[str, Any]:
        """Détecte la surconfiance"""
        detection_result = {
            "detected": False,
            "confidence": 0.0
        }
        
        # Comparaison entre auto-évaluation et performance réelle
        self_assessed_ability = self_model.cognitive_abilities.get("reasoning_speed", 0.5)
        
        if reasoning_system:
            reasoning_stats = reasoning_system.get_reasoning_stats()
            actual_performance = reasoning_stats.get("average_confidence", 0.5)
            
            # Surconfiance si auto-évaluation > performance réelle + marge
            confidence_gap = self_assessed_ability - actual_performance
            if confidence_gap > 0.3:  # Écart significatif
                detection_result["detected"] = True
                detection_result["confidence"] = min(confidence_gap * 2, 1.0)
        
        return detection_result

class ResourceMonitoringSystem:
    """Système de surveillance des ressources cognitives"""

    def __init__(self) -> None:
        self._last_kernel_state: Dict[str, Any] = {}
        self._last_kernel_ts: float = 0.0
        self._last_machine_snapshot: Dict[str, Any] = {}

    def _refresh_kernel_state(self, cognitive_architecture) -> Dict[str, Any]:
        kernel_state = None
        if cognitive_architecture is not None:
            try:
                kernel_state = getattr(cognitive_architecture, "phenomenal_kernel_state", None)
                if not kernel_state:
                    kernel_state = getattr(cognitive_architecture, "_phenomenal_kernel_state", None)
            except Exception:
                kernel_state = None
        if isinstance(kernel_state, dict) and kernel_state:
            self._last_kernel_state = dict(kernel_state)
            self._last_kernel_ts = time.time()
        return dict(self._last_kernel_state)

    def sample_machine_state(self) -> Dict[str, Any]:
        """Expose the last phenomenal kernel snapshot."""

        state = dict(self._last_kernel_state)
        if self._last_machine_snapshot:
            state["machine"] = dict(self._last_machine_snapshot)
        return state

    @property
    def last_snapshot(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_kernel_state) if self._last_kernel_state else None

    @property
    def last_machine_snapshot(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_machine_snapshot) if self._last_machine_snapshot else None

    def register_machine_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if isinstance(snapshot, dict):
            self._last_machine_snapshot = dict(snapshot)

    def assess_cognitive_load(self, cognitive_architecture, reasoning_system) -> float:
        """Évalue la charge cognitive actuelle"""

        load_indicators = []

        if not hasattr(cognitive_architecture, "global_activation"):
            cognitive_architecture.global_activation = 0.5

        if not hasattr(cognitive_architecture, "get_cognitive_status"):
            return 0.5

        if isinstance(reasoning_system, str) or not hasattr(reasoning_system, "reasoning_history"):
            return 0.5

        global_activation = getattr(cognitive_architecture, "global_activation", 0.5)
        load_indicators.append(global_activation)

        kernel_state = self._refresh_kernel_state(cognitive_architecture)
        energy = kernel_state.get("energy")
        if isinstance(energy, (int, float)):
            load_indicators.append(1.0 - float(_clip(energy)))
        slowdown = kernel_state.get("global_slowdown")
        if isinstance(slowdown, (int, float)):
            load_indicators.append(float(_clip(slowdown)))

        machine = dict(self._last_machine_snapshot)
        cpu_load = machine.get("cpu", {}).get("load") if isinstance(machine.get("cpu"), dict) else None
        mem_percent = machine.get("memory", {}).get("percent") if isinstance(machine.get("memory"), dict) else None
        gpu_util = machine.get("gpu", {}).get("util_pct") if isinstance(machine.get("gpu"), dict) else None

        if isinstance(cpu_load, (int, float)):
            load_indicators.append(max(0.0, min(1.0, float(cpu_load) / 100.0)))
        if isinstance(mem_percent, (int, float)):
            load_indicators.append(max(0.0, min(1.0, float(mem_percent) / 100.0)))
        if isinstance(gpu_util, (int, float)):
            load_indicators.append(max(0.0, min(1.0, float(gpu_util) / 100.0)))

        try:
            wm_load = cognitive_architecture.get_cognitive_status().get("working_memory_load", 0)
            normalized_wm_load = min(wm_load / 10.0, 1.0)
            load_indicators.append(normalized_wm_load)
        except Exception:
            load_indicators.append(0.5)

        try:
            recent_inferences = reasoning_system.reasoning_history.get("recent_inferences", [])
            if recent_inferences:
                avg_complexity = np.mean(
                    [inf.get("complexity", 0.5) for inf in list(recent_inferences)[-3:]]
                )
                load_indicators.append(avg_complexity)
        except Exception:
            load_indicators.append(0.5)

        return float(np.mean(load_indicators)) if load_indicators else 0.5

    def _assess_performance_decline(self, performance_history: list) -> float:
        """
        Évalue une éventuelle dégradation des performances cognitives.
        Retourne une valeur entre 0 (aucune baisse) et 1 (baisse significative).
        """

        if not performance_history or len(performance_history) < 2:
            return 0.0

        try:
            recent_scores = [p.get("score", 0.5) for p in performance_history[-3:]]
            diffs = [recent_scores[i + 1] - recent_scores[i] for i in range(len(recent_scores) - 1)]
            decline = -np.mean([d for d in diffs if d < 0]) if any(d < 0 for d in diffs) else 0.0
            return float(min(decline, 1.0))
        except Exception:
            return 0.0

    def assess_fatigue(self, metacognitive_history, cognitive_architecture) -> float:
        """Évalue le niveau de fatigue cognitive"""

        fatigue_indicators = []

        kernel_state = self._refresh_kernel_state(cognitive_architecture)
        kernel_fatigue = kernel_state.get("fatigue")
        if isinstance(kernel_fatigue, (int, float)):
            fatigue_indicators.append(0.4 * float(_clip(kernel_fatigue)))
        slowdown = kernel_state.get("global_slowdown")
        if isinstance(slowdown, (int, float)):
            fatigue_indicators.append(0.3 * float(_clip(slowdown)))

        machine = dict(self._last_machine_snapshot)
        cpu_load = machine.get("cpu", {}).get("load") if isinstance(machine.get("cpu"), dict) else None
        mem_percent = machine.get("memory", {}).get("percent") if isinstance(machine.get("memory"), dict) else None
        if isinstance(cpu_load, (int, float)):
            fatigue_indicators.append(0.1 * max(0.0, min(1.0, float(cpu_load) / 100.0)))
        if isinstance(mem_percent, (int, float)):
            fatigue_indicators.append(0.1 * max(0.0, min(1.0, float(mem_percent) / 100.0)))

        operation_time = time.time() - metacognitive_history.get("system_start_time", time.time())
        time_fatigue = min(operation_time / 3600.0, 1.0)
        fatigue_indicators.append(time_fatigue * 0.2)

        recent_events = len(metacognitive_history.get("events", []))
        event_fatigue = min(recent_events / 100.0, 1.0)
        fatigue_indicators.append(event_fatigue * 0.2)

        if cognitive_architecture:
            performance_decline = self._assess_performance_decline(cognitive_architecture)
            fatigue_indicators.append(performance_decline * 0.2)

        return float(sum(fatigue_indicators))

class ProgressTrackingSystem:
    """Système de suivi des progrès cognitifs"""
    
    def track_learning_progress(self, metacognitive_system) -> Dict[str, float]:
        """Suit les progrès d'apprentissage"""
        progress_metrics = {}
        
        # Progrès métacognitif
        metacognitive_states = metacognitive_system.metacognitive_states
        progress_metrics["metacognitive_awareness"] = metacognitive_states["awareness_level"]
        progress_metrics["self_understanding"] = metacognitive_states["self_understanding"]
        
        # Progrès des capacités cognitives
        cognitive_abilities = metacognitive_system.self_model.cognitive_abilities
        for ability, level in cognitive_abilities.items():
            progress_metrics[f"ability_{ability}"] = level
        
        return progress_metrics

class StrategySelector:
    """Sélecteur de stratégies cognitives adaptatives"""
    
    def select_learning_strategy(self, domain: CognitiveDomain, context: Dict[str, Any]) -> str:
        """Sélectionne une stratégie d'apprentissage adaptée"""
        # Sélection basée sur le domaine et le contexte
        if domain == CognitiveDomain.MEMORY:
            return "spaced_repetition"
        elif domain == CognitiveDomain.REASONING:
            return "elaborative_interrogation"
        elif domain == CognitiveDomain.LEARNING:
            return "self_explanation"
        else:
            return "default_strategy"

class MetacognitiveAttention:
    """Système d'attention métacognitive"""
    
    def allocate_metacognitive_attention(self, events: List[MetacognitiveEvent]) -> Dict[str, float]:
        """Alloue l'attention métacognitive aux événements"""
        attention_allocation = {}
        
        for event in list(events)[-10:]:  # Derniers 10 événements
            attention_score = event.significance * (1.0 - event.confidence)
            attention_allocation[event.description] = attention_score
        
        # Normalisation
        total_attention = sum(attention_allocation.values())
        if total_attention > 0:
            attention_allocation = {k: v/total_attention for k, v in attention_allocation.items()}
        
        return attention_allocation

class EffortRegulator:
    """Régulateur d'effort cognitif"""
    
    def adjust_effort_level(self, current_load: float, target_performance: float) -> float:
        """Ajuste le niveau d'effort cognitif"""
        if current_load > 0.8:
            # Réduction d'effort si charge trop élevée
            return max(0.3, target_performance - 0.2)
        elif current_load < 0.3:
            # Augmentation d'effort si charge trop faible
            return min(1.0, target_performance + 0.2)
        else:
            return target_performance

class MetacognitiveGoalManager:
    """Gestionnaire de buts métacognitifs"""
    
    def __init__(self):
        self.metacognitive_goals = {
            "improve_self_awareness": 0.8,
            "enhance_error_detection": 0.7,
            "develop_better_strategies": 0.6,
            "increase_learning_efficiency": 0.9
        }
    
    def update_goal_priorities(self, metacognitive_status: Dict[str, Any]):
        """Met à jour les priorités des buts métacognitifs"""
        # Ajustement basé sur les états actuels
        awareness_level = metacognitive_status["metacognitive_states"]["awareness_level"]
        if awareness_level < 0.5:
            self.metacognitive_goals["improve_self_awareness"] = 0.9
        else:
            self.metacognitive_goals["improve_self_awareness"] = 0.6

# Test du système métacognitif
if __name__ == "__main__":
    print("🧠 TEST DU SYSTÈME MÉTACOGNITIF")
    print("=" * 50)
    
    # Création du système
    metacognitive_system = MetacognitiveSystem()
    
    # Test de surveillance de base
    print("\n🔍 Test de surveillance cognitive en cours")
    time.sleep(3)
    
    # Test de réflexion déclenchée
    print("\n💭 Test de réflexion métacognitive en cours")
    reflection = metacognitive_system.trigger_reflection(
        trigger="test_performance_review",
        domain=CognitiveDomain.REASONING,
        urgency=0.7,
        depth=2
    )
    
    print(f"Reflexion terminée - Durée: {reflection.duration:.2f}s")
    print(f"Qualité: {reflection.quality_score:.2f}")
    print(f"Insights générés: {len(reflection.insights)}")
    print(f"Plans d'action: {len(reflection.action_plans)}")
    
    # Affichage du statut
    print("\n📊 Statut métacognitif:")
    status = metacognitive_system.get_metacognitive_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f" - {key}:")
            for subkey, subvalue in value.items():
                print(f"   - {subkey}: {subvalue}")
        else:
            print(f" - {key}: {value}")
    
    # Arrêt propre
    metacognitive_system.stop_metacognitive_system()
    
    print("\n✅ Test du système métacognitif terminé avec succès!")
    
    