# -*- coding: utf-8 -*-
"""
Système de Créativité Avancée de l'AGI Évolutive
- Génération d'idées (divergente, latérale, transfert de domaines, etc.)
- Mélange conceptuel
- Détection d'insights
- Gestion de projets d'innovation
Implémente des gardes robustes + normalisation silencieuse pour éviter les erreurs de type
"string indices must be integers, not 'str'".
"""
from __future__ import annotations

import logging
import math
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Mapping
from collections import defaultdict, deque

try:
    import networkx as nx  # utilisé pour l'espace conceptuel
except Exception:  # fallback minimal si networkx indisponible
    class _MiniGraph(dict):
        def add_node(self, n, **attrs): self.setdefault(n, {}).update(attrs)
        def add_edge(self, a, b, **attrs):
            self.setdefault(a, {}); self.setdefault(b, {})
            self.setdefault("_edges", {}).setdefault((a,b), {}).update(attrs)
        def nodes(self, data=False):
            if data: return [(k, v) for k,v in self.items() if k!="_edges"]
            return [k for k in self.keys() if k!="_edges"]
        def neighbors(self, n):
            ed = self.get("_edges", {}); res = []
            for (a,b) in ed:
                if a==n: res.append(b)
                if b==n: res.append(a)
            return res
        def get_edge_data(self, a, b): return self.get("_edges", {}).get((a,b), {})
        def degree(self, n): return len(self.neighbors(n))
        def number_of_nodes(self): return len([k for k in self.keys() if k!="_edges"])
        def number_of_edges(self): return len(self.get("_edges", {}))
    nx = type("nx", (), {"Graph": _MiniGraph})

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)

# -------------------- Types & Data --------------------

class IdeaState(Enum):
    RAW = "brute"
    DEVELOPED = "developpee"
    REFINED = "affinee"
    INTEGRATED = "integree"
    IMPLEMENTED = "implementee"
    ARCHIVED = "archivee"

class InsightType(Enum):
    ASSOCIATIVE = "associatif"
    RESTRUCTURING = "restructuration"
    ANALOGICAL = "analogique"
    EMERGENT = "emergent"
    INTUITIVE = "intuitif"

@dataclass
class CreativeIdea:
    id: str
    concept_core: str
    description: str
    state: IdeaState
    novelty: float
    usefulness: float
    feasibility: float
    elaboration: float
    domains: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    associations: List[str] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    development_history: List[Dict[str, Any]] = field(default_factory=list)
    emotional_affinity: float = 0.3
    activation_level: float = 0.1

@dataclass
class ConceptualBlend:
    blend_id: str
    input_spaces: List[str]
    generic_space: List[str]
    blended_space: str
    emergent_structure: List[str]
    creativity_score: float
    coherence: float
    elaboration_potential: float

@dataclass
class CreativeInsight:
    insight_id: str
    type: InsightType
    content: str
    significance: float
    clarity: float
    surprise: float
    emotional_intensity: float
    preceding_incubation: float
    trigger: str
    timestamp: float
    related_ideas: List[str]
    verification_status: str

@dataclass
class InnovationProject:
    project_id: str
    core_idea: str
    objectives: List[str]
    constraints: List[str]
    resources_needed: List[str]
    development_phases: List[Dict[str, Any]]
    current_phase: int
    success_metrics: Dict[str, float]
    risk_assessment: Dict[str, float]
    team_dynamics: Dict[str, Any]

# -------------------- Utils & Guards --------------------

def _clip(v: float, lo: float=0.0, hi: float=1.0) -> float:
    try:
        v = float(v)
    except Exception:
        v = lo
    return max(lo, min(hi, v))

def _is_ci(x: Any) -> bool:
    return isinstance(x, CreativeIdea)

def _ensure_ci(x: Any, topic: str="") -> CreativeIdea:
    if isinstance(x, CreativeIdea):
        return x
    s = str(x)
    return CreativeIdea(
        id=f"idea_{int(time.time()*1000)}_{random.randint(100,999)}",
        concept_core=s,
        description=f"Idée: {s}",
        state=IdeaState.RAW,
        novelty=0.5, usefulness=0.5, feasibility=0.5, elaboration=0.3,
        domains=[], components=[], associations=[]
    )

def crea_normalize(self: "CreativitySystem") -> None:
    # Containers dict
    for name in ("creative_states","creative_history","idea_generation","conceptual_blending",
                 "insight_detection","innovation_engine","creative_processes",
                 "processing_threads","creative_knowledge","conceptual_space"):
        if not isinstance(getattr(self, name, None), dict):
            setattr(self, name, {})
    # Keys
    self.creative_states.setdefault("creative_flow", 0.3)
    self.creative_states.setdefault("cognitive_flexibility", 0.5)
    self.creative_states.setdefault("inspiration_level", 0.4)
    self.creative_states.setdefault("creative_confidence", 0.5)
    self.creative_history.setdefault("ideas_generated", deque(maxlen=5000))
    self.creative_history.setdefault("blends_created", deque(maxlen=1000))
    self.creative_history.setdefault("insights_experienced", deque(maxlen=500))
    self.creative_history.setdefault("projects_completed", deque(maxlen=100))
    self.creative_history.setdefault("creative_breakthroughs", deque(maxlen=50))
    self.creative_history.setdefault("learning_trajectory", deque(maxlen=1000))

    # idea_generation
    self.idea_generation.setdefault("idea_pool", deque(maxlen=1000))
    self.idea_generation.setdefault("generation_strategies", {})
    # coerce pool to CreativeIdea
    pool = list(self.idea_generation["idea_pool"])
    fixed = []
    for it in pool:
        fixed.append(_ensure_ci(it))
    self.idea_generation["idea_pool"].clear()
    self.idea_generation["idea_pool"].extend(fixed)

    # conceptual space and knowledge
    self.conceptual_space.setdefault("concept_network", nx.Graph())
    self.conceptual_blending.setdefault("blend_history", deque(maxlen=500))
    self.creative_knowledge.setdefault("analogical_sources", {})
    self.creative_knowledge.setdefault("heuristics", {})
    self.creative_knowledge.setdefault("constraint_templates", {})
    self.creative_knowledge.setdefault("innovation_patterns", {})

# -------------------- Adaptive helpers --------------------

class OnlineLogisticModel:
    """Modèle logistique en ligne avec facteur d'oubli et bornes."""

    def __init__(self, dim: int, lr: float = 0.2, l2: float = 0.01,
                 forget: float = 0.01, bounds: Tuple[float, float] = (-3.0, 3.0)) -> None:
        self.dim = dim
        self.lr = lr
        self.l2 = l2
        self.forget = forget
        self.bounds = bounds
        self.weights: List[float] = [0.0] * dim

    def _clip_weight(self, w: float) -> float:
        return max(self.bounds[0], min(self.bounds[1], w))

    def predict(self, features: List[float]) -> float:
        z = sum(w * x for w, x in zip(self.weights, features))
        # Protection contre overflow
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, features: List[float], target: float) -> float:
        target = _clip(target, 0.0, 1.0)
        pred = self.predict(features)
        err = target - pred
        max_shift = 0.0
        for idx, x in enumerate(features):
            grad = err * x - self.l2 * self.weights[idx]
            updated = (1.0 - self.forget) * self.weights[idx] + self.lr * grad
            updated = self._clip_weight(updated)
            shift = abs(updated - self.weights[idx])
            if shift > max_shift:
                max_shift = shift
            self.weights[idx] = updated
        return max_shift


class AdaptiveCreativeMetrics:
    """Apprentissage en ligne des pondérations de mise à jour des états créatifs."""

    def __init__(self, states: Dict[str, float], strategy_names: List[str]) -> None:
        self.state_names = [k for k in ("creative_flow", "creative_confidence",
                                        "cognitive_flexibility", "inspiration_level")
                            if k in states]
        self.strategy_index: Dict[str, float] = {}
        if strategy_names:
            denom = max(1, len(strategy_names) - 1)
            for idx, name in enumerate(strategy_names):
                self.strategy_index[name] = idx / denom if denom else 0.0
        self.default_strategy_value = 0.5
        self.default_ttl_days = 7.0
        self.feature_dim = 12
        self.models: Dict[str, OnlineLogisticModel] = {
            name: OnlineLogisticModel(self.feature_dim, lr=0.25, l2=0.02, forget=0.015)
            for name in self.state_names
        }
        self.max_delta: Dict[str, float] = {
            "creative_flow": 0.15,
            "creative_confidence": 0.12,
            "cognitive_flexibility": 0.1,
            "inspiration_level": 0.14,
        }
        self.drift_threshold = 0.08
        self.drift_log: deque = deque(maxlen=200)

    def _strategy_value(self, strategy: Optional[str]) -> float:
        if strategy in self.strategy_index:
            return self.strategy_index[strategy]
        return self.default_strategy_value

    def _features(self, stats: Dict[str, Any]) -> List[float]:
        ideas_norm = min(1.0, stats.get("ideas_generated", 0) / 10.0)
        efficiency_norm = min(1.0, stats.get("efficiency", 0.0) / 10.0)
        constraints_norm = min(1.0, stats.get("constraints_count", 0) / 5.0)
        ttl_norm = stats.get("ttl_days", self.default_ttl_days) / 30.0
        reward = stats.get("reward", 0.5)
        novelty = stats.get("avg_novelty", 0.5)
        usefulness = stats.get("avg_usefulness", 0.5)
        feasibility = stats.get("avg_feasibility", 0.5)
        elaboration = stats.get("avg_elaboration", 0.3)
        diversity = stats.get("diversity", 0.5)
        strategy_val = self._strategy_value(stats.get("strategy"))
        features = [
            1.0,
            ideas_norm,
            novelty,
            usefulness,
            feasibility,
            elaboration,
            efficiency_norm,
            reward,
            constraints_norm,
            ttl_norm,
            diversity,
            strategy_val,
        ]
        return features

    def _target(self, state: str, stats: Dict[str, Any]) -> float:
        if state == "creative_flow":
            base = 0.6 * stats.get("reward", 0.5) + 0.4 * min(1.0, stats.get("efficiency", 0.0) / 5.0)
        elif state == "creative_confidence":
            base = 0.6 * stats.get("avg_usefulness", 0.5) + 0.4 * stats.get("avg_feasibility", 0.5)
        elif state == "inspiration_level":
            base = 0.6 * stats.get("avg_novelty", 0.5) + 0.4 * stats.get("avg_elaboration", 0.3)
        else:  # cognitive_flexibility
            inv_constraints = 1.0 - min(1.0, stats.get("constraints_count", 0) / 10.0)
            base = 0.5 * stats.get("diversity", 0.5) + 0.5 * inv_constraints
        return _clip(base, 0.0, 1.0)

    def apply(self, stats: Dict[str, Any], states: Dict[str, float], scale: float = 1.0) -> None:
        features = self._features(stats)
        for state in self.state_names:
            model = self.models[state]
            max_delta = self.max_delta.get(state, 0.1) * scale
            pred = model.predict(features)
            delta = (pred - 0.5) * 2.0 * max_delta
            states[state] = _clip(states.get(state, 0.5) + delta)
            drift = model.update(features, self._target(state, stats))
            if drift > self.drift_threshold:
                self.drift_log.append({
                    "timestamp": time.time(),
                    "state": state,
                    "drift": drift,
                    "features": features,
                    "stats": {k: stats.get(k) for k in ("strategy", "reward", "ideas_generated")},
                })


class ContextualThompsonBandit:
    """Bandit contextuel simple avec Thompson Sampling discret."""

    def __init__(self, strategies: List[str], decay: float = 0.01) -> None:
        self.strategies = strategies
        self.decay = decay
        self.posteriors: Dict[Tuple[str, str], List[float]] = defaultdict(lambda: [1.0, 1.0])

    def select(self, context_key: str) -> Tuple[str, str]:
        if not self.strategies:
            return "random_association", context_key
        best_strategy = None
        best_score = -1.0
        for strat in self.strategies:
            alpha, beta = self.posteriors[(strat, context_key)]
            sample = random.betavariate(alpha, beta)
            if sample > best_score:
                best_score = sample
                best_strategy = strat
        return best_strategy or self.strategies[0], context_key

    def update(self, strategy: str, context_key: str, reward: float) -> None:
        key = (strategy, context_key)
        alpha, beta = self.posteriors[key]
        alpha = max(1.0, alpha * (1.0 - self.decay))
        beta = max(1.0, beta * (1.0 - self.decay))
        success = _clip(reward, 0.0, 1.0)
        failure = 1.0 - success
        self.posteriors[key] = [alpha + success, beta + failure]


class DiscreteThompsonSelector:
    """Sélectionneur Thompson Sampling pour options discrètes (ex. TTL)."""

    def __init__(self, options: List[int], default: int = 7, decay: float = 0.01) -> None:
        self.options = options
        self.default = default
        self.decay = decay
        self.posteriors: Dict[Tuple[int, str], List[float]] = defaultdict(lambda: [1.0, 1.0])
        self.last_choice: int = default

    def select(self, context_key: str) -> Tuple[int, str]:
        if not self.options:
            return self.default, context_key
        choice = None
        best_score = -1.0
        for option in self.options:
            alpha, beta = self.posteriors[(option, context_key)]
            sample = random.betavariate(alpha, beta)
            if sample > best_score:
                best_score = sample
                choice = option
        self.last_choice = choice or self.default
        return self.last_choice, context_key

    def update(self, option: int, context_key: str, reward: float) -> None:
        key = (option, context_key)
        alpha, beta = self.posteriors[key]
        alpha = max(1.0, alpha * (1.0 - self.decay))
        beta = max(1.0, beta * (1.0 - self.decay))
        success = _clip(reward, 0.0, 1.0)
        failure = 1.0 - success
        self.posteriors[key] = [alpha + success, beta + failure]
# -------------------- Subsystems (light) --------------------

class ActivationSpreadingSystem:
    def __init__(self, factor: float=0.3, decay: float=0.99):
        self.factor = factor
        self.decay = decay
    def tick(self, G):
        for node in G.nodes():
            act = G.nodes[node].get("activation", 0.1) * self.decay
            G.nodes[node]["activation"] = max(0.01, act)
    def boost(self, G, start: str, amount: float=0.2):
        if start in G.nodes():
            G.nodes[start]["activation"] = _clip(G.nodes[start].get("activation",0.1)+amount)
            for nb in getattr(G, "neighbors", lambda n: [])(start):
                w = G.get_edge_data(start, nb).get("weight", 0.5)
                G.nodes[nb]["activation"] = _clip(G.nodes[nb].get("activation",0.1)+w*self.factor*amount)

# -------------------- Main System --------------------

class CreativitySystem:
    """
    Système de créativité robuste, avec normalisation silencieuse et threads d'arrière-plan.
    """
    def __init__(self, cognitive_architecture: Any=None, memory_system: Any=None,
                 reasoning_system: Any=None, emotional_system: Any=None,
                 metacognitive_system: Any=None):
        self.cognitive_architecture = cognitive_architecture
        self.memory_system = memory_system or (getattr(cognitive_architecture, "memory", None) if cognitive_architecture else None)
        self.reasoning_system = reasoning_system or (getattr(cognitive_architecture, "reasoning", None) if cognitive_architecture else None)
        self.emotional_system = emotional_system or (getattr(cognitive_architecture, "emotions", None) if cognitive_architecture else None)
        self.metacognitive_system = metacognitive_system or (getattr(cognitive_architecture, "metacognition", None) if cognitive_architecture else None)

        # Core containers
        self.conceptual_space: Dict[str, Any] = {"concept_network": nx.Graph(), "act": ActivationSpreadingSystem()}
        self.idea_generation: Dict[str, Any] = {
            "idea_pool": deque(maxlen=1000),
            "generation_strategies": {},
        }
        self.conceptual_blending: Dict[str, Any] = {"blend_history": deque(maxlen=500)}
        self.insight_detection: Dict[str, Any] = {"insight_history": deque(maxlen=200)}
        self.innovation_engine: Dict[str, Any] = {"innovation_pipeline": deque(maxlen=50)}
        self.creative_processes: Dict[str, Any] = {"current_phase": "preparation", "phase_transitions": deque(maxlen=100)}
        self.creative_states: Dict[str, Any] = {
            "creative_flow": 0.3, "cognitive_flexibility": 0.5,
            "inspiration_level": 0.4, "creative_confidence": 0.5
        }
        self.contextual_influences: Dict[str, Any] = {
            "environmental_stimuli": [], "cultural_constraints": [], "domain_knowledge": {},
            "emotional_climate": 0.5, "cognitive_load": 0.3, "time_pressure": 0.2
        }
        self.creative_knowledge: Dict[str, Any] = {
            "analogical_sources": {
                "nature": ["évolution", "symbiose", "adaptation"],
                "technologie": ["réseau", "interface", "automatisation"],
                "art": ["composition", "contraste", "rythme"],
            },
            "heuristics": {},
            "constraint_templates": {},
            "innovation_patterns": {}
        }
        self.creative_history: Dict[str, Any] = {
            "ideas_generated": deque(maxlen=5000),
            "blends_created": deque(maxlen=1000),
            "insights_experienced": deque(maxlen=500),
            "projects_completed": deque(maxlen=100),
            "creative_breakthroughs": deque(maxlen=50),
            "learning_trajectory": deque(maxlen=1000),
        }
        self.processing_threads: Dict[str, Any] = {}
        self.running = True

        # Init
        crea_normalize(self)
        self._initialize_basic_conceptual_network()
        self._initialize_generation_strategies()
        self._initialize_innate_creativity()
        self._initialize_adaptive_modules()

        # Start background
        self._start_creative_monitoring()
        self._start_incubation_process()
        self._start_insight_detection()

        # First cycle
        self._initiate_first_creative_cycle()

    # -------------------- Initialization helpers --------------------

    def _initialize_basic_conceptual_network(self) -> None:
        G = self.conceptual_space["concept_network"]
        base = ["espace","temps","mouvement","énergie","forme","couleur","son","texture","quantité","qualité",
                "relation","causalité","similarité","différence","partie","tout","ordre","chaos","symétrie","asymétrie"]
        for c in base:
            G.add_node(c, activation=0.1, domain="fondamental")
        for a,b in [("espace","temps"),("mouvement","énergie"),("forme","couleur"),
                    ("partie","tout"),("ordre","chaos"),("symétrie","asymétrie"),
                    ("relation","causalité"),("similarité","différence")]:
            G.add_edge(a,b, weight=0.8, type="fondamental")

    def _initialize_generation_strategies(self) -> None:
        self.idea_generation["generation_strategies"] = {
            "random_association": self._strat_random_association,
            "domain_transfer": self._strat_domain_transfer,
            "constraint_challenge": self._strat_constraint_challenge,
            "attribute_listing": self._strat_attribute_listing,
            "forced_relationship": self._strat_forced_relationship,
        }

    def _initialize_innate_creativity(self) -> None:
        self.creative_knowledge["heuristics"].update({
            "analogy": {"description": "Similarités entre domaines", "effectiveness": 0.7},
            "combination": {"description": "Combiner des éléments", "effectiveness": 0.8},
            "transformation": {"description": "Transformer un concept", "effectiveness": 0.6},
        })
        self.creative_knowledge["innovation_patterns"].update({
            "problem_solution": {"description": "Problème → Solution", "success_rate": 0.6},
            "improvement": {"description": "Amélioration incrémentale", "success_rate": 0.7},
        })

    def _initialize_adaptive_modules(self) -> None:
        strategies = list(self.idea_generation.get("generation_strategies", {}).keys())
        self.strategy_selector = ContextualThompsonBandit(strategies)
        self.ttl_selector = DiscreteThompsonSelector([3, 7, 14, 30], default=7)
        self.adaptive_metrics = AdaptiveCreativeMetrics(self.creative_states, strategies)
        self.adaptive_drift_log = self.adaptive_metrics.drift_log

    # -------------------- Background threads --------------------

    def _creative_monitoring_loop(self) -> None:
        while self.running:
            try:
                crea_normalize(self)
                self._monitor_creative_state()
                self._update_conceptual_space()
                self._evaluate_ongoing_ideas()
            except Exception:
                pass
            time.sleep(2)

    def _incubation_loop(self) -> None:
        while self.running:
            try:
                crea_normalize(self)
                self._process_incubation_phase()
            except Exception:
                pass
            time.sleep(10)

    def _insight_detection_loop(self) -> None:
        while self.running:
            try:
                crea_normalize(self)
                self._monitor_insight_conditions()
            except Exception:
                pass
            time.sleep(5)

    def _start_creative_monitoring(self) -> None:
        import threading

        t = threading.Thread(target=self._creative_monitoring_loop, daemon=True)
        t.start()
        self.processing_threads["creative_monitoring"] = t

    def _start_incubation_process(self) -> None:
        import threading

        t = threading.Thread(target=self._incubation_loop, daemon=True)
        t.start()
        self.processing_threads["incubation_process"] = t

    def _start_insight_detection(self) -> None:
        import threading

        t = threading.Thread(target=self._insight_detection_loop, daemon=True)
        t.start()
        self.processing_threads["insight_detection"] = t

    # -------------------- First cycle --------------------

    def _initiate_first_creative_cycle(self) -> None:
        crea_normalize(self)
        ideas = self.generate_ideas("créativité", constraints=[], num_ideas=5, strategy="random_association")
        self.creative_processes["current_phase"] = "preparation"
        self.creative_history["learning_trajectory"].append({
            "timestamp": time.time(),
            "event": "first_cycle",
            "ideas_generated": len(ideas),
            "creative_state": dict(self.creative_states),
        })

    # -------------------- Idea generation --------------------

    def generate_ideas(self, topic: str, constraints: List[str], num_ideas: int=10, strategy: str="auto") -> List[CreativeIdea]:
        crea_normalize(self)
        context_key = self._context_signature(topic, constraints)
        strategy_context = context_key
        strategy_guidance = None
        if strategy == "auto":
            strategy, strategy_context, strategy_guidance = self._select_strategy(topic, constraints)
        ttl_choice, ttl_context = self.ttl_selector.select(context_key)
        self._enforce_ttl(ttl_choice)
        func = self.idea_generation["generation_strategies"].get(strategy, self._strat_random_association)
        start_time = time.time()
        raw = func(topic, constraints, num_ideas)
        generation_time = max(0.001, time.time() - start_time)
        out: List[CreativeIdea] = []
        for s in raw:
            ci = self._develop_raw_idea(s, topic, constraints)
            out.append(ci)
            self.idea_generation["idea_pool"].append(ci)
            self.creative_history["ideas_generated"].append(ci)
        llm_new, llm_guidance = self._llm_generate_ideas(
            topic,
            constraints,
            strategy,
            out,
            context_key,
            ttl_choice,
        )
        if llm_new:
            for idea in llm_new:
                out.append(idea)
                self.idea_generation["idea_pool"].append(idea)
                self.creative_history["ideas_generated"].append(idea)
        stats = self._compute_generation_stats(out, strategy, generation_time, len(constraints), ttl_choice, context_key, strategy_context, ttl_context)
        if strategy_guidance:
            stats.setdefault("llm_strategy_guidance", strategy_guidance)
        if llm_guidance:
            stats.setdefault("llm_idea_guidance", llm_guidance)
        self._update_creative_metrics(stats)
        reward = stats.get("reward", 0.0)
        self.strategy_selector.update(strategy, strategy_context, reward)
        self.ttl_selector.update(ttl_choice, ttl_context, reward)
        return out

    def _select_strategy(self, topic: str, constraints: List[str]) -> Tuple[str, str, Optional[Mapping[str, Any]]]:
        context_key = self._context_signature(topic, constraints)
        strategy, ctx = self.strategy_selector.select(context_key)
        guidance = self._llm_select_strategy(topic, constraints, context_key, strategy, ctx)
        if guidance and isinstance(guidance.get("strategy"), str):
            strategy = guidance["strategy"]
        if guidance and isinstance(guidance.get("variant"), str):
            ctx = guidance["variant"]
        return strategy, ctx, guidance

    def _context_signature(self, topic: str, constraints: List[str]) -> str:
        topic = topic or ""
        words = topic.split()
        char_bucket = min(5, len(topic) // 4)
        word_bucket = min(5, len(words))
        constraint_bucket = min(5, len(constraints))
        mode = "multi" if word_bucket > 1 else "mono"
        time_bucket = int((time.time() % 86400) // 3600) // 4
        return f"len:{char_bucket}|words:{word_bucket}|constraints:{constraint_bucket}|mode:{mode}|t:{time_bucket}"

    def _llm_select_strategy(
        self,
        topic: str,
        constraints: List[str],
        context_key: str,
        fallback_strategy: str,
        fallback_context: str,
    ) -> Optional[Mapping[str, Any]]:
        payload = {
            "topic": topic,
            "constraints": constraints,
            "context_key": context_key,
            "fallback": {
                "strategy": fallback_strategy,
                "context": fallback_context,
            },
            "recent_metrics": getattr(self, "_last_generation_stats", {}),
        }

        response = try_call_llm_dict(
            "creativity_strategy_selector",
            input_payload=payload,
            logger=LOGGER,
        )

        if isinstance(response, Mapping):
            return dict(response)
        return None

    def _enforce_ttl(self, ttl_days: int) -> None:
        ttl_seconds = max(1, ttl_days) * 86400
        cutoff = time.time() - ttl_seconds
        pool = self.idea_generation["idea_pool"]
        retained = []
        for idea in list(pool):
            if _is_ci(idea) and getattr(idea, "created_time", 0.0) < cutoff:
                continue
            retained.append(idea)
        if len(retained) != len(pool):
            pool.clear()
            pool.extend(retained)

    def _aggregate_idea_metrics(self, ideas: List[CreativeIdea]) -> Dict[str, float]:
        if not ideas:
            return {
                "count": 0,
                "avg_novelty": 0.5,
                "avg_usefulness": 0.5,
                "avg_feasibility": 0.5,
                "avg_elaboration": 0.3,
                "diversity": 0.5,
            }
        total = len(ideas)
        novelty = sum(getattr(i, "novelty", 0.5) for i in ideas) / total
        usefulness = sum(getattr(i, "usefulness", 0.5) for i in ideas) / total
        feasibility = sum(getattr(i, "feasibility", 0.5) for i in ideas) / total
        elaboration = sum(getattr(i, "elaboration", 0.3) for i in ideas) / total
        unique_signatures = {str(getattr(i, "concept_core", ""))[:64] for i in ideas}
        diversity = min(1.0, len(unique_signatures) / max(1, total))
        return {
            "count": total,
            "avg_novelty": _clip(novelty),
            "avg_usefulness": _clip(usefulness),
            "avg_feasibility": _clip(feasibility),
            "avg_elaboration": _clip(elaboration),
            "diversity": diversity,
        }

    def _compute_generation_stats(self, ideas: List[CreativeIdea], strategy: str, generation_time: float,
                                   constraints_count: int, ttl_days: int, context_key: str,
                                   strategy_context: str, ttl_context: str) -> Dict[str, Any]:
        metrics = self._aggregate_idea_metrics(ideas)
        efficiency = metrics["count"] / max(generation_time, 1e-6)
        reward = _clip(0.4 * metrics["avg_novelty"] + 0.4 * metrics["avg_usefulness"] + 0.2 * metrics["avg_feasibility"])
        stats = {
            "ideas_generated": metrics["count"],
            "generation_time": generation_time,
            "strategy": strategy,
            "strategy_context": strategy_context,
            "avg_novelty": metrics["avg_novelty"],
            "avg_usefulness": metrics["avg_usefulness"],
            "avg_feasibility": metrics["avg_feasibility"],
            "avg_elaboration": metrics["avg_elaboration"],
            "diversity": metrics["diversity"],
            "reward": reward,
            "efficiency": efficiency,
            "constraints_count": constraints_count,
            "ttl_days": ttl_days,
            "ttl_context": ttl_context,
            "context": context_key,
        }
        self._last_generation_stats = stats
        return stats

    def _llm_generate_ideas(
        self,
        topic: str,
        constraints: List[str],
        strategy: str,
        current: List[CreativeIdea],
        context_key: str,
        ttl_days: int,
    ) -> Tuple[List[CreativeIdea], Optional[Mapping[str, Any]]]:
        payload = {
            "topic": topic,
            "constraints": constraints,
            "strategy": strategy,
            "existing_ideas": [
                {
                    "title": idea.concept_core,
                    "novelty": getattr(idea, "novelty", 0.0),
                    "usefulness": getattr(idea, "usefulness", 0.0),
                }
                for idea in current
            ],
            "context_key": context_key,
            "ttl_days": ttl_days,
        }

        response = try_call_llm_dict(
            "creativity_pipeline",
            input_payload=payload,
            logger=LOGGER,
            extra_instructions=None,
        )

        ideas: List[CreativeIdea] = []
        if isinstance(response, Mapping):
            raw_ideas = response.get("ideas")
            if isinstance(raw_ideas, list):
                for idea_payload in raw_ideas:
                    if not isinstance(idea_payload, Mapping):
                        continue
                    idea = self._idea_from_llm(idea_payload, topic, constraints)
                    if idea:
                        ideas.append(idea)
        guidance = dict(response) if isinstance(response, Mapping) else None
        return ideas, guidance

    def _idea_from_llm(
        self,
        idea_payload: Mapping[str, Any],
        topic: str,
        constraints: List[str],
    ) -> Optional[CreativeIdea]:
        title = idea_payload.get("title") or idea_payload.get("name")
        description = idea_payload.get("description") or title
        if not isinstance(description, str):
            return None
        base = self._develop_raw_idea(description, topic, constraints)
        if isinstance(title, str):
            base.concept_core = title
            base.description = f"Idée créative: {title} — {description}"
        try:
            if "novelty" in idea_payload:
                base.novelty = _clip(float(idea_payload["novelty"]))
            if "usefulness" in idea_payload:
                base.usefulness = _clip(float(idea_payload["usefulness"]))
            if "feasibility" in idea_payload:
                base.feasibility = _clip(float(idea_payload.get("feasibility", base.feasibility)))
            if "elaboration" in idea_payload:
                base.elaboration = _clip(float(idea_payload.get("elaboration", base.elaboration)))
        except Exception:
            pass
        return base

    # Strategies (return list[str])
    def _strat_random_association(self, topic: str, constraints: List[str], n: int) -> List[str]:
        G = self.conceptual_space["concept_network"]
        nodes = list(G.nodes()) or [topic]
        ideas = []
        for _ in range(n*3):
            a = random.choice(nodes)
            b = random.choice(nodes)
            if a==b and len(nodes)>1:
                b = random.choice([x for x in nodes if x!=a])
            ideas.append(f"{a} + {b} → {a}-{b} hybride")
            if len(ideas)>=n: break
        return ideas[:n]

    def _strat_domain_transfer(self, topic: str, constraints: List[str], n: int) -> List[str]:
        src = list(self.creative_knowledge["analogical_sources"].keys())
        ideas = []
        for _ in range(n):
            d = random.choice(src) if src else "source"
            c = random.choice(self.creative_knowledge["analogical_sources"].get(d, ["concept"]))
            ideas.append(f"Transférer {c} ({d}) vers {topic}")
        return ideas

    def _strat_constraint_challenge(self, topic: str, constraints: List[str], n: int) -> List[str]:
        ideas = []
        for i in range(max(1, min(n, len(constraints) or 1))):
            if constraints:
                c = constraints[i % len(constraints)]
                ideas.append(f"{topic} sans '{c}'")
            else:
                ideas.append(f"{topic} sans contrainte X")
        while len(ideas)<n:
            ideas.append(f"{topic} approche radicale #{len(ideas)+1}")
        return ideas[:n]

    def _strat_attribute_listing(self, topic: str, constraints: List[str], n: int) -> List[str]:
        attrs = ["taille","couleur","forme","texture","poids","durée","intensité","complexité","vitesse","coût"]
        mods = ["amplifier","réduire","inverser","combiner","transformer"]
        ideas = []
        for _ in range(n):
            ideas.append(f"{topic} avec {random.choice(attrs)} à {random.choice(mods)}")
        return ideas

    def _strat_forced_relationship(self, topic: str, constraints: List[str], n: int) -> List[str]:
        G = self.conceptual_space["concept_network"]
        nodes = list(G.nodes()) or [topic]
        random.shuffle(nodes)
        ideas = []
        for c in nodes[:n]:
            if c==topic: continue
            ideas.append(f"Relation forcée {topic} ↔ {c}")
        while len(ideas)<n: ideas.append(f"Relation forcée {topic} ↔ concept_{len(ideas)+1}")
        return ideas[:n]

    # Develop raw idea into CreativeIdea
    def _develop_raw_idea(self, raw: str, topic: str, constraints: List[str]) -> CreativeIdea:
        novelty = _clip(random.gauss(0.65, 0.2))
        usefulness = _clip(random.gauss(0.6, 0.2))
        feasibility = _clip(random.gauss(0.55, 0.2))
        elaboration = _clip(random.gauss(0.35, 0.15))
        return CreativeIdea(
            id=f"idea_{int(time.time()*1000)}_{random.randint(100,999)}",
            concept_core=str(raw),
            description=f"Idée créative: {raw}",
            state=IdeaState.RAW,
            novelty=novelty,
            usefulness=usefulness,
            feasibility=feasibility,
            elaboration=elaboration,
            domains=[],
            components=[],
            associations=[],
            emotional_affinity=_clip(random.random()),
            activation_level=0.1
        )

    # -------------------- Monitoring --------------------

    def _monitor_creative_state(self) -> None:
        ideas = [i for i in list(self.idea_generation["idea_pool"])[-10:] if _is_ci(i)]
        if ideas:
            ttl_days = getattr(self.ttl_selector, "last_choice", 7)
            stats = self._compute_generation_stats(
                ideas,
                strategy="monitor",
                generation_time=1.0,
                constraints_count=0,
                ttl_days=ttl_days,
                context_key="monitoring",
                strategy_context="monitoring",
                ttl_context="monitoring",
            )
            self.adaptive_metrics.apply(stats, self.creative_states, scale=0.35)

    def _update_conceptual_space(self) -> None:
        G = self.conceptual_space["concept_network"]
        act = self.conceptual_space.get("act")
        if act:
            act.tick(G)

    def _evaluate_ongoing_ideas(self) -> None:
        ideas = [i for i in list(self.idea_generation["idea_pool"])[-20:] if _is_ci(i)]
        for i in ideas:
            if i.state == IdeaState.RAW and random.random()<0.1:
                i.state = IdeaState.DEVELOPED
                i.last_modified = time.time()
                i.development_history.append({"timestamp": time.time(), "action": "promotion_auto", "state": "developed"})

    def _process_incubation_phase(self) -> None:
        ideas = [i for i in list(self.idea_generation["idea_pool"]) if _is_ci(i)]
        for i in ideas:
            if i.state in (IdeaState.RAW, IdeaState.DEVELOPED):
                i.activation_level = _clip(i.activation_level + random.uniform(0.01, 0.05))

    def _monitor_insight_conditions(self) -> None:
        cond = 0
        if self.creative_states.get("creative_flow",0.3) > 0.6: cond += 1
        if self.creative_states.get("cognitive_flexibility",0.5) > 0.5: cond += 1
        if self.creative_states.get("inspiration_level",0.4) > 0.4: cond += 1
        if 0.3 < self.contextual_influences.get("cognitive_load",0.5) < 0.7: cond += 1
        if cond >=3 and random.random()<0.1:
            self.experience_insight({"problem":"optimisation_processus_créatif"})

    # -------------------- Blending --------------------

    def create_conceptual_blend(self, concept1: str, concept2: str) -> ConceptualBlend:
        generic = ["relation","structure","processus"]
        blended = f"{concept1}-{concept2} hybride via {random.choice(generic)}"
        emergent = [f"structure émergente autour de {concept1} & {concept2}"]
        creativity = _clip(len(blended)/80.0)
        coherence = _clip(0.5 + 0.1*random.random())
        elabor = _clip(0.4 + 0.2*random.random())
        blend = ConceptualBlend(
            blend_id=f"blend_{int(time.time()*1000)}_{random.randint(100,999)}",
            input_spaces=[concept1, concept2],
            generic_space=generic,
            blended_space=blended,
            emergent_structure=emergent,
            creativity_score=creativity,
            coherence=coherence,
            elaboration_potential=elabor
        )
        self.conceptual_blending["blend_history"].append(blend)
        self.creative_history["blends_created"].append(blend)
        return blend

    # -------------------- Insights --------------------

    def experience_insight(self, problem_context: Dict[str, Any]) -> Optional[CreativeInsight]:
        content = f"Perspective nouvelle sur {problem_context.get('problem','un problème')}"
        t = time.time()
        insight = CreativeInsight(
            insight_id=f"insight_{int(t*1000)}_{random.randint(100,999)}",
            type=InsightType.EMERGENT,
            content=content,
            significance=_clip(random.uniform(0.5,0.9)),
            clarity=_clip(random.uniform(0.6,0.9)),
            surprise=_clip(random.uniform(0.6,0.9)),
            emotional_intensity=_clip(random.uniform(0.5,0.9)),
            preceding_incubation=random.uniform(10.0,60.0),
            trigger=random.choice(["changement de perspective","association distante","relaxation"]),
            timestamp=t,
            related_ideas=[i.id for i in list(self.idea_generation["idea_pool"])[-2:] if _is_ci(i)],
            verification_status="unverified"
        )
        self.insight_detection.setdefault("insight_history", deque(maxlen=200)).append(insight)
        self.creative_history["insights_experienced"].append(insight)
        self._update_after_insight(insight)
        return insight

    def _update_after_insight(self, insight: CreativeInsight) -> None:
        self.creative_states["creative_confidence"] = _clip(self.creative_states.get("creative_confidence",0.5) + 0.2*insight.significance)
        self.creative_states["inspiration_level"] = _clip(self.creative_states.get("inspiration_level",0.4) + 0.15*insight.emotional_intensity)

    # -------------------- Innovation --------------------

    def develop_innovation_project(self, core_idea: str, objectives: List[str], constraints: List[str]) -> InnovationProject:
        phases = [
            {"name":"exploration","duration_estimate":random.uniform(5,15),"resources":["recherche","brainstorming"]},
            {"name":"prototype","duration_estimate":random.uniform(10,30),"resources":["conception","test"]},
            {"name":"validation","duration_estimate":random.uniform(8,20),"resources":["expérimentation","feedback"]},
            {"name":"implémentation","duration_estimate":random.uniform(15,40),"resources":["déploiement","documentation"]},
        ]
        risks = {"technique": random.uniform(0.3,0.8), "marché": random.uniform(0.2,0.7), "ressources": _clip(len(constraints)/10.0), "temporel": random.uniform(0.4,0.9)}
        metrics = {"satisfaction_utilisateur": random.uniform(0.6,0.9), "impact_innovation": random.uniform(0.5,0.8)}
        res = ["temps","attention","énergie_cognitive"]
        project = InnovationProject(
            project_id=f"project_{int(time.time()*1000)}_{random.randint(100,999)}",
            core_idea=core_idea, objectives=objectives, constraints=constraints,
            resources_needed=res, development_phases=phases, current_phase=0,
            success_metrics=metrics, risk_assessment=risks,
            team_dynamics={"auto_collaboration": True, "roles":["générateur","évaluateur","développeur"]}
        )
        self.innovation_engine["innovation_pipeline"].append(project)
        self.creative_history["projects_completed"].append(project)
        return project

    # -------------------- Metrics & Status --------------------

    def _update_creative_metrics(self, stats: Dict[str, Any]) -> None:
        self.adaptive_metrics.apply(stats, self.creative_states)
        log_entry = {
            "timestamp": time.time(),
            "strategy": stats.get("strategy"),
            "strategy_context": stats.get("strategy_context"),
            "ttl_days": stats.get("ttl_days"),
            "ttl_context": stats.get("ttl_context"),
            "ideas_generated": stats.get("ideas_generated"),
            "generation_time": stats.get("generation_time"),
            "reward": stats.get("reward"),
            "avg_novelty": stats.get("avg_novelty"),
            "avg_usefulness": stats.get("avg_usefulness"),
            "avg_feasibility": stats.get("avg_feasibility"),
            "avg_elaboration": stats.get("avg_elaboration"),
            "diversity": stats.get("diversity"),
            "efficiency": stats.get("efficiency"),
            "creative_state": dict(self.creative_states),
        }
        if self.adaptive_metrics.drift_log:
            log_entry["recent_drift"] = self.adaptive_metrics.drift_log[-1]
        self.creative_history["learning_trajectory"].append(log_entry)

    def get_creative_status(self) -> Dict[str, Any]:
        pool = [i for i in list(self.idea_generation["idea_pool"])[-10:] if _is_ci(i)]
        return {
            "creative_states": dict(self.creative_states),
            "current_phase": self.creative_processes.get("current_phase","preparation"),
            "recent_activity": {
                "ideas_generated": len(pool),
                "ideas_by_state": { s.value: sum(1 for i in pool if i.state==s) for s in IdeaState }
            },
            "conceptual_space": {
                "concepts_count": self.conceptual_space["concept_network"].number_of_nodes(),
                "connections_count": self.conceptual_space["concept_network"].number_of_edges(),
            },
            "innovation_pipeline": {
                "projects_count": len(self.innovation_engine.get("innovation_pipeline", [])),
            },
            "historical_metrics": {
                "total_ideas": len(self.creative_history["ideas_generated"]),
                "total_blends": len(self.creative_history["blends_created"]),
                "total_insights": len(self.creative_history["insights_experienced"]),
                "total_projects": len(self.creative_history["projects_completed"]),
                "breakthroughs": len(self.creative_history["creative_breakthroughs"]),
            }
        }

    def stop_creativity_system(self) -> None:
        self.running = False

__all__ = ["CreativitySystem","CreativeIdea","ConceptualBlend","CreativeInsight","InnovationProject","IdeaState","InsightType"]
