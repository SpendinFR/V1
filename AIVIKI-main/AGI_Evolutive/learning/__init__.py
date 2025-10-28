# learning/__init__.py
"""
Système d'Apprentissage Évolutif de l'AGI - module unique et auto-contenu.

Ce fichier regroupe TOUTES les classes d'apprentissage en un seul module :
- ExperientialLearning (apprentissage expérientiel, cycle de Kolb)
- MetaLearning (méta-apprentissage, ajuste les hyper-paramètres)
- TransferLearning (transfert inter-domaines, mapping analogique)
- ReinforcementLearning (apprentissage par renforcement tabulaire simple)
- CuriosityEngine (récompense intrinsèque, exploration)

Points clés :
- AUCUNE importation de sous-modules (évite l'erreur d'import).
- Méthodes to_state()/from_state() pour la persistance.
- Auto-wiring sécurisé via getattr(self.cognitive_architecture, "<module_name>").
- Idempotent : si un sous-composant n'existe pas, le code reste stable.
"""

from __future__ import annotations
import logging
import time, math, random, hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


# ============================================================
# 🌱 Utilitaires communs
# ============================================================

def _now() -> float:
    return time.time()

def _safe_mean(xs: List[float], default: float = 0.0) -> float:
    return sum(xs) / len(xs) if xs else default

def _hash_str(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


# ============================================================
# 🧩 1) Apprentissage EXPÉRIENTIEL (cycle de Kolb)
# ============================================================

@dataclass
class LearningEpisode:
    id: str
    timestamp: float
    raw: Dict[str, Any] = field(default_factory=dict)
    reflections: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    outcomes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    emotional_valence: float = 0.0
    confidence_gain: float = 0.0
    integration_level: float = 0.0
    contextual_features: List[float] = field(default_factory=list)
    contextual_metadata: Dict[str, Any] = field(default_factory=dict)


class ContextFeatureEncoder:
    """Encode les expériences en un vecteur de caractéristiques compact."""

    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self._feature_names: List[str] = [
            "bias",
            "summary_length",
            "has_error",
            "has_success",
            "has_time",
            "concept_count",
            "reflection_count",
            "novelty_signal",
            "emotion_valence",
            "confidence_bias",
        ]

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def encode(
        self,
        raw: Dict[str, Any],
        reflections: Iterable[str],
        concepts: Iterable[str],
        emotional_valence: float = 0.0,
        confidence_gain: float = 0.0,
    ) -> Tuple[List[float], Dict[str, Any]]:
        summary = str(raw.get("summary", raw))
        summary_len = min(len(summary) / 100.0, 5.0)
        lower_summary = summary.lower()
        has_error = 1.0 if ("erreur" in lower_summary or "error" in lower_summary) else 0.0
        has_success = 1.0 if ("succès" in lower_summary or "success" in lower_summary) else 0.0
        has_time = 1.0 if ("temps" in lower_summary or "delai" in lower_summary or "delay" in lower_summary) else 0.0
        concept_count = float(len(list(concepts)))
        reflection_count = float(len(list(reflections)))

        novelty_signal = 0.0
        arch = self.cognitive_architecture
        if arch and getattr(arch, "memory", None) and hasattr(arch.memory, "novelty_score"):
            try:
                novelty_signal = float(arch.memory.novelty_score(summary))
            except Exception:
                novelty_signal = 0.0

        emotion_valence = emotional_valence
        if arch and getattr(arch, "emotions", None):
            try:
                emotion_valence = float(getattr(arch.emotions, "current_valence", emotion_valence))
            except Exception:
                emotion_valence = emotional_valence

        confidence_bias = confidence_gain
        meta = {
            "summary_length": summary_len,
            "concept_count": concept_count,
            "reflection_count": reflection_count,
            "novelty_signal": novelty_signal,
        }
        features = [
            1.0,
            summary_len,
            has_error,
            has_success,
            has_time,
            concept_count,
            reflection_count,
            novelty_signal,
            emotion_valence,
            confidence_bias,
        ]
        enriched = self._llm_enrich_features(
            raw=raw,
            reflections=list(reflections),
            concepts=list(concepts),
            features=features,
            meta=meta,
            emotional_valence=emotion_valence,
            confidence_gain=confidence_gain,
        )
        if enriched:
            features, meta = enriched
        return features, meta

    def _llm_enrich_features(
        self,
        *,
        raw: Mapping[str, Any],
        reflections: List[str],
        concepts: List[str],
        features: List[float],
        meta: Dict[str, Any],
        emotional_valence: float,
        confidence_gain: float,
    ) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        payload = {
            "summary": raw,
            "reflections": reflections,
            "concepts": concepts,
            "features": features,
            "emotional_valence": emotional_valence,
            "confidence_gain": confidence_gain,
        }

        response = try_call_llm_dict(
            "context_feature_encoder",
            input_payload=payload,
            logger=LOGGER,
        )
        if not isinstance(response, Mapping):
            return None

        llm_features = response.get("features")
        merged = list(features)
        if isinstance(llm_features, list) and len(llm_features) == len(features):
            try:
                merged = [
                    float(a + float(b)) / 2.0
                    for a, b in zip(features, llm_features)
                ]
            except Exception:
                merged = list(features)
        llm_tags = response.get("tags")
        updated_meta = dict(meta)
        if llm_tags:
            updated_meta["llm_tags"] = llm_tags
        if response.get("notes"):
            updated_meta.setdefault("llm_notes", response.get("notes"))
        return merged, updated_meta


class OnlineLinearModel:
    """Régression linéaire online bornée (GLM simplifié)."""

    def __init__(
        self,
        feature_dim: int,
        learning_rate: float = 0.05,
        l2: float = 1e-3,
        bounds: Tuple[float, float] = (0.0, 1.0),
        activation: Optional[Callable[[float], float]] = None,
    ):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.bounds = bounds
        self.weights: List[float] = [0.0] * max(1, feature_dim)
        self.bias: float = 0.0
        self.activation = activation or (lambda x: 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x)))))

    def _ensure_dim(self, n: int):
        if n <= len(self.weights):
            return
        self.weights.extend([0.0] * (n - len(self.weights)))

    def predict(self, features: Iterable[float]) -> float:
        feats = list(features)
        self._ensure_dim(len(feats))
        z = self.bias
        for w, x in zip(self.weights, feats):
            z += w * x
        activated = self.activation(z)
        lo, hi = self.bounds
        return max(lo, min(hi, activated))

    def update(self, features: Iterable[float], target: float) -> Tuple[float, float]:
        feats = list(features)
        self._ensure_dim(len(feats))
        prediction = self.predict(feats)
        lo, hi = self.bounds
        clipped_target = max(lo, min(hi, float(target)))
        error = clipped_target - prediction
        lr = self.learning_rate
        for i, x in enumerate(feats):
            self.weights[i] += lr * (error * x - self.l2 * self.weights[i])
        self.bias += lr * (error - self.l2 * self.bias)
        return prediction, error

    def reset(self):
        for i in range(len(self.weights)):
            self.weights[i] = 0.0
        self.bias = 0.0

    def feature_importances(self) -> List[float]:
        return [abs(w) for w in self.weights]


class ThompsonBandit:
    """Thompson Sampling Beta-Bernoulli pour l'exploration contextuelle."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.parameters: Dict[str, Tuple[float, float]] = {}

    def _params(self, action: str) -> Tuple[float, float]:
        if action not in self.parameters:
            self.parameters[action] = (self.prior_alpha, self.prior_beta)
        return self.parameters[action]

    def select(
        self,
        actions: Iterable[str],
        priors: Optional[Dict[str, float]] = None,
        fallback: Optional[Callable[[], str]] = None,
    ) -> str:
        best_action = None
        best_score = -float("inf")
        for action in actions:
            alpha, beta = self._params(action)
            sample = random.betavariate(alpha, beta)
            if priors and action in priors:
                # moyenne pondérée : moitié exploitation, moitié exploration
                sample = 0.5 * sample + 0.5 * float(priors[action])
            if sample > best_score:
                best_score = sample
                best_action = action
        if best_action is None:
            return fallback() if fallback else ""
        return best_action

    def update(self, action: Optional[str], reward: float):
        if not action:
            return
        alpha, beta = self._params(action)
        reward = max(0.0, min(1.0, float(reward)))
        alpha += reward
        beta += (1.0 - reward)
        self.parameters[action] = (alpha, beta)

    def expected(self, action: str) -> float:
        alpha, beta = self._params(action)
        return alpha / (alpha + beta)


class ExperientialLearning:
    """
    Cycle complet : expérience concrète → observation réflexive → conceptualisation → expérimentation.
    Conçu pour intégrer des épisodes et enrichir la mémoire & les politiques.
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        # journal d'expériences (persistant)
        self.learning_episodes: List[LearningEpisode] = []
        # compétences / états (adaptent les comportements)
        self.learning_competencies: Dict[str, float] = {
            "observation": 0.55,
            "reflection": 0.5,
            "abstraction": 0.45,
            "experimentation": 0.5,
            "pattern_detection": 0.6,
            "analogy": 0.5,
        }
        self.learning_states: Dict[str, float] = {
            "engagement": 0.65,
            "frustration_tolerance": 0.6,
            "openness": 0.75,
            "momentum": 0.5,
        }
        self.metrics: Dict[str, Any] = {
            "episodes_completed": 0,
            "concepts_formed": 0,
            "skills_compiled": 0,
            "errors_corrected": 0,
            "insights": 0,
            "feature_updates": 0,
            "prediction_error": 0.0,
        }
        self.learning_rate: float = 0.1
        self.feature_encoder = ContextFeatureEncoder(cognitive_architecture)
        self.contextual_model = OnlineLinearModel(
            feature_dim=len(self.feature_encoder.feature_names),
            learning_rate=0.08,
            bounds=(0.0, 1.0),
        )
        self._recent_prediction_error: List[float] = []
        self.auto_curriculum: Dict[str, Dict[str, Any]] = {}

        # auto-wiring (lecture uniquement, pas d'import croisé)
        ca = self.cognitive_architecture
        if ca:
            self.memory = getattr(ca, "memory", None)
            self.emotions = getattr(ca, "emotions", None)
            self.metacognition = getattr(ca, "metacognition", None)
            self.goals = getattr(ca, "goals", None)
            self.reasoning = getattr(ca, "reasoning", None)
            self.perception = getattr(ca, "perception", None)
            self.world_model = getattr(ca, "world_model", None)
            self.creativity = getattr(ca, "creativity", None)

    # --------- pipeline principal ---------

    def process_experience(self, raw: Dict[str, Any]) -> LearningEpisode:
        eid = f"ep::{int(_now())}::{_hash_str(str(raw))}"
        reflections = self._reflect(raw)
        concepts = self._abstract(raw, reflections)
        experiments = self._design_experiments(concepts)
        outcomes = self._execute_and_evaluate(experiments)
        emotional_valence = self._valence(outcomes)
        confidence_gain = self._confidence(outcomes)
        integration = self._integration(concepts, outcomes)
        llm_guidance = self._llm_summarise_experience(
            raw,
            reflections=reflections,
            concepts=concepts,
            experiments=experiments,
            outcomes=outcomes,
            emotional_valence=emotional_valence,
            confidence_gain=confidence_gain,
            integration=integration,
        )
        if llm_guidance:
            add_refs = llm_guidance.get("reflections")
            if isinstance(add_refs, list):
                reflections.extend(str(r) for r in add_refs if r)
            add_concepts = llm_guidance.get("concepts")
            if isinstance(add_concepts, list):
                concepts.extend(str(c) for c in add_concepts if c)
            add_experiments = llm_guidance.get("experiments")
            if isinstance(add_experiments, list):
                for exp in add_experiments:
                    if isinstance(exp, Mapping):
                        experiments.append(dict(exp))
            if isinstance(llm_guidance.get("integration_score"), (int, float)):
                integration = float(llm_guidance["integration_score"])
        features, feature_meta = self.feature_encoder.encode(
            raw,
            reflections,
            concepts,
            emotional_valence=emotional_valence,
            confidence_gain=confidence_gain,
        )
        prediction, error = self.contextual_model.update(features, integration)
        self._recent_prediction_error.append(error)
        if len(self._recent_prediction_error) > 50:
            self._recent_prediction_error.pop(0)
        self.metrics["feature_updates"] += 1
        self.metrics["prediction_error"] = float(
            _safe_mean([abs(e) for e in self._recent_prediction_error], 0.0)
        )
        self._adjust_states_from_prediction(prediction, error)

        episode = LearningEpisode(
            id=eid, timestamp=_now(), raw=raw, reflections=reflections,
            concepts=concepts, experiments=experiments, outcomes=outcomes,
            emotional_valence=emotional_valence, confidence_gain=confidence_gain,
            integration_level=integration, contextual_features=features,
            contextual_metadata=feature_meta,
        )
        if llm_guidance:
            episode.contextual_metadata.setdefault("llm_guidance", llm_guidance)
        self.learning_episodes.append(episode)
        self.metrics["episodes_completed"] += 1
        self.metrics["concepts_formed"] += len(concepts)
        if outcomes:
            self.metrics["insights"] += 1

        # consolidation minimale en mémoire
        self._consolidate_episode(episode)
        self._bridge_with_meta_learning(episode, prediction, error)
        # feedback à la méta-cognition
        if getattr(self, "metacognition", None) and hasattr(self.metacognition, "register_learning_event"):
            try:
                self.metacognition.register_learning_event(episode.id, confidence_gain, integration)
            except Exception:
                pass
        return episode

    def _reflect(self, raw: Dict[str, Any]) -> List[str]:
        qs = [
            "Qu'est-ce qui s'est réellement passé ?",
            "Qu'est-ce que j'attendais ?",
            "Qu'est-ce qui a surpris ?",
            "Qu'est-ce que je dois vérifier ensuite ?",
        ]
        refs = [f"{q} → " + str(raw.get("summary", raw))[:120] for q in qs]
        # légère progression
        self.learning_competencies["reflection"] = min(1.0, self.learning_competencies["reflection"] + 0.01)
        return refs

    def _abstract(self, raw: Dict[str, Any], refs: List[str]) -> List[str]:
        concepts = []
        s = str(raw).lower()
        if "erreur" in s or "error" in s:
            concepts.append("Principe d'erreur : causes → effets (prévenir plutôt que corriger)")
        if "réussite" in s or "success" in s:
            concepts.append("Principe de réussite : répéter les conditions gagnantes")
        if "temps" in s or "delai" in s:
            concepts.append("Principe temporel : estimer/contraindre le temps utile")
        if not concepts:
            concepts.append("Principe de parcimonie : tester l'hypothèse la plus simple d'abord")
        self.learning_competencies["abstraction"] = min(1.0, self.learning_competencies["abstraction"] + 0.01)
        return concepts

    def _design_experiments(self, concepts: List[str]) -> List[Dict[str, Any]]:
        exps = []
        for c in concepts:
            exps.append({
                "concept": c,
                "type": "prediction_check",
                "risk": 0.3,
                "expected": "comportement cohérent avec le concept",
            })
            exps.append({
                "concept": c,
                "type": "generalization",
                "risk": 0.2,
                "expected": "fonctionne dans un contexte voisin",
            })
        self.learning_competencies["experimentation"] = min(1.0, self.learning_competencies["experimentation"] + 0.005)
        return exps

    def _execute_and_evaluate(self, exps: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        outcomes: Dict[str, Dict[str, float]] = {}
        for e in exps:
            risk = float(e.get("risk", 0.3))
            base = 0.7 - 0.4 * risk
            noise = random.uniform(-0.1, 0.1)
            success = max(0.05, min(0.95, base + noise))
            outcomes.setdefault(e["concept"], {"success_rate": 0.0, "learning_gain": 0.0})
            # agrégation simple
            outcomes[e["concept"]]["success_rate"] = (outcomes[e["concept"]]["success_rate"] + success) / 2 or success
            outcomes[e["concept"]]["learning_gain"] = (outcomes[e["concept"]]["learning_gain"] + (0.5 + success*0.3)) / 2
        return outcomes

    def _valence(self, outcomes: Dict[str, Dict[str, float]]) -> float:
        vals = [o.get("success_rate", 0.0) for o in outcomes.values()]
        return _safe_mean(vals, 0.5)

    def _confidence(self, outcomes: Dict[str, Dict[str, float]]) -> float:
        vals = [o.get("learning_gain", 0.0) for o in outcomes.values()]
        return _safe_mean(vals, 0.0) * 0.8

    def _adjust_states_from_prediction(self, prediction: float, error: float):
        # momentum suit la qualité de la prédiction (peu d'erreur => plus de momentum)
        delta = max(-0.05, min(0.05, -error))
        self.learning_states["momentum"] = max(
            0.0,
            min(1.0, self.learning_states.get("momentum", 0.5) + delta),
        )
        self.learning_states["engagement"] = max(
            0.0,
            min(1.0, self.learning_states.get("engagement", 0.65) + delta * 0.5),
        )

    def _llm_summarise_experience(
        self,
        raw: Mapping[str, Any],
        *,
        reflections: List[str],
        concepts: List[str],
        experiments: List[Mapping[str, Any]],
        outcomes: Mapping[str, Mapping[str, float]],
        emotional_valence: float,
        confidence_gain: float,
        integration: float,
    ) -> Optional[Mapping[str, Any]]:
        payload = {
            "raw": raw,
            "reflections": reflections,
            "concepts": concepts,
            "experiments": experiments,
            "outcomes": outcomes,
            "emotional_valence": emotional_valence,
            "confidence_gain": confidence_gain,
            "integration": integration,
        }

        response = try_call_llm_dict(
            "learning_encoder",
            input_payload=payload,
            logger=LOGGER,
        )
        if isinstance(response, Mapping):
            return dict(response)
        return None
        # Ajuste légèrement les compétences en fonction de la qualité du modèle
        impact = max(-0.02, min(0.02, prediction - 0.5))
        for key in ("reflection", "abstraction", "experimentation"):
            if key in self.learning_competencies:
                self.learning_competencies[key] = max(
                    0.1,
                    min(1.0, self.learning_competencies[key] + impact),
                )

    def _bridge_with_meta_learning(
        self,
        episode: LearningEpisode,
        prediction: float,
        error: float,
    ):
        arch = self.cognitive_architecture
        if not arch:
            return
        meta = getattr(arch, "meta_learning", None) or getattr(arch, "metalearning", None)
        if meta and hasattr(meta, "register_feedback"):
            try:
                meta.register_feedback(
                    module="experiential",
                    score=float(episode.integration_level),
                    prediction=float(prediction),
                    error=float(error),
                    context=episode.contextual_metadata,
                )
            except Exception:
                pass

    def _integration(self, concepts: List[str], outcomes: Dict[str, Dict[str, float]]) -> float:
        if not concepts or not outcomes:
            return 0.0
        c = min(1.0, len(concepts) / 6.0)
        o = _safe_mean([x["success_rate"] for x in outcomes.values()], 0.5)
        return min(1.0, 0.4 * c + 0.6 * o)

    def _consolidate_episode(self, ep: LearningEpisode):
        mem = getattr(self, "memory", None)
        if not mem:
            return
        try:
            # stocke une trace épisodique simple (compatible avec la persistance maison)
            LTM = getattr(mem, "long_term_memory", {})
            key = f"learn::{ep.id}"
            if isinstance(LTM, dict):
                bucket = "EPISODIC"
                LTM.setdefault(bucket, {})[key] = {
                    "timestamp": ep.timestamp,
                    "concepts": ep.concepts,
                    "valence": ep.emotional_valence,
                    "confidence": ep.confidence_gain,
                    "integration": ep.integration_level,
                }
                if hasattr(mem, "memory_metadata") and isinstance(mem.memory_metadata, dict):
                    mem.memory_metadata["total_memories"] = mem.memory_metadata.get("total_memories", 0) + 1
        except Exception:
            pass

    # ---- API publique utile ----

    def to_state(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "learning_competencies": self.learning_competencies,
            "learning_states": self.learning_states,
            "metrics": self.metrics,
            "episodes": [ep.__dict__ for ep in self.learning_episodes[-200:]],  # limite snapshot
            "contextual_model": {
                "weights": self.contextual_model.weights,
                "bias": self.contextual_model.bias,
            },
        }

    def from_state(self, state: Dict[str, Any]):
        self.learning_rate = state.get("learning_rate", 0.1)
        self.learning_competencies.update(state.get("learning_competencies", {}))
        self.learning_states.update(state.get("learning_states", {}))
        self.metrics.update(state.get("metrics", {}))
        self.learning_episodes = []
        for d in state.get("episodes", []):
            try:
                self.learning_episodes.append(LearningEpisode(**d))
            except Exception:
                continue
        model_state = state.get("contextual_model", {})
        if model_state:
            self.contextual_model._ensure_dim(len(model_state.get("weights", [])))
            self.contextual_model.weights[: len(model_state.get("weights", []))] = list(model_state.get("weights", []))
            self.contextual_model.bias = float(model_state.get("bias", 0.0))

    def summarize(self) -> str:
        return f"{self.metrics.get('episodes_completed',0)} épisodes, {self.metrics.get('concepts_formed',0)} concepts."

    def self_assess_concept(self, concept: str) -> Dict[str, Any]:
        """
        Auto-évaluation heuristique de la compréhension d'un concept via la mémoire.
        Retourne: { 'confidence': 0..1, 'coverage': {definition, examples, counterexample}, 'evidence': [...] }
        Critères: 1 définition, >=2 exemples, >=1 contre-exemple.
        """
        concept = (concept or '').strip()
        coverage = {'definition': 0.0, 'examples': 0.0, 'counterexample': 0.0}
        evidence: List[str] = []
        try:
            mem = getattr(self.cognitive_architecture, 'memory', None)
            if not mem or not hasattr(mem, 'get_recent_memories'):
                return {'confidence': 0.0, 'coverage': coverage, 'evidence': evidence}
            recent = mem.get_recent_memories(150)
            defin_re = ('définition', 'def:', 'definition')
            ex_re = ('ex:', 'exemple', 'exemple ', 'exemples', 'ex ')
            contra_re = ('contre-ex', 'contreex', 'contre exemple', 'contre-exemple', 'counterexample')
            def_hits = ex_hits = contra_hits = 0
            for item in recent:
                text = str(item.get('content') or item.get('text') or '').lower()
                meta = item.get('metadata') or {}
                if concept.lower() in text:
                    snippet = text[:180]
                    if any(k in text for k in defin_re) or meta.get('definition'):
                        def_hits += 1; evidence.append('def:' + snippet)
                    if any(k in text for k in ex_re) or meta.get('example') or meta.get('examples'):
                        ex_hits += 1; evidence.append('ex:' + snippet)
                    if any(k in text for k in contra_re) or meta.get('counterexample'):
                        contra_hits += 1; evidence.append('cx:' + snippet)
            coverage['definition'] = 1.0 if def_hits >= 1 else 0.0
            coverage['examples'] = min(1.0, ex_hits / 2.0)
            coverage['counterexample'] = 1.0 if contra_hits >= 1 else 0.0
            confidence = 0.5*coverage['definition'] + 0.3*coverage['examples'] + 0.2*coverage['counterexample']
            if (def_hits + ex_hits + contra_hits) >= 4:
                confidence = min(1.0, confidence + 0.05)
            result = {'confidence': float(confidence), 'coverage': coverage, 'evidence': evidence[:10]}
            try:
                arch = getattr(self, 'cognitive_architecture', None)
                calibration = getattr(arch, 'calibration', None) if arch else None
                if calibration:
                    event_id = calibration.log_prediction(
                        domain='concepts',
                        p=float(confidence),
                        meta={
                            'concept': concept,
                            'coverage': dict(coverage),
                            'evidence_count': len(evidence),
                        }
                    )
                    success = bool(
                        coverage.get('definition', 0.0) >= 1.0
                        and coverage.get('examples', 0.0) >= 0.5
                        and coverage.get('counterexample', 0.0) >= 1.0
                        and confidence >= 0.6
                    )
                    calibration.log_outcome(event_id, success=success)
                    mem = getattr(arch, 'memory', None)
                    if mem and hasattr(mem, 'add_memory'):
                        mem.add_memory(
                            kind='calibration_observation',
                            content='concept_self_assess',
                            metadata={
                                'event_id': event_id,
                                'domain': 'concepts',
                                'concept': concept,
                                'p': float(confidence),
                                'auto_success': bool(success),
                            }
                        )
            except Exception:
                pass
            return result
        except Exception:
            return {'confidence': 0.0, 'coverage': coverage, 'evidence': evidence}


# ============================================================
# 🧠 1 bis) Boucle d'auto-évolution
# ============================================================

    def on_auto_intention_promoted(
        self,
        event: Mapping[str, Any],
        evaluation: Optional[Mapping[str, Any]] = None,
        self_assessment: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(event, Mapping):
            return
        action_type = str(event.get("action_type") or "").strip()
        if not action_type:
            return
        significance = float((evaluation or {}).get("significance", 0.5) or 0.5)
        self.learning_competencies[action_type] = max(
            0.3, min(1.0, 0.4 + 0.5 * significance)
        )
        self.learning_states[action_type] = max(0.2, min(1.0, 0.45 + 0.4 * significance))
        self.metrics["skills_compiled"] += 1
        self.auto_curriculum[action_type] = {
            "description": event.get("description"),
            "signals": list(event.get("signals", [])),
            "requirements": list(event.get("requirements", [])),
            "score": (evaluation or {}).get("score"),
        }
        if self_assessment and isinstance(self_assessment, Mapping):
            self.auto_curriculum[action_type]["checkpoints"] = list(
                self_assessment.get("checkpoints", [])
            )
        if self.reasoning and hasattr(self.reasoning, "reasoning_history"):
            trajectory = self.reasoning.reasoning_history.setdefault("learning_trajectory", [])
            trajectory.append(
                {
                    "ts": _now(),
                    "source": "auto_evolution",
                    "action_type": action_type,
                    "integration_target": self.auto_curriculum[action_type].get("checkpoints"),
                }
            )


# ============================================================
# 🧠 2) MÉTA-APPRENTISSAGE
# ============================================================

class MetaLearning:
    """
    Ajuste dynamiquement les hyper-paramètres d'apprentissage des autres composantes
    selon la performance agrégée (succès récents, confiance/integration).
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.performance: List[float] = []  # scores 0..1
        self.adjustment_rate: float = 0.05
        self.last_adjust_ts: float = 0.0
        self.performance_window: int = 200
        self.module_feedback: Dict[str, Dict[str, Any]] = {}
        self.curriculum_history: List[Dict[str, Any]] = []

    def register_performance(self, score: float):
        self.performance.append(max(0.0, min(1.0, float(score))))
        if len(self.performance) > self.performance_window:
            self.performance.pop(0)
        self._maybe_adjust()

    def register_feedback(
        self,
        module: str,
        score: float,
        prediction: float,
        error: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.module_feedback[module] = {
            "score": max(0.0, min(1.0, float(score))),
            "prediction": max(0.0, min(1.0, float(prediction))),
            "error": float(error),
            "context": dict(context or {}),
            "timestamp": _now(),
        }
        self.register_performance(score)

    def _maybe_adjust(self):
        now = _now()
        if now - self.last_adjust_ts < 5.0:
            return
        avg = _safe_mean(self.performance, 0.5)
        ca = self.cognitive_architecture
        if not ca:
            return
        # Ajuste le taux d'apprentissage expérientiel
        xl = getattr(ca, "learning", None)
        if xl and hasattr(xl, "learning_rate"):
            base = float(getattr(xl, "learning_rate", 0.1))
            delta = self.adjustment_rate * (0.5 - avg)
            setattr(xl, "learning_rate", max(0.01, min(1.0, base + delta)))
            # Ajuste aussi l'état momentum via feedback récent
            feedback = self.module_feedback.get("experiential")
            if feedback and hasattr(xl, "learning_states"):
                momentum_target = feedback["score"]
                xl.learning_states["momentum"] = max(
                    0.0,
                    min(
                        1.0,
                        0.7 * xl.learning_states.get("momentum", 0.5)
                        + 0.3 * momentum_target,
                    ),
                )
        # Ajuste aussi la curiosité si présente
        cur = getattr(ca, "curiosity", None) or getattr(ca, "learning", None)
        if cur and hasattr(cur, "curiosity_level"):
            cur.curiosity_level = max(0.1, min(1.0, cur.curiosity_level + (0.5 - avg) * 0.05))
        # Coordonne le transfert de connaissances en fonction des erreurs
        transfer = getattr(ca, "transfer", None) or getattr(ca, "transfer_learning", None)
        feedback = self.module_feedback.get("transfer")
        if transfer and feedback and hasattr(transfer, "success_rate"):
            if feedback["error"] > 0:
                transfer.success_rate = max(
                    0.0,
                    min(1.0, transfer.success_rate * 0.95),
                )
            else:
                transfer.success_rate = max(
                    transfer.success_rate,
                    feedback["score"],
                )
        self.last_adjust_ts = now

    # aide à la persistance
    def to_state(self) -> Dict[str, Any]:
        return {
            "performance": self.performance,
            "last_adjust_ts": self.last_adjust_ts,
            "module_feedback": self.module_feedback,
            "curriculum_history": self.curriculum_history[-50:],
        }

    def from_state(self, state: Dict[str, Any]):
        self.performance = state.get("performance", [])
        self.last_adjust_ts = state.get("last_adjust_ts", 0.0)
        self.module_feedback = state.get("module_feedback", {})
        self.curriculum_history = state.get("curriculum_history", [])

    def propose_curriculum(self) -> Dict[str, Any]:
        """Génère un plan simple basé sur les erreurs récentes des modules."""
        modules_sorted = sorted(
            self.module_feedback.items(),
            key=lambda kv: abs(kv[1].get("error", 0.0)),
            reverse=True,
        )
        focus_modules = [m for m, _ in modules_sorted[:3]]
        curriculum = {
            "focus_modules": focus_modules,
            "timestamp": _now(),
            "recommendations": [],
        }
        for module_name in focus_modules:
            fb = self.module_feedback.get(module_name, {})
            if not fb:
                continue
            curriculum["recommendations"].append(
                {
                    "module": module_name,
                    "target_score": fb.get("score", 0.5) + 0.1,
                    "context": fb.get("context", {}),
                }
            )
        self.curriculum_history.append(curriculum)
        if len(self.curriculum_history) > 50:
            self.curriculum_history.pop(0)
        return curriculum


# ============================================================
# 🔄 3) TRANSFERT DE CONNAISSANCES
# ============================================================

@dataclass
class KnowledgeDomain:
    name: str
    concepts: Dict[str, Any]
    procedures: List[str] = field(default_factory=list)
    principles: List[str] = field(default_factory=list)

class TransferLearning:
    """
    Cherche des analogies et mappings structurels entre domaines,
    pour réutiliser des concepts/procédures/principes.
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.domains: Dict[str, KnowledgeDomain] = {}
        self.transfer_log: List[Dict[str, Any]] = []
        self.success_rate: float = 0.0
        self.mapping_model = OnlineLinearModel(feature_dim=6, learning_rate=0.06)

    def register_domain(self, domain_name: str, concepts: Dict[str, Any],
                        procedures: Optional[List[str]] = None,
                        principles: Optional[List[str]] = None) -> KnowledgeDomain:
        d = KnowledgeDomain(domain_name, concepts, procedures or [], principles or [])
        self.domains[domain_name] = d
        return d

    def _similarity(self, a: KnowledgeDomain, b: KnowledgeDomain) -> float:
        a_c, b_c = set(a.concepts.keys()), set(b.concepts.keys())
        if not a_c or not b_c:
            return 0.0
        inter = len(a_c & b_c)
        union = len(a_c | b_c)
        return inter / union

    def _domain_features(
        self, source: KnowledgeDomain, target: KnowledgeDomain, similarity: float
    ) -> List[float]:
        shared_proc = len(set(source.procedures) & set(target.procedures))
        shared_principles = len(set(source.principles) & set(target.principles))
        features = [
            1.0,
            similarity,
            float(len(source.concepts)),
            float(len(target.concepts)),
            float(shared_proc),
            float(shared_principles),
        ]
        return features

    def learn_mapping(
        self,
        source: str,
        target: str,
        success: float,
    ):
        if source not in self.domains or target not in self.domains:
            return
        src, tgt = self.domains[source], self.domains[target]
        sim = self._similarity(src, tgt)
        feats = self._domain_features(src, tgt, sim)
        prediction, error = self.mapping_model.update(feats, success)
        arch = self.cognitive_architecture
        if arch:
            meta = getattr(arch, "meta_learning", None) or getattr(arch, "metalearning", None)
            if meta and hasattr(meta, "register_feedback"):
                try:
                    meta.register_feedback(
                        module="transfer",
                        score=success,
                        prediction=prediction,
                        error=error,
                        context={"source": source, "target": target},
                    )
                except Exception:
                    pass

    def attempt_transfer(self, source: str, target: str, kinds: Optional[List[str]] = None) -> Dict[str, Any]:
        if source not in self.domains or target not in self.domains:
            raise ValueError("Domaines inconnus")
        src, tgt = self.domains[source], self.domains[target]
        sim = self._similarity(src, tgt)
        features = self._domain_features(src, tgt, sim)
        predicted = self.mapping_model.predict(features)
        kinds = kinds or ["concepts", "procedures", "principles"]
        transferred: List[str] = []
        difficulty: List[str] = []

        if "concepts" in kinds:
            for k, v in src.concepts.items():
                threshold = 0.25 + 0.35 * predicted
                if k not in tgt.concepts and sim > threshold:
                    tgt.concepts[k] = {"adapted_from": source, "desc": str(v)[:200]}
                    transferred.append(f"concept::{k}")
                else:
                    difficulty.append(f"concept::{k}")
        if "procedures" in kinds:
            for p in src.procedures:
                threshold = 0.2 + 0.3 * predicted
                if p not in tgt.procedures and sim > threshold:
                    tgt.procedures.append(p + " (adapté)")
                    transferred.append(f"procedure::{p}")
                else:
                    difficulty.append(f"procedure::{p}")
        if "principles" in kinds:
            for pr in src.principles:
                threshold = 0.15 + 0.25 * predicted
                if pr not in tgt.principles and sim > threshold:
                    tgt.principles.append(pr + " (généralisé)")
                    transferred.append(f"principle::{pr}")
                else:
                    difficulty.append(f"principle::{pr}")

        overall_success = max(0.0, min(1.0, 0.5 * sim + 0.5 * (len(transferred) / max(1, len(transferred) + len(difficulty)))))
        self.learn_mapping(source, target, overall_success)
        self.transfer_log.append({
            "when": _now(),
            "source": source, "target": target,
            "similarity": sim, "success": overall_success,
            "transferred": transferred, "difficulty": difficulty,
        })
        # MAJ taux global
        self.success_rate = _safe_mean([t["success"] for t in self.transfer_log], 0.0)
        return self.transfer_log[-1]

    def to_state(self) -> Dict[str, Any]:
        return {
            "domains": {k: {"concepts": v.concepts, "procedures": v.procedures, "principles": v.principles} for k, v in self.domains.items()},
            "transfer_log": self.transfer_log,
            "success_rate": self.success_rate,
            "mapping_model": {
                "weights": self.mapping_model.weights,
                "bias": self.mapping_model.bias,
            },
        }

    def from_state(self, state: Dict[str, Any]):
        self.domains = {}
        for name, d in state.get("domains", {}).items():
            self.domains[name] = KnowledgeDomain(name, d.get("concepts", {}), d.get("procedures", []), d.get("principles", []))
        self.transfer_log = state.get("transfer_log", [])
        self.success_rate = state.get("success_rate", 0.0)
        mapping_state = state.get("mapping_model", {})
        if mapping_state:
            self.mapping_model._ensure_dim(len(mapping_state.get("weights", [])))
            self.mapping_model.weights[: len(mapping_state.get("weights", []))] = list(mapping_state.get("weights", []))
            self.mapping_model.bias = float(mapping_state.get("bias", 0.0))


# ============================================================
# 🏆 4) APPRENTISSAGE PAR RENFORCEMENT (tabulaire)
# ============================================================

class ReinforcementLearning:
    """
    Table de valeurs simple (state/action) avec mise à jour TD(0).
    Suffit pour moduler des choix locaux dans l'AGI.
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.value_table: Dict[str, float] = {}
        self.alpha: float = 0.1
        self.gamma: float = 0.9
        self.last_state: Optional[str] = None
        self.last_action: Optional[str] = None
        self.last_option: Optional[str] = None
        self.last_context: Optional[List[float]] = None
        self.bandit = ThompsonBandit()
        self.option_bandit = ThompsonBandit()
        self.options: Dict[str, Dict[str, Any]] = {}
        self.context_model = OnlineLinearModel(feature_dim=12, learning_rate=0.07)

    def register_option(self, name: str, actions: List[str]):
        if not actions:
            return
        self.options[name] = {
            "actions": list(dict.fromkeys(actions)),
            "model": OnlineLinearModel(feature_dim=len(self.context_model.weights), learning_rate=0.05),
        }

    def _action_features(self, action: str, context_features: Optional[List[float]]) -> List[float]:
        features = [1.0]
        if context_features:
            features.extend(list(context_features))
        h = int(hashlib.sha1(action.encode("utf-8", errors="ignore")).hexdigest(), 16)
        features.append((h % 997) / 997.0)
        features.append(((h // 997) % 997) / 997.0)
        return features

    def update_value(
        self,
        state: str,
        reward: float,
        next_state: Optional[str] = None,
        context_features: Optional[List[float]] = None,
    ) -> float:
        state_key = f"{state}|{self.last_action}" if self.last_action else state
        old = self.value_table.get(state_key, 0.0)
        next_key = f"{next_state}|{self.last_action}" if next_state and self.last_action else next_state
        next_val = self.value_table.get(next_key, 0.0) if next_state else 0.0
        new = old + self.alpha * (reward + self.gamma * next_val - old)
        self.value_table[state_key] = new
        self.last_state = state
        norm_reward = max(0.0, min(1.0, (reward + 1.0) / 2.0))
        self.bandit.update(self.last_action, norm_reward)
        if context_features is None:
            context_features = self.last_context
        if self.last_action and context_features is not None:
            self.context_model.update(self._action_features(self.last_action, context_features), norm_reward)
        if self.last_option and self.last_option in self.options:
            option = self.options[self.last_option]
            option_model = option.get("model")
            if option_model and context_features is not None:
                option_model.update(context_features, norm_reward)
            self.option_bandit.update(self.last_option, norm_reward)
        return new

    def choose_action(
        self,
        state: str,
        actions: List[str],
        eps: float = 0.2,
        context_features: Optional[List[float]] = None,
    ) -> str:
        if not actions:
            return ""
        if context_features is not None:
            self.last_context = list(context_features)
        context_features = context_features if context_features is not None else self.last_context
        option_name = None
        if self.options:
            option_names = list(self.options.keys())
            priors: Dict[str, float] = {}
            if context_features is not None:
                for name, payload in self.options.items():
                    option_model = payload.get("model")
                    if option_model:
                        priors[name] = option_model.predict(context_features)
            option_name = self.option_bandit.select(
                option_names,
                priors=priors if priors else None,
                fallback=lambda: option_names[0],
            )
            candidate_actions = self.options.get(option_name, {}).get("actions", [])
            if candidate_actions:
                actions = list(candidate_actions)
        if random.random() < eps:
            act = random.choice(actions)
        else:
            priors: Dict[str, float] = {}
            if context_features is not None:
                for a in actions:
                    priors[a] = self.context_model.predict(self._action_features(a, context_features))
            act = self.bandit.select(
                actions,
                priors=priors if priors else None,
                fallback=lambda: max(actions, key=lambda a: self.value_table.get(f"{state}|{a}", 0.0)),
            )
        self.last_action = act
        self.last_option = option_name
        return act

    def to_state(self) -> Dict[str, Any]:
        return {"value_table": self.value_table, "alpha": self.alpha, "gamma": self.gamma}

    def from_state(self, state: Dict[str, Any]):
        self.value_table = state.get("value_table", {})
        self.alpha = float(state.get("alpha", 0.1))
        self.gamma = float(state.get("gamma", 0.9))


# ============================================================
# 🔭 5) MOTEUR DE CURIOSITÉ (récompense intrinsèque)
# ============================================================

class CuriosityEngine:
    """
    Génère des récompenses intrinsèques basées sur la nouveauté et l'imprévu.

    Ce moteur appartient au package ``learning`` : il renvoie un score
    scalaire que les autres modules peuvent utiliser pour ajuster la
    motivation ou la vitesse d'apprentissage.  Il ne crée pas de nouveaux
    objectifs ; cette responsabilité est assurée par
    :class:`AGI_Evolutive.goals.curiosity.CuriosityEngine`.
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.curiosity_level: float = 0.5
        self.seen_hashes: Dict[str, int] = {}
        self.history: List[Tuple[float, str, float]] = []  # (time, stimulus, reward)
        self.prediction_model = OnlineLinearModel(feature_dim=5, learning_rate=0.05)

    def _novelty(self, stimulus: str) -> float:
        h = _hash_str(stimulus, 8)
        c = self.seen_hashes.get(h, 0)
        self.seen_hashes[h] = c + 1
        # nouveauté: décroît avec répétition
        return 1.0 / (1.0 + c)

    def _stimulus_features(self, stimulus: str) -> List[float]:
        text = stimulus or ""
        length = min(len(text) / 80.0, 5.0)
        digits = sum(ch.isdigit() for ch in text) / max(1, len(text))
        uppercase = sum(ch.isupper() for ch in text) / max(1, len(text))
        entropy = 0.0
        if text:
            seen: Dict[str, int] = {}
            for ch in text[:200]:
                seen[ch] = seen.get(ch, 0) + 1
            total = sum(seen.values())
            for count in seen.values():
                p = count / total
                entropy -= p * math.log(p + 1e-9)
        return [1.0, length, digits, uppercase, entropy]

    def stimulate(self, stimulus: str) -> float:
        n = self._novelty(stimulus)
        features = self._stimulus_features(stimulus)
        prediction = self.prediction_model.predict(features)
        surprise = abs(n - prediction)
        reward = self.curiosity_level * (0.6 * n + 0.4 * surprise)
        self.history.append((_now(), stimulus[:120], reward))
        self.prediction_model.update(features, n)
        # hook facultatif : booster la motivation des goals si présent
        ca = self.cognitive_architecture
        if ca and getattr(ca, "goals", None) and hasattr(ca.goals, "motivation_system"):
            try:
                ca.goals.motivation_system.boost_motivation(reward)
            except Exception:
                pass
        if ca:
            meta = getattr(ca, "meta_learning", None) or getattr(ca, "metalearning", None)
            if meta and hasattr(meta, "register_feedback"):
                try:
                    meta.register_feedback(
                        module="curiosity",
                        score=max(0.0, min(1.0, reward)),
                        prediction=prediction,
                        error=n - prediction,
                        context={"stimulus_len": len(stimulus or "")},
                    )
                except Exception:
                    pass
        return reward

    def adjust(self, success_rate: float):
        self.curiosity_level = max(0.1, min(1.0, self.curiosity_level + (0.5 - success_rate) * 0.05))

    def to_state(self) -> Dict[str, Any]:
        return {
            "curiosity_level": self.curiosity_level,
            "seen_hashes": self.seen_hashes,
            "history": self.history[-200:],
            "prediction_model": {
                "weights": self.prediction_model.weights,
                "bias": self.prediction_model.bias,
            },
        }

    def from_state(self, state: Dict[str, Any]):
        self.curiosity_level = float(state.get("curiosity_level", 0.5))
        self.seen_hashes = dict(state.get("seen_hashes", {}))
        self.history = list(state.get("history", []))
        model_state = state.get("prediction_model", {})
        if model_state:
            self.prediction_model._ensure_dim(len(model_state.get("weights", [])))
            self.prediction_model.weights[: len(model_state.get("weights", []))] = list(model_state.get("weights", []))
            self.prediction_model.bias = float(model_state.get("bias", 0.0))


# ============================================================
# 🔗 Exports publics pour import direct: from learning import X
# ============================================================

__all__ = [
    "ExperientialLearning",
    "MetaLearning",
    "TransferLearning",
    "ReinforcementLearning",
    "CuriosityEngine",
]
