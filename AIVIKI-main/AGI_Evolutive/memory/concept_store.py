from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import logging
import os
import json
import time
import math
import uuid
import random
from collections import deque

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


class OnlineLinearModel:
    """Simple online linear model with forgetting for adaptive scoring."""

    def __init__(
        self,
        feature_names: List[str],
        lr: float = 0.05,
        l2: float = 1e-4,
        decay: float = 0.995,
        initial_weights: Optional[Dict[str, float]] = None,
        bias: float = 0.0,
    ) -> None:
        self.feature_names = feature_names
        self.lr = lr
        self.l2 = l2
        self.decay = decay
        self.weights: Dict[str, float] = {name: 0.0 for name in feature_names}
        if initial_weights:
            for name, value in initial_weights.items():
                if name in self.weights:
                    self.weights[name] = value
        self.bias = bias

    def predict(self, features: Dict[str, float]) -> float:
        score = self.bias
        for name, value in features.items():
            score += self.weights.get(name, 0.0) * value
        return score

    def update(self, features: Dict[str, float], target: float) -> None:
        prediction = self.predict(features)
        error = prediction - target
        for name in self.feature_names:
            weight = self.weights.get(name, 0.0)
            feature_value = features.get(name, 0.0)
            # Apply decay/drift handling
            weight *= self.decay
            gradient = error * feature_value + self.l2 * weight
            weight -= self.lr * gradient
            self.weights[name] = weight
        self.bias *= self.decay
        self.bias -= self.lr * error

    def state(self) -> Dict[str, float]:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "lr": self.lr,
            "l2": self.l2,
            "decay": self.decay,
            "features": self.feature_names,
        }

    def load_state(self, state: Optional[Dict[str, float]]) -> None:
        if not state:
            return
        self.lr = state.get("lr", self.lr)
        self.l2 = state.get("l2", self.l2)
        self.decay = state.get("decay", self.decay)
        self.bias = state.get("bias", self.bias)
        weights = state.get("weights", {})
        for name in self.feature_names:
            if name in weights:
                self.weights[name] = weights[name]

    @classmethod
    def from_state(cls, state: Dict[str, float]) -> "OnlineLinearModel":
        feature_names = state.get("features", [])
        model = cls(
            feature_names=feature_names,
            lr=state.get("lr", 0.05),
            l2=state.get("l2", 1e-4),
            decay=state.get("decay", 0.995),
            initial_weights=state.get("weights"),
            bias=state.get("bias", 0.0),
        )
        return model


class DiscreteThompsonSampler:
    """Discrete Thompson Sampling helper for adaptive half-life selection."""

    def __init__(
        self,
        candidates: List[float],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        drift: float = 0.995,
    ) -> None:
        if not candidates:
            raise ValueError("At least one candidate value is required")
        self.candidates = candidates
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.drift = drift
        self.success: Dict[float, float] = {candidate: prior_alpha for candidate in candidates}
        self.failure: Dict[float, float] = {candidate: prior_beta for candidate in candidates}

    def sample(self) -> float:
        best_value = self.candidates[0]
        best_score = -1.0
        for candidate in self.candidates:
            alpha = max(self.success.get(candidate, self.prior_alpha), 1e-3)
            beta = max(self.failure.get(candidate, self.prior_beta), 1e-3)
            score = random.betavariate(alpha, beta)
            if score > best_score:
                best_score = score
                best_value = candidate
        return best_value

    def update(self, candidate: float, reward: float) -> None:
        reward = max(0.0, min(1.0, reward))
        if candidate not in self.success:
            # Allow dynamic extension while remaining robust.
            self.success[candidate] = self.prior_alpha
            self.failure[candidate] = self.prior_beta
        self.success[candidate] = self.success[candidate] * self.drift + reward
        self.failure[candidate] = self.failure[candidate] * self.drift + (1.0 - reward)

    def state(self) -> Dict[str, Dict[float, float]]:
        return {
            "candidates": self.candidates,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "drift": self.drift,
            "success": self.success,
            "failure": self.failure,
        }

    def load_state(self, state: Optional[Dict[str, Dict[float, float]]]) -> None:
        if not state:
            return
        candidates = state.get("candidates") or self.candidates
        if candidates:
            self.candidates = candidates
        self.prior_alpha = state.get("prior_alpha", self.prior_alpha)
        self.prior_beta = state.get("prior_beta", self.prior_beta)
        self.drift = state.get("drift", self.drift)
        success = {float(k): v for k, v in state.get("success", {}).items()}
        failure = {float(k): v for k, v in state.get("failure", {}).items()}
        for candidate in self.candidates:
            self.success[candidate] = success.get(float(candidate), success.get(candidate, self.prior_alpha))
            self.failure[candidate] = failure.get(float(candidate), failure.get(candidate, self.prior_beta))

    @classmethod
    def from_state(cls, state: Dict[str, Dict[float, float]]) -> "DiscreteThompsonSampler":
        candidates = state.get("candidates", [])
        sampler = cls(
            candidates=candidates,
            prior_alpha=state.get("prior_alpha", 1.0),
            prior_beta=state.get("prior_beta", 1.0),
            drift=state.get("drift", 0.995),
        )
        success = {float(k): v for k, v in state.get("success", {}).items()}
        failure = {float(k): v for k, v in state.get("failure", {}).items()}
        for candidate in sampler.candidates:
            sampler.success[candidate] = success.get(float(candidate), success.get(candidate, sampler.prior_alpha))
            sampler.failure[candidate] = failure.get(float(candidate), failure.get(candidate, sampler.prior_beta))
        return sampler


@dataclass
class Concept:
    id: str
    label: str
    support: float = 0.0
    recency: float = 1.0
    salience: float = 0.0
    confidence: float = 0.5
    last_seen: float = field(default_factory=time.time)
    examples: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    half_life_choice: float = 48 * 3600


@dataclass
class Relation:
    id: str
    src: str
    dst: str
    rtype: str
    weight: float = 0.0
    confidence: float = 0.5
    updated_at: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)
    half_life_choice: float = 72 * 3600


class ConceptStore:
    """Petite base graphe concepts/relations avec persistance JSON."""

    def __init__(self, path_concepts: str = "data/concepts.json", path_dashboard: str = "data/concepts_dashboard.json"):
        self.path_concepts = path_concepts
        self.path_dashboard = path_dashboard
        os.makedirs(os.path.dirname(self.path_concepts), exist_ok=True)
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[str, Relation] = {}
        self._concept_guidance: Dict[str, Dict[str, Any]] = {}
        self._relation_guidance: Dict[str, Dict[str, Any]] = {}
        # Adaptive scoring and decay configuration
        concept_feature_names = [
            "support",
            "support_sq",
            "salience",
            "salience_sq",
            "recency",
            "confidence",
            "examples",
            "support_salience",
        ]
        self.concept_score_model = OnlineLinearModel(
            feature_names=concept_feature_names,
            lr=0.05,
            l2=1e-4,
            decay=0.998,
            initial_weights={
                "support": 0.6,
                "salience": 0.4,
                "support_salience": 0.1,
                "recency": 0.05,
                "confidence": 0.05,
            },
        )
        self.concept_half_life_sampler = DiscreteThompsonSampler(
            candidates=[12 * 3600, 24 * 3600, 36 * 3600, 48 * 3600, 72 * 3600, 7 * 24 * 3600],
            prior_alpha=5.0,
            prior_beta=2.0,
            drift=0.999,
        )
        self.relation_half_life_sampler = DiscreteThompsonSampler(
            candidates=[24 * 3600, 48 * 3600, 72 * 3600, 96 * 3600, 10 * 24 * 3600],
            prior_alpha=5.0,
            prior_beta=2.0,
            drift=0.999,
        )
        self._load()

    def upsert_concept(
        self,
        label: str,
        support_delta: float,
        salience_delta: float,
        example_mem_id: Optional[str],
        confidence: float = 0.6,
    ) -> Concept:
        cid = self._find_by_label(label) or str(uuid.uuid4())[:8]
        concept = self.concepts.get(cid)
        if not concept:
            concept = Concept(id=cid, label=label, support=0.0, salience=0.0, confidence=confidence)
            self.concepts[cid] = concept
        now = time.time()
        half_life = self.concept_half_life_sampler.sample()
        concept.half_life_choice = half_life
        decay = self._exp_decay(concept.last_seen, now, half_life=half_life)
        concept.support = concept.support * decay + support_delta
        concept.salience = max(0.0, min(1.0, concept.salience * decay + salience_delta))
        concept.recency = 1.0
        concept.last_seen = now
        concept.confidence = max(concept.confidence, confidence)
        if example_mem_id and len(concept.examples) < 10:
            concept.examples.append(example_mem_id)
        features = self._concept_features(concept)
        target = self._concept_target(support_delta, salience_delta)
        self.concept_score_model.update(features, target)
        reward = self._concept_reward(support_delta, salience_delta)
        self.concept_half_life_sampler.update(half_life, reward)
        payload = {
            "concept": {
                "id": concept.id,
                "label": concept.label,
                "support": concept.support,
                "salience": concept.salience,
                "confidence": concept.confidence,
                "examples": list(concept.examples),
            },
            "delta": {
                "support": support_delta,
                "salience": salience_delta,
                "confidence": confidence,
            },
        }
        guidance = try_call_llm_dict(
            "memory_concept_curation",
            input_payload=payload,
            logger=LOGGER,
        )
        if guidance:
            for item in guidance.get("concepts", []):
                if not isinstance(item, Mapping):
                    continue
                identifier = item.get("id")
                if identifier in {concept.id, concept.label}:
                    self._concept_guidance[concept.id] = dict(item)
                    break
            for rel in guidance.get("relations", []):
                if isinstance(rel, Mapping) and rel.get("id"):
                    self._relation_guidance[str(rel["id"])] = dict(rel)
        return concept

    def upsert_relation(
        self,
        src_cid: str,
        dst_cid: str,
        rtype: str,
        weight_delta: float,
        mem_id: Optional[str],
        confidence: float = 0.6,
    ) -> Relation:
        rid = f"{src_cid}::{rtype}::{dst_cid}"
        relation = self.relations.get(rid)
        if not relation:
            relation = Relation(id=rid, src=src_cid, dst=dst_cid, rtype=rtype, weight=0.0, confidence=confidence)
            self.relations[rid] = relation
        now = time.time()
        half_life = self.relation_half_life_sampler.sample()
        relation.half_life_choice = half_life
        decay = self._exp_decay(relation.updated_at, now, half_life=half_life)
        relation.weight = relation.weight * decay + weight_delta
        relation.confidence = max(relation.confidence, confidence)
        relation.updated_at = now
        if mem_id and len(relation.evidence) < 20:
            relation.evidence.append(mem_id)
        reward = self._relation_reward(weight_delta)
        self.relation_half_life_sampler.update(half_life, reward)
        payload = {
            "relation": {
                "id": relation.id,
                "src": relation.src,
                "dst": relation.dst,
                "rtype": relation.rtype,
                "weight": relation.weight,
                "confidence": relation.confidence,
            },
            "delta": {
                "weight": weight_delta,
                "confidence": confidence,
            },
        }
        guidance = try_call_llm_dict(
            "memory_concept_curation",
            input_payload=payload,
            logger=LOGGER,
        )
        if guidance:
            for rel in guidance.get("relations", []):
                if not isinstance(rel, Mapping):
                    continue
                identifier = str(rel.get("id"))
                if identifier in {relation.id, f"{relation.src}::{relation.rtype}::{relation.dst}"}:
                    self._relation_guidance[relation.id] = dict(rel)
                    break
        return relation

    def get_top_concepts(self, k: int = 20) -> List[Concept]:
        return sorted(
            self.concepts.values(),
            key=lambda c: self.concept_score_model.predict(self._concept_features(c)),
            reverse=True,
        )[:k]

    def neighbors(self, cid: str, rtype: Optional[str] = None, k: int = 10) -> List[Tuple[Relation, Concept]]:
        results: List[Tuple[Relation, Concept]] = []
        for relation in self.relations.values():
            if relation.src != cid:
                continue
            if rtype is not None and relation.rtype != rtype:
                continue
            dst = self.concepts.get(relation.dst)
            if dst:
                results.append((relation, dst))
        results.sort(key=lambda item: item[0].weight, reverse=True)
        return results[:k]

    def walk_associations(
        self,
        start_label: str,
        *,
        max_depth: int = 2,
        relation_types: Optional[Iterable[str]] = None,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        if not start_label:
            return []
        start_id = self._find_by_label(start_label)
        if start_id is None:
            if start_label in self.concepts:
                start_id = start_label
            else:
                return []

        allowed = None
        if relation_types is not None:
            allowed = {str(rel).lower() for rel in relation_types}

        queue = deque([(start_id, 0)])
        visited = {start_id}
        results: List[Dict[str, Any]] = []

        while queue and len(results) < limit:
            cid, depth = queue.popleft()
            if depth >= max_depth:
                continue
            concept = self.concepts.get(cid)
            source_label = concept.label if concept else str(cid)
            for relation in self.relations.values():
                if relation.src != cid:
                    continue
                rtype = relation.rtype or "related_to"
                if allowed and rtype.lower() not in allowed:
                    continue
                dst = self.concepts.get(relation.dst)
                if not dst:
                    continue
                results.append(
                    {
                        "source": source_label,
                        "target": dst.label,
                        "relation": rtype,
                        "weight": relation.weight,
                        "confidence": relation.confidence,
                        "depth": depth + 1,
                    }
                )
                if len(results) >= limit:
                    break
                if relation.dst not in visited and depth + 1 < max_depth:
                    visited.add(relation.dst)
                    queue.append((relation.dst, depth + 1))

        return results

    def concept_guidance(self, concept_id: Optional[str] = None) -> Dict[str, Any]:
        if concept_id is not None:
            return dict(self._concept_guidance.get(concept_id, {}))
        return {cid: dict(payload) for cid, payload in self._concept_guidance.items()}

    def relation_guidance(self, relation_id: Optional[str] = None) -> Dict[str, Any]:
        if relation_id is not None:
            return dict(self._relation_guidance.get(relation_id, {}))
        return {rid: dict(payload) for rid, payload in self._relation_guidance.items()}

    def find_by_label_prefix(self, prefix: str, k: int = 10) -> List[Concept]:
        lower_prefix = prefix.lower()
        matches = [concept for concept in self.concepts.values() if concept.label.lower().startswith(lower_prefix)]
        matches.sort(key=lambda concept: concept.support, reverse=True)
        return matches[:k]

    def register_concept_feedback(self, concept_id: str, reward: float) -> None:
        concept = self.concepts.get(concept_id)
        if not concept:
            return
        bounded_reward = max(0.0, min(1.0, reward))
        self.concept_half_life_sampler.update(concept.half_life_choice, bounded_reward)
        features = self._concept_features(concept)
        self.concept_score_model.update(features, bounded_reward)

    def register_relation_feedback(self, relation_id: str, reward: float) -> None:
        relation = self.relations.get(relation_id)
        if not relation:
            return
        bounded_reward = max(0.0, min(1.0, reward))
        self.relation_half_life_sampler.update(relation.half_life_choice, bounded_reward)

    def _concept_features(self, concept: Concept) -> Dict[str, float]:
        support_norm = math.tanh(concept.support)
        salience = concept.salience
        recency = concept.recency
        confidence = concept.confidence
        examples = min(1.0, len(concept.examples) / 10.0)
        features = {
            "support": support_norm,
            "support_sq": support_norm * support_norm,
            "salience": salience,
            "salience_sq": salience * salience,
            "recency": recency,
            "confidence": confidence,
            "examples": examples,
            "support_salience": support_norm * salience,
        }
        return features

    @staticmethod
    def _concept_target(support_delta: float, salience_delta: float) -> float:
        raw = max(0.0, support_delta) + max(0.0, salience_delta)
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _concept_reward(support_delta: float, salience_delta: float) -> float:
        return max(0.0, min(1.0, support_delta + salience_delta))

    @staticmethod
    def _relation_reward(weight_delta: float) -> float:
        return max(0.0, min(1.0, weight_delta))

    def save(self) -> None:
        data = {
            "concepts": {cid: asdict(concept) for cid, concept in self.concepts.items()},
            "relations": {rid: asdict(relation) for rid, relation in self.relations.items()},
            "saved_at": time.time(),
            "adaptive": {
                "concept_score_model": self.concept_score_model.state(),
                "concept_half_life": self.concept_half_life_sampler.state(),
                "relation_half_life": self.relation_half_life_sampler.state(),
            },
        }
        with open(self.path_concepts, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(data), handle, ensure_ascii=False, indent=2)
        dashboard = {
            "t": time.time(),
            "top_concepts": [asdict(concept) for concept in self.get_top_concepts(25)],
            "counts": {"concepts": len(self.concepts), "relations": len(self.relations)},
        }
        with open(self.path_dashboard, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(dashboard), handle, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self.path_concepts):
            return
        try:
            with open(self.path_concepts, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.concepts = {cid: Concept(**payload) for cid, payload in data.get("concepts", {}).items()}
            self.relations = {rid: Relation(**payload) for rid, payload in data.get("relations", {}).items()}
            adaptive_state = data.get("adaptive", {})
            score_state = adaptive_state.get("concept_score_model")
            if score_state:
                try:
                    if score_state.get("features"):
                        self.concept_score_model = OnlineLinearModel.from_state(score_state)
                    else:
                        self.concept_score_model.load_state(score_state)
                except Exception:
                    pass
            concept_half_life_state = adaptive_state.get("concept_half_life")
            if concept_half_life_state:
                try:
                    self.concept_half_life_sampler.load_state(concept_half_life_state)
                except Exception:
                    pass
            relation_half_life_state = adaptive_state.get("relation_half_life")
            if relation_half_life_state:
                try:
                    self.relation_half_life_sampler.load_state(relation_half_life_state)
                except Exception:
                    pass
        except Exception:
            self.concepts = {}
            self.relations = {}

    def _find_by_label(self, label: str) -> Optional[str]:
        lower_label = label.lower()
        for cid, concept in self.concepts.items():
            if concept.label.lower() == lower_label:
                return cid
        return None

    @staticmethod
    def _exp_decay(last_t: float, now_t: float, half_life: float) -> float:
        dt = max(0.0, now_t - last_t)
        if half_life <= 0:
            return 1.0
        return math.pow(0.5, dt / half_life)
