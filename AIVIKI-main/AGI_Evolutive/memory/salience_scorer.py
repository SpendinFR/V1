"""Scoring de saillance pour les items mémoire."""

from __future__ import annotations

import dataclasses
import logging
import math
import random
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

try:  # Modules optionnels (config) peuvent manquer selon l'environnement
    from config import memory_flags as _mem_flags
except Exception:  # pragma: no cover - robustesse import
    _mem_flags = None  # type: ignore

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


def _norm01(value: float) -> float:
    """Normalise ``value`` dans [0, 1]."""

    if math.isnan(value):  # type: ignore[arg-type]
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclasses.dataclass
class _LearningConfig:
    """Configuration pour l'apprentissage en ligne."""

    enabled: bool = False
    learning_rate: float = 0.05
    regularization: float = 1e-3
    exploration: float = 0.0  # epsilon-greedy
    nonlinear_features: bool = False
    bias: float = 0.0
    initial_weight: float = 0.0


class _OnlineLogit:
    """GLM (logistique) très léger pour mettre à jour les poids."""

    def __init__(self, feature_names: Iterable[str], cfg: _LearningConfig) -> None:
        self.cfg = cfg
        self.bias = cfg.bias
        self.weights: Dict[str, float] = {name: cfg.initial_weight for name in feature_names}

    # ------------------------------------------------------------------
    def _linear_response(self, features: Dict[str, float]) -> float:
        score = self.bias
        for name, value in features.items():
            score += self.weights.get(name, 0.0) * value
        return score

    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def predict(self, features: Dict[str, float]) -> float:
        return _norm01(self._sigmoid(self._linear_response(features)))

    def update(self, features: Dict[str, float], target: float, weight: float = 1.0) -> None:
        """Descente de gradient stochastique avec L2."""

        pred = self.predict(features)
        error = (target - pred) * weight
        lr = self.cfg.learning_rate
        reg = self.cfg.regularization

        self.bias += lr * error
        for name, value in features.items():
            w = self.weights.get(name, 0.0)
            self.weights[name] = w + lr * (error * value - reg * w)


class SalienceScorer:
    """Combine plusieurs signaux (récence, affect, etc.) en score 0..1."""

    def __init__(
        self,
        *,
        now: Optional[Callable[[], float]] = None,
        reward: Optional[Any] = None,
        goals: Optional[Any] = None,
        prefs: Optional[Any] = None,
    ) -> None:
        self.now: Callable[[], float] = now or time.time
        self.reward = reward
        self.goals = goals
        self.prefs = prefs
        self._llm_cache: Dict[str, Dict[str, Any]] = {}

        self._weights = getattr(_mem_flags, "SALIENCE_WEIGHTS", {
            "recency": 0.25,
            "affect": 0.20,
            "reward": 0.15,
            "goal_rel": 0.15,
            "prefs": 0.15,
            "novelty": 0.07,
            "usage": 0.03,
        })
        self._half_lives = getattr(_mem_flags, "HALF_LIVES", {
            "default": 3 * 24 * 3600,
            "interaction": 2 * 24 * 3600,
            "episode": 7 * 24 * 3600,
            "digest.daily": 14 * 24 * 3600,
            "digest.weekly": 30 * 24 * 3600,
            "digest.monthly": 90 * 24 * 3600,
        })

        learning_flags = getattr(_mem_flags, "SALIENCE_LEARNING", None)
        self._learning_cfg = _LearningConfig(**self._filter_learning_kwargs(learning_flags)) if isinstance(learning_flags, dict) else _LearningConfig()
        self._model: Optional[_OnlineLogit] = None
        if self._learning_cfg.enabled:
            feature_names = sorted(self._base_feature_names())
            if self._learning_cfg.nonlinear_features:
                feature_names.extend(self._interaction_names(feature_names))
            self._model = _OnlineLogit(feature_names, self._learning_cfg)

    # ------------------------------------------------------------------
    def score(self, item: Dict[str, Any]) -> float:
        """Retourne la saillance globale d'un item mémoire."""

        if not item:
            return 0.0

        parts = self._feature_parts(item)
        if self._model:
            features = self._feature_vector(parts)
            score = self._model.predict(features)
            if self._learning_cfg.exploration > 0.0 and random.random() < self._learning_cfg.exploration:
                # epsilon-greedy : petite exploration en ajoutant bruit uniform
                score = _norm01(score + random.uniform(-0.1, 0.1))
            return self._maybe_llm_enrich(item, parts, score)

        total_weight = sum(self._weights.values()) or 1.0
        score = 0.0
        for key, weight in self._weights.items():
            score += weight * parts.get(key, 0.0)
        base_score = _norm01(score / total_weight)
        return self._maybe_llm_enrich(item, parts, base_score)

    # ------------------------------------------------------------------
    def mutate_static_weights(self, spread: float = 0.05) -> Dict[str, float]:
        """Crée une mutation légère des poids statiques et les applique.

        Cette méthode soutient un mécanisme d'évolution simple : en injectant
        une perturbation contrôlée, on peut tester des variantes puis
        conserver la configuration qui donne les meilleurs résultats.
        """

        base_total = sum(self._weights.values()) or 1.0
        mutated = {}
        for key, value in self._weights.items():
            delta = random.uniform(-spread, spread)
            mutated[key] = max(0.0, value + delta)
        total = sum(mutated.values()) or 1.0
        self._weights = {k: v * (base_total / total) for k, v in mutated.items()}
        return dict(self._weights)

    # ------------------------------------------------------------------
    def learn(self, item: Dict[str, Any], target: float, weight: float = 1.0) -> None:
        """Met à jour le modèle en ligne à partir d'un feedback.

        ``target`` doit être dans [0, 1] (1 = item très utile, 0 = inutile).
        Cette méthode est optionnelle : si l'apprentissage est désactivé, elle
        n'a aucun effet. Elle garantit la compatibilité avec l'API existante.
        """

        if not self._model or not item:
            return
        parts = self._feature_parts(item)
        features = self._feature_vector(parts)
        self._model.update(features, _norm01(target), weight)

    # ------------------------------------------------------------------
    # LLM integration helpers

    def _memory_identifier(self, item: Mapping[str, Any]) -> Optional[str]:
        for key in ("id", "uuid", "memory_id"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _llm_payload(
        self,
        item: Mapping[str, Any],
        parts: Mapping[str, float],
        heuristic_score: float,
    ) -> Dict[str, Any]:
        text = (item.get("text") or item.get("content") or "").strip()
        snippet = text[:320]
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        safe_meta = {
            key: metadata[key]
            for key in ("kind", "source", "channel", "topic")
            if metadata.get(key) is not None
        }
        return {
            "memory_id": self._memory_identifier(item),
            "kind": item.get("kind"),
            "text": snippet,
            "features": {name: round(float(value), 4) for name, value in parts.items()},
            "heuristic_score": round(float(heuristic_score), 4),
            "metadata": safe_meta,
        }

    def _maybe_llm_enrich(
        self,
        item: Mapping[str, Any],
        parts: Mapping[str, float],
        heuristic_score: float,
    ) -> float:
        identifier = self._memory_identifier(item)
        if not identifier:
            return float(heuristic_score)

        cached = self._llm_cache.get(identifier)
        if cached is None and 0.25 <= heuristic_score <= 0.85:
            response = try_call_llm_dict(
                "salience_scorer",
                input_payload=self._llm_payload(item, parts, heuristic_score),
                logger=LOGGER,
            )
            if isinstance(response, Mapping):
                cached = dict(response)
                cached.setdefault("memory_id", identifier)
                self._llm_cache[identifier] = cached
            else:
                self._llm_cache[identifier] = {}
                cached = self._llm_cache[identifier]

        if cached:
            llm_score = cached.get("salience")
            try:
                llm_value = float(llm_score)
            except (TypeError, ValueError):
                llm_value = heuristic_score
            else:
                llm_value = max(0.0, min(1.0, llm_value))
            blended = 0.6 * heuristic_score + 0.4 * llm_value
            if hasattr(item, "setdefault"):
                try:
                    container = item.setdefault("llm", {})  # type: ignore[assignment]
                    if isinstance(container, dict):
                        container.setdefault("salience", cached)
                except Exception:  # pragma: no cover - defensive
                    pass
            return float(_norm01(blended))

        return float(heuristic_score)

    # ------------------------------------------------------------------
    def _filter_learning_kwargs(self, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed = set(_LearningConfig.__annotations__)
        return {k: v for k, v in (values or {}).items() if k in allowed}

    # ------------------------------------------------------------------
    def _feature_parts(self, item: Dict[str, Any]) -> Dict[str, float]:
        return {
            "recency": self._recency(item),
            "affect": self._affect(item),
            "reward": self._reward(item),
            "goal_rel": self._goal_rel(item),
            "prefs": self._prefs(item),
            "novelty": self._novelty(item),
            "usage": self._usage(item),
        }

    def _base_feature_names(self) -> Tuple[str, ...]:
        return tuple(sorted({
            "recency",
            "affect",
            "reward",
            "goal_rel",
            "prefs",
            "novelty",
            "usage",
        }))

    def _interaction_names(self, base_names: Iterable[str]) -> Tuple[str, ...]:
        names = []
        base = list(base_names)
        for i, left in enumerate(base):
            for right in base[i + 1 :]:
                names.append(f"{left}*{right}")
        return tuple(names)

    def _feature_vector(self, parts: Dict[str, float]) -> Dict[str, float]:
        features = dict(parts)
        if self._learning_cfg.nonlinear_features:
            for combo, value in self._pairwise(parts):
                features[combo] = value
        return features

    def _pairwise(self, parts: Dict[str, float]) -> Iterable[Tuple[str, float]]:
        keys = sorted(parts)
        for i, left in enumerate(keys):
            lval = parts[left]
            for right in keys[i + 1 :]:
                features_name = f"{left}*{right}"
                rval = parts[right]
                yield features_name, lval * rval

    # ------------------------------------------------------------------
    def _recency(self, item: Dict[str, Any]) -> float:
        ts = self._timestamp(item)
        if ts is None:
            return 0.5
        age = max(0.0, self.now() - ts)
        label = str(item.get("kind") or item.get("type") or "default")
        half = float(self._half_lives.get(label, self._half_lives.get("default", 1.0)))
        if half <= 0.0:
            return 1.0
        return _norm01(math.pow(0.5, age / half))

    def _affect(self, item: Dict[str, Any]) -> float:
        affect = item.get("affect") or {}
        if isinstance(affect, dict):
            val = float(affect.get("valence", affect.get("val", 0.0)))
            aro = float(affect.get("arousal", affect.get("aro", 0.0)))
        elif isinstance(affect, (int, float)):
            val = float(affect)
            aro = 0.0
        else:
            val = 0.0
            aro = 0.0
        # map [-1,1]x[0,1] -> [0,1]
        base = (val + 1.0) / 2.0
        return _norm01(0.7 * base + 0.3 * max(0.0, aro))

    def _reward(self, item: Dict[str, Any]) -> float:
        # chercher un champ direct sinon interroger reward_engine
        if isinstance(item.get("reward"), (int, float)):
            r = float(item["reward"])  # attendu -1..+1
            return _norm01((r + 1.0) / 2.0)
        if self.reward and hasattr(self.reward, "recent_for"):
            try:
                r = float(self.reward.recent_for(item))  # duck-typed
                return _norm01((r + 1.0) / 2.0)
            except Exception:
                pass
        return 0.5  # neutre

    def _goal_rel(self, item: Dict[str, Any]) -> float:
        if self.goals and hasattr(self.goals, "relevance"):
            try:
                return _norm01(float(self.goals.relevance(item)))
            except Exception:
                return 0.0
        # heuristique: tags/metadata goal:true
        if item.get("metadata", {}).get("goal_related"):
            return 0.8
        return 0.0

    def _prefs(self, item: Dict[str, Any]) -> float:
        if not self.prefs:
            return 0.0
        try:
            concepts = item.get("concepts", []) or []
            tags = item.get("tags", []) or []
            return _norm01(float(self.prefs.get_affinity(concepts, tags)))
        except Exception:
            return 0.0

    def _novelty(self, item: Dict[str, Any]) -> float:
        # Par défaut: 0.5 (ni nouveau ni redondant). À améliorer (simhash/embeddings) si dispo.
        if isinstance(item.get("novelty"), (int, float)):
            return _norm01(float(item["novelty"]))
        similarity = item.get("similarity") or item.get("embedding_similarity")
        if isinstance(similarity, (int, float)):
            # similarité 1 -> pas nouveau, 0 -> très nouveau
            return _norm01(1.0 - float(similarity))
        novelty_features = item.get("metadata", {}).get("novelty", {})
        if isinstance(novelty_features, dict):
            # exemple: {"nearest_distance": 0.3}
            distance = novelty_features.get("nearest_distance")
            if isinstance(distance, (int, float)):
                return _norm01(float(distance))
        return 0.5

    def _usage(self, item: Dict[str, Any]) -> float:
        acc = float(item.get("access_count", 0))
        # boost saturé à 1.0 vers 10 accès
        usage = max(0.0, min(1.0, acc / 10.0))
        last_access = item.get("last_access_ts")
        if isinstance(last_access, (int, float)):
            age = max(0.0, self.now() - float(last_access))
            half = float(self._half_lives.get("usage", self._half_lives.get("default", 1.0)))
            if half > 0.0:
                decay = math.pow(0.5, age / half)
                usage = 0.5 * usage + 0.5 * decay
        return _norm01(usage)

    # ------------------------------------------------------------------
    def _timestamp(self, item: Dict[str, Any]) -> Optional[float]:
        ts = item.get("ts") or item.get("timestamp") or item.get("created_at")
        if isinstance(ts, (int, float)):
            return float(ts)
        return None
