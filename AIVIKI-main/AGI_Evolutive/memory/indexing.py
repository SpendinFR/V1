import json
import math
import random
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable

from AGI_Evolutive.utils.jsonsafe import json_sanitize

from .encoders import TinyEncoder, cosine
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


@dataclass
class MatchFeatures:
    """Lightweight container used to track the features of a hit."""

    doc_id: int
    query: str
    features: Dict[str, float]
    timestamp: float


class OnlineLinear:
    """Simple online GLM-like regressor with bounded weights.

    The goal is not to be the most sophisticated learner but to offer a
    lightweight mechanism that can continuously adapt how different features
    (cosine similarity, recency, salience, novelty, …) are combined.
    """

    def __init__(
        self,
        feature_names: Iterable[str],
        lr: float = 0.1,
        bounds: Tuple[float, float] = (0.0, 2.0),
        l2: float = 1e-4,
    ):
        self._feature_names = list(feature_names)
        self.lr = lr
        self.bounds = bounds
        self.l2 = l2
        # Initial weights favour cosine similarity so behaviour stays close to
        # the historical implementation until enough feedback is collected.
        self._weights: Dict[str, float] = {
            name: 1.0 if name == "cosine" else 0.1 for name in self._feature_names
        }
        self._bias = 0.0

    def predict(self, features: Dict[str, float]) -> float:
        score = self._bias
        for name in self._feature_names:
            score += self._weights.get(name, 0.0) * features.get(name, 0.0)
        return score

    def update(self, features: Dict[str, float], reward: float):
        # Reward is assumed to be in [0, 1].
        reward = max(0.0, min(1.0, reward))
        pred = self.predict(features)
        error = reward - pred
        for name in self._feature_names:
            value = features.get(name, 0.0)
            if value == 0.0:
                continue
            grad = error * value - self.l2 * self._weights.get(name, 0.0)
            new_weight = self._weights.get(name, 0.0) + self.lr * grad
            self._weights[name] = max(self.bounds[0], min(self.bounds[1], new_weight))
        # Update the bias separately.
        self._bias = max(self.bounds[0], min(self.bounds[1], self._bias + self.lr * error))

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)


class DiscreteThompsonSampler:
    """Discrete Thompson sampling on a small pool of weight presets."""

    def __init__(self, candidates: List[Dict[str, float]]):
        self._candidates = list(candidates)
        self._stats: List[Tuple[float, float]] = [
            (1.0, 1.0) for _ in self._candidates
        ]  # (successes, failures)

    def sample(self) -> Tuple[int, Dict[str, float]]:
        draws = []
        for idx, (s, f) in enumerate(self._stats):
            draws.append((random.betavariate(s, f), idx))
        _, best_idx = max(draws)
        return best_idx, self._candidates[best_idx]

    def update(self, candidate_idx: int, reward: float):
        reward = max(0.0, min(1.0, reward))
        successes, failures = self._stats[candidate_idx]
        if reward >= 0.5:
            successes += 1.0
        else:
            failures += 1.0
        self._stats[candidate_idx] = (successes, failures)


class InMemoryIndex:
    """
    Index vectoriel simple (RAM) + métadonnées.
    - add_document(text, meta)
    - search_text(query, top_k)
    Persistance JSON optionnelle (sans dépendances).
    """

    def __init__(self, encoder: Optional[TinyEncoder] = None):
        self.encoder = encoder or TinyEncoder()
        self._docs: List[Dict[str, Any]] = []  # {"id": int, "text": str, "meta": {"source": str}, "vec": List[float]}
        self._next_id: int = 1
        self._by_id: Dict[int, Dict[str, Any]] = {}
        self._usage: Dict[int, int] = {}
        self._topics: Dict[str, Dict[str, Any]] = {}

        feature_names = ["cosine", "recency", "salience", "novelty", "bridge_boost"]
        self._scorer = OnlineLinear(feature_names)
        self._sampler = DiscreteThompsonSampler(
            [
                {"cosine": 1.0, "recency": 0.1, "salience": 0.1, "novelty": 0.2, "bridge_boost": 0.05},
                {"cosine": 0.8, "recency": 0.4, "salience": 0.4, "novelty": 0.4, "bridge_boost": 0.2},
                {"cosine": 1.2, "recency": 0.2, "salience": 0.6, "novelty": 0.3, "bridge_boost": 0.3},
            ]
        )
        self._last_matches: Dict[int, MatchFeatures] = {}
        self._last_candidate: Optional[int] = None

    def add_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
        vec = self.encoder.encode(text or "")
        doc_id = self._next_id
        self._next_id += 1
        record = {"id": doc_id, "text": text or "", "meta": meta or {}, "vec": vec}
        self._docs.append(record)
        self._by_id[doc_id] = record
        self._usage.setdefault(doc_id, 0)
        self._update_topics(record)
        return doc_id

    def _search_vec(self, qvec: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        scores: List[Tuple[int, float]] = []
        matches: Dict[int, MatchFeatures] = {}
        candidate_idx, candidate_weights = self._sampler.sample()
        for d in self._docs:
            features = self._compute_features(qvec, d)
            # Base adaptive score (GLM online)
            adaptive_score = self._scorer.predict(features)
            # Exploration branch: weight preset sampled via Thompson sampling
            exploratory_score = 0.0
            for name, value in features.items():
                exploratory_score += candidate_weights.get(name, 0.0) * value
            # We combine both so that the online learner remains the main driver
            # while still exploring alternative weightings.
            score = 0.7 * adaptive_score + 0.3 * exploratory_score
            matches[d["id"]] = MatchFeatures(
                doc_id=d["id"],
                query="",
                features=features,
                timestamp=time.time(),
            )
            scores.append((d["id"], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        # Track for feedback updates
        self._last_matches = matches
        self._last_candidate = candidate_idx
        return scores[: max(1, top_k)]

    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qvec = self.encoder.encode(query or "", train=False)
        hits = self._search_vec(qvec, top_k=top_k)
        # Résolution en docs
        out = []
        llm_candidates: List[Dict[str, Any]] = []
        for did, score in hits:
            d = self._by_id.get(did)
            if not d:
                continue
            self._usage[did] = self._usage.get(did, 0) + 1
            if did in self._last_matches:
                self._last_matches[did].query = query or ""
                features = dict(self._last_matches[did].features)
            else:
                features = {}
            payload = {
                "id": did,
                "score": float(round(score, 4)),
                "text": d["text"],
                "meta": dict(d["meta"]) if d["meta"] else {},
            }
            out.append(payload)
            llm_candidates.append(
                {
                    "id": did,
                    "score": float(round(score, 4)),
                    "features": features,
                    "text": d["text"],
                    "meta": dict(d["meta"]) if d["meta"] else {},
                }
            )
        llm_response = try_call_llm_dict(
            "memory_index_optimizer",
            input_payload={"query": query, "candidates": llm_candidates},
            logger=LOGGER,
        )
        if llm_response:
            adjustments: Dict[int, Dict[str, Any]] = {}
            for item in llm_response.get("reranked", []):
                if not isinstance(item, dict):
                    continue
                cid = item.get("id")
                if isinstance(cid, int):
                    adjustments[cid] = item
            if adjustments:
                for entry in out:
                    adj = adjustments.get(entry["id"])
                    if not adj:
                        continue
                    boost = adj.get("boost")
                    if isinstance(boost, (int, float)):
                        entry["score"] = float(round(entry["score"] + float(boost), 4))
                    justification = adj.get("justification")
                    if justification:
                        entry.setdefault("llm_justifications", []).append(str(justification))
                out.sort(key=lambda item: item["score"], reverse=True)
        return out

    # ---------- Adaptive feedback ----------
    def register_feedback(self, doc_id: int, reward: float):
        """Update the adaptive components given a reward in [0, 1]."""

        match = self._last_matches.get(doc_id)
        if not match:
            return
        self._scorer.update(match.features, reward)
        if self._last_candidate is not None:
            self._sampler.update(self._last_candidate, reward)
        # Maintain bridge boosts by marking links between topics and queries.
        self._record_bridge(match, reward)

    def batch_feedback(self, feedbacks: List[Tuple[int, float]]):
        for doc_id, reward in feedbacks:
            self.register_feedback(doc_id, reward)

    def _record_bridge(self, match: MatchFeatures, reward: float):
        if reward <= 0:
            return
        doc = self._by_id.get(match.doc_id)
        if not doc:
            return
        topic = self._resolve_topic(doc)
        if not topic:
            return
        bridge = self._topics.setdefault(topic, {"centroid": doc["vec"], "bridges": {}})
        bridges = bridge.setdefault("bridges", {})
        key = match.query.strip().lower()
        if not key:
            return
        score, count = bridges.get(key, (0.0, 0))
        bridges[key] = (score + reward, count + 1)

    # ---------- Feature engineering ----------
    def _compute_features(self, qvec: List[float], doc: Dict[str, Any]) -> Dict[str, float]:
        meta = doc.get("meta", {}) or {}
        now = time.time()
        cos = cosine(qvec, doc["vec"])
        timestamp = meta.get("timestamp")
        if isinstance(timestamp, (int, float)) and timestamp > 0:
            age = max(0.0, now - float(timestamp))
            # Recency between 0 and 1 (recent docs closer to 1)
            recency = math.exp(-age / (60.0 * 60.0 * 24.0))
        else:
            recency = 0.0
        salience_value = meta.get("salience", meta.get("priority", 0.0))
        try:
            salience = float(salience_value)
        except (TypeError, ValueError):
            salience = 0.0
        salience = max(0.0, min(1.0, salience))
        novelty = 1.0 / (1.0 + float(self._usage.get(doc["id"], 0)))
        bridge = self._bridge_signal(doc, meta)
        return {
            "cosine": cos,
            "recency": recency,
            "salience": salience,
            "novelty": novelty,
            "bridge_boost": bridge,
        }

    def _bridge_signal(self, doc: Dict[str, Any], meta: Dict[str, Any]) -> float:
        topic = self._resolve_topic(doc, meta)
        if not topic:
            return 0.0
        bridges = self._topics.get(topic, {}).get("bridges", {})
        # A high-level bridge is how well the topic is connected to other
        # queries historically.
        total = 0.0
        for score, count in bridges.values():
            if count > 0:
                total += score / count
        if not bridges:
            return 0.0
        return min(1.0, total / (len(bridges) + 1e-9))

    def _resolve_topic(self, doc: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
        meta = meta if meta is not None else doc.get("meta", {})
        topic = meta.get("topic") or meta.get("tag") or meta.get("source")
        if isinstance(topic, str):
            topic = topic.strip().lower()
        return topic or None

    def _update_topics(self, doc: Dict[str, Any]):
        topic = self._resolve_topic(doc)
        if not topic:
            return
        centroid, count = self._topics.get(topic, {}).get("centroid"), self._topics.get(topic, {}).get("count")
        if centroid is None:
            self._topics[topic] = {
                "centroid": list(doc["vec"]),
                "count": 1,
                "bridges": {},
            }
            return
        # Incremental centroid update
        count = int(count or 0) + 1
        new_centroid = []
        for old, new in zip(centroid, doc["vec"]):
            new_centroid.append(old + (new - old) / float(count))
        self._topics[topic]["centroid"] = new_centroid
        self._topics[topic]["count"] = count

    def get_topic_summary(self, topic: str) -> Optional[Dict[str, Any]]:
        topic = (topic or "").strip().lower()
        if not topic:
            return None
        info = self._topics.get(topic)
        if not info:
            return None
        bridges = {
            query: score / max(1, count)
            for query, (score, count) in info.get("bridges", {}).items()
        }
        return {
            "topic": topic,
            "count": info.get("count", 0),
            "bridges": bridges,
        }

    # ---------- Persistance (optionnelle) ----------
    def save(self, path: str):
        payload = {
            "next_id": self._next_id,
            "docs": [{"id": d["id"], "text": d["text"], "meta": d["meta"], "vec": d["vec"]} for d in self._docs],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_sanitize(payload), f, ensure_ascii=False)

    def load(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._next_id = int(payload.get("next_id", 1))
            self._docs = []
            self._by_id = {}
            for d in payload.get("docs", []):
                # sécurité minimale
                if not isinstance(d, dict):
                    continue
                record = {
                    "id": int(d.get("id", 0)),
                    "text": str(d.get("text", "")),
                    "meta": dict(d.get("meta", {})),
                    "vec": list(d.get("vec", [])),
                }
                self._docs.append(record)
                self._by_id[record["id"]] = record
        except FileNotFoundError:
            # premier run: normal
            pass

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        return self._by_id.get(int(doc_id))
