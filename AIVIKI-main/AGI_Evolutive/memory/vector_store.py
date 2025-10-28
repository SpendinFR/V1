"""Adaptive semantic vector store.

This module replaces the historical bag-of-words implementation with a
semantic encoder coupled with an online learner that continuously adapts the
weighting of similarity features.  The goal is to preserve the public API of
``VectorStore`` while offering a much richer representation and ranking
strategy.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from AGI_Evolutive.core.config import cfg
from AGI_Evolutive.memory.encoders import TinyEncoder, cosine as cosine_similarity
from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

_DIR = cfg()["VECTOR_DIR"]

LOGGER = logging.getLogger(__name__)


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    low, high = bounds
    return max(low, min(high, value))


@dataclass
class _MatchContext:
    doc_id: str
    features: Dict[str, float]
    timestamp: float


class _BoundedOnlineLinear:
    """Simple GLM-like online regressor with bounded weights."""

    def __init__(
        self,
        feature_names: Iterable[str],
        lr: float = 0.08,
        bounds: Tuple[float, float] = (0.0, 1.5),
        l2: float = 1e-4,
    ) -> None:
        self._feature_names = list(feature_names)
        self.lr = lr
        self.bounds = bounds
        self.l2 = l2
        self._weights: Dict[str, float] = {
            name: 1.0 if name == "cosine" else 0.2 for name in self._feature_names
        }
        self._bias = 0.0

    def predict(self, features: Dict[str, float]) -> float:
        score = self._bias
        for name in self._feature_names:
            score += self._weights.get(name, 0.0) * features.get(name, 0.0)
        return score

    def update(self, features: Dict[str, float], reward: float) -> None:
        reward = _clamp(reward, (0.0, 1.0))
        prediction = self.predict(features)
        error = reward - prediction
        for name in self._feature_names:
            value = features.get(name, 0.0)
            if value == 0.0:
                continue
            gradient = error * value - self.l2 * self._weights.get(name, 0.0)
            new_weight = self._weights.get(name, 0.0) + self.lr * gradient
            self._weights[name] = _clamp(new_weight, self.bounds)
        self._bias = _clamp(self._bias + self.lr * error, self.bounds)

    def snapshot(self) -> Dict[str, float]:
        data: Dict[str, float] = {f"w_{name}": value for name, value in self._weights.items()}
        data["bias"] = self._bias
        return data

    def restore(self, payload: Optional[Dict[str, float]]) -> None:
        if not payload:
            return
        for name in self._feature_names:
            if f"w_{name}" in payload:
                self._weights[name] = _clamp(float(payload[f"w_{name}"]), self.bounds)
        if "bias" in payload:
            self._bias = _clamp(float(payload["bias"]), self.bounds)


class _DiscreteThompsonSampler:
    """Discrete Thompson sampling over a set of candidate weightings."""

    def __init__(self, candidates: Iterable[Dict[str, float]]) -> None:
        self._candidates = [dict(candidate) for candidate in candidates]
        self._stats: List[Tuple[float, float]] = [(1.0, 1.0) for _ in self._candidates]

    def sample(self) -> Tuple[int, Dict[str, float]]:
        draws: List[Tuple[float, int]] = []
        for idx, (successes, failures) in enumerate(self._stats):
            draws.append((random.betavariate(successes, failures), idx))
        _, best_idx = max(draws)
        return best_idx, self._candidates[best_idx]

    def update(self, idx: int, reward: float) -> None:
        reward = _clamp(reward, (0.0, 1.0))
        success, failure = self._stats[idx]
        if reward >= 0.5:
            success += 1.0
        else:
            failure += 1.0
        self._stats[idx] = (success, failure)

    def snapshot(self) -> List[Tuple[float, float]]:
        return [(float(s), float(f)) for s, f in self._stats]

    def restore(self, payload: Optional[List[Tuple[float, float]]]) -> None:
        if not payload:
            return
        if len(payload) != len(self._candidates):
            return
        self._stats = [(max(1e-3, float(s)), max(1e-3, float(f))) for s, f in payload]


class VectorStore:
    """Adaptive semantic vector index with online ranking."""

    def __init__(self) -> None:
        os.makedirs(_DIR, exist_ok=True)
        self.idx_path = os.path.join(_DIR, "index.json")
        self._encoder = TinyEncoder()
        self._recency_half_life = 3600.0 * 12.0  # 12 hours decay by default
        feature_names = ("cosine", "recency", "novelty", "bridge")
        self._weight_model = _BoundedOnlineLinear(feature_names, bounds=(0.0, 1.5))
        self._sampler = _DiscreteThompsonSampler(
            [
                {"cosine": 1.0, "recency": 0.2, "novelty": 0.1, "bridge": 0.1},
                {"cosine": 0.8, "recency": 0.4, "novelty": 0.3, "bridge": 0.2},
                {"cosine": 1.2, "recency": 0.1, "novelty": 0.2, "bridge": 0.4},
            ]
        )
        self._documents: Dict[str, Dict[str, object]] = {}
        self._last_matches: Dict[str, _MatchContext] = {}
        self._last_candidate: Optional[int] = None
        self._load()
        self._last_llm_ranking: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self.idx_path):
            return
        try:
            with open(self.idx_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return
        docs = raw.get("documents", {})
        for doc_id, payload in docs.items():
            vector = [float(x) for x in payload.get("vector", [])]
            timestamp = float(payload.get("timestamp", time.time()))
            tokens = list(payload.get("tokens", []))
            usage = int(payload.get("usage", 0))
            text = payload.get("text", "")
            self._documents[str(doc_id)] = {
                "text": text,
                "vector": vector,
                "timestamp": timestamp,
                "tokens": tokens,
                "usage": usage,
            }
        self._weight_model.restore(raw.get("weight_model"))
        self._sampler.restore(raw.get("sampler"))
        encoder_state = raw.get("encoder")
        if encoder_state:
            self._encoder.restore(encoder_state)

    def _save(self) -> None:
        payload = {
            "documents": {
                doc_id: {
                    "text": data.get("text", ""),
                    "vector": list(data.get("vector", [])),
                    "timestamp": data.get("timestamp", time.time()),
                    "tokens": list(data.get("tokens", [])),
                    "usage": data.get("usage", 0),
                }
                for doc_id, data in self._documents.items()
            },
            "weight_model": self._weight_model.snapshot(),
            "sampler": self._sampler.snapshot(),
            "encoder": self._encoder.snapshot(),
        }
        with open(self.idx_path, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(payload), handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upsert(self, doc_id: str, text: str) -> None:
        vector = self._encoder.encode(text or "")
        tokens, _ = self._encoder.tokenizer.tokenize(text or "", update=False)
        timestamp = time.time()
        self._documents[str(doc_id)] = {
            "text": text or "",
            "vector": vector,
            "timestamp": timestamp,
            "tokens": list(tokens),
            "usage": self._documents.get(str(doc_id), {}).get("usage", 0),
        }
        self._save()

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if not self._documents:
            return []
        q_vector = self._encoder.encode(query or "", train=False)
        q_tokens, _ = self._encoder.tokenizer.tokenize(query or "", update=False)
        now = time.time()

        candidate_idx, candidate_weights = self._sampler.sample()
        scores: List[Tuple[str, float]] = []
        matches: Dict[str, _MatchContext] = {}

        for doc_id, payload in self._documents.items():
            d_vector = payload.get("vector", [])
            if not d_vector:
                continue
            cosine = cosine_similarity(q_vector, d_vector)
            recency = self._compute_recency(now, float(payload.get("timestamp", now)))
            novelty = 1.0 / (1.0 + float(payload.get("usage", 0)))
            bridge = self._bridge_score(q_tokens, list(payload.get("tokens", [])))
            features = {
                "cosine": cosine,
                "recency": recency,
                "novelty": novelty,
                "bridge": bridge,
            }
            adaptive = self._weight_model.predict(features)
            exploratory = sum(candidate_weights.get(name, 0.0) * value for name, value in features.items())
            score = 0.7 * adaptive + 0.3 * exploratory
            scores.append((doc_id, score))
            matches[doc_id] = _MatchContext(doc_id=doc_id, features=features, timestamp=now)

        scores.sort(key=lambda item: item[1], reverse=True)
        self._last_matches = matches
        self._last_candidate = candidate_idx

        top_hits: List[Tuple[str, float]] = []
        hit_details: List[Dict[str, Any]] = []
        for doc_id, score in scores[: max(1, k)]:
            data = self._documents.get(doc_id)
            if not data:
                continue
            data["usage"] = int(data.get("usage", 0)) + 1
            rounded = float(round(score, 6))
            top_hits.append((doc_id, rounded))
            context = self._last_matches.get(doc_id)
            hit_details.append(
                {
                    "id": doc_id,
                    "score": rounded,
                    "text": data.get("text", ""),
                    "features": context.features if context else {},
                }
            )
        llm_response = try_call_llm_dict(
            "memory_vector_guidance",
            input_payload={"query": query, "candidates": hit_details},
            logger=LOGGER,
        )
        if llm_response:
            self._last_llm_ranking = dict(llm_response)
            adjustments: Dict[str, Dict[str, Any]] = {}
            for entry in llm_response.get("reranked", []):
                if isinstance(entry, Mapping) and entry.get("id"):
                    adjustments[str(entry["id"])] = dict(entry)
            if adjustments:
                augmented: List[Tuple[str, float]] = []
                for detail in hit_details:
                    doc_id = detail["id"]
                    boost = 0.0
                    comment = None
                    if doc_id in adjustments:
                        boost = float(adjustments[doc_id].get("boost", 0.0))
                        comment = adjustments[doc_id].get("comment")
                    new_score = float(round(detail["score"] + boost, 6))
                    detail["score"] = new_score
                    if comment:
                        context = self._documents.get(doc_id, {})
                        notes = context.setdefault("llm_comments", [])
                        if isinstance(notes, list):
                            notes.append(comment)
                    augmented.append((doc_id, new_score))
                augmented.sort(key=lambda item: item[1], reverse=True)
                top_hits = augmented
        else:
            self._last_llm_ranking = {}
        self._save()
        return top_hits

    def register_feedback(self, doc_id: str, reward: float) -> None:
        """Optional hook to adapt the learner based on external feedback."""

        context = self._last_matches.get(str(doc_id))
        if context is None:
            return
        reward = _clamp(reward, (0.0, 1.0))
        self._weight_model.update(context.features, reward)
        if self._last_candidate is not None:
            self._sampler.update(self._last_candidate, reward)
        self._save()

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _compute_recency(self, now: float, timestamp: float) -> float:
        age = max(0.0, now - timestamp)
        if age == 0.0:
            return 1.0
        decay = math.exp(-math.log(2.0) * age / max(1.0, self._recency_half_life))
        return float(decay)

    @staticmethod
    def _bridge_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        q_set = set(query_tokens)
        d_set = set(doc_tokens)
        intersection = len(q_set & d_set)
        union = len(q_set | d_set)
        if union == 0:
            return 0.0
        return intersection / union

