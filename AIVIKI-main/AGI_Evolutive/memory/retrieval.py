import logging
import math
import re
import time
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from config.memory_flags import ENABLE_RERANKING  # type: ignore
except Exception:
    ENABLE_RERANKING = True


try:  # pragma: no cover - optional integration
    from memory.salience_scorer import SalienceScorer  # type: ignore
    from memory.preferences_adapter import PreferencesAdapter  # type: ignore
except Exception:  # pragma: no cover - graceful degradation
    SalienceScorer = None
    PreferencesAdapter = None


from AGI_Evolutive.utils.llm_service import try_call_llm_dict

from .encoders import TinyEncoder
from .indexing import DiscreteThompsonSampler, InMemoryIndex
from .vector_store import VectorStore


LOGGER = logging.getLogger(__name__)


_EMOJI_RE = re.compile("[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]")
_FRENCH_PATTERN = re.compile(
    r"\best\s*(?:-|\s)?\s*(un|une|le|la|l'|des|de|du|d’|d')\s+[\w\-']{2,}",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "").lower()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


class _BoundedOnlineLinear:
    """Simple GLM-like regressor with bounded weights."""

    def __init__(
        self,
        feature_names: List[str],
        *,
        bounds: Tuple[float, float] = (0.0, 1.0),
        lr: float = 0.08,
        l2: float = 0.001,
        max_grad: float = 0.5,
    ) -> None:
        self._feature_names = list(feature_names)
        self._weights: Dict[str, float] = {
            name: 0.5 if name in ("lexical", "vector") else 0.1
            for name in self._feature_names
        }
        self._bias = 0.0
        self.bounds = (min(bounds), max(bounds))
        self.lr = lr
        self.l2 = l2
        self.max_grad = max_grad

    def predict(self, features: Dict[str, float]) -> float:
        score = self._bias
        for name in self._feature_names:
            score += self._weights.get(name, 0.0) * features.get(name, 0.0)
        return max(self.bounds[0], min(self.bounds[1], score))

    def update(self, features: Dict[str, float], reward: float) -> None:
        reward = max(self.bounds[0], min(self.bounds[1], float(reward)))
        pred = self.predict(features)
        error = reward - pred
        for name in self._feature_names:
            value = features.get(name, 0.0)
            if value == 0.0:
                continue
            grad = error * value - self.l2 * self._weights.get(name, 0.0)
            grad = max(-self.max_grad, min(self.max_grad, grad))
            new_weight = self._weights.get(name, 0.0) + self.lr * grad
            self._weights[name] = max(self.bounds[0], min(self.bounds[1], new_weight))
        bias_grad = max(-self.max_grad, min(self.max_grad, error))
        self._bias = max(self.bounds[0], min(self.bounds[1], self._bias + self.lr * bias_grad))

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)


class _OnlineFallbackClassifier:
    """Lightweight online text classifier used as recall-oriented fallback."""

    def __init__(self) -> None:
        self.lr = 0.1
        self.l2 = 0.0005
        self._weights: Dict[str, float] = defaultdict(float)
        self._bias = 0.0
        self._last_features: Dict[int, Dict[str, float]] = {}

    def _extract_features(self, query: str, text: str) -> Dict[str, float]:
        norm_query = _normalize_text(query)
        norm_text = _normalize_text(text)
        features: Dict[str, float] = {}

        tokens = norm_text.split()
        for token in tokens[:6]:
            features[f"uni:{token}"] = 1.0
        bigrams = [" ".join(tokens[i : i + 2]) for i in range(max(0, len(tokens) - 1))]
        for bigram in bigrams[:5]:
            features[f"bi:{bigram}"] = 1.0

        query_tokens = norm_query.split()
        for token in query_tokens[:4]:
            features[f"q:{token}"] = 1.0

        features["question_mark"] = 1.0 if "?" in query else 0.0
        features["exclamation"] = 1.0 if "!" in text else 0.0
        features["emoji"] = 1.0 if _EMOJI_RE.search(text) else 0.0
        pattern_match = _FRENCH_PATTERN.search(text) or _FRENCH_PATTERN.search(norm_text)
        features["french_pattern"] = 1.0 if pattern_match else 0.0
        features["short_text"] = 1.0 if len(tokens) <= 12 else 0.0
        features["long_text"] = 1.0 if len(tokens) >= 40 else 0.0

        return features

    @staticmethod
    def _sigmoid(raw: float) -> float:
        if raw >= 0.0:
            z = math.exp(-raw)
            return 1.0 / (1.0 + z)
        z = math.exp(raw)
        return z / (1.0 + z)

    def predict(self, doc_id: int, query: str, text: str) -> float:
        features = self._extract_features(query, text)
        raw = self._bias
        for name, value in features.items():
            raw += self._weights.get(name, 0.0) * value
        score = self._sigmoid(raw)
        self._last_features[doc_id] = features
        return score

    def update(self, doc_id: int, reward: float) -> None:
        features = self._last_features.get(doc_id)
        if not features:
            return
        reward = max(0.0, min(1.0, float(reward)))
        raw = self._bias
        for name, value in features.items():
            raw += self._weights.get(name, 0.0) * value
        pred = self._sigmoid(raw)
        error = reward - pred
        for name, value in features.items():
            grad = error * value - self.l2 * self._weights.get(name, 0.0)
            self._weights[name] += self.lr * grad
        self._bias += self.lr * error
        self._last_features.pop(doc_id, None)


class _AdaptiveCandidateReranker:
    """Adaptive reranker mixing online GLM and discrete Thompson sampling."""

    _FEATURE_NAMES = ["lexical", "vector", "salience", "recency", "affinity", "fallback"]

    def __init__(self) -> None:
        self._weight_model = _BoundedOnlineLinear(self._FEATURE_NAMES)
        self._bandit = DiscreteThompsonSampler(
            [
                {"lexical": 0.6, "vector": 0.6, "salience": 0.2, "recency": 0.2, "affinity": 0.1, "fallback": 0.1},
                {"lexical": 0.4, "vector": 0.8, "salience": 0.3, "recency": 0.4, "affinity": 0.2, "fallback": 0.15},
                {"lexical": 0.7, "vector": 0.3, "salience": 0.4, "recency": 0.3, "affinity": 0.25, "fallback": 0.2},
            ]
        )
        self._fallback = _OnlineFallbackClassifier()
        self._feedback_trace: Dict[int, Dict[str, Any]] = {}
        self._last_llm_payload: Optional[Dict[str, Any]] = None
        self._last_llm_response: Optional[Mapping[str, Any]] = None

    def _clip_feature(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _prepare_features(
        self,
        candidate: Dict[str, Any],
        *,
        query: str,
        salience_scorer: Optional["SalienceScorer"],
        preferences: Optional[Any],
    ) -> Tuple[Dict[str, float], float]:
        now = time.time()
        salience = candidate.get("salience")
        if salience is None and salience_scorer is not None:
            try:
                salience = float(salience_scorer.score(candidate))  # type: ignore[arg-type]
            except Exception:
                salience = 0.0
            candidate["salience"] = salience

        affinity = candidate.get("_affinity")
        if affinity is None and preferences is not None:
            try:
                concepts = candidate.get("concepts", [])
                tags = candidate.get("tags", [])
                affinity = float(preferences.get_affinity(concepts, tags))
            except Exception:
                affinity = 0.0
            candidate["_affinity"] = affinity

        recency = candidate.get("_recency")
        if recency is None:
            try:
                ts = float(candidate.get("ts", now))
                recency = max(0.0, min(1.0, 1.0 - ((now - ts) / (30 * 24 * 3600))))
            except Exception:
                recency = 0.0
            candidate["_recency"] = recency

        lexical_score = self._clip_feature(float(candidate.get("lexical", 0.0)))
        vector_score = self._clip_feature(float(candidate.get("vector", 0.0)))
        salience_score = self._clip_feature(float(salience or 0.0))
        recency_score = self._clip_feature(float(recency or 0.0))
        affinity_score = self._clip_feature(float(affinity or 0.0))

        fallback_score = self._fallback.predict(
            int(candidate.get("id", 0)), query, candidate.get("text", "")
        )

        features = {
            "lexical": lexical_score,
            "vector": vector_score,
            "salience": salience_score,
            "recency": recency_score,
            "affinity": affinity_score,
            "fallback": self._clip_feature(fallback_score),
        }
        candidate["_fallback_score"] = float(features["fallback"])
        candidate["_adaptive_features"] = dict(features)
        return features, float(fallback_score)

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        *,
        query: str,
        salience_scorer: Optional["SalienceScorer"],
        preferences: Optional[Any],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates

        bandit_idx, bandit_weights = self._bandit.sample()
        self._feedback_trace.clear()

        for candidate in candidates:
            doc_id = int(candidate.get("id", 0))
            features, _ = self._prepare_features(
                candidate,
                query=query,
                salience_scorer=salience_scorer,
                preferences=preferences,
            )
            base_score = self._weight_model.predict(features)
            exploratory = 0.0
            for name, value in features.items():
                exploratory += bandit_weights.get(name, 0.0) * value
            final_score = 0.8 * base_score + 0.2 * exploratory
            candidate["final"] = float(final_score)
            self._feedback_trace[doc_id] = {
                "features": dict(features),
                "bandit_idx": bandit_idx,
                "vector_kind": candidate.get("meta", {}).get("vector_kind"),
            }

        llm_response = self._call_llm_rerank(query, candidates)
        if llm_response:
            self._apply_llm_adjustments(candidates, llm_response)

        return candidates

    def register_feedback(self, doc_id: int, reward: float) -> None:
        ctx = self._feedback_trace.get(int(doc_id))
        if not ctx:
            self._fallback.update(int(doc_id), reward)
            return
        features = ctx.get("features", {})
        self._weight_model.update(features, float(reward))
        self._fallback.update(int(doc_id), reward)
        bandit_idx = ctx.get("bandit_idx")
        if bandit_idx is not None:
            self._bandit.update(bandit_idx, float(reward))
        self._feedback_trace.pop(int(doc_id), None)

    def vector_key_for(self, doc_id: int) -> Optional[str]:
        ctx = self._feedback_trace.get(int(doc_id))
        if not ctx:
            return None
        kind = ctx.get("vector_kind")
        if not kind:
            return None
        return f"{kind}::{int(doc_id)}"

    # ------------------------------------------------------------------
    def _call_llm_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Mapping[str, Any]]:
        if len(candidates) <= 1:
            return None

        payload_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            try:
                doc_id = int(candidate.get("id", 0))
            except (TypeError, ValueError):
                continue
            text = str(candidate.get("text", ""))
            text = text.strip()
            if len(text) > 480:
                text = text[:480] + "…"
            meta = candidate.get("meta", {})
            if not isinstance(meta, Mapping):
                meta = {}
            allowed_meta = {
                key: meta[key]
                for key in ("type", "title", "source", "tags", "ts", "vector_kind")
                if key in meta
            }
            adaptive_features = candidate.get("_adaptive_features", {})
            if not isinstance(adaptive_features, Mapping):
                adaptive_features = {}
            payload_candidates.append(
                {
                    "id": doc_id,
                    "text_preview": text,
                    "scores": {
                        "heuristic": float(candidate.get("final", 0.0)),
                        "lexical": float(candidate.get("lexical", 0.0)),
                        "vector": float(candidate.get("vector", 0.0)),
                        "salience": float(candidate.get("salience", adaptive_features.get("salience", 0.0))),
                        "recency": float(candidate.get("_recency", adaptive_features.get("recency", 0.0))),
                        "affinity": float(candidate.get("_affinity", adaptive_features.get("affinity", 0.0))),
                        "fallback": float(candidate.get("_fallback_score", adaptive_features.get("fallback", 0.0))),
                    },
                    "metadata": allowed_meta,
                }
            )

        if not payload_candidates:
            return None

        payload = {
            "query": query,
            "candidates": payload_candidates,
        }

        self._last_llm_payload = payload
        response = try_call_llm_dict(
            "memory_retrieval_ranking",
            input_payload=payload,
            logger=LOGGER,
            max_retries=2,
        )
        self._last_llm_response = response
        return response

    def _apply_llm_adjustments(
        self,
        candidates: List[Dict[str, Any]],
        response: Mapping[str, Any],
    ) -> None:
        rankings = response.get("rankings") if isinstance(response, Mapping) else None
        if not isinstance(rankings, Sequence):
            return

        adjustments: Dict[int, Dict[str, Any]] = {}
        for idx, entry in enumerate(rankings, start=1):
            if not isinstance(entry, Mapping):
                continue
            try:
                doc_id = int(entry.get("id"))
            except (TypeError, ValueError):
                continue
            try:
                score = float(entry.get("adjusted_score", entry.get("score", 0.0)))
            except (TypeError, ValueError):
                continue
            score = self._clip_feature(score)
            adjustments[doc_id] = {
                "score": score,
                "rationale": str(entry.get("rationale") or entry.get("reason") or "").strip(),
                "priority": entry.get("priority"),
                "rank": int(entry.get("rank", idx)),
            }

        if not adjustments:
            return

        for candidate in candidates:
            try:
                doc_id = int(candidate.get("id", 0))
            except (TypeError, ValueError):
                continue
            adjustment = adjustments.get(doc_id)
            if not adjustment:
                continue
            heuristic = float(candidate.get("final", 0.0))
            llm_score = adjustment["score"]
            candidate["llm_adjusted_score"] = llm_score
            if adjustment["rationale"]:
                candidate["llm_rationale"] = adjustment["rationale"]
            if adjustment.get("priority") is not None:
                candidate["llm_priority"] = adjustment["priority"]
            candidate["llm_rank"] = adjustment["rank"]
            candidate["final"] = 0.65 * heuristic + 0.35 * llm_score
            ctx = self._feedback_trace.get(doc_id)
            if ctx is not None:
                ctx["llm_adjusted_score"] = llm_score
                ctx["llm_rank"] = adjustment["rank"]


class MemoryRetrieval:
    """
    Façade Retrieval:
    - add_interaction(user, agent, extra)
    - add_document(text, title, source)
    - search_text(query, top_k)
    """

    def __init__(
        self,
        encoder: Optional[TinyEncoder] = None,
        index: Optional[InMemoryIndex] = None,
        vector_store: Optional[VectorStore] = None,
        *,
        salience_scorer: Optional["SalienceScorer"] = None,
        preferences: Optional[Any] = None,
    ):
        self.encoder = encoder or TinyEncoder()
        self.index = index or InMemoryIndex(self.encoder)
        self.vector_store = vector_store
        self.salience_scorer = salience_scorer
        self.preferences = None
        self._reranker: Optional[_AdaptiveCandidateReranker] = None

        if preferences is not None:
            self.preferences = preferences
            if PreferencesAdapter and not hasattr(preferences, "get_affinity"):
                try:
                    self.preferences = PreferencesAdapter(preferences)  # type: ignore[call-arg]
                except Exception:
                    self.preferences = preferences

        if ENABLE_RERANKING:
            self._reranker = _AdaptiveCandidateReranker()

    # -------- Ajout ----------
    def add_interaction(self, user: str, agent: str, extra: Optional[Dict[str, Any]] = None) -> int:
        text = f"[USER] {user}\n[AGENT] {agent}"
        meta = {"type": "interaction", "ts": time.time()}
        if extra:
            meta.update(extra)
        doc_id = self.index.add_document(text, meta=meta)
        if self.vector_store:
            try:
                self.vector_store.upsert(f"interaction::{doc_id}", text)
            except Exception:
                pass
        return doc_id

    def add_document(self, text: str, title: Optional[str] = None, source: Optional[str] = None) -> int:
        meta = {"type": "doc"}
        if title:
            meta["title"] = title
        if source:
            meta["source"] = source
        meta["ts"] = time.time()
        doc_id = self.index.add_document(text, meta=meta)
        if self.vector_store:
            try:
                self.vector_store.upsert(f"doc::{doc_id}", text)
            except Exception:
                pass
        return doc_id

    # -------- Recherche ----------
    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        lexical_hits: List[Dict[str, Any]] = []
        try:
            lexical_hits = self.index.search_text(query, top_k=top_k)
        except Exception:
            lexical_hits = []

        combined: Dict[int, Dict[str, Any]] = {}

        for entry in lexical_hits:
            doc_id = entry.get("id")
            if doc_id is None:
                continue
            doc_key = int(doc_id)
            payload = combined.setdefault(
                doc_key,
                {
                    "id": doc_key,
                    "text": entry.get("text", ""),
                    "meta": dict(entry.get("meta", {})),
                },
            )
            payload["lexical"] = float(entry.get("score", 0.0))
            payload.setdefault("ts", payload.get("meta", {}).get("ts"))

        vector_hits: List[Tuple[str, float]] = []
        if self.vector_store:
            try:
                vector_hits = self.vector_store.search(query or "", k=top_k)
            except Exception:
                vector_hits = []

        for doc_identifier, score in vector_hits:
            kind, doc_record = self._resolve_hit(doc_identifier)
            if not doc_record:
                continue
            doc_key = int(doc_record["id"])
            payload = combined.setdefault(
                doc_key,
                {
                    "id": doc_key,
                    "text": doc_record.get("text", ""),
                    "meta": dict(doc_record.get("meta", {})),
                },
            )
            payload["vector"] = max(float(score), float(payload.get("vector", 0.0)))
            meta = dict(doc_record.get("meta", {}))
            if meta:
                payload.setdefault("meta", {}).update({k: v for k, v in meta.items() if k not in payload["meta"]})
            if kind:
                payload.setdefault("meta", {})["vector_kind"] = kind
            payload.setdefault("ts", payload.get("meta", {}).get("ts"))

        candidates = list(combined.values())

        if ENABLE_RERANKING and self._reranker:
            candidates = self._reranker.rerank(
                candidates,
                query=query,
                salience_scorer=self.salience_scorer,
                preferences=self.preferences,
            )

        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            final_score = candidate.get("final")
            if final_score is None:
                final_score = max(
                    float(candidate.get("lexical", 0.0)),
                    float(candidate.get("vector", 0.0)),
                )
                candidate["final"] = final_score
            candidate["score"] = float(round(candidate.get("final", final_score), 4))
            results.append(candidate)

        results.sort(key=lambda item: item.get("final", item.get("score", 0.0)), reverse=True)

        if not results:
            return lexical_hits

        return results[: max(1, top_k)]

    def register_feedback(self, doc_id: int, reward: float) -> None:
        reward = max(0.0, min(1.0, float(reward)))
        doc_key = int(doc_id)
        vector_identifier: Optional[str] = None
        if self._reranker:
            vector_identifier = self._reranker.vector_key_for(doc_key)
            self._reranker.register_feedback(doc_key, reward)

        try:
            self.index.register_feedback(doc_key, reward)
        except Exception:
            pass

        if self.vector_store:
            keys: List[str] = []
            if vector_identifier:
                keys.append(vector_identifier)
            else:
                keys.extend([f"doc::{doc_key}", f"interaction::{doc_key}"])
            for key in keys:
                try:
                    self.vector_store.register_feedback(key, reward)
                    break
                except Exception:
                    continue

    # -------- Persistance (optionnelle) ----------
    def save(self, path: str):
        self.index.save(path)

    def load(self, path: str):
        self.index.load(path)

    # -------- Helpers ----------
    def _resolve_hit(self, identifier: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not identifier:
            return None, None
        raw = identifier
        kind: Optional[str] = None
        if "::" in raw:
            kind, raw = raw.split("::", 1)
        try:
            doc_id = int(raw)
        except ValueError:
            return kind, None
        record = self.index.get_document(doc_id)
        return kind, record
