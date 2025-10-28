"""Embedding helpers bridging lexical and LLM-backed semantic features."""
from __future__ import annotations

import logging
import math
from collections import Counter, OrderedDict
from threading import Lock
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

from AGI_Evolutive.memory.encoders import tokenize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


def _normalize_term(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    term = str(value).strip().lower()
    return term or None


class AdaptiveSemanticEmbedder:
    """Combine lexical counts with LLM-provided concepts and relations."""

    def __init__(
        self,
        *,
        lexical_weight: float = 0.6,
        keyword_weight: float = 1.4,
        relation_weight: float = 1.2,
        cache_size: int = 256,
    ) -> None:
        self.lexical_weight = max(0.1, float(lexical_weight))
        self.keyword_weight = max(0.1, float(keyword_weight))
        self.relation_weight = max(0.1, float(relation_weight))
        self.cache_size = max(16, int(cache_size))
        self._cache: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
        self._lock = Lock()

    # ------------------------------------------------------------------
    def __call__(self, text: str) -> Dict[str, float]:
        key = (text or "").strip().lower()
        if not key:
            return {}
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                # move to end (LRU)
                self._cache.move_to_end(key)
                return dict(cached)

        features = self._build_features(text)
        with self._lock:
            self._cache[key] = dict(features)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return features

    # ------------------------------------------------------------------
    def _build_features(self, text: str) -> Dict[str, float]:
        tokens = tokenize(text)
        counter = Counter(tokens)
        features: Dict[str, float] = {}
        for token, count in counter.items():
            features[f"lex:{token}"] = self.lexical_weight * float(count)

        llm_payload = self._call_llm(text)
        if llm_payload:
            self._inject_keywords(features, llm_payload.get("keywords"))
            self._inject_keywords(features, llm_payload.get("related_terms"), prefix="rel:")
            self._inject_topics(features, llm_payload.get("topics"))
            self._inject_relations(features, llm_payload.get("relations"))

        norm = math.sqrt(sum(value * value for value in features.values())) or 1.0
        return {name: value / norm for name, value in features.items()}

    # ------------------------------------------------------------------
    def _call_llm(self, text: str) -> Optional[Mapping[str, object]]:
        try:
            return try_call_llm_dict(
                "memory_semantic_embedding",
                input_payload={"text": text},
                logger=LOGGER,
            )
        except Exception:  # pragma: no cover - integration fallback
            LOGGER.debug("LLM semantic embedding unavailable", exc_info=True)
            return None

    # ------------------------------------------------------------------
    def _inject_keywords(
        self,
        features: MutableMapping[str, float],
        payload: Optional[Iterable[Mapping[str, object]]],
        *,
        prefix: str = "kw:",
    ) -> None:
        if not payload:
            return
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            term = _normalize_term(entry.get("term") or entry.get("label") or entry.get("keyword"))
            if not term:
                continue
            raw_weight = entry.get("weight", 0.7)
            try:
                weight = max(0.1, min(1.0, float(raw_weight)))
            except (TypeError, ValueError):
                weight = 0.7
            features[f"{prefix}{term}"] = features.get(f"{prefix}{term}", 0.0) + self.keyword_weight * weight

    def _inject_topics(
        self,
        features: MutableMapping[str, float],
        payload: Optional[Iterable[Mapping[str, object]]],
    ) -> None:
        if not payload:
            return
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            term = _normalize_term(entry.get("label") or entry.get("topic") or entry.get("name"))
            if not term:
                continue
            try:
                prominence = float(entry.get("weight", entry.get("priority", 0.6)))
            except (TypeError, ValueError):
                prominence = 0.6
            prominence = max(0.1, min(1.0, prominence))
            features[f"topic:{term}"] = features.get(f"topic:{term}", 0.0) + self.keyword_weight * prominence

    def _inject_relations(
        self,
        features: MutableMapping[str, float],
        payload: Optional[Iterable[Mapping[str, object]]],
    ) -> None:
        if not payload:
            return
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            subject = _normalize_term(entry.get("subject") or entry.get("from"))
            obj = _normalize_term(entry.get("object") or entry.get("to"))
            rel = _normalize_term(entry.get("verb") or entry.get("relation"))
            if not subject or not obj:
                continue
            key = f"edge:{subject}->{obj}" if not rel else f"edge:{subject}:{rel}:{obj}"
            try:
                confidence = float(entry.get("confidence", entry.get("weight", 0.6)))
            except (TypeError, ValueError):
                confidence = 0.6
            confidence = max(0.1, min(1.0, confidence))
            features[key] = features.get(key, 0.0) + self.relation_weight * confidence


__all__ = ["AdaptiveSemanticEmbedder"]
