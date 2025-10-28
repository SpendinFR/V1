"""Facade exposing the canonical belief ontology to the knowledge layer."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Optional, Tuple

from AGI_Evolutive.beliefs.entity_linker import EntityLinker as _BeliefEntityLinker
from AGI_Evolutive.beliefs.graph import BeliefGraph
from AGI_Evolutive.beliefs.ontology import Ontology as _BeliefOntology
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)
_LLM_SPEC_KEY = "knowledge_entity_typing"


class Ontology(_BeliefOntology):
    """
    Thin wrapper re-exporting the belief ontology for the knowledge layer.

    Il fournit un constructeur sans argument qui clone les types par
    défaut afin d'éviter les effets de bord lorsque l'on manipule
    l'ontologie côté *knowledge* sans toucher aux structures de croyances.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if not args and not kwargs:
            default = _BeliefOntology.default()
            self.entity_types = dict(default.entity_types)
            self.relation_types = dict(default.relation_types)
            self.event_types = dict(default.event_types)


class EntityLinker(_BeliefEntityLinker):
    """
    Entity linker aware of the ontology and the belief graph.

    Cette façade enrichit :class:`AGI_Evolutive.beliefs.entity_linker.EntityLinker`
    avec l'accès à une ontologie et – optionnellement – au graphe de
    croyances pour pré-remplir la table d'alias.
    """

    def __init__(
        self,
        ontology: Optional[Ontology] = None,
        beliefs: Optional[BeliefGraph] = None,
    ) -> None:
        super().__init__()
        self.ontology = ontology or Ontology()
        self.beliefs = beliefs
        self._type_priors: Counter[str] = Counter()
        self._feature_likelihoods: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self._feature_values: Dict[str, set[str]] = defaultdict(set)
        if beliefs is not None:
            self._ingest_from_beliefs(beliefs)

    # ------------------------------------------------------------------
    def link(self, text: str, *, hint_type: Optional[str] = None) -> Dict[str, str]:
        """Resolve ``text`` into a canonical entity and type."""

        entity_type = hint_type or self._infer_type(text)
        canonical, resolved_type = self.resolve(text, entity_type=entity_type)
        return {"text": text, "canonical": canonical, "type": resolved_type}

    # ------------------------------------------------------------------
    def _infer_type(self, text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return "Entity"

        llm_type = self._llm_infer_type(stripped)
        if llm_type:
            return llm_type

        if not self._type_priors:
            return self._fallback_type(stripped)

        features = self._extract_features(stripped)
        log_scores: Dict[str, float] = {}
        total_types = sum(self._type_priors.values())
        if total_types <= 0:
            return self._fallback_type(stripped)

        for entity_type, count in self._type_priors.items():
            log_prob = math.log((count + 1.0) / (total_types + len(self._type_priors)))
            for feature_name, feature_value in features:
                key = (feature_name, feature_value)
                likelihoods = self._feature_likelihoods.get(key)
                cardinality = max(1, len(self._feature_values.get(feature_name, {feature_value})))
                observed = 0.0
                if likelihoods is not None:
                    observed = likelihoods.get(entity_type, 0.0)
                log_prob += math.log((observed + 1.0) / (self._type_priors[entity_type] + cardinality))
            log_scores[entity_type] = log_prob

        if not log_scores:
            return self._fallback_type(stripped)

        best_type = max(log_scores.items(), key=lambda item: item[1])[0]
        return best_type

    def _fallback_type(self, stripped: str) -> str:
        if stripped.istitle() and " " in stripped:
            return "Person"
        if stripped and stripped[0].isupper():
            return "Place"
        if any(ch.isdigit() for ch in stripped):
            return "Identifier"
        return "Entity"

    def _extract_features(self, text: str) -> Iterable[Tuple[str, str]]:
        tokens = text.split()
        features = [
            ("has_space", "1" if len(tokens) > 1 else "0"),
            ("is_title", "1" if text.istitle() else "0"),
            ("is_upper", "1" if text.isupper() else "0"),
            ("has_digit", "1" if any(ch.isdigit() for ch in text) else "0"),
            ("prefix", text[:3].lower()),
            ("suffix", text[-3:].lower() if len(text) >= 3 else text.lower()),
            ("token_count", str(len(tokens))),
        ]
        if tokens:
            features.append(("last_token", tokens[-1].lower()))
        return features

    def _llm_infer_type(self, stripped: str) -> Optional[str]:
        payload = {
            "label": stripped,
            "known_types": sorted(self.ontology.entity_types.keys()),
            "feature_snapshot": [
                {"name": name, "value": value} for name, value in self._extract_features(stripped)
            ],
            "priors": [
                {"type": type_name, "count": count}
                for type_name, count in self._type_priors.most_common(6)
            ],
        }

        response = try_call_llm_dict(
            _LLM_SPEC_KEY,
            input_payload=payload,
            logger=LOGGER,
        )
        if not response:
            return None

        candidate = self._normalize_type_name(response.get("type"))
        fallback = self._normalize_type_name(response.get("fallback_type"))
        confidence = self._safe_float(response.get("confidence"))

        chosen = self._select_llm_type(candidate, fallback, confidence)
        return chosen

    def _normalize_type_name(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _select_llm_type(
        self,
        candidate: Optional[str],
        fallback: Optional[str],
        confidence: Optional[float],
    ) -> Optional[str]:
        known_types = set(self.ontology.entity_types.keys())
        threshold = 0.45

        if candidate and (not known_types or candidate in known_types):
            if confidence is None or confidence >= threshold:
                return candidate

        if fallback and fallback in known_types:
            return fallback

        if candidate and not known_types:
            return candidate

        return None

    def _record_features(self, entity_type: str, label: str) -> None:
        features = list(self._extract_features(label))
        self._type_priors[entity_type] += 1
        for feature_name, feature_value in features:
            key = (feature_name, feature_value)
            likelihoods = self._feature_likelihoods[key]
            likelihoods[entity_type] += 1
            self._feature_values[feature_name].add(feature_value)

    def register(
        self,
        name: str,
        entity_type: str,
        *,
        canonical_id: Optional[str] = None,
        weight: Optional[float] = None,
        context: Optional[str] = None,
    ) -> str:
        canonical = super().register(
            name,
            entity_type,
            canonical_id=canonical_id,
            weight=weight,
            context=context,
        )
        if entity_type:
            self._record_features(entity_type, name)
        return canonical

    def alias(
        self,
        alias: str,
        canonical_id: str,
        *,
        weight: Optional[float] = None,
        context: Optional[str] = None,
    ) -> None:
        record = self.get(canonical_id)
        super().alias(alias, canonical_id, weight=weight, context=context)
        if record:
            self._record_features(record.entity_type, alias)

    def _compute_weight(self, count: int, avg_confidence: float, recency: float) -> float:
        base = math.log1p(count)
        confidence_boost = avg_confidence
        recency_factor = math.exp(-recency / 900.0)  # ~15 minutes half-life
        weight = (base * 0.4) + (confidence_boost * 0.4) + (recency_factor * 0.2)
        return max(0.05, min(weight, 1.0))

    def _ingest_from_beliefs(self, beliefs: BeliefGraph) -> None:
        now = time.time()
        stats: Dict[str, Dict[str, float]] = {}

        def update_stats(key: str, belief_time: float, confidence: float) -> None:
            record = stats.setdefault(key, {"count": 0.0, "confidence": 0.0, "updated_at": 0.0})
            record["count"] += 1.0
            record["confidence"] += confidence
            record["updated_at"] = max(record["updated_at"], belief_time)

        try:
            for belief in beliefs.query(active_only=False):
                update_stats(f"subject::{belief.subject}", belief.updated_at, belief.confidence)
                update_stats(f"value::{belief.value}", belief.updated_at, belief.confidence)

                subject_key = f"subject::{belief.subject}"
                subject_stats = stats[subject_key]
                recency_subject = max(0.0, now - subject_stats.get("updated_at", belief.updated_at))
                subject_weight = self._compute_weight(
                    int(subject_stats["count"]),
                    subject_stats["confidence"] / max(1.0, subject_stats["count"]),
                    recency_subject,
                )
                self.register(
                    belief.subject_label,
                    belief.subject_type,
                    canonical_id=belief.subject,
                    weight=subject_weight,
                    context=belief.relation_type,
                )

                value_key = f"value::{belief.value}"
                value_stats = stats[value_key]
                recency_value = max(0.0, now - value_stats.get("updated_at", belief.updated_at))
                value_weight = self._compute_weight(
                    int(value_stats["count"]),
                    value_stats["confidence"] / max(1.0, value_stats["count"]),
                    recency_value,
                )
                self.register(
                    belief.value_label,
                    belief.value_type,
                    canonical_id=belief.value,
                    weight=value_weight,
                    context=belief.relation_type,
                )
        except Exception:
            # Belief graph may not be initialised yet.
            return

