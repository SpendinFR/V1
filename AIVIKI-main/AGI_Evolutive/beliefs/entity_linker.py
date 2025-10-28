"""Simple entity linker handling aliases and synonym resolution for the belief graph."""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@dataclass
class EntityRecord:
    canonical_id: str
    name: str
    entity_type: str
    popularity: float = 0.5
    activation: float = 0.0
    last_update: float = field(default_factory=time.monotonic)
    context_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.activation == 0.0:
            self.activation = _logit(self.popularity)
        else:
            self.popularity = _sigmoid(self.activation)

    def apply_decay(self, decay_rate: float, now: Optional[float] = None) -> None:
        if decay_rate <= 0.0:
            return
        now = now or time.monotonic()
        elapsed = max(0.0, now - self.last_update)
        if elapsed == 0.0:
            return
        factor = math.exp(-decay_rate * elapsed)
        self.activation *= factor
        for key in list(self.context_weights):
            self.context_weights[key] *= factor
        self.popularity = _sigmoid(self.activation)
        self.last_update = now

    def bump(
        self,
        weight: float,
        *,
        decay_rate: float = 0.0,
        context: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        now = now or time.monotonic()
        if decay_rate:
            self.apply_decay(decay_rate, now)
        self.activation += weight
        self.popularity = _sigmoid(self.activation)
        if context:
            self.context_weights[context] = self.context_weights.get(context, 0.0) + weight
        self.last_update = now

    def contextual_popularity(self, context: Optional[str]) -> float:
        if not context:
            return self.popularity
        extra = self.context_weights.get(context)
        if extra is None:
            return self.popularity
        return _sigmoid(self.activation + extra)


class AdaptiveWeighter:
    """Simple online learner that adapts weights per entity type."""

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        *,
        learning_rate: float = 0.25,
        adapt_gain: float = 0.6,
        decay: float = 0.02,
    ) -> None:
        self._base_weights = base_weights or {"register": 0.1, "alias": 0.05, "merge": 0.05, "resolve": 0.05}
        self._learning_rate = learning_rate
        self._adapt_gain = adapt_gain
        self._decay = decay
        self._weights: Dict[Tuple[str, str], float] = {}
        self._last_seen: Dict[Tuple[str, str], float] = {}

    def _base(self, event: str) -> float:
        return self._base_weights.get(event, 0.05)

    def _apply_decay(self, key: Tuple[str, str]) -> float:
        now = time.monotonic()
        weight = self._weights.get(key, self._base(key[0]))
        last = self._last_seen.get(key)
        if last is not None and self._decay > 0.0:
            elapsed = max(0.0, now - last)
            if elapsed:
                weight *= math.exp(-self._decay * elapsed)
        self._last_seen[key] = now
        return weight

    def score(self, event: str, entity_type: Optional[str], record: Optional[EntityRecord]) -> float:
        key = (event, entity_type or "*")
        current = self._apply_decay(key)
        base = self._base(event)
        target = base
        if record:
            deficiency = max(0.0, 1.0 - record.popularity)
            target += deficiency * self._adapt_gain
        updated = (1.0 - self._learning_rate) * current + self._learning_rate * target
        updated = max(0.01, min(updated, 1.0))
        self._weights[key] = updated
        return updated


class EntityLinker:
    """
    Maintains the low-level alias table for the belief graph.

    Ce composant ne dépend que des structures de croyances et sert de
    fondation à la façade ``knowledge.EntityLinker`` qui ajoute une
    intégration avec l'ontologie et la mémoire déclarative.
    """

    def __init__(
        self,
        *,
        decay_half_life: float = 600.0,
        base_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._entities: Dict[str, EntityRecord] = {}
        self._aliases: Dict[str, str] = {}
        self._context_usage: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._weighter = AdaptiveWeighter(base_weights)
        self._decay_rate = math.log(2) / decay_half_life if decay_half_life > 0 else 0.0

    def _normalize(self, text: str) -> str:
        return text.strip().lower()

    def _bump_context(self, canonical_id: str, context: Optional[str], weight: float) -> None:
        if not context:
            return
        self._context_usage[context][canonical_id] += weight

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        entity_type: str,
        *,
        canonical_id: Optional[str] = None,
        weight: Optional[float] = None,
        context: Optional[str] = None,
    ) -> str:
        """Registers an entity and returns its canonical identifier."""

        key = self._normalize(name)
        canonical_id = canonical_id or key
        record = self._entities.get(canonical_id)
        now = time.monotonic()
        adaptive_weight = weight if weight is not None else self._weighter.score("register", entity_type, record)
        if record:
            record.bump(adaptive_weight, decay_rate=self._decay_rate, context=context, now=now)
        else:
            initial_popularity = _sigmoid(adaptive_weight)
            record = EntityRecord(
                canonical_id=canonical_id,
                name=name,
                entity_type=entity_type,
                popularity=initial_popularity,
                activation=_logit(initial_popularity),
                last_update=now,
            )
            self._entities[canonical_id] = record
        self._aliases[key] = canonical_id
        self._bump_context(canonical_id, context, adaptive_weight)
        return canonical_id

    def alias(
        self,
        alias: str,
        canonical_id: str,
        *,
        weight: Optional[float] = None,
        context: Optional[str] = None,
    ) -> None:
        record = self._entities.get(canonical_id)
        if not record:
            return
        self._aliases[self._normalize(alias)] = canonical_id
        adaptive_weight = weight if weight is not None else self._weighter.score("alias", record.entity_type, record)
        record.bump(adaptive_weight, decay_rate=self._decay_rate, context=context)
        self._bump_context(canonical_id, context, adaptive_weight)

    # ------------------------------------------------------------------
    def resolve(
        self,
        name: str,
        *,
        entity_type: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Returns (canonical_id, entity_type)."""

        norm = self._normalize(name)
        canonical = self._aliases.get(norm)
        now = time.monotonic()
        if canonical:
            record = self._entities[canonical]
            record.bump(self._weighter.score("resolve", record.entity_type, record), decay_rate=self._decay_rate, context=context, now=now)
            if entity_type and record.entity_type != entity_type:
                record.entity_type = entity_type
            self._bump_context(canonical, context, 0.01)
            return canonical, record.entity_type

        llm_choice = self._llm_resolve(
            name,
            entity_type=entity_type,
            context=context,
        )
        if llm_choice:
            canonical_id, resolved_type, justification = llm_choice
            if canonical_id in self._entities:
                record = self._entities[canonical_id]
            else:
                record = EntityRecord(
                    canonical_id=canonical_id,
                    name=name,
                    entity_type=resolved_type or entity_type or "Entity",
                    popularity=0.6,
                )
                self._entities[canonical_id] = record
            record.bump(
                self._weighter.score("resolve", record.entity_type, record),
                decay_rate=self._decay_rate,
                context=context,
                now=now,
            )
            if resolved_type and record.entity_type != resolved_type:
                record.entity_type = resolved_type
            if justification:
                self._bump_context(canonical_id, context, 0.03)
            self._aliases[norm] = canonical_id
            return canonical_id, record.entity_type

        candidates = []
        if context:
            context_scores = self._context_usage.get(context, {})
            for canonical_id, record in self._entities.items():
                if entity_type and record.entity_type != entity_type:
                    continue
                score = record.contextual_popularity(context) + context_scores.get(canonical_id, 0.0)
                candidates.append((score, canonical_id, record))
        if candidates:
            score, canonical_id, record = max(candidates, key=lambda item: item[0])
            if score > 0.5:
                self._aliases[norm] = canonical_id
                record.bump(self._weighter.score("alias", record.entity_type, record), decay_rate=self._decay_rate, context=context, now=now)
                self._bump_context(canonical_id, context, 0.02)
                if entity_type and record.entity_type != entity_type:
                    record.entity_type = entity_type
                return canonical_id, record.entity_type

        canonical_id = self.register(name, entity_type or "Entity", canonical_id=norm, context=context)
        record = self._entities[canonical_id]
        if entity_type and record.entity_type != entity_type:
            record.entity_type = entity_type
        return canonical_id, record.entity_type

    def _llm_resolve(
        self,
        name: str,
        *,
        entity_type: Optional[str],
        context: Optional[str],
    ) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        payload = {
            "mention": name,
            "entity_type": entity_type,
            "context": context,
            "known_entities": [
                {
                    "canonical_id": canonical_id,
                    "name": record.name,
                    "entity_type": record.entity_type,
                    "popularity": record.popularity,
                }
                for canonical_id, record in list(self._entities.items())[:50]
            ],
        }
        response = try_call_llm_dict(
            "entity_linker",
            input_payload=payload,
            logger=LOGGER,
        )
        if not response:
            return None
        canonical = response.get("canonical_entity")
        if not canonical or not isinstance(canonical, str):
            return None
        resolved_type = response.get("resolved_type") or response.get("entity_type")
        if resolved_type is not None and not isinstance(resolved_type, str):
            resolved_type = None
        justification = response.get("justification")
        if justification is not None and not isinstance(justification, str):
            justification = None
        return canonical, resolved_type, justification

    # ------------------------------------------------------------------
    def merge(self, preferred: str, duplicate: str) -> None:
        """Fuses two entity identifiers, updating aliases."""

        preferred_norm = self._normalize(preferred)
        dup_norm = self._normalize(duplicate)
        if preferred_norm == dup_norm:
            return

        pref_id = self._aliases.get(preferred_norm, preferred_norm)
        dup_id = self._aliases.get(dup_norm, dup_norm)

        if dup_id == pref_id:
            return

        pref_record = self._entities.get(pref_id)
        dup_record = self._entities.pop(dup_id, None)
        if not pref_record or not dup_record:
            return

        weight = self._weighter.score("merge", pref_record.entity_type, pref_record)
        pref_record.bump(weight + dup_record.popularity, decay_rate=self._decay_rate)
        self._aliases[dup_norm] = pref_id
        for alias, canonical in list(self._aliases.items()):
            if canonical == dup_id:
                self._aliases[alias] = pref_id
        for context, mapping in self._context_usage.items():
            if dup_id in mapping:
                mapping[pref_id] += mapping.pop(dup_id)

    def get(self, canonical_id: str) -> Optional[EntityRecord]:
        return self._entities.get(canonical_id)

    def known_entities(self) -> Dict[str, EntityRecord]:
        return dict(self._entities)

    def canonical_form(self, name: str) -> str:
        norm = self._normalize(name)
        return self._aliases.get(norm, norm)
