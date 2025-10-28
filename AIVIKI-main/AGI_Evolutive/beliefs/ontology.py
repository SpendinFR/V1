"""Lightweight ontology definitions for entity and relation types used by the belief graph."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Set

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


_LLM_SPEC_KEY = "ontology_enrichment"
_LLM_CONFIDENCE_THRESHOLD = 0.55
_LLM_SNAPSHOT_LIMIT = 32
_CACHE_MISS = object()


logger = logging.getLogger(__name__)


def _coerce_confidence(value: Any, *, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence))


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "oui", "yes"}:
            return True
        if lowered in {"false", "0", "non", "no"}:
            return False
    return default


def _ensure_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if isinstance(value, Iterable):
        result: list[str] = []
        for candidate in value:
            if isinstance(candidate, str):
                stripped = candidate.strip()
                if stripped:
                    result.append(stripped)
        return result
    return []


def _normalized_name(value: str) -> str:
    return value.strip().lower()


@dataclass(frozen=True)
class EntityType:
    """Represents a semantic type for an entity."""

    name: str
    parent: Optional[str] = None

    def is_a(self, other: str, *, registry: Dict[str, "EntityType"]) -> bool:
        if self.name == other:
            return True
        parent = self.parent
        while parent:
            if parent == other:
                return True
            parent = registry.get(parent).parent if registry.get(parent) else None
        return False


@dataclass(frozen=True)
class RelationType:
    """Represents the schema for a relation in the belief graph."""

    name: str
    domain: Set[str]
    range: Set[str]
    polarity_sensitive: bool = True
    temporal: bool = False
    stability: str = "anchor"  # "anchor" or "episode"

    def allows(self, subject_type: str, object_type: str, *, entities: Dict[str, EntityType]) -> bool:
        return any(
            entities.get(subject_type, EntityType(subject_type)).is_a(domain, registry=entities)
            for domain in self.domain
        ) and any(
            entities.get(object_type, EntityType(object_type)).is_a(rng, registry=entities)
            for rng in self.range
        )


@dataclass(frozen=True)
class EventType:
    """Schema for n-ary events."""

    name: str
    roles: Dict[str, Set[str]]  # role -> allowed entity types

    def validate_roles(self, assignments: Dict[str, str], *, entities: Dict[str, EntityType]) -> bool:
        for role, entity_type in assignments.items():
            allowed = self.roles.get(role)
            if not allowed:
                return False
            if not any(
                entities.get(entity_type, EntityType(entity_type)).is_a(option, registry=entities)
                for option in allowed
            ):
                return False
        return True


class Ontology:
    """Central registry for entity, relation and event types.

    C'est l'implémentation canonique côté ``beliefs`` ; le package
    ``knowledge`` se contente d'en fournir une enveloppe qui clone les
    types par défaut pour un usage read-only.
    """

    def __init__(self) -> None:
        self.entity_types: Dict[str, EntityType] = {}
        self.relation_types: Dict[str, RelationType] = {}
        self.event_types: Dict[str, EventType] = {}
        self._entity_suggestions: Dict[str, Optional[Mapping[str, Any]]] = {}
        self._relation_suggestions: Dict[str, Optional[Mapping[str, Any]]] = {}
        self._event_suggestions: Dict[str, Optional[Mapping[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    def _llm_snapshot(self) -> Dict[str, Any]:
        def _slice(values: Iterable[Any]) -> list[Any]:
            return list(values)[:_LLM_SNAPSHOT_LIMIT]

        entities_payload = [
            {
                "name": entity.name,
                "parent": entity.parent,
            }
            for entity in _slice(self.entity_types.values())
        ]
        relations_payload = [
            {
                "name": relation.name,
                "domain": sorted(relation.domain),
                "range": sorted(relation.range),
                "polarity_sensitive": relation.polarity_sensitive,
                "temporal": relation.temporal,
                "stability": relation.stability,
            }
            for relation in _slice(self.relation_types.values())
        ]
        events_payload = [
            {
                "name": event.name,
                "roles": {role: sorted(options) for role, options in event.roles.items()},
            }
            for event in _slice(self.event_types.values())
        ]
        return {
            "entities": entities_payload,
            "relations": relations_payload,
            "events": events_payload,
        }

    def _llm_enrich(
        self,
        *,
        entity_candidates: Optional[list[Mapping[str, Any]]] = None,
        relation_candidates: Optional[list[Mapping[str, Any]]] = None,
        event_candidates: Optional[list[Mapping[str, Any]]] = None,
    ) -> Optional[Mapping[str, Any]]:
        payload = {
            "candidates": {
                "entities": list(entity_candidates or []),
                "relations": list(relation_candidates or []),
                "events": list(event_candidates or []),
            },
            "existing_snapshot": self._llm_snapshot(),
        }
        return try_call_llm_dict(
            _LLM_SPEC_KEY,
            input_payload=payload,
            logger=logger,
            max_retries=2,
        )

    def _lookup_cached(self, cache: Dict[str, Optional[Mapping[str, Any]]], name: str) -> Any:
        key = _normalized_name(name)
        if key in cache:
            return cache[key]
        return _CACHE_MISS

    def _store_cached(
        self,
        cache: Dict[str, Optional[Mapping[str, Any]]],
        name: str,
        suggestion: Optional[Mapping[str, Any]],
    ) -> Optional[Mapping[str, Any]]:
        cache[_normalized_name(name)] = suggestion
        return suggestion

    def _match_suggestion(
        self,
        items: Iterable[Mapping[str, Any]],
        name: str,
    ) -> Optional[Mapping[str, Any]]:
        lowered = _normalized_name(name)
        for item in items:
            candidate_name = item.get("name")
            if isinstance(candidate_name, str) and _normalized_name(candidate_name) == lowered:
                return item
        return None

    def _entity_suggestion(self, name: str) -> Optional[Mapping[str, Any]]:
        cached = self._lookup_cached(self._entity_suggestions, name)
        if cached is not _CACHE_MISS:
            return cached

        response = self._llm_enrich(entity_candidates=[{"name": name}])
        if not isinstance(response, Mapping):
            return self._store_cached(self._entity_suggestions, name, None)

        suggestion = self._match_suggestion(
            [item for item in response.get("entities", []) if isinstance(item, Mapping)],
            name,
        )
        if suggestion is None:
            return self._store_cached(self._entity_suggestions, name, None)

        confidence = _coerce_confidence(suggestion.get("confidence"), default=1.0)
        if confidence < _LLM_CONFIDENCE_THRESHOLD:
            logger.debug(
                "LLM suggestion for entity '%s' rejected (confidence %.2f)",
                name,
                confidence,
            )
            return self._store_cached(self._entity_suggestions, name, None)

        return self._store_cached(self._entity_suggestions, name, suggestion)

    def _relation_suggestion(self, name: str) -> Optional[Mapping[str, Any]]:
        cached = self._lookup_cached(self._relation_suggestions, name)
        if cached is not _CACHE_MISS:
            return cached

        response = self._llm_enrich(relation_candidates=[{"name": name}])
        if not isinstance(response, Mapping):
            return self._store_cached(self._relation_suggestions, name, None)

        suggestion = self._match_suggestion(
            [item for item in response.get("relations", []) if isinstance(item, Mapping)],
            name,
        )
        if suggestion is None:
            return self._store_cached(self._relation_suggestions, name, None)

        confidence = _coerce_confidence(suggestion.get("confidence"), default=1.0)
        if confidence < _LLM_CONFIDENCE_THRESHOLD:
            logger.debug(
                "LLM suggestion for relation '%s' rejected (confidence %.2f)",
                name,
                confidence,
            )
            return self._store_cached(self._relation_suggestions, name, None)

        return self._store_cached(self._relation_suggestions, name, suggestion)

    def _event_suggestion(self, name: str) -> Optional[Mapping[str, Any]]:
        cached = self._lookup_cached(self._event_suggestions, name)
        if cached is not _CACHE_MISS:
            return cached

        response = self._llm_enrich(event_candidates=[{"name": name}])
        if not isinstance(response, Mapping):
            return self._store_cached(self._event_suggestions, name, None)

        suggestion = self._match_suggestion(
            [item for item in response.get("events", []) if isinstance(item, Mapping)],
            name,
        )
        if suggestion is None:
            return self._store_cached(self._event_suggestions, name, None)

        confidence = _coerce_confidence(suggestion.get("confidence"), default=1.0)
        if confidence < _LLM_CONFIDENCE_THRESHOLD:
            logger.debug(
                "LLM suggestion for event '%s' rejected (confidence %.2f)",
                name,
                confidence,
            )
            return self._store_cached(self._event_suggestions, name, None)

        return self._store_cached(self._event_suggestions, name, suggestion)

    def register_entity(self, name: str, *, parent: Optional[str] = None) -> None:
        if parent and parent not in self.entity_types:
            logger.debug("Auto-registering missing parent entity '%s' for '%s'", parent, name)
            self.entity_types[parent] = EntityType(name=parent, parent="Entity" if parent != "Entity" else None)
        self.entity_types[name] = EntityType(name=name, parent=parent)

    def register_relation(
        self,
        name: str,
        *,
        domain: Iterable[str],
        range: Iterable[str],
        polarity_sensitive: bool = True,
        temporal: bool = False,
        stability: str = "anchor",
    ) -> None:
        self.relation_types[name] = RelationType(
            name=name,
            domain=set(domain),
            range=set(range),
            polarity_sensitive=polarity_sensitive,
            temporal=temporal,
            stability=stability,
        )

    def register_event(self, name: str, *, roles: Dict[str, Iterable[str]]) -> None:
        self.event_types[name] = EventType(
            name=name,
            roles={role: set(options) for role, options in roles.items()},
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    def load_from_mapping(
        self,
        config: Mapping[str, Any],
        *,
        clear_existing: bool = False,
    ) -> None:
        """Populate the ontology from a mapping structure.

        The mapping follows a lightweight schema::

            {
                "entities": [
                    {"name": "Agent", "parent": "Entity"},
                    "CustomEntity"
                ],
                "relations": [
                    {
                        "name": "knows",
                        "domain": ["Agent"],
                        "range": ["Agent"],
                        "polarity_sensitive": false,
                        "temporal": true,
                        "stability": "episode"
                    }
                ],
                "events": [
                    {
                        "name": "meeting",
                        "roles": {"host": ["Agent"]}
                    }
                ]
            }

        Missing optional fields fall back to sensible defaults, so the method
        is resilient to partial configurations.
        """

        if clear_existing:
            self.entity_types.clear()
            self.relation_types.clear()
            self.event_types.clear()

        for entity_entry in config.get("entities", []) or []:
            if isinstance(entity_entry, str):
                self.register_entity(entity_entry)
            else:
                name = entity_entry.get("name")
                if not name:
                    logger.warning("Skipping entity definition without a name: %s", entity_entry)
                    continue
                self.register_entity(name, parent=entity_entry.get("parent"))

        for relation_entry in config.get("relations", []) or []:
            if not isinstance(relation_entry, Mapping):
                logger.warning("Skipping invalid relation definition: %s", relation_entry)
                continue
            name = relation_entry.get("name")
            domain = relation_entry.get("domain", ["Entity"])
            range_ = relation_entry.get("range", ["Entity"])
            if not name:
                logger.warning("Skipping relation definition without a name: %s", relation_entry)
                continue
            self.register_relation(
                name,
                domain=domain,
                range=range_,
                polarity_sensitive=relation_entry.get("polarity_sensitive", True),
                temporal=relation_entry.get("temporal", False),
                stability=relation_entry.get("stability", "anchor"),
            )

        for event_entry in config.get("events", []) or []:
            if not isinstance(event_entry, Mapping):
                logger.warning("Skipping invalid event definition: %s", event_entry)
                continue
            name = event_entry.get("name")
            roles = event_entry.get("roles", {})
            if not name:
                logger.warning("Skipping event definition without a name: %s", event_entry)
                continue
            if not isinstance(roles, Mapping):
                logger.warning("Skipping event '%s' with invalid roles: %s", name, roles)
                continue
            self.register_event(name, roles={role: list(options) for role, options in roles.items()})

    def load_from_file(self, path: Path | str, *, clear_existing: bool = False) -> None:
        """Load ontology definitions from a JSON file."""

        data: Mapping[str, Any]
        raw_path = Path(path)
        with raw_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ValueError(f"Expected a mapping at the top-level of {raw_path}")
        self.load_from_mapping(data, clear_existing=clear_existing)

    # ------------------------------------------------------------------
    # Lookup helpers
    def entity(self, name: str) -> Optional[EntityType]:
        return self.entity_types.get(name)

    def relation(self, name: str) -> Optional[RelationType]:
        return self.relation_types.get(name)

    def event(self, name: str) -> Optional[EventType]:
        event_type = self.event_types.get(name)
        if event_type:
            return event_type
        suggestion = self._event_suggestion(name)
        if not suggestion:
            return None
        roles_data = suggestion.get("roles")
        if not isinstance(roles_data, Mapping):
            logger.debug(  # pragma: no cover - logging
                "LLM suggestion for event '%s' missing roles: %s",
                name,
                suggestion,
            )
            return None
        roles: Dict[str, Set[str]] = {}
        for role, allowed in roles_data.items():
            if not isinstance(role, str):
                continue
            allowed_list = _ensure_str_list(allowed)
            if not allowed_list:
                continue
            roles[role] = set(allowed_list)
            for ent_name in allowed_list:
                if ent_name not in self.entity_types:
                    self.infer_entity_type(ent_name)
        if not roles:
            logger.debug(  # pragma: no cover - logging
                "LLM suggestion for event '%s' produced empty roles: %s",
                name,
                suggestion,
            )
            return None
        event_type = EventType(name=name, roles=roles)
        self.event_types[name] = event_type
        logger.debug(  # pragma: no cover - logging
            "LLM provided schema for event '%s' with roles=%s",
            name,
            sorted(roles.keys()),
        )
        return event_type

    # ------------------------------------------------------------------
    def infer_relation_type(self, name: str) -> RelationType:
        rel = self.relation(name)
        if rel:
            return rel
        suggestion = self._relation_suggestion(name)
        if suggestion:
            domain_candidates = _ensure_str_list(
                suggestion.get("domain")
                or suggestion.get("domain_types")
                or suggestion.get("subjects")
            )
            range_candidates = _ensure_str_list(
                suggestion.get("range")
                or suggestion.get("range_types")
                or suggestion.get("objects")
            )
            domain = {candidate for candidate in domain_candidates if candidate}
            range_ = {candidate for candidate in range_candidates if candidate}
            if domain and range_:
                for entity_name in domain.union(range_):
                    if entity_name not in self.entity_types:
                        self.infer_entity_type(entity_name)
                relation = RelationType(
                    name=name,
                    domain=domain,
                    range=range_,
                    polarity_sensitive=_coerce_bool(
                        suggestion.get("polarity_sensitive"), default=True
                    ),
                    temporal=_coerce_bool(
                        suggestion.get("temporal"), default=False
                    ),
                    stability=str(suggestion.get("stability", "anchor")).strip() or "anchor",
                )
                self.relation_types[name] = relation
                logger.debug(
                    "LLM provided schema for relation '%s': domain=%s range=%s",  # pragma: no cover - logging
                    name,
                    sorted(relation.domain),
                    sorted(relation.range),
                )
                return relation
            logger.debug(  # pragma: no cover - logging
                "LLM suggestion for relation '%s' missing domain/range: %s",
                name,
                suggestion,
            )
        # Fallback relation for unknown entries
        fallback = RelationType(
            name=name,
            domain={"Entity"},
            range={"Entity"},
            polarity_sensitive=True,
            temporal=False,
            stability="anchor",
        )
        self.relation_types[name] = fallback
        return fallback

    def infer_entity_type(self, name: str) -> EntityType:
        ent = self.entity(name)
        if ent:
            return ent
        suggestion = self._entity_suggestion(name)
        if suggestion:
            parent_raw = suggestion.get("parent") or suggestion.get("parent_type")
            parent: Optional[str]
            if isinstance(parent_raw, str) and parent_raw.strip():
                parent = parent_raw.strip()
            else:
                parent = "Entity"
            if parent == name:
                parent = "Entity"
            if parent and parent not in self.entity_types:
                inferred_parent = EntityType(name=parent, parent="Entity" if parent != "Entity" else None)
                self.entity_types[parent] = inferred_parent
            entity = EntityType(name=name, parent=parent)
            self.entity_types[name] = entity
            logger.debug(  # pragma: no cover - logging
                "LLM provided parent '%s' for entity '%s' (%s)",
                parent,
                name,
                suggestion.get("justification"),
            )
            return entity
        fallback = EntityType(name=name, parent="Entity")
        self.entity_types[name] = fallback
        return fallback

    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "Ontology":
        onto = cls()
        # Base hierarchy
        onto.register_entity("Entity")
        onto.register_entity("Agent", parent="Entity")
        onto.register_entity("Person", parent="Agent")
        onto.register_entity("Organization", parent="Agent")
        onto.register_entity("Place", parent="Entity")
        onto.register_entity("Object", parent="Entity")
        onto.register_entity("Food", parent="Object")
        onto.register_entity("Habit", parent="Entity")
        onto.register_entity("Activity", parent="Entity")
        onto.register_entity("TemporalSegment", parent="Entity")
        onto.register_entity("Context", parent="Entity")
        onto.register_entity("Emotion", parent="Entity")
        onto.register_entity("Goal", parent="Entity")
        onto.register_entity("Intention", parent="Goal")
        onto.register_entity("Resource", parent="Object")
        onto.register_entity("Tool", parent="Resource")
        onto.register_entity("Knowledge", parent="Entity")
        onto.register_entity("Experience", parent="Entity")
        onto.register_entity("Communication", parent="Activity")

        # Relations
        onto.register_relation("likes", domain=["Agent"], range=["Entity"], stability="anchor")
        onto.register_relation(
            "does_often",
            domain=["Agent"],
            range=["Activity"],
            temporal=True,
            stability="anchor",
        )
        onto.register_relation(
            "causes",
            domain=["Entity"],
            range=["Entity"],
            stability="anchor",
        )
        onto.register_relation(
            "part_of",
            domain=["Entity"],
            range=["Entity"],
            stability="anchor",
        )
        onto.register_relation(
            "opposes",
            domain=["Entity"],
            range=["Entity"],
            stability="episode",
        )
        onto.register_relation(
            "temporal",
            domain=["Activity", "TemporalSegment"],
            range=["TemporalSegment", "Activity"],
            temporal=True,
            stability="episode",
        )
        onto.register_relation(
            "related_to",
            domain=["Entity"],
            range=["Entity"],
            stability="anchor",
        )
        onto.register_relation(
            "located_in",
            domain=["Entity"],
            range=["Place"],
            stability="anchor",
        )
        onto.register_relation(
            "uses",
            domain=["Agent"],
            range=["Tool", "Resource"],
            temporal=True,
            stability="episode",
        )
        onto.register_relation(
            "has_goal",
            domain=["Agent"],
            range=["Goal", "Intention"],
            stability="anchor",
        )
        onto.register_relation(
            "influences",
            domain=["Agent", "Organization", "Context"],
            range=["Agent", "Context", "Emotion"],
            temporal=True,
            stability="episode",
        )
        onto.register_relation(
            "reports_on",
            domain=["Agent", "Organization"],
            range=["Knowledge", "Experience"],
            stability="anchor",
        )
        onto.register_relation(
            "expresses",
            domain=["Agent"],
            range=["Emotion", "Intention"],
            temporal=True,
            stability="episode",
        )
        onto.register_relation(
            "precedes",
            domain=["Activity", "TemporalSegment"],
            range=["Activity", "TemporalSegment"],
            temporal=True,
            stability="episode",
        )

        # Events
        onto.register_event(
            "interaction",
            roles={
                "actor": ["Agent"],
                "target": ["Agent", "Entity"],
                "medium": ["Object", "Activity"],
                "location": ["Place"],
            },
        )
        onto.register_event(
            "consumption",
            roles={"actor": ["Agent"], "item": ["Food", "Object"], "location": ["Place"]},
        )
        onto.register_event(
            "routine",
            roles={"actor": ["Agent"], "activity": ["Activity"], "timeslot": ["TemporalSegment"]},
        )
        onto.register_event(
            "collaboration",
            roles={
                "participants": ["Agent", "Organization"],
                "goal": ["Goal", "Intention"],
                "resource": ["Resource", "Tool"],
                "context": ["Context", "Place"],
            },
        )
        onto.register_event(
            "observation",
            roles={
                "observer": ["Agent"],
                "subject": ["Entity"],
                "insight": ["Knowledge", "Experience"],
            },
        )
        onto.register_event(
            "goal_progress",
            roles={
                "agent": ["Agent"],
                "goal": ["Goal", "Intention"],
                "milestone": ["TemporalSegment", "Experience"],
            },
        )
        return onto
