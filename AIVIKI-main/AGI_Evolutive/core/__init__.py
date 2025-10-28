"""Core orchestration helpers enriched with LLM guidance."""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Iterable, MutableMapping, Sequence

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CoreComponent:
    name: str
    module: str
    description: str

    def load(self) -> MutableMapping[str, object]:
        info: MutableMapping[str, object] = {
            "name": self.name,
            "module": self.module,
            "description": self.description,
            "available": False,
            "exported": [],
        }
        try:
            module = importlib.import_module(self.module)
        except Exception:
            return info

        info["available"] = True
        exports = []
        for attr_name, attr_value in inspect.getmembers(module):
            if attr_name.startswith("_"):
                continue
            if inspect.isclass(attr_value) or inspect.isfunction(attr_value):
                exports.append(attr_name)
        info["exported"] = sorted(exports)[:12]
        return info


_CORE_COMPONENTS: tuple[_CoreComponent, ...] = (
    _CoreComponent(
        name="Autopilot",
        module="AGI_Evolutive.core.autopilot",
        description="Boucle d'orchestration haute-niveau (ingestion → cognition → persistance).",
    ),
    _CoreComponent(
        name="CognitiveArchitecture",
        module="AGI_Evolutive.core.cognitive_architecture",
        description="Coordination centrale des sous-systèmes cognitifs et de la mémoire.",
    ),
    _CoreComponent(
        name="PersistenceManager",
        module="AGI_Evolutive.core.persistence",
        description="Gestion des snapshots, migrations et journalisation de l'état cognitif.",
    ),
    _CoreComponent(
        name="SelfhoodEngine",
        module="AGI_Evolutive.core.selfhood_engine",
        description="Suivi des traits identitaires et adaptation méta-cognitive.",
    ),
    _CoreComponent(
        name="MAI Structures",
        module="AGI_Evolutive.core.structures.mai",
        description="Structures Mechanistic Actionable Insight et logique d'émission des bids.",
    ),
    _CoreComponent(
        name="TriggerTypes",
        module="AGI_Evolutive.core.trigger_types",
        description="Taxonomie des triggers internes et externes activant les pipelines.",
    ),
)


def core_overview(*, extra_context: Sequence[str] | None = None) -> MutableMapping[str, object]:
    """Return a structured description of the core layer."""

    components = [_component_snapshot(comp) for comp in _CORE_COMPONENTS]
    heuristics = _fallback_overview(components)

    payload = {
        "components": components,
        "hints": list(extra_context or ()),
    }
    response = try_call_llm_dict(
        "core_overview",
        input_payload=json_sanitize(payload),
        logger=LOGGER,
        max_retries=2,
    )

    enriched = _normalise_llm_overview(response, heuristics)
    if enriched is not None:
        return enriched
    return heuristics


def _component_snapshot(component: _CoreComponent) -> MutableMapping[str, object]:
    data = component.load()
    data.setdefault("name", component.name)
    data.setdefault("module", component.module)
    data.setdefault("description", component.description)
    return data


def _fallback_overview(components: Iterable[MutableMapping[str, object]]) -> MutableMapping[str, object]:
    components = [dict(item) for item in components]
    available = [c for c in components if c.get("available")]
    missing = [c for c in components if not c.get("available")]
    summary = (
        "Couche core partiellement chargée : "
        f"{len(available)} composants actifs, {len(missing)} inaccessibles."
    )
    alerts = [f"Module introuvable: {c.get('module')}" for c in missing]
    recommendations = []
    if missing:
        recommendations.append("Vérifier les dépendances et le packaging des modules core.")
    if not missing:
        summary = "Couche core entièrement opérationnelle avec orchestration prête."
    return {
        "source": "heuristic",
        "summary": summary,
        "alerts": alerts,
        "recommended_focus": recommendations or ["Surveiller les métriques d'orchestration."],
        "components": components,
        "confidence": 0.55 if missing else 0.72,
        "notes": "Profil généré sans LLM, basé sur l'import des modules présents.",
    }


def _normalise_llm_overview(
    response: object,
    fallback: MutableMapping[str, object],
) -> MutableMapping[str, object] | None:
    if not isinstance(response, MutableMapping):
        return None

    summary = _clean_str(response.get("summary")) or fallback.get("summary")
    alerts = _clean_list(response.get("alerts"))
    focus = _clean_list(response.get("recommended_focus"))
    components = response.get("components")
    if isinstance(components, list):
        try:
            components = [dict(item) for item in components]
        except Exception:
            components = fallback.get("components")
    else:
        components = fallback.get("components")

    confidence = response.get("confidence")
    try:
        confidence = float(confidence)
    except Exception:
        confidence = fallback.get("confidence", 0.6)

    notes = _clean_str(response.get("notes")) or fallback.get("notes")

    enriched = dict(fallback)
    enriched.update(
        {
            "source": "llm",
            "summary": summary,
            "alerts": alerts,
            "recommended_focus": focus or fallback.get("recommended_focus", []),
            "components": components,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "notes": notes,
        }
    )
    return enriched


def _clean_str(value: object) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return ""


def _clean_list(value: object) -> list[str]:
    items: list[str] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            text = _clean_str(item)
            if text:
                items.append(text)
    return items


__all__ = ["core_overview", "_CoreComponent"]

