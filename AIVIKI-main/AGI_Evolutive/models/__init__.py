"""User modeling primitives and LLM-assisted synthesis helpers."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping, Sequence

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

from .intent import IntentModel
from .user import UserModel


LOGGER = logging.getLogger(__name__)


def _sorted_items(
    items: Mapping[str, Any] | None,
    *,
    key: str,
    value_getter: Callable[[Any], Any] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not isinstance(items, Mapping):
        return []
    collected: list[dict[str, Any]] = []
    for label, payload in items.items():
        if isinstance(payload, Mapping):
            data = dict(payload)
        else:
            data = {key: payload}
        data.setdefault("label", label)
        if value_getter is not None:
            value = value_getter(payload)
        elif isinstance(payload, (int, float)):
            value = float(payload)
        else:
            value = data.get(key)
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        data[key] = max(0.0, min(1.0, score))
        collected.append(data)
    collected.sort(key=lambda entry: entry.get(key, 0.0), reverse=True)
    return collected[:limit]


def _sanitize_interactions(interactions: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if not interactions:
        return []
    cleaned: list[Mapping[str, Any]] = []
    for entry in interactions:
        if not isinstance(entry, Mapping):
            continue
        text = str(entry.get("text") or entry.get("content") or "").strip()
        if not text:
            continue
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        cleaned.append(
            {
                "text": text[:240],
                "metadata": {
                    key: metadata[key]
                    for key in ("channel", "sentiment", "topic", "partner")
                    if metadata.get(key) is not None
                },
            }
        )
        if len(cleaned) >= 12:
            break
    return cleaned


def _extract_state(user_model_or_state: Any) -> Mapping[str, Any]:
    if hasattr(user_model_or_state, "describe"):
        try:
            state = user_model_or_state.describe()  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive guard
            state = {}
    else:
        state = user_model_or_state
    if isinstance(state, Mapping):
        return state
    return {}


def _build_llm_payload(
    state: Mapping[str, Any],
    recent_interactions: Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    persona = state.get("persona") if isinstance(state.get("persona"), Mapping) else {}
    preferences = _sorted_items(
        state.get("preferences"),
        key="probability",
        value_getter=lambda payload: payload.get("prob") if isinstance(payload, Mapping) else payload,
        limit=6,
    )
    routines = []
    routines_map = state.get("routines")
    if isinstance(routines_map, Mapping):
        for bucket, entries in routines_map.items():
            if not isinstance(entries, Mapping):
                continue
            top_entries = _sorted_items(
                entries,
                key="probability",
                value_getter=lambda payload: payload.get("prob") if isinstance(payload, Mapping) else payload,
                limit=2,
            )
            for payload in top_entries:
                routines.append(
                    {
                        "time_bucket": bucket,
                        "activity": payload.get("label"),
                        "probability": payload.get("probability", 0.0),
                    }
                )
    routines.sort(key=lambda entry: entry.get("probability", 0.0), reverse=True)
    return json_sanitize(
        {
            "persona": persona,
            "values": persona.get("values") if isinstance(persona, Mapping) else {},
            "top_preferences": preferences,
            "routine_candidates": routines[:6],
            "recent_interactions": _sanitize_interactions(recent_interactions),
            "updated_at": state.get("last_update"),
        }
    )


def _fallback_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    persona = payload.get("persona") if isinstance(payload.get("persona"), Mapping) else {}
    values = payload.get("values") if isinstance(payload.get("values"), Mapping) else {}
    sorted_values = sorted(
        (
            (str(label), float(value))
            for label, value in values.items()
            if isinstance(label, str) and isinstance(value, (int, float))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    key_traits = [
        {
            "trait": label.replace("_", " "),
            "confidence": max(0.0, min(1.0, value)),
            "evidence": "Score de valeur interne",
        }
        for label, value in sorted_values[:5]
    ]

    preference_highlights = [
        {"label": pref.get("label"), "probability": pref.get("probability", 0.0)}
        for pref in payload.get("top_preferences", [])[:5]
        if isinstance(pref, Mapping)
    ]

    routine_insights = [
        {
            "time_bucket": routine.get("time_bucket"),
            "activity": routine.get("activity"),
            "probability": routine.get("probability", 0.0),
        }
        for routine in payload.get("routine_candidates", [])[:3]
        if isinstance(routine, Mapping)
    ]

    tone = str(persona.get("tone", "neutre")) if isinstance(persona, Mapping) else "neutre"
    tone_lower = tone.lower()
    if any(term in tone_lower for term in ("joy", "posit", "enthous", "heureux")):
        satisfaction_trend = "hausse"
    elif any(term in tone_lower for term in ("stress", "fatigue", "anx", "baisse")):
        satisfaction_trend = "baisse"
    else:
        satisfaction_trend = "stable"

    recommended_actions: list[Mapping[str, Any]] = []
    if preference_highlights:
        first_pref = preference_highlights[0]
        label = first_pref.get("label")
        if isinstance(label, str):
            recommended_actions.append(
                {
                    "action": f"adapter les propositions vers {label}",
                    "reason": "préférence prioritaire détectée",
                }
            )
    if not recommended_actions and routine_insights:
        routine = routine_insights[0]
        activity = routine.get("activity")
        if isinstance(activity, str):
            recommended_actions.append(
                {
                    "action": f"proposer un accompagnement autour de {activity}",
                    "reason": "routine fréquente identifiée",
                }
            )

    persona_summary = "Profil utilisateur partiellement déterminé."
    if key_traits:
        main_trait = key_traits[0]["trait"]
        persona_summary = f"Utilisateur {main_trait} avec ton {tone_lower}."

    return {
        "persona_summary": persona_summary,
        "key_traits": key_traits,
        "preference_highlights": preference_highlights,
        "routine_insights": routine_insights,
        "recommended_actions": recommended_actions,
        "satisfaction_trend": satisfaction_trend,
        "notes": "Synthèse heuristique faute d'intégration LLM disponible.",
    }


def summarize_user_models(
    user_model_or_state: UserModel | Mapping[str, Any],
    *,
    recent_interactions: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Produce an actionable summary of user modelling signals.

    The function attempts to leverage the dedicated LLM integration. When the LLM
    is unavailable the heuristics from the existing state are used as a fallback
    to preserve current behaviour.
    """

    state = _extract_state(user_model_or_state)
    payload = _build_llm_payload(state, recent_interactions)
    response = try_call_llm_dict(
        "user_models_overview",
        input_payload=payload,
        logger=LOGGER,
    )
    if response:
        sanitized = json_sanitize(response)
        sanitized.setdefault("notes", "")
        return sanitized
    return _fallback_summary(payload)


__all__ = ["IntentModel", "UserModel", "summarize_user_models"]
