"""Runtime helpers to enforce critical LLM output contracts.

The integration specs rely on structured payloads that are consumed directly
by Python modules.  When a model drifts or returns malformed content we want
to fail soft by cleaning the response instead of letting exceptions ripple
through the runtime.  The helpers in this module normalise the handful of
high-impact integrations called out in ``docs/llm_runtime_contracts.md`` so
callers can simply opt-in via :func:`enforce_llm_contract`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional


Sanitizer = Callable[[Mapping[str, Any]], Dict[str, Any]]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value).strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    return ""


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _iter_sequence(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        return value.values()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _sanitize_dialogue_state(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _as_text(payload.get("state_summary") or payload.get("summary"))
    commitments: list[Dict[str, Any]] = []
    for raw in _iter_sequence(payload.get("open_commitments")):
        if not isinstance(raw, Mapping):
            continue
        label = _as_text(raw.get("commitment") or raw.get("label") or raw.get("title"))
        if not label:
            continue
        entry: Dict[str, Any] = {"commitment": label}
        deadline = _as_text(raw.get("deadline") or raw.get("due"))
        if deadline:
            entry["deadline"] = deadline
        status = _as_text(raw.get("status"))
        if status:
            entry["status"] = status
        confidence = _as_float(raw.get("confidence"))
        if confidence is not None:
            entry["confidence"] = _clip_unit(confidence)
        origin = _as_text(raw.get("origin"))
        if origin:
            entry["origin"] = origin
        commitments.append(entry)
        if len(commitments) >= 12:
            break

    pending: list[str] = []
    raw_pending = payload.get("pending_questions")
    if isinstance(raw_pending, Mapping):
        raw_pending = raw_pending.values()
    for item in _iter_sequence(raw_pending):
        text = _as_text(item)
        if text:
            pending.append(text)
        if len(pending) >= 10:
            break
    if not pending:
        single = _as_text(raw_pending)
        if single:
            pending.append(single)

    notes = _as_text(payload.get("notes"))

    cleaned: Dict[str, Any] = {
        "state_summary": summary,
        "open_commitments": commitments,
        "pending_questions": pending,
    }
    if notes:
        cleaned["notes"] = notes
    return cleaned


def _sanitize_episodic_linker(payload: Mapping[str, Any]) -> Dict[str, Any]:
    links: list[Dict[str, Any]] = []
    for raw in _iter_sequence(payload.get("links")):
        if not isinstance(raw, Mapping):
            continue
        src = _as_text(raw.get("from") or raw.get("src"))
        dst = _as_text(raw.get("to") or raw.get("dst"))
        relation = _as_text(raw.get("type_lien") or raw.get("relation") or raw.get("rel"))
        if not src or not dst or not relation:
            continue
        entry: Dict[str, Any] = {
            "from": src,
            "to": dst,
            "type_lien": relation,
        }
        confidence = _as_float(raw.get("confidence"))
        if confidence is not None:
            entry["confidence"] = _clip_unit(confidence)
        notes = _as_text(raw.get("notes"))
        if notes:
            entry["notes"] = notes
        links.append(entry)
        if len(links) >= 32:
            break

    cleaned: Dict[str, Any] = {"links": links}
    notes = _as_text(payload.get("notes"))
    if notes:
        cleaned["notes"] = notes
    return cleaned


def _sanitize_identity_mission(payload: Mapping[str, Any]) -> Dict[str, Any]:
    mission_block = payload.get("mission")
    if not isinstance(mission_block, Mapping):
        mission_block = {}
    axes: Dict[str, str] = {}
    axis_aliases = {
        "prioritaire": ("prioritaire", "priority", "focus"),
        "support": ("support", "backbone", "soutien"),
        "vision": ("vision", "aspiration", "north_star"),
    }
    for axis, aliases in axis_aliases.items():
        value = ""
        for alias in aliases:
            value = _as_text(mission_block.get(alias) or payload.get(alias))
            if value:
                break
        if value:
            axes[axis] = value

    cleaned: Dict[str, Any] = {"mission": axes}
    mission_text = _as_text(payload.get("mission_text") or payload.get("mission_statement"))
    if mission_text:
        cleaned["mission_text"] = mission_text
    notes = _as_text(payload.get("notes"))
    if notes:
        cleaned["notes"] = notes
    return cleaned


def _sanitize_goal_interpreter(payload: Mapping[str, Any]) -> Dict[str, Any]:
    actions: list[Dict[str, Any]] = []
    for raw in _iter_sequence(payload.get("candidate_actions")):
        if not isinstance(raw, Mapping):
            continue
        action_label = _as_text(raw.get("action"))
        if not action_label:
            continue
        entry: Dict[str, Any] = {"action": action_label}
        rationale = _as_text(raw.get("rationale") or raw.get("reason"))
        if rationale:
            entry["rationale"] = rationale
        confidence = _as_float(raw.get("confidence"))
        if confidence is not None:
            entry["confidence"] = _clip_unit(confidence)
        context = raw.get("context")
        if isinstance(context, Mapping):
            entry["context"] = {str(k): context[k] for k in context if isinstance(k, str)}
        actions.append(entry)
        if len(actions) >= 8:
            break

    cleaned: Dict[str, Any] = {"candidate_actions": actions}
    normalized_goal = _as_text(payload.get("normalized_goal"))
    if normalized_goal:
        cleaned["normalized_goal"] = normalized_goal
    notes = _as_text(payload.get("notes"))
    if notes:
        cleaned["notes"] = notes
    return cleaned


def _sanitize_trigger_router(payload: Mapping[str, Any]) -> Dict[str, Any]:
    pipelines = [
        item
        for item in (
            _as_text(candidate)
            for candidate in _iter_sequence(payload.get("pipelines"))
        )
        if item
    ][:6]
    secondary = [
        item
        for item in (
            _as_text(candidate)
            for candidate in _iter_sequence(payload.get("secondary"))
        )
        if item
    ][:6]
    notes = _as_text(payload.get("notes"))
    cleaned: Dict[str, Any] = {
        "pipelines": pipelines,
        "secondary": secondary,
    }
    if notes:
        cleaned["notes"] = notes
    return cleaned


def _sanitize_planner_support(payload: Mapping[str, Any]) -> Dict[str, Any]:
    steps: list[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, raw in enumerate(_iter_sequence(payload.get("plan"))):
        if not isinstance(raw, Mapping):
            continue
        description = _as_text(raw.get("description"))
        if not description:
            continue
        step_id = _as_text(raw.get("id"))
        if not step_id:
            step_id = f"llm_step_{idx + 1}"
        if step_id in seen_ids:
            step_id = f"{step_id}_{idx + 1}"
        seen_ids.add(step_id)
        entry: Dict[str, Any] = {"id": step_id, "description": description}
        depends_on: list[str] = []
        for candidate in _iter_sequence(raw.get("depends_on")):
            if not isinstance(candidate, str):
                continue
            dep = _as_text(candidate)
            if dep:
                depends_on.append(dep)
        if depends_on:
            entry["depends_on"] = depends_on[:10]
        priority = _as_float(raw.get("priority"))
        if priority is not None:
            entry["priority"] = _clip_unit(priority)
        action_type = _as_text(raw.get("action_type"))
        if action_type:
            entry["action_type"] = action_type
        context = raw.get("context")
        if isinstance(context, Mapping):
            entry["context"] = {str(k): context[k] for k in context if isinstance(k, str)}
        steps.append(entry)
        if len(steps) >= 12:
            break

    cleaned: Dict[str, Any] = {"plan": steps}
    if "risks" in payload:
        cleaned["risks"] = payload.get("risks")
    if "notes" in payload:
        cleaned["notes"] = payload.get("notes")
    return cleaned


_SANITIZERS: Dict[str, Sanitizer] = {
    "dialogue_state": _sanitize_dialogue_state,
    "episodic_linker": _sanitize_episodic_linker,
    "identity_mission": _sanitize_identity_mission,
    "goal_interpreter": _sanitize_goal_interpreter,
    "trigger_router": _sanitize_trigger_router,
    "planner_support": _sanitize_planner_support,
}


def enforce_llm_contract(
    spec_key: str, payload: Optional[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Return a cleaned mapping that complies with the runtime contract.

    ``None`` inputs are propagated as-is so callers can fall back to their
    heuristics.  For specs without a dedicated sanitizer we simply coerce the
    mapping into a ``dict`` to avoid leaking non-serialisable objects.
    """

    if not isinstance(payload, Mapping):
        return None
    sanitizer = _SANITIZERS.get(spec_key)
    if sanitizer is None:
        return dict(payload)
    try:
        return sanitizer(payload)
    except Exception:
        # Fail soft – if sanitisation goes wrong, allow the caller to fall
        # back to heuristics by signalling ``None``.
        return None


__all__ = ["enforce_llm_contract"]

