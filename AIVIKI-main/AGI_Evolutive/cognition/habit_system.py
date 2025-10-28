"""Context-cued habit routines with lightweight scheduling logic."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

__all__ = ["HabitSystem", "HabitRoutine"]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _parse_time_of_day(value: str) -> dtime:
    parts = str(value or "").strip().split(":")
    try:
        hour = int(parts[0]) if parts and parts[0] else 0
        minute = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        second = int(parts[2]) if len(parts) > 2 and parts[2] else 0
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid time specification: {value!r}") from exc
    if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
        raise ValueError(f"Time out of range: {value!r}")
    return dtime(hour=hour, minute=minute, second=second)


@dataclass
class HabitRoutine:
    """Definition of a habit routine.

    The schedule dictionary accepts the following fields:

    ``kind`` (str)
        One of ``daily``, ``interval`` or ``deadline``. Defaults to ``interval``.
    ``at`` (str)
        Time-of-day string (``HH:MM`` or ``HH:MM:SS``) used for the ``daily`` kind.
    ``lead_time_sec`` (float)
        How long before the anchor the cue becomes eligible.
    ``window_sec`` (float)
        Grace period after the anchor before the routine is considered overdue.
    ``every_sec`` / ``every_hours`` (float)
        Interval expressed in seconds or hours for the ``interval`` kind.
    ``start_after_sec`` (float)
        Optional activation delay applied on top of the creation timestamp.
    ``habit_strength`` (float)
        Base strength emitted when the routine is due.
    ``due_ts`` / ``due_in_days`` / ``due_in_sec`` (float)
        Deadline configuration for the ``deadline`` kind.
    ``reminder_every_sec`` (float)
        Minimum delay between successive reminders for deadline routines.
    """

    name: str
    description: str
    schedule: Dict[str, Any]
    steps: Sequence[Dict[str, Any]] = field(default_factory=list)
    tags: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": list(self.steps),
            "tags": list(self.tags),
            "schedule": dict(self.schedule),
            "metadata": dict(self.metadata),
        }

    # ------------------------------------------------------------------
    # Scheduling helpers
    def evaluate(self, now: float, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        schedule = self.schedule
        if not schedule:
            return None

        created_at = float(state.get("created_at", self.metadata.get("created_at", now)))
        activation_delay = float(schedule.get("start_after_sec") or 0.0)
        if activation_delay > 0 and now < created_at + activation_delay:
            return None

        kind = str(schedule.get("kind", "interval")).lower()
        if kind == "daily":
            return self._evaluate_daily(now, state)
        if kind == "deadline":
            return self._evaluate_deadline(now, state)
        return self._evaluate_interval(now, state)

    def _evaluate_daily(self, now: float, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            anchor = _parse_time_of_day(self.schedule.get("at", "21:00"))
        except ValueError:
            return None

        lead = float(self.schedule.get("lead_time_sec") or 0.0)
        window = float(self.schedule.get("window_sec") or 3600.0)
        dt_now = datetime.fromtimestamp(now)
        anchor_dt = datetime.combine(dt_now.date(), anchor)
        due_ts = anchor_dt.timestamp()

        period_key = anchor_dt.strftime("%Y-%m-%d")
        last_period = state.get("last_period")
        if last_period == period_key:
            return None

        if now < due_ts - lead:
            return None

        status = "due" if now <= due_ts + window else "overdue"
        base_strength = float(self.schedule.get("habit_strength") or 0.7)
        strength = _clamp(base_strength)
        if status == "overdue":
            overdue_ratio = min(1.0, (now - due_ts) / max(window, 1.0))
            strength = _clamp(strength + 0.15 * overdue_ratio)

        return {
            "period": period_key,
            "due_ts": due_ts,
            "status": status,
            "strength": strength,
        }

    def _evaluate_interval(self, now: float, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        every_sec = self.schedule.get("every_sec")
        if every_sec is None:
            hours = self.schedule.get("every_hours")
            if hours is None:
                return None
            every_sec = float(hours) * 3600.0
        every = max(60.0, float(every_sec))

        last_completed = float(state.get("last_completed") or 0.0)
        last_triggered = float(state.get("last_triggered") or 0.0)
        last_reference = max(last_completed, last_triggered, state.get("created_at", now))
        if now - last_reference < every:
            return None

        period_index = int((now - state.get("created_at", now)) // every)
        base_strength = float(self.schedule.get("habit_strength") or 0.6)
        return {
            "period": f"interval:{period_index}",
            "due_ts": now,
            "status": "due",
            "strength": _clamp(base_strength),
        }

    def _evaluate_deadline(self, now: float, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        schedule = self.schedule
        due_ts = schedule.get("due_ts")
        if due_ts is None:
            due_ts = self._resolve_due_ts(state)
            if due_ts is None:
                return None
            schedule["due_ts"] = due_ts
        else:
            due_ts = float(due_ts)

        if state.get("completed_ts") and not schedule.get("repeat", False):
            return None

        reminder_interval = float(schedule.get("reminder_every_sec") or 3600.0)
        last_triggered = float(state.get("last_triggered") or 0.0)
        if last_triggered and now - last_triggered < reminder_interval:
            return None

        status = "due" if now <= due_ts else "overdue"
        base_strength = float(schedule.get("habit_strength") or 0.75)
        strength = _clamp(base_strength)
        if status == "overdue":
            overdue_hours = max(0.0, now - due_ts) / 3600.0
            strength = _clamp(strength + min(0.25, 0.05 * overdue_hours))

        return {
            "period": f"deadline:{int(due_ts)}",
            "due_ts": due_ts,
            "status": status,
            "strength": strength,
        }

    def _resolve_due_ts(self, state: Dict[str, Any]) -> Optional[float]:
        schedule = self.schedule
        now = time.time()
        if "due_in_sec" in schedule:
            return float(state.get("created_at", self.metadata.get("created_at", now))) + float(
                schedule["due_in_sec"]
            )
        if "due_in_days" in schedule:
            days = float(schedule["due_in_days"])
            return float(state.get("created_at", self.metadata.get("created_at", now))) + days * 86400.0
        if "due_at" in schedule:
            try:
                return datetime.fromisoformat(str(schedule["due_at"]).replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        if "due_ts" in schedule:
            return float(schedule["due_ts"])
        return None


class HabitSystem:
    """Stateful habit manager that surfaces context cues when routines are due."""

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        state_path: str = "data/habit_state.json",
    ) -> None:
        self._config_path = config_path
        self._state_path = state_path
        self._routines: Dict[str, HabitRoutine] = {}
        self._state: Dict[str, Dict[str, Any]] = {}
        self._load_state()
        if config_path:
            self._load_config(config_path)

    # ------------------------------------------------------------------
    # LLM integration helpers
    def _llm_refine_cue(
        self,
        name: str,
        payload: Dict[str, Any],
        state_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        llm_payload = {
            "routine": {
                "name": name,
                "schedule": payload.get("schedule"),
                "tags": payload.get("tags", []),
                "evaluation": {
                    "status": payload.get("status"),
                    "strength": payload.get("strength"),
                    "due_ts": payload.get("due_ts"),
                    "period": payload.get("period"),
                },
                "steps": payload.get("steps"),
            },
            "state": {
                "created_at": state_snapshot.get("created_at"),
                "last_triggered": state_snapshot.get("last_triggered"),
                "last_completed": state_snapshot.get("last_completed"),
                "completed_ts": state_snapshot.get("completed_ts"),
            },
            "now": payload.get("triggered_at"),
        }
        logger = getattr(self, "_logger", None)
        llm_response = try_call_llm_dict(
            "cognition_habit_system",
            input_payload=llm_payload,
            logger=logger,
        )
        if not isinstance(llm_response, dict):
            return payload

        refined = dict(payload)
        if "strength" in llm_response:
            try:
                refined["strength"] = _clamp(float(llm_response["strength"]))
            except (TypeError, ValueError):
                pass
        if isinstance(llm_response.get("status"), str) and llm_response["status"].strip():
            refined["status"] = llm_response["status"].strip()
        if isinstance(llm_response.get("message"), str) and llm_response["message"].strip():
            refined["llm_message"] = llm_response["message"].strip()
        if isinstance(llm_response.get("notes"), str) and llm_response["notes"].strip():
            refined["llm_notes"] = llm_response["notes"].strip()
        if "confidence" in llm_response:
            try:
                refined["llm_confidence"] = _clamp(float(llm_response["confidence"]))
            except (TypeError, ValueError):
                pass
        if isinstance(llm_response.get("tags"), list):
            refined["tags"] = list({*refined.get("tags", []), *llm_response["tags"]})
        refined["llm"] = llm_response
        return refined

    # ------------------------------------------------------------------
    # Public API
    def register_routine(
        self,
        name: str,
        description: str,
        *,
        schedule: Optional[Dict[str, Any]] = None,
        steps: Optional[Sequence[Dict[str, Any]]] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HabitRoutine:
        schedule = dict(schedule or {})
        now = time.time()
        metadata = dict(metadata or {})
        metadata.setdefault("created_at", now)

        routine = HabitRoutine(
            name=name,
            description=description,
            schedule=schedule,
            steps=list(steps or []),
            tags=list(tags or []),
            metadata=metadata,
        )
        self._routines[name] = routine
        state_entry = self._state.setdefault(name, {"created_at": metadata["created_at"]})
        state_entry.setdefault("created_at", metadata["created_at"])
        self._persist_state()
        return routine

    def remove_routine(self, name: str) -> None:
        self._routines.pop(name, None)
        if name in self._state:
            del self._state[name]
            self._persist_state()

    def list_routines(self) -> List[str]:
        return list(self._routines.keys())

    def poll_context_cue(self, *, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        now = float(now or time.time())
        best_payload: Optional[Dict[str, Any]] = None
        best_name: Optional[str] = None
        for name, routine in self._routines.items():
            state_entry = self._state.setdefault(
                name, {"created_at": routine.metadata.get("created_at", now)}
            )
            evaluation = routine.evaluate(now, state_entry)
            if not evaluation:
                continue
            payload = routine.to_payload()
            payload.update(evaluation)
            payload["triggered_at"] = now
            if best_payload is None or payload["due_ts"] <= best_payload["due_ts"]:
                best_payload = payload
                best_name = name
        if best_payload and best_name:
            entry = self._state.setdefault(
                best_name, {"created_at": self._routines[best_name].metadata.get("created_at", now)}
            )
            snapshot = dict(entry)
            entry["last_triggered"] = now
            if "period" in best_payload:
                entry["last_period"] = best_payload["period"]
            best_payload = self._llm_refine_cue(best_name, best_payload, snapshot)
            self._persist_state()
            return best_payload
        return None

    def record_completion(self, name: str, *, when: Optional[float] = None) -> None:
        if name not in self._routines:
            return
        entry = self._state.setdefault(
            name, {"created_at": self._routines[name].metadata.get("created_at", time.time())}
        )
        ts = float(when or time.time())
        entry["last_completed"] = ts
        entry["completed_ts"] = ts
        self._persist_state()

    # ------------------------------------------------------------------
    # Persistence helpers
    def _load_state(self) -> None:
        path = self._state_path
        if not path or not os.path.exists(path):
            self._state = {}
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._state = {
                    str(k): (v if isinstance(v, dict) else {})
                    for k, v in data.items()
                }
            else:
                self._state = {}
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corruption fallback
            self._state = {}

    def _persist_state(self) -> None:
        path = self._state_path
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass  # pragma: no cover - IO errors are ignored but do not break execution

    def _load_config(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError):  # pragma: no cover - invalid config is ignored
            return
        routines = config.get("routines") if isinstance(config, dict) else None
        if not isinstance(routines, list):
            return
        for routine in routines:
            if not isinstance(routine, dict):
                continue
            name = routine.get("name")
            description = routine.get("description")
            if not (name and description):
                continue
            schedule = routine.get("schedule") if isinstance(routine.get("schedule"), dict) else {}
            steps = routine.get("steps") if isinstance(routine.get("steps"), list) else []
            tags = routine.get("tags") if isinstance(routine.get("tags"), list) else []
            metadata = routine.get("metadata") if isinstance(routine.get("metadata"), dict) else {}
            self.register_routine(
                name,
                description,
                schedule=schedule,
                steps=steps,
                tags=tags,
                metadata=metadata,
            )
