
from __future__ import annotations

"""
Unified MechanismStore for Mechanistic Actionable Insights (MAIs).

- Append-only JSONL event log for persistence (ops: add/update/retire/delete).
- Thread-safe cache with RLock.
- Backward compatible with legacy paths and ops.
- Rehydrates nested dataclasses (ImpactHypothesis, EvidenceRef) on load.
- Discovery helper: scan_applicable(..., include_status={'draft','active','ready'})

Environment variable override:
    MAI_STORE_PATH: absolute or relative path to the JSONL store.
Default path (if env var not set):
    data/runtime/mai_store.jsonl
If that file does not exist but legacy 'data/mai_store.jsonl' exists, it will use the legacy file.
"""

import getpass
import json
import logging
import os
import platform
import threading
import time
from collections import defaultdict
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional

# Import MAI and nested dataclasses. We only use types; persistence is via dicts.
from AGI_Evolutive.core.structures.mai import (
    EvidenceRef,
    ImpactHypothesis,
    MAI,
)
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

SCHEMA_VERSION = 1

# Default + legacy locations
_ENV_PATH = os.environ.get("MAI_STORE_PATH")
if _ENV_PATH:
    _DEFAULT_PATH = Path(_ENV_PATH).expanduser()
else:
    _DEFAULT_PATH = Path("data/runtime/mai_store.jsonl")

_LEGACY_PATH = Path("data/mai_store.jsonl")

LOGGER = logging.getLogger(__name__)
_LLM_SPEC_KEY = "knowledge_mechanism_screening"


class MechanismStore:
    """Append-only JSONL store for :class:`MAI` objects (thread-safe)."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        audit_context: Optional[Mapping[str, object]] = None,
    ):
        # Resolve path with env/legacy fallback
        if path is not None:
            self.path = Path(path)
        else:
            # If default doesn't exist but legacy does, use legacy
            self.path = _DEFAULT_PATH
            if (not self.path.exists()) and _LEGACY_PATH.exists():
                self.path = _LEGACY_PATH

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._cache: Dict[str, MAI] = {}
        self._audit_context = self._build_audit_context(audit_context)
        self._hooks: Dict[str, List[Callable[[Mapping[str, object]], None]]] = defaultdict(list)

        if self.path.exists():
            self._load_all()

    # ------------------------------------------------------------------
    # Public API
    def add(self, mai: MAI) -> None:
        """Insert a new MAI into the store (or replace if same id)."""
        with self._lock:
            self._cache[mai.id] = mai
            self._append("add", mai)
            self._fire_hooks("add", {"id": mai.id, "mai": mai})

    def update(self, mai: MAI) -> None:
        """Update/replace an existing MAI by id."""
        with self._lock:
            self._cache[mai.id] = mai
            self._append("update", mai)
            self._fire_hooks("update", {"id": mai.id, "mai": mai})

    def retire(self, mai_id: str, reason: Optional[str] = None) -> None:
        """Soft-delete a MAI (kept in the event log, removed from cache)."""
        with self._lock:
            # Log whatever we know about it for audit (if present in cache, serialize full MAI; else id only)
            payload: Dict[str, object]
            if mai_id in self._cache:
                payload = asdict(self._cache[mai_id])
                # record retirement reason if provided
                if reason:
                    payload.setdefault("meta", {})
                    if isinstance(payload["meta"], dict):
                        payload["meta"]["retire_reason"] = reason
            else:
                payload = {"id": mai_id}
                if reason:
                    payload["retire_reason"] = reason

            self._append("retire", payload)
            self._cache.pop(mai_id, None)
            self._fire_hooks("retire", {"id": mai_id, "reason": reason})

    def delete(self, mai_id: str) -> None:
        """Hard-delete from the in-memory index. Logged in the JSONL as 'delete'."""
        with self._lock:
            self._cache.pop(mai_id, None)
            self._append("delete", {"id": mai_id})
            self._fire_hooks("delete", {"id": mai_id})

    def get(self, mai_id: str) -> Optional[MAI]:
        with self._lock:
            return self._cache.get(mai_id)

    def all(self) -> Iterator[MAI]:
        with self._lock:
            # Return an iterator over a snapshot list to avoid concurrent modification issues
            return iter(list(self._cache.values()))

    def scan_applicable(
        self,
        state: Mapping[str, object],
        predicate_registry: Mapping[str, object],
        *,
        include_status: Optional[Iterable[str]] = None,
    ) -> List[MAI]:
        """Return MAIs whose preconditions hold in the given state.

        Args:
            state: The current world-state (symbols/features) available to preconditions.
            predicate_registry: Functions/predicates available to evaluate preconditions.
            include_status: Optional set of allowed MAI.status values; defaults to {'draft','active','ready'}.
        """
        allowed_status = set(include_status or {"draft", "active", "ready"})
        winners: List[MAI] = []
        with self._lock:
            for mai in self._cache.values():
                if getattr(mai, "status", "active") not in allowed_status:
                    continue
                try:
                    if mai.is_applicable(state, predicate_registry):
                        winners.append(mai)
                except Exception:
                    # Defensive: skip malformed MAIs rather than crashing
                    continue
        if not winners:
            return winners

        refined = self._llm_refine_applicable(winners, state, predicate_registry)
        if refined is not None:
            return refined

        return winners

    # ------------------------------------------------------------------
    # LLM-assisted prioritisation
    def _llm_refine_applicable(
        self,
        candidates: List[MAI],
        state: Mapping[str, object],
        predicate_registry: Mapping[str, object],
    ) -> Optional[List[MAI]]:
        payload = {
            "state": self._sanitize_for_json(dict(state)),
            "predicate_names": sorted(str(name) for name in predicate_registry.keys()),
            "candidates": [self._summarize_mai(mai) for mai in candidates],
        }

        response = try_call_llm_dict(
            _LLM_SPEC_KEY,
            input_payload=payload,
            logger=LOGGER,
        )
        if not response:
            return None

        decisions = response.get("decisions")
        if not isinstance(decisions, (list, tuple)):
            return None

        mapping = {mai.id: mai for mai in candidates}
        accepted_entries: List[tuple[int, Optional[float], str]] = []
        rejected_ids: set[str] = set()
        seen_accepts: set[str] = set()

        for index, entry in enumerate(decisions):
            if not isinstance(entry, Mapping):
                continue
            identifier_raw = entry.get("id") or entry.get("mai_id")
            if identifier_raw is None:
                continue
            identifier = str(identifier_raw).strip()
            if not identifier or identifier not in mapping:
                continue

            decision = str(entry.get("decision") or entry.get("status") or "").strip().lower()
            if decision in {"reject", "drop", "exclude"}:
                rejected_ids.add(identifier)
                continue
            if decision in {"accept", "select", "keep", "prioritize", "priorise"}:
                priority = self._safe_float(entry.get("priority"))
                if identifier not in seen_accepts:
                    accepted_entries.append((index, priority, identifier))
                    seen_accepts.add(identifier)
                continue
            if decision in {"defer", "abstain", "ignore"}:
                continue

        if not accepted_entries and not rejected_ids:
            return None

        ordered_ids: List[str] = []

        if accepted_entries:
            accepted_entries.sort(
                key=lambda item: (
                    item[1] if item[1] is not None else float("inf"),
                    item[0],
                )
            )
            for _, _priority, identifier in accepted_entries:
                ordered_ids.append(identifier)

        if rejected_ids:
            for identifier in list(ordered_ids):
                if identifier in rejected_ids:
                    ordered_ids.remove(identifier)

        for mai in candidates:
            if mai.id in rejected_ids:
                continue
            if mai.id in ordered_ids:
                continue
            ordered_ids.append(mai.id)

        if not ordered_ids:
            return []

        return [mapping[identifier] for identifier in ordered_ids]

    def _summarize_mai(self, mai: MAI) -> Dict[str, object]:
        tags: List[str]
        if isinstance(mai.tags, (list, tuple, set)):
            tags = [str(tag) for tag in mai.tags]
        elif mai.tags is None:
            tags = []
        else:
            tags = [str(mai.tags)]

        expected_impact = mai.expected_impact
        if expected_impact is None:
            expected_payload: object = None
        elif is_dataclass(expected_impact):
            expected_payload = asdict(expected_impact)
        elif isinstance(expected_impact, Mapping):
            expected_payload = dict(expected_impact)
        else:
            expected_payload = expected_impact

        preconditions_raw = mai.preconditions
        if isinstance(preconditions_raw, (list, tuple, set)):
            preconditions = list(preconditions_raw)
        elif preconditions_raw is None:
            preconditions = []
        else:
            preconditions = [preconditions_raw]

        summary: Dict[str, object] = {
            "id": mai.id,
            "title": mai.title,
            "summary": mai.summary,
            "status": mai.status,
            "tags": self._sanitize_for_json(tags, depth=1),
            "owner": mai.owner,
            "expected_impact": self._sanitize_for_json(expected_payload, depth=1),
            "metadata": self._sanitize_for_json(mai.metadata, depth=1),
            "preconditions": self._sanitize_for_json(preconditions, depth=1),
            "precondition_expr": self._sanitize_for_json(mai.precondition_expr, depth=1),
        }
        return summary

    def _sanitize_for_json(self, value: object, *, depth: int = 0, limit: int = 8) -> object:
        if depth > 3:
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            result: Dict[str, object] = {}
            for index, (key, sub_value) in enumerate(value.items()):
                if index >= limit:
                    break
                result[str(key)] = self._sanitize_for_json(sub_value, depth=depth + 1)
            return result
        if isinstance(value, (list, tuple, set)):
            sanitized_items = []
            for index, item in enumerate(value):
                if index >= limit:
                    break
                sanitized_items.append(self._sanitize_for_json(item, depth=depth + 1))
            return sanitized_items
        return str(value)

    def _safe_float(self, value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Extensibility / observability helpers
    def register_hook(self, event: str, callback: Callable[[Mapping[str, object]], None]) -> None:
        """Register a callback invoked with event payloads.

        Hooks receive a mapping containing at minimum the ``id`` of the MAI concerned.
        They execute within the store lock, so implementations should be fast/non-blocking.
        """

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._hooks[event].append(callback)

    def update_audit_context(self, extra: Mapping[str, object]) -> None:
        """Merge additional audit context that will be appended to future log records."""

        if not isinstance(extra, Mapping):
            raise TypeError("audit context must be a mapping")
        with self._lock:
            self._audit_context.update(extra)

    def annotate_metrics(
        self,
        mai_id: str,
        metrics: Mapping[str, float],
        *,
        freshness: Optional[float] = None,
        ttl: Optional[float] = None,
        extra: Optional[Mapping[str, object]] = None,
        history_limit: int = 50,
    ) -> None:
        """Annotate a MAI with external evaluation metrics without imposing aggregation logic."""

        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        cleaned_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                cleaned_metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        timestamp = time.time()
        with self._lock:
            mai = self._cache.get(mai_id)
            if not mai:
                raise KeyError(f"MAI '{mai_id}' not found")

            metadata = mai.metadata if isinstance(mai.metadata, dict) else {}
            metrics_block = metadata.setdefault("metrics", {})
            metrics_block["values"] = {**metrics_block.get("values", {}), **cleaned_metrics}
            metrics_block["updated_at"] = timestamp
            if freshness is not None:
                metrics_block["freshness"] = float(freshness)
            if ttl is not None:
                metrics_block["ttl"] = float(ttl)
            if extra and isinstance(extra, Mapping):
                extra_section = metrics_block.setdefault("extra", {})
                extra_section.update({str(k): v for k, v in extra.items()})

            history = metrics_block.setdefault("history", [])
            history.append(
                {
                    "time": timestamp,
                    "values": cleaned_metrics,
                    "freshness": freshness,
                    "ttl": ttl,
                }
            )
            if history_limit > 0 and len(history) > history_limit:
                del history[:-history_limit]

            mai.metadata = metadata
            mai.updated_at = timestamp
            self._cache[mai.id] = mai
            self._append("update", mai)
            self._fire_hooks(
                "annotate",
                {
                    "id": mai.id,
                    "metrics": cleaned_metrics,
                    "freshness": freshness,
                    "ttl": ttl,
                },
            )

    # ------------------------------------------------------------------
    # Persistence
    def _append(self, op: str, mai: Optional[MAI | Mapping[str, object]] = None) -> None:
        record: Dict[str, object] = {
            "schema": SCHEMA_VERSION,
            "op": op,
            "time": time.time(),
        }
        audit_snapshot = dict(self._audit_context)
        audit_snapshot["pid"] = os.getpid()
        record["audit"] = audit_snapshot
        if mai is not None:
            if isinstance(mai, Mapping):
                record["mai"] = dict(mai)
            else:
                record["mai"] = asdict(mai)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_all(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                op = rec.get("op")
                payload = rec.get("mai") or {}
                if payload is None:
                    payload = {}

                if op == "delete" and isinstance(payload, dict) and payload.get("id") and not payload.get("status") and not payload.get("preconditions"):
                    # Minimal delete record {id: ...}
                    self._cache.pop(payload.get("id"), None)
                    continue

                # Accept both full MAI dict and minimal {'id': ...}
                mai: Optional[MAI] = None
                if isinstance(payload, dict) and payload.get("id"):
                    try:
                        payload = self._validate_payload(payload)
                        mai = self._rehydrate_mai(payload)
                    except Exception:
                        mai = None

                if op in {"add", "update"} and mai is not None:
                    self._cache[mai.id] = mai
                elif op in {"retire", "delete"}:
                    mai_id = payload.get("id") or (mai.id if mai else None)
                    if isinstance(mai_id, str):
                        self._cache.pop(mai_id, None)

    # ------------------------------------------------------------------
    # Maintenance
    def snapshot(self) -> None:
        """Rewrite the store to the current in-memory state (best-effort compaction)."""

        with self._lock:
            tmp_path = self.path.with_suffix(".snapshot")
            with tmp_path.open("w", encoding="utf-8") as fh:
                for mai in self._cache.values():
                    record = {
                        "schema": SCHEMA_VERSION,
                        "op": "add",
                        "time": time.time(),
                        "audit": dict(self._audit_context),
                        "mai": asdict(mai),
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_path.replace(self.path)

    # ------------------------------------------------------------------
    # Rehydration helpers
    def _rehydrate_mai(self, payload: Mapping[str, object]) -> MAI:
        """Coerce a dict payload back into an MAI, rebuilding nested dataclasses."""
        data: Dict[str, object] = dict(payload)

        # Rehydrate expected_impact
        impact = data.get("expected_impact")
        if isinstance(impact, dict):
            data["expected_impact"] = ImpactHypothesis(**self._filter_fields(ImpactHypothesis, impact))

        # Rehydrate provenance_docs
        docs = data.get("provenance_docs") or []
        if isinstance(docs, list):
            data["provenance_docs"] = [
                d if isinstance(d, EvidenceRef) else EvidenceRef(**self._filter_fields(EvidenceRef, d))
                for d in docs
                if isinstance(d, (dict, EvidenceRef))
            ]

        # Filter to MAI dataclass fields to avoid constructor errors
        return MAI(**self._filter_fields(MAI, data))

    @staticmethod
    def _filter_fields(cls, payload: Mapping[str, object]) -> Dict[str, object]:
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in payload.items() if k in allowed}

    def _build_audit_context(
        self, audit_context: Optional[Mapping[str, object]]
    ) -> Dict[str, object]:
        default_context: Dict[str, object] = {
            "user": self._safe_getpass(),
            "host": platform.node() or "unknown-host",
            "runtime_version": os.environ.get("AGI_RUNTIME_VERSION", "unknown"),
        }
        if audit_context:
            default_context.update({str(k): v for k, v in audit_context.items()})
        return default_context

    @staticmethod
    def _safe_getpass() -> str:
        try:
            return getpass.getuser()
        except Exception:
            return os.environ.get("USER", "unknown")

    def _fire_hooks(self, event: str, payload: Mapping[str, object]) -> None:
        callbacks = list(self._hooks.get(event, ()))
        if not callbacks:
            return
        for cb in callbacks:
            try:
                cb(payload)
            except Exception:
                continue

    def _validate_payload(self, payload: Mapping[str, object]) -> Dict[str, object]:
        """Lightweight schema validation to guard against corrupted records."""

        data = dict(payload)

        if not isinstance(data.get("id"), str) or not data["id"].strip():
            raise ValueError("MAI payload missing valid 'id'")

        for key in ("tags", "provenance_docs", "provenance_episodes", "safety_invariants"):
            value = data.get(key)
            if value is None:
                data[key] = []
            elif not isinstance(value, list):
                data[key] = [value]

        metadata = data.get("metadata")
        if metadata is None:
            data["metadata"] = {}
        elif not isinstance(metadata, Mapping):
            data["metadata"] = {"_coerced_from": type(metadata).__name__}

        runtime_counters = data.get("runtime_counters")
        if runtime_counters is None or not isinstance(runtime_counters, Mapping):
            data["runtime_counters"] = {
                "activation": 0.0,
                "wins": 0.0,
                "benefit": 0.0,
                "regret": 0.0,
                "rollbacks": 0.0,
            }

        for ts_field in ("created_at", "updated_at"):
            value = data.get(ts_field)
            if value is not None:
                try:
                    data[ts_field] = float(value)
                except (TypeError, ValueError):
                    data.pop(ts_field, None)

        return data


__all__ = ["MAI", "ImpactHypothesis", "EvidenceRef", "MechanismStore"]
