from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)

LOGGER = logging.getLogger(__name__)


class EventPipeline:
    """Pipeline asynchrone pour traiter les événements loggés.

    Les gestionnaires enregistrés reçoivent une copie de l'événement brut sans
    modifier le flux d'écriture principal.
    """

    def __init__(
        self,
        *,
        handlers: Optional[Sequence[Callable[[Dict[str, Any]], None]]] = None,
        max_queue_size: int = 2048,
    ) -> None:
        self._handlers: List[Callable[[Dict[str, Any]], None]] = list(handlers or [])
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="event-pipeline", daemon=True)
        self._thread.start()

    def add_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        self._handlers.append(handler)

    def submit(self, event: Dict[str, Any]) -> None:
        if self._stop_event.is_set():
            return
        payload = dict(event)
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            # Politique simple: on évacue l'élément le plus ancien pour conserver la cadence.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                # si le queue est encore full, abandonner silencieusement
                return

    def close(self, *, wait: bool = True) -> None:
        self._stop_event.set()
        if wait and self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                for handler in self._handlers:
                    try:
                        handler(dict(event))
                    except Exception:
                        # un handler ne doit pas faire tomber toute la pipeline
                        continue
            finally:
                self._queue.task_done()


class RollingMetricAggregator:
    """Agrégateur simple pour calculer statistiques cumulées sur des champs numériques."""

    def __init__(self, fields: Iterable[str]) -> None:
        self._fields = list(fields)
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, float]] = {
            name: {"count": 0.0, "sum": 0.0, "min": float("inf"), "max": float("-inf")}
            for name in self._fields
        }

    def __call__(self, event: Dict[str, Any]) -> None:
        with self._lock:
            for field in self._fields:
                value = event.get(field)
                if isinstance(value, (int, float)):
                    stats = self._stats[field]
                    stats["count"] += 1
                    stats["sum"] += float(value)
                    stats["min"] = min(stats["min"], float(value))
                    stats["max"] = max(stats["max"], float(value))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            result: Dict[str, Dict[str, float]] = {}
            for name, stats in self._stats.items():
                count = stats["count"]
                avg = stats["sum"] / count if count else 0.0
                result[name] = {
                    "count": count,
                    "sum": stats["sum"],
                    "min": stats["min"] if count else 0.0,
                    "max": stats["max"] if count else 0.0,
                    "avg": avg,
                }
            return result


class SnapshotDriftTracker:
    """Compare les snapshots successifs et journalise les dérives."""

    def __init__(self, log_path: str = "runtime/snapshot_drifts.jsonl") -> None:
        self.log_path = log_path
        directory = os.path.dirname(self.log_path) or "."
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None
        self._last_snapshot: Optional[Dict[str, Any]] = None

    def record(self, snapshot_path: str, data: Dict[str, Any]) -> None:
        sanitized = json_sanitize(data or {})
        payload = json.dumps(sanitized, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        with self._lock:
            drift_event: Optional[Dict[str, Any]] = None
            if self._last_hash is None:
                drift_event = {
                    "t": time.time(),
                    "snapshot": snapshot_path,
                    "initial": True,
                    "changed_keys": [],
                }
            elif digest != self._last_hash:
                changed = sorted(self._diff_keys(self._last_snapshot or {}, sanitized))
                drift_event = {
                    "t": time.time(),
                    "snapshot": snapshot_path,
                    "initial": False,
                    "changed_keys": changed,
                }

            self._last_hash = digest
            self._last_snapshot = sanitized

            if drift_event is None:
                return

            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(drift_event, ensure_ascii=False) + "\n")

    def _diff_keys(self, previous: Dict[str, Any], current: Dict[str, Any], prefix: str = "") -> List[str]:
        keys = set(previous.keys()) | set(current.keys())
        diffs: List[str] = []
        for key in keys:
            path = f"{prefix}.{key}" if prefix else key
            if key not in previous or key not in current:
                diffs.append(path)
                continue
            old_value = previous[key]
            new_value = current[key]
            if isinstance(old_value, dict) and isinstance(new_value, dict):
                diffs.extend(self._diff_keys(old_value, new_value, path))
            elif old_value != new_value:
                diffs.append(path)
        return diffs


class LLMAnalyticsInterpreter:
    """Batch events and request a semantic analytics summary from the local LLM."""

    def __init__(
        self,
        *,
        manager: Optional[Any] = None,
        enabled: Optional[bool] = None,
        batch_size: int = 24,
        flush_interval: float = 45.0,
        log_path: str = "runtime/llm_analytics.jsonl",
    ) -> None:
        self._provided_manager = manager
        self._manager: Optional[Any] = None
        self._enabled = is_llm_enabled() if enabled is None else bool(enabled)
        self.batch_size = max(1, int(batch_size))
        self.flush_interval = float(flush_interval)
        self.log_path = log_path
        directory = os.path.dirname(self.log_path) or "."
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = time.time()
        self._last_output: Optional[Mapping[str, Any]] = None

    @property
    def last_output(self) -> Optional[Mapping[str, Any]]:
        return self._last_output

    def __call__(self, event: Dict[str, Any]) -> None:
        if not self._enabled:
            return

        sanitized = json_sanitize(event or {})
        with self._lock:
            self._buffer.append(sanitized)
            now = time.time()
            should_flush = len(self._buffer) >= self.batch_size or (now - self._last_flush) >= self.flush_interval
            if not should_flush:
                return
            batch = list(self._buffer)
            self._buffer.clear()
            self._last_flush = now

        self._interpret_batch(batch)

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()
        self._interpret_batch(batch)

    def _get_manager(self) -> Optional[Any]:
        if self._manager is not None:
            return self._manager
        if self._provided_manager is not None:
            self._manager = self._provided_manager
            return self._manager
        try:
            self._manager = get_llm_manager()
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Unable to resolve LLM manager", exc_info=True)
            return None
        return self._manager

    def _interpret_batch(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        manager = self._get_manager()
        if manager is None or not self._enabled:
            return
        try:
            response = manager.call_dict("runtime_analytics", input_payload={"events": batch})
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM analytics interpretation failed", exc_info=True)
            return
        if not isinstance(response, Mapping):
            return

        record = {
            "t": time.time(),
            "events": batch,
            "analysis": response,
        }
        self._last_output = response
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_sanitize(record), ensure_ascii=False) + "\n")

