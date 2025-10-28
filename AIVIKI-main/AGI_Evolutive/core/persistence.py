
# core/persistence.py
"""Persistence helpers for evolutionary state management.

Cette version étend le `PersistenceManager` originel avec plusieurs capacités :
- versionnage/migrations des snapshots,
- backends de stockage interchangeables,
- journalisation incrémentale des drifts,
- autosave adaptatif déclenché sur dérive significative,
- instrumentation riche avec alertes optionnelles.
"""
import hashlib
import inspect
import json
import logging
import os
import pickle
import time
import types
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".agi_state")
DEFAULT_DIR = os.path.abspath(DEFAULT_DIR)
DEFAULT_FILE = os.path.join(DEFAULT_DIR, "snapshot.pkl")


class StorageBackend:
    """Abstract storage backend for snapshots and history artifacts."""

    def write_bytes(self, path: str, payload: bytes, *, atomic: bool = False) -> None:
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def list_files(self, path: str, suffix: Optional[str] = None) -> Iterable[str]:
        raise NotImplementedError

    def remove(self, path: str) -> None:
        raise NotImplementedError

    def ensure_dir(self, path: str) -> None:
        raise NotImplementedError

    def append_text(self, path: str, text: str) -> None:
        raise NotImplementedError

    def resolve(self, path: str) -> str:
        raise NotImplementedError


class FileStorageBackend(StorageBackend):
    """Simple filesystem backend with atomic writes."""

    def __init__(self, root: Optional[str] = None):
        self.root = os.path.abspath(root or DEFAULT_DIR)
        os.makedirs(self.root, exist_ok=True)

    def resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.root, path)

    def ensure_dir(self, path: str) -> None:
        os.makedirs(self.resolve(path), exist_ok=True)

    def write_bytes(self, path: str, payload: bytes, *, atomic: bool = False) -> None:
        full_path = self.resolve(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if atomic:
            tmpfile = f"{full_path}.tmp"
            with open(tmpfile, "wb") as fh:
                fh.write(payload)
            os.replace(tmpfile, full_path)
        else:
            with open(full_path, "wb") as fh:
                fh.write(payload)

    def read_bytes(self, path: str) -> bytes:
        full_path = self.resolve(path)
        with open(full_path, "rb") as fh:
            return fh.read()

    def exists(self, path: str) -> bool:
        return os.path.exists(self.resolve(path))

    def list_files(self, path: str, suffix: Optional[str] = None) -> Iterable[str]:
        full_path = self.resolve(path)
        try:
            for entry in os.listdir(full_path):
                if suffix and not entry.endswith(suffix):
                    continue
                yield entry
        except Exception:
            return

    def remove(self, path: str) -> None:
        full_path = self.resolve(path)
        if os.path.exists(full_path):
            os.remove(full_path)

    def append_text(self, path: str, text: str) -> None:
        full_path = self.resolve(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "a", encoding="utf-8") as fh:
            fh.write(text)


MigrationCallable = Callable[[Dict[str, Any]], Dict[str, Any]]
_MIGRATIONS: Dict[int, Tuple[int, MigrationCallable]] = {}


def register_migration(from_version: int, to_version: int) -> Callable[[MigrationCallable], MigrationCallable]:
    """Register a migration that upgrades a snapshot to a newer version."""

    def decorator(func: MigrationCallable) -> MigrationCallable:
        _MIGRATIONS[from_version] = (to_version, func)
        return func

    return decorator

def _is_picklable(x):
    try:
        pickle.dumps(x)
        return True
    except Exception:
        return False

def _to_state(obj):
    """
    Essaie d'extraire un dict sérialisable depuis 'obj'.
    - Si l'objet expose .to_state(), on l'utilise.
    - Sinon, on tente __dict__ en filtrant les valeurs non picklables.
    """
    if hasattr(obj, "to_state") and callable(getattr(obj, "to_state")):
        try:
            state = obj.to_state()
            if _is_picklable(state):
                return state
        except Exception:
            pass
    d = {}
    # fallback sur __dict__ si dispo
    src = getattr(obj, "__dict__", {})
    for k, v in src.items():
        # ignorer méthodes, modules, fonctions, générateurs et coroutines
        if isinstance(v, (types.ModuleType, types.FunctionType, types.GeneratorType)):
            continue
        if inspect.isroutine(v) or inspect.isclass(v):
            continue
        if _is_picklable(v):
            d[k] = v
        else:
            d[k] = f"<non_picklable:{type(v).__name__}>"
    return d

def _from_state(obj, state: Dict[str, Any]):
    """
    Restaure un état simple dans l'objet (best-effort).
    - Si l'objet expose .from_state(state), on l'utilise.
    - Sinon on met à jour __dict__ avec les clés existantes uniquement.
    """
    if hasattr(obj, "from_state") and callable(getattr(obj, "from_state")):
        try:
            obj.from_state(state)
            return
        except Exception:
            pass
    if not hasattr(obj, "__dict__"):
        return
    for k, v in state.items():
        # Les valeurs marquées comme non sérialisables servent uniquement de trace dans le
        # snapshot. Il ne faut pas écraser l'objet original (ex: moteurs, connexions)
        # avec cette chaîne sentinelle lors du chargement.
        if isinstance(v, str) and v.startswith("<non_picklable:") and v.endswith(">"):
            continue
        try:
            setattr(obj, k, v)
        except Exception:
            pass

class PersistenceManager:
    def __init__(
        self,
        arch,
        directory: str = DEFAULT_DIR,
        filename: str = DEFAULT_FILE,
        *,
        schema_version: int = 1,
        backend: Optional[StorageBackend] = None,
        autosave_interval: float = 60.0,
        autosave_min_interval: float = 10.0,
        autosave_drift_threshold: float = 0.5,
        alert_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.arch = arch
        self.directory = os.path.abspath(directory)
        self.backend = backend or FileStorageBackend(self.directory)
        self.filename = filename if os.path.isabs(filename) else os.path.join(self.directory, filename)
        self.schema_version = int(schema_version)
        self.autosave_interval = float(max(autosave_interval, 1.0))
        self.autosave_min_interval = float(max(autosave_min_interval, 0.5))
        self.autosave_drift_threshold = float(max(autosave_drift_threshold, 0.0))
        self.alert_hook = alert_hook
        self.logger = logging.getLogger(__name__)
        self._last_save = time.time()
        self.backend.ensure_dir(self.directory)
        self.history_dir = os.path.join(self.directory, "history")
        self.backend.ensure_dir(self.history_dir)
        self.history_retention = 20
        self.journal_path = os.path.join(self.history_dir, "events.log")
        self._last_snapshot_meta: Optional[Dict[str, Any]] = None
        self._last_snapshot_hash: Optional[str] = None
        self._last_drift: Optional[Dict[str, Any]] = None
        self._pending_snapshot_cache: Optional[Tuple[Dict[str, Any], bytes, Dict[str, Any]]] = None
        self._pending_snapshot_time: float = 0.0
        self.snapshot_cache_ttl = 2.0

    def make_snapshot(self) -> Dict[str, Any]:
        subs = [
            "memory","perception","reasoning","goals","emotions",
            "learning","metacognition","creativity","world_model","language"
        ]
        snap = {"timestamp": time.time(), "version": self.schema_version}
        for name in subs:
            comp = getattr(self.arch, name, None)
            if comp is None:
                snap[name] = None
            else:
                snap[name] = _to_state(comp)
        return snap
    
    def save(self):
        start = time.perf_counter()
        cache = self._consume_snapshot_cache()
        if cache is None:
            snap = self.make_snapshot()
            payload = pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL)
            digest = hashlib.sha256(payload).hexdigest()
            summary = self._summarize_snapshot(snap, digest)
        else:
            snap, payload, summary = cache
            digest = summary.get("hash") or hashlib.sha256(payload).hexdigest()
        try:
            self.backend.write_bytes(self.filename, payload, atomic=True)
        except Exception as exc:
            self._alert("snapshot_write_failed", {"error": repr(exc), "path": self.filename})
            raise
        self._last_save = time.time()
        prev_summary = self._last_snapshot_meta
        self._last_drift = self._compute_drift(prev_summary, summary)
        self._last_snapshot_meta = summary
        previous_hash = self._last_snapshot_hash
        self._last_snapshot_hash = digest
        if digest != previous_hash:
            self._record_history(payload, summary)
            self._append_journal(
                "snapshot",
                digest=digest,
                severity=self._last_drift.get("severity", 0.0) if self._last_drift else 0.0,
                components=list(summary.get("components", {}).keys()),
            )
            self._prune_history()
        duration = time.perf_counter() - start
        self.logger.info("Snapshot saved", extra={"digest": digest, "duration": duration})
        return self.filename

    def load(self) -> bool:
        if not self.backend.exists(self.filename):
            return False
        try:
            payload = self.backend.read_bytes(self.filename)
            snap = pickle.loads(payload)
            snap = self._apply_migrations(snap)
            for name, state in snap.items():
                if name in ("timestamp", "version"):
                    continue
                comp = getattr(self.arch, name, None)
                if comp is not None and isinstance(state, dict):
                    _from_state(comp, state)
            digest = hashlib.sha256(payload).hexdigest()
            self._last_snapshot_meta = self._summarize_snapshot(snap, digest)
            self._last_snapshot_hash = digest
            self._last_drift = None
            self._append_journal("load", digest=digest, version=snap.get("version"))
            self.logger.info("Snapshot loaded", extra={"digest": digest})
            return True
        except Exception as exc:
            self._alert("snapshot_load_failed", {"error": repr(exc), "path": self.filename})
            return False

    def autosave_tick(self):
        now = time.time()
        elapsed = now - self._last_save
        if elapsed < self.autosave_min_interval:
            return
        if elapsed >= self.autosave_interval:
            self.save()
            return
        severity = self._estimate_drift_severity()
        if severity >= self.autosave_drift_threshold:
            self.logger.debug(
                "Autosave triggered on drift",
                extra={"severity": severity, "threshold": self.autosave_drift_threshold},
            )
            self.save()

    def save_on_exit(self):
        try:
            self.save()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Snapshot orchestration helpers
    def _consume_snapshot_cache(self) -> Optional[Tuple[Dict[str, Any], bytes, Dict[str, Any]]]:
        if not self._pending_snapshot_cache:
            return None
        if (time.time() - self._pending_snapshot_time) > self.snapshot_cache_ttl:
            self._pending_snapshot_cache = None
            return None
        cache = self._pending_snapshot_cache
        self._pending_snapshot_cache = None
        return cache

    def _estimate_drift_severity(self) -> float:
        try:
            snap = self.make_snapshot()
            payload = pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL)
            digest = hashlib.sha256(payload).hexdigest()
            summary = self._summarize_snapshot(snap, digest)
        except Exception as exc:
            self._alert("snapshot_probe_failed", {"error": repr(exc)})
            return 0.0
        prev_summary = self._last_snapshot_meta
        drift = self._compute_drift(prev_summary, summary)
        self._pending_snapshot_cache = (snap, payload, summary)
        self._pending_snapshot_time = time.time()
        return float(drift.get("severity", 0.0))

    def _apply_migrations(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        current_version = int(snap.get("version", 1) or 1)
        target = self.schema_version
        if current_version == target:
            return snap
        visited = set()
        while current_version != target:
            if current_version in visited:
                raise RuntimeError("Cyclic snapshot migrations detected")
            visited.add(current_version)
            migration = _MIGRATIONS.get(current_version)
            if migration is None:
                self._alert(
                    "migration_missing",
                    {"from_version": current_version, "to_version": target},
                )
                break
            next_version, func = migration
            try:
                snap = func(snap)
            except Exception as exc:
                self._alert(
                    "migration_failed",
                    {"from_version": current_version, "to_version": next_version, "error": repr(exc)},
                )
                raise
            snap["version"] = next_version
            current_version = next_version
        return snap

    def _alert(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        payload = context or {}
        if self.alert_hook:
            try:
                self.alert_hook(message, payload)
            except Exception as exc:
                extra = {"message": message, **payload}
                self.logger.warning("Alert hook failure: %s", exc, extra=extra)
        else:
            self.logger.warning("Persistence alert", extra={"message": message, **payload})

    # ------------------------------------------------------------------
    # History & drift reporting
    def _summarize_snapshot(self, snap: Dict[str, Any], digest: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "timestamp": snap.get("timestamp", time.time()),
            "version": snap.get("version"),
            "hash": digest,
            "components": {},
        }
        components: Dict[str, Any] = {}
        for name, payload in snap.items():
            if name in ("timestamp", "version"):
                continue
            if isinstance(payload, dict):
                components[name] = {
                    "size": len(payload),
                    "keys_sample": sorted(list(payload.keys()))[:8],
                }
            else:
                components[name] = {"size": 0, "type": type(payload).__name__}
        summary["components"] = components
        return summary

    def _compute_drift(
        self,
        prev: Optional[Dict[str, Any]],
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not prev:
            components = current.get("components", {})
            return {
                "severity": 1.0 if components else 0.0,
                "new": sorted(components.keys()),
                "removed": [],
                "deltas": {name: data.get("size", 0) for name, data in components.items()},
            }
        prev_components = prev.get("components", {})
        curr_components = current.get("components", {})
        deltas: Dict[str, int] = {}
        for name, info in curr_components.items():
            prev_size = int(prev_components.get(name, {}).get("size", 0))
            curr_size = int(info.get("size", 0))
            delta = curr_size - prev_size
            if delta:
                deltas[name] = delta
        removed = [name for name in prev_components.keys() if name not in curr_components]
        new = [name for name in curr_components.keys() if name not in prev_components]
        if not deltas and not new and not removed:
            severity = 0.0
        else:
            norm = max(len(curr_components) or 1, 1)
            severity = sum(abs(v) for v in deltas.values()) / norm
            if new or removed:
                severity = min(1.0, severity + 0.2)
        return {
            "severity": round(float(severity), 4),
            "new": sorted(new),
            "removed": sorted(removed),
            "deltas": deltas,
        }

    def _record_history(self, payload: bytes, summary: Dict[str, Any]) -> None:
        ts = summary.get("timestamp", time.time())
        try:
            stamp = datetime.utcfromtimestamp(float(ts)).strftime("%Y%m%dT%H%M%SZ")
        except Exception:
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        digest = summary.get("hash", "snapshot")
        base = f"{stamp}_{digest[:12]}"
        data_path = os.path.join(self.history_dir, f"{base}.pkl")
        meta_path = os.path.join(self.history_dir, f"{base}.json")
        try:
            self.backend.write_bytes(data_path, payload, atomic=True)
        except Exception as exc:
            self._alert("history_write_failed", {"error": repr(exc), "path": data_path})
            return
        meta_payload = dict(summary)
        meta_payload["path"] = os.path.relpath(data_path, self.directory)
        try:
            self.backend.write_bytes(meta_path, json.dumps(meta_payload, indent=2, sort_keys=True).encode("utf-8"))
        except Exception as exc:
            self._alert("history_meta_failed", {"error": repr(exc), "path": meta_path})

    def _prune_history(self) -> None:
        try:
            entries = [
                f
                for f in self.backend.list_files(self.history_dir, suffix=".json")
                if self.backend.exists(os.path.join(self.history_dir, f))
            ]
        except Exception as exc:
            self._alert("history_list_failed", {"error": repr(exc)})
            return
        if len(entries) <= self.history_retention:
            return
        entries.sort()
        obsolete = entries[: len(entries) - self.history_retention]
        for meta_name in obsolete:
            base, _ = os.path.splitext(meta_name)
            for suffix in (".json", ".pkl"):
                path = os.path.join(self.history_dir, f"{base}{suffix}")
                try:
                    if self.backend.exists(path):
                        self.backend.remove(path)
                except Exception as exc:
                    self._alert("history_prune_failed", {"error": repr(exc), "path": path})

    def _append_journal(self, event_type: str, **data: Any) -> None:
        event = {
            "timestamp": time.time(),
            "type": event_type,
            **data,
        }
        try:
            self.backend.append_text(self.journal_path, json.dumps(event, sort_keys=True) + "\n")
        except Exception as exc:
            self._alert("journal_append_failed", {"error": repr(exc)})

    def get_last_snapshot_metadata(self) -> Dict[str, Any]:
        """Return metadata describing the latest persisted snapshot."""

        return dict(self._last_snapshot_meta or {})

    def get_last_drift(self) -> Dict[str, Any]:
        """Return the most recent drift analysis between snapshots."""

        return dict(self._last_drift or {})

    def generate_health_report(self) -> MutableMapping[str, Any]:
        """Produce a diagnostic summary for observability dashboards."""

        status = {
            "schema_version": self.schema_version,
            "seconds_since_last_save": max(0.0, time.time() - self._last_save),
            "last_snapshot": self.get_last_snapshot_metadata(),
            "last_drift": self.get_last_drift(),
        }
        heuristic = self._fallback_health(status)

        response = try_call_llm_dict(
            "persistence_healthcheck",
            input_payload=json_sanitize(status),
            logger=self.logger,
            max_retries=2,
        )
        if not isinstance(response, MutableMapping):
            return heuristic

        summary = self._clean_text(response.get("summary")) or heuristic["summary"]
        alerts = self._clean_list(response.get("alerts"))
        recommendations = self._clean_list(response.get("recommended_actions"))
        notes = self._clean_text(response.get("notes")) or heuristic["notes"]
        confidence = heuristic.get("confidence", 0.6)
        try:
            confidence = float(response.get("confidence", confidence))
        except Exception:
            confidence = float(confidence)

        enriched = dict(heuristic)
        enriched.update(
            {
                "source": "llm",
                "summary": summary,
                "alerts": alerts,
                "recommended_actions": recommendations or heuristic["recommended_actions"],
                "confidence": max(0.0, min(1.0, confidence)),
                "notes": notes,
            }
        )
        return enriched

    def _fallback_health(self, status: Mapping[str, Any]) -> MutableMapping[str, Any]:
        last_drift = status.get("last_drift", {})
        severity = float(last_drift.get("severity", 0.0) or 0.0)
        seconds_since_save = float(status.get("seconds_since_last_save", 0.0) or 0.0)
        alerts: list[str] = []
        recommendations: list[str] = []
        if severity >= 0.5:
            alerts.append("Dérive importante détectée sur le dernier snapshot.")
            recommendations.append("Planifier un audit des composantes les plus impactées.")
        if seconds_since_save > max(self.autosave_interval * 3, 900):
            alerts.append("Dernière sauvegarde trop ancienne.")
            recommendations.append("Déclencher une sauvegarde manuelle ou ajuster l'autosave.")
        if not alerts:
            recommendations.append("Continuer la surveillance standard des snapshots.")
        summary = "Persistance stable." if not alerts else "Persistance à surveiller."
        return {
            "source": "heuristic",
            "summary": summary,
            "alerts": alerts,
            "recommended_actions": recommendations,
            "confidence": 0.58 if alerts else 0.72,
            "notes": "Synthèse heuristique basée sur la dernière dérive et le rythme de sauvegarde.",
        }

    @staticmethod
    def _clean_text(value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        return ""

    @staticmethod
    def _clean_list(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        results: list[str] = []
        for item in value:
            text = PersistenceManager._clean_text(item)
            if text:
                results.append(text)
        return results
