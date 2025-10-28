from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from .analytics import EventPipeline, SnapshotDriftTracker
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

if TYPE_CHECKING:
    from AGI_Evolutive.core.persistence import PersistenceManager


logger = logging.getLogger(__name__)

class JSONLLogger:
    """
    Logger JSONL thread-safe des événements agent.
    Ecrit dans runtime/agent_events.jsonl + snapshots optionnels.

    L'instance peut enrichir automatiquement les événements grâce à un
    ``metadata_provider`` et les rediriger vers une ``EventPipeline``
    asynchrone pour des calculs de métriques.
    """

    def __init__(
        self,
        path: str = "runtime/agent_events.jsonl",
        persistence: Optional["PersistenceManager"] = None,
        *,
        metadata_provider: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        pipeline: Optional[EventPipeline] = None,
        drift_tracker: Optional[SnapshotDriftTracker] = None,
    ):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self.persistence: Optional["PersistenceManager"] = persistence
        self._metadata_provider = metadata_provider
        self._pipeline = pipeline
        self._drift_tracker = drift_tracker

    def write(self, event_type: str, **fields: Any) -> None:
        meta: Dict[str, Any] = {}
        if self._metadata_provider is not None:
            try:
                extra = self._metadata_provider(event_type, dict(fields))
            except Exception:
                extra = None
            if extra:
                meta = {k: v for k, v in extra.items() if k not in fields}

        llm_fields: Dict[str, Any] = {}
        llm_payload = {
            "event_type": event_type,
            "fields": json_sanitize(fields),
            "metadata": json_sanitize(meta),
        }
        response = try_call_llm_dict(
            "jsonl_logger",
            input_payload=llm_payload,
            logger=logger,
        )
        if response:
            llm_fields = {
                f"llm_{key}": value for key, value in response.items() if f"llm_{key}" not in fields
            }

        rec = {
            "t": time.time(),
            "type": event_type,
            **fields,
            **meta,
            **llm_fields,
        }
        line = json.dumps(json_sanitize(rec), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self._pipeline is not None:
            try:
                self._pipeline.submit(rec)
            except Exception:
                # la collecte analytique ne doit pas perturber le logging principal
                pass

    def snapshot(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        persistence: Optional["PersistenceManager"] = None,
    ) -> str:
        snap_dir = "runtime/snapshots"
        os.makedirs(snap_dir, exist_ok=True)
        ts = int(time.time())
        out = os.path.join(snap_dir, f"{ts}_{name}.json")
        manager = persistence or self.persistence
        data = payload
        if data is None and manager is not None:
            try:
                data = manager.make_snapshot()
            except Exception:
                data = {}
        if data is None:
            data = {}
        with self._lock:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(json_sanitize(data), f, ensure_ascii=False, indent=2)
        if self._drift_tracker is not None and data is not None:
            try:
                self._drift_tracker.record(out, data)
            except Exception:
                # Ne pas interrompre l'appelant si la détection de drift échoue
                pass

        return out

    def rotate(self, keep_last: int = 5) -> None:
        if keep_last is not None and keep_last < 0:
            raise ValueError("keep_last doit être >= 0")

        # basique: ne supprime rien par défaut
        if not os.path.exists(self.path):
            return

        directory = os.path.dirname(self.path) or "."
        basename = os.path.basename(self.path)
        timestamp = int(time.time())
        rotated_path = os.path.join(directory, f"{basename}.{timestamp}")

        with self._lock:
            if not os.path.exists(self.path):
                return

            # éviter collisions si plusieurs rotations dans la même seconde
            suffix = 0
            candidate = rotated_path
            while os.path.exists(candidate):
                suffix += 1
                candidate = os.path.join(directory, f"{basename}.{timestamp}.{suffix}")
            os.replace(self.path, candidate)

            # créer un nouveau fichier vide pour le log courant
            open(self.path, "a", encoding="utf-8").close()

            if keep_last is None:
                return

            # supprimer les plus anciens fichiers archivés au-delà de la limite
            entries = []
            for name in os.listdir(directory):
                if name == basename:
                    continue
                if name.startswith(f"{basename}."):
                    full_path = os.path.join(directory, name)
                    entries.append((os.path.getmtime(full_path), full_path))

            entries.sort(reverse=True)
            for _, path in entries[keep_last:]:
                try:
                    os.remove(path)
                except OSError:
                    # ne pas interrompre la rotation si suppression impossible
                    pass
