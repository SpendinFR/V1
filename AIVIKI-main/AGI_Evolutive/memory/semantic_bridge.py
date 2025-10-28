"""Glue code wiring MemoryStore events to semantic processors."""
from __future__ import annotations

import logging
import queue
import threading
from contextlib import nullcontext
from typing import Any, Dict, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


class SemanticMemoryBridge:
    """Background worker that forwards new memories to semantic components."""

    def __init__(
        self,
        memory_store: Any,
        *,
        concept_extractor: Optional[Any] = None,
        episodic_linker: Optional[Any] = None,
        semantic_manager: Optional[Any] = None,
        batch_size: int = 16,
        idle_sleep: float = 0.4,
        synchronization_lock: Optional[Any] = None,
    ) -> None:
        self._memory = memory_store
        self._concept = concept_extractor
        self._episodic = episodic_linker
        self._manager = semantic_manager
        self._batch_size = max(1, int(batch_size))
        self._idle_sleep = max(0.05, float(idle_sleep))
        self._lock = synchronization_lock
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="semantic-bridge", daemon=True)

        if hasattr(memory_store, "register_hook"):
            memory_store.register_hook("semantic_bridge", self._on_event)
        self._thread.start()

    # ------------------------------------------------------------------
    def shutdown(self, timeout: Optional[float] = None) -> None:
        self._stop.set()
        self._thread.join(timeout)

    # ------------------------------------------------------------------
    def _on_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if event not in {"add", "update"}:
            return
        try:
            self._queue.put_nowait(dict(payload))
        except queue.Full:
            LOGGER.debug("Semantic bridge queue full, dropping memory %s", payload.get("id"))

    # ------------------------------------------------------------------
    def _worker(self) -> None:
        backoff = self._idle_sleep
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=backoff)
            except queue.Empty:
                backoff = min(1.5, backoff * 1.3)
                continue

            backoff = self._idle_sleep
            batch = [item]
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            self._process_batch(batch)
            if self._manager and hasattr(self._manager, "on_new_items"):
                try:
                    urgency = min(1.0, 0.4 + 0.1 * len(batch))
                    self._manager.on_new_items(urgency=urgency)
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.debug("Semantic manager hook failed", exc_info=True)

    # ------------------------------------------------------------------
    def _process_batch(self, batch: Sequence[Mapping[str, Any]]) -> None:
        if not batch:
            return
        try:
            memories = list(batch)
            lock = self._lock if self._lock is not None else nullcontext()
            with lock:
                if self._concept and hasattr(self._concept, "process_memories"):
                    self._concept.process_memories(memories)
                if self._episodic and hasattr(self._episodic, "process_memories"):
                    self._episodic.process_memories(memories)
            llm_payload = {
                "memories": [
                    {
                        "id": item.get("id"),
                        "kind": item.get("kind"),
                        "text": item.get("text"),
                        "tags": item.get("tags"),
                        "salience": item.get("salience"),
                    }
                    for item in memories
                ]
            }
            llm_response = try_call_llm_dict(
                "memory_semantic_bridge",
                input_payload=llm_payload,
                logger=LOGGER,
            )
            if llm_response and self._manager and hasattr(self._manager, "on_llm_annotations"):
                try:
                    self._manager.on_llm_annotations(llm_response)
                except Exception:
                    LOGGER.debug("Semantic manager annotations hook failed", exc_info=True)
        except Exception:
            LOGGER.debug("Semantic processing hook failed", exc_info=True)


__all__ = ["SemanticMemoryBridge"]
