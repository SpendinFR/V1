import json
import logging
import math
import os
import re
import time
import uuid
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

try:  # pragma: no cover - optional import during bootstrap
    from .embedding_adapters import AdaptiveSemanticEmbedder
except Exception:  # pragma: no cover - fallback when module unavailable
    AdaptiveSemanticEmbedder = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class _VectorIndex:
    """Very small in-memory vector store for semantic lookups.

    The index keeps an L2-normalised bag-of-words embedding per memory item so we can
    provide inexpensive semantic retrieval without new dependencies. Callers may
    provide a custom embedding function if a richer representation is desired.
    """

    def __init__(self, embed_fn: Optional[Callable[[str], Dict[str, float]]] = None) -> None:
        self._embed_fn = embed_fn or self._default_embed
        self._vectors: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall(r"[\w']+", text.lower())

    def _default_embed(self, text: str) -> Dict[str, float]:
        tokens = self._tokenise(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        norm = math.sqrt(sum(float(v) ** 2 for v in counts.values())) or 1.0
        return {token: value / norm for token, value in counts.items()}

    # ------------------------------------------------------------------
    def rebuild(self, items: Iterable[Dict[str, Any]]) -> None:
        self._vectors = {}
        for item in items:
            self.upsert(item)

    def upsert(self, item: Mapping[str, Any]) -> None:
        item_id = str(item.get("id")) if item.get("id") else None
        if not item_id:
            return
        text = str(item.get("text") or "")
        if not text.strip():
            self._vectors.pop(item_id, None)
            return
        try:
            embedding = self._embed_fn(text)
        except Exception as exc:  # pragma: no cover - defensive log
            logger.debug("Vector index embedding failed for %s: %s", item_id, exc)
            return
        self._vectors[item_id] = dict(embedding)

    def remove(self, item_id: str) -> None:
        self._vectors.pop(item_id, None)

    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        keys = a.keys() & b.keys()
        return sum(a[k] * b[k] for k in keys)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not text.strip() or not self._vectors:
            return []
        try:
            query_vec = self._embed_fn(text)
        except Exception as exc:  # pragma: no cover - defensive log
            logger.debug("Vector index query embedding failed: %s", exc)
            return []
        results: List[Tuple[str, float]] = []
        for item_id, vector in self._vectors.items():
            score = self._cosine_similarity(query_vec, vector)
            if score <= 0:
                continue
            results.append((item_id, score))
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[: max(0, top_k)]


class MemoryStore:
    """Persistent append-only memory buffer with lightweight retrieval helpers."""

    def __init__(
        self,
        path: str = "data/memory_store.json",
        max_items: int = 5000,
        flush_every: int = 10,
        *,
        embed_fn: Optional[Callable[[str], Dict[str, float]]] = None,
    ):
        self.path = path
        self.max_items = max_items
        self.flush_every = max(1, flush_every)
        self.state: Dict[str, Any] = {"memories": []}
        self.metrics: Dict[str, Any] = {
            "added": 0,
            "evicted": 0,
            "updated": 0,
            "flushed": 0,
            "last_flush_ts": None,
        }
        self._dirty = 0
        self._hooks: Dict[str, Callable[[str, Dict[str, Any]], None]] = {}

        if embed_fn is not None:
            self._embedder = embed_fn
        elif AdaptiveSemanticEmbedder is not None:
            self._embedder = AdaptiveSemanticEmbedder()
        else:  # pragma: no cover - theoretical fallback path
            self._embedder = None

        effective_embed = embed_fn or getattr(self, "_embedder", None)
        self._index = _VectorIndex(effective_embed)
        self._load()
        self._index.rebuild(self.state.get("memories", []))

    # ------------------------------------------------------------------
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.state = json.load(fh)
            except Exception:
                self.state = {"memories": []}
        else:
            self.state = {"memories": []}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(json_sanitize(self.state), fh, ensure_ascii=False, indent=2)
        self._dirty = 0
        self.metrics["flushed"] = self.metrics.get("flushed", 0) + 1
        self.metrics["last_flush_ts"] = time.time()

    # ------------------------------------------------------------------
    def register_hook(self, name: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a callback invoked as ``callback(event, payload)``.

        Hooks are invoked for ``add`` and ``update`` events. Re-registering the same
        name overwrites the previous callback.
        """

        self._hooks[name] = callback

    def unregister_hook(self, name: str) -> None:
        self._hooks.pop(name, None)

    def _emit_hooks(self, event: str, payload: Dict[str, Any]) -> None:
        for name, hook in list(self._hooks.items()):
            try:
                hook(event, dict(payload))
            except Exception as exc:  # pragma: no cover - best-effort hooks
                logger.debug("MemoryStore hook '%s' failed on %s: %s", name, event, exc)

    # ------------------------------------------------------------------
    def add_memory(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(entry)
        ts = data.get("ts", time.time())
        data.setdefault("ts", ts)
        data.setdefault("id", f"mem_{int(ts*1000)}_{uuid.uuid4().hex[:6]}")
        data.setdefault("kind", data.get("kind", "generic"))
        data.setdefault("score", _safe_float(data.get("score"), 0.0))
        data.setdefault("uses", int(data.get("uses", 0)))
        data.setdefault("tags", list(data.get("tags", [])))
        data.setdefault("metadata", dict(data.get("metadata", {})))
        llm_payload = {
            "memory": {
                "kind": data.get("kind"),
                "text": data.get("text"),
                "tags": list(data.get("tags", [])),
                "salience": data.get("salience"),
                "metadata": {k: v for k, v in data.get("metadata", {}).items() if isinstance(k, str)},
            }
        }
        llm_response = try_call_llm_dict(
            "memory_store_strategy",
            input_payload=llm_payload,
            logger=logger,
        )
        if llm_response:
            normalized_kind = llm_response.get("normalized_kind")
            if isinstance(normalized_kind, str) and normalized_kind.strip():
                data["kind"] = normalized_kind.strip()
            suggested_tags = llm_response.get("tags")
            if isinstance(suggested_tags, list):
                existing = list(data.get("tags", []))
                for tag in suggested_tags:
                    if isinstance(tag, str) and tag not in existing:
                        existing.append(tag)
                data["tags"] = existing
            metadata_updates = llm_response.get("metadata_updates")
            if isinstance(metadata_updates, Mapping):
                data.setdefault("metadata", {}).update(metadata_updates)
            retention = llm_response.get("retention_priority")
            if isinstance(retention, str) and retention:
                data.setdefault("metadata", {}).setdefault("retention_priority", retention)
            notes = llm_response.get("notes")
            if notes:
                metadata = data.setdefault("metadata", {})
                notes_list = metadata.setdefault("llm_notes", [])
                if isinstance(notes_list, list):
                    notes_list.append(notes)
        self.state.setdefault("memories", []).append(data)
        if len(self.state["memories"]) > self.max_items:
            overflow = len(self.state["memories"]) - self.max_items
            if overflow > 0:
                evicted = self.state["memories"][:overflow]
                for item in evicted:
                    item_id = item.get("id")
                    if item_id:
                        self._index.remove(str(item_id))
                self.state["memories"] = self.state["memories"][-self.max_items:]
                self.metrics["evicted"] = self.metrics.get("evicted", 0) + len(evicted)
        self._index.upsert(data)
        self._dirty += 1
        self.metrics["added"] = self.metrics.get("added", 0) + 1
        self._emit_hooks("add", data)
        if self._dirty >= self.flush_every:
            self._save()
        return data

    def get_recent_memories(self, n: int = 50) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        return list(self.state.get("memories", [])[-n:])

    # ------------------------------------------------------------------
    def query_memories(
        self,
        query: str,
        *,
        top_k: int = 10,
        kind: Optional[Iterable[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return memories ranked by semantic similarity and optional filters."""

        memories = self.state.get("memories", [])
        kind_set = set(k for k in (kind or []) if k)
        min_score = float(min_score) if min_score is not None else None

        if query:
            ranked_ids = self._index.query(query, top_k=top_k * 3 if top_k else 10)
        else:
            ranked_ids = []

        ranked_lookup = {item_id: score for item_id, score in ranked_ids}
        candidates: List[Tuple[Dict[str, Any], float]] = []

        for item in reversed(memories):
            item_id = str(item.get("id")) if item.get("id") else None
            if kind_set and (item.get("kind") not in kind_set):
                continue
            if min_score is not None and _safe_float(item.get("score")) < min_score:
                continue
            score = ranked_lookup.get(item_id, 0.0)
            if score <= 0 and ranked_ids:
                continue
            candidates.append((item, score))
            if not ranked_ids and len(candidates) >= top_k:
                break

        if ranked_ids:
            candidates.sort(key=lambda pair: (pair[1], _safe_float(pair[0].get("score"))), reverse=True)
        else:
            candidates.sort(key=lambda pair: (_safe_float(pair[0].get("score")), pair[0].get("ts", 0.0)), reverse=True)

        return [dict(item) for item, _ in candidates[:top_k]] if top_k else [dict(item) for item, _ in candidates]

    def all_memories(self) -> List[Dict[str, Any]]:
        return list(self.state.get("memories", []))

    def flush(self):
        if self._dirty:
            self._save()

    # ------------------------------------------------------------------
    def add_item(self, item: Dict[str, Any]) -> str:
        """Compatibility adapter for summarizers expecting an ``add_item`` API."""

        new_item = self.add_memory(item)
        return str(new_item.get("id"))

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        if not item_id:
            return None
        for item in self.state.get("memories", []):
            if str(item.get("id")) == str(item_id):
                return dict(item)
        return None

    def update_item(self, item_id: str, patch: MutableMapping[str, Any]) -> None:
        if not item_id or not isinstance(patch, MutableMapping):
            return
        for item in self.state.get("memories", []):
            if str(item.get("id")) == str(item_id):
                before = dict(item)
                item.update(patch)
                self._dirty += 1
                self.metrics["updated"] = self.metrics.get("updated", 0) + 1
                self._index.upsert(item)
                self._emit_hooks("update", item)
                if self._dirty >= self.flush_every:
                    self._save()
                logger.debug("Memory item %s updated: %s -> %s", item_id, before, item)
                return

    def list_items(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        filters = dict(filters or {})
        memories = list(self.state.get("memories", []))
        kind = filters.get("kind")
        older_than = filters.get("older_than_ts")
        newer_than = filters.get("newer_than_ts")
        not_compressed = bool(filters.get("not_compressed"))
        limit = int(filters.get("limit") or 0)

        if kind is not None:
            kinds = {kind} if isinstance(kind, str) else set(kind)
            memories = [m for m in memories if m.get("kind") in kinds]
        if older_than is not None:
            threshold = _safe_float(older_than)
            memories = [m for m in memories if _safe_float(m.get("ts")) < threshold]
        if newer_than is not None:
            threshold = _safe_float(newer_than)
            memories = [m for m in memories if _safe_float(m.get("ts")) > threshold]
        if not_compressed:
            memories = [m for m in memories if not m.get("compressed_into")]

        memories.sort(key=lambda item: _safe_float(item.get("ts")), reverse=True)
        if limit > 0:
            memories = memories[:limit]
        return [dict(item) for item in memories]

    def now(self) -> float:
        return time.time()

    # ------------------------------------------------------------------
    def bump_score(self, item_id: str, delta: float = 0.1) -> None:
        """Increment the usefulness score of a memory and track usage."""

        if not item_id:
            return
        for item in self.state.get("memories", []):
            if str(item.get("id")) == str(item_id):
                item["score"] = _safe_float(item.get("score")) + _safe_float(delta)
                item["uses"] = int(item.get("uses", 0)) + 1
                self._dirty += 1
                self._index.upsert(item)
                self._emit_hooks("update", item)
                if self._dirty >= self.flush_every:
                    self._save()
                return

    def decay_scores(self, *, rate: float = 0.95, floor: float = 0.0) -> None:
        """Apply multiplicative decay to stored scores to control drift."""

        rate = float(rate)
        floor = float(floor)
        modified = False
        for item in self.state.get("memories", []):
            score = _safe_float(item.get("score")) * rate
            if score < floor:
                score = floor
            if score != item.get("score"):
                item["score"] = score
                modified = True
        if modified:
            self._dirty += 1
            if self._dirty >= self.flush_every:
                self._save()

    # ------------------------------------------------------------------
    # Persistence helpers used by :class:`PersistenceManager`
    def to_state(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "max_items": self.max_items,
            "memories": list(self.state.get("memories", [])),
        }

    def from_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self.path = str(payload.get("path", self.path))
        self.max_items = int(payload.get("max_items", self.max_items))
        memories = payload.get("memories")
        if isinstance(memories, list):
            self.state["memories"] = list(memories)
        self._index.rebuild(self.state.get("memories", []))
        self._dirty = 0
        try:
            self._save()
        except Exception:
            pass
