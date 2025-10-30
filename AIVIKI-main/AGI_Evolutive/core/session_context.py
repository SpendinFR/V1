"""Session-scoped context helpers.

The orchestrator historically stored a large amount of mutable state on the
instance itself.  This made multi-session hosting extremely fragile because
values such as the last prediction error, cached beliefs or temporary
priorities leaked between users.  The helper defined in this module centralises
all the mutable pieces per ``(user_id, session_id)`` tuple and provides a small
set of utilities used by the orchestrator and high-level runtime modules.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple


SessionKey = Tuple[str, str]


_THREAD_CONTEXT = threading.local()


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float = 0.0
    scope: str = "session"
    trace_id: Optional[str] = None
    topic: Optional[str] = None


@dataclass
class SessionContext:
    """Represents the mutable state for one user/session pair."""

    user_id: str
    session_id: str
    state: Dict[str, Any] = field(default_factory=dict)
    caches: Dict[str, _CacheEntry] = field(default_factory=dict)
    last_trace_id: Optional[str] = None
    trace_started_at: float = 0.0
    quiet_until: float = 0.0
    locked_model: Optional[str] = None
    debounce_window: float = 0.3
    _recent_events: Dict[str, float] = field(default_factory=dict)
    current_topic: Optional[str] = None

    def ensure_default(self, name: str, value: Any) -> Any:
        if name not in self.state:
            self.state[name] = value
        return self.state[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.state.get(name, default)

    def set(self, name: str, value: Any) -> None:
        self.state[name] = value

    # --- Trace management -------------------------------------------------
    def start_trace(self, topic: Optional[str] = None) -> str:
        trace_id = f"trace::{uuid.uuid4().hex[:16]}"
        previous_topic = self.current_topic
        self.last_trace_id = trace_id
        self.trace_started_at = time.time()
        if topic is not None:
            if topic != previous_topic:
                self._invalidate_caches(scope="topic")
            self.current_topic = topic
        self._invalidate_caches(scope="trace")
        return trace_id

    def clear_trace(self) -> None:
        self.last_trace_id = None
        self.trace_started_at = 0.0
        self._invalidate_caches(scope="trace")

    # --- Cache helpers ----------------------------------------------------
    def _invalidate_caches(self, *, scope: Optional[str] = None) -> None:
        for key in list(self.caches.keys()):
            entry = self.caches.get(key)
            if entry is None:
                continue
            if scope is None or entry.scope == scope:
                self.caches.pop(key, None)

    def _is_entry_stale(self, entry: _CacheEntry, scope: str) -> bool:
        now = time.time()
        if entry.expires_at and entry.expires_at < now:
            return True
        if scope == "trace" and entry.trace_id != self.last_trace_id:
            return True
        if scope == "topic" and entry.topic != self.current_topic:
            return True
        return False

    def get_cache(
        self,
        name: str,
        *,
        default_factory: Optional[Callable[[], Any]] = None,
        ttl: Optional[float] = None,
        scope: str = "session",
    ) -> Any:
        scope = scope or "session"
        entry = self.caches.get(name)
        if entry is not None and self._is_entry_stale(entry, entry.scope):
            entry = None
            self.caches.pop(name, None)
        if entry is None:
            value = default_factory() if default_factory is not None else None
            expires_at = time.time() + float(ttl) if ttl else 0.0
            entry = _CacheEntry(
                value=value,
                expires_at=expires_at,
                scope=scope,
                trace_id=self.last_trace_id,
                topic=self.current_topic,
            )
            self.caches[name] = entry
            return entry.value
        if ttl:
            entry.expires_at = time.time() + float(ttl)
        if scope in {"trace", "topic"} and self._is_entry_stale(entry, scope):
            # Stale due to scope mismatch, rebuild lazily
            value = default_factory() if default_factory is not None else None
            expires_at = time.time() + float(ttl) if ttl else 0.0
            entry = _CacheEntry(
                value=value,
                expires_at=expires_at,
                scope=scope,
                trace_id=self.last_trace_id,
                topic=self.current_topic,
            )
            self.caches[name] = entry
        return entry.value

    def set_cache(
        self,
        name: str,
        value: Any,
        *,
        ttl: Optional[float] = None,
        scope: str = "session",
    ) -> Any:
        expires_at = time.time() + float(ttl) if ttl else 0.0
        self.caches[name] = _CacheEntry(
            value=value,
            expires_at=expires_at,
            scope=scope or "session",
            trace_id=self.last_trace_id,
            topic=self.current_topic,
        )
        return value

    def clear_cache(self, name: Optional[str] = None, *, scope: Optional[str] = None) -> None:
        if name is not None:
            self.caches.pop(name, None)
            return
        if scope is None:
            self.caches.clear()
            return
        self._invalidate_caches(scope=scope)

    # --- Quiet periods ----------------------------------------------------
    def start_quiet_period(self, duration: float) -> None:
        self.quiet_until = max(self.quiet_until, time.time() + max(0.0, duration))

    def is_quiet(self) -> bool:
        return time.time() < self.quiet_until

    # --- Model locking ----------------------------------------------------
    def lock_model(self, model_name: str) -> None:
        if not model_name:
            return
        if self.locked_model is None:
            self.locked_model = model_name

    def allow_model(self, candidate: str) -> bool:
        if self.locked_model is None:
            return True
        return self.locked_model == candidate

    # --- Debounce ---------------------------------------------------------
    def debounce(self, signature: str, window: Optional[float] = None) -> bool:
        if not signature:
            return False
        now = time.time()
        win = float(window if window is not None else self.debounce_window)
        last = self._recent_events.get(signature)
        if last is not None and (now - last) < win:
            return True
        self._recent_events[signature] = now
        # prune old entries lazily
        if len(self._recent_events) > 64:
            cutoff = now - win
            stale = [key for key, ts in self._recent_events.items() if ts < cutoff]
            for key in stale:
                self._recent_events.pop(key, None)
        return False


class SessionContextManager:
    """Thread-safe registry for :class:`SessionContext` instances."""

    def __init__(self) -> None:
        self._contexts: Dict[SessionKey, SessionContext] = {}
        self._defaults: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, user_id: str, session_id: str) -> SessionContext:
        key = (user_id or "default", session_id or "default")
        with self._lock:
            ctx = self._contexts.get(key)
            if ctx is None:
                ctx = SessionContext(user_id=key[0], session_id=key[1])
                ctx.state.update(self._defaults)
                self._contexts[key] = ctx
            else:
                for name, value in self._defaults.items():
                    ctx.state.setdefault(name, value)
            return ctx

    def set_default(self, name: str, value: Any) -> None:
        with self._lock:
            self._defaults[name] = value
            for ctx in self._contexts.values():
                ctx.state.setdefault(name, value)

    def snapshot(self) -> Dict[SessionKey, Dict[str, Any]]:
        with self._lock:
            return {key: dict(ctx.state) for key, ctx in self._contexts.items()}


class SessionStateProxy(MutableMapping[str, Any]):
    """Mapping proxy exposing a session context as a dict-like object."""

    def __init__(self, context: SessionContext) -> None:
        self._context = context

    def __getitem__(self, key: str) -> Any:
        return self._context.state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._context.state[key] = value

    def __delitem__(self, key: str) -> None:
        del self._context.state[key]

    def __iter__(self):  # type: ignore[override]
        return iter(self._context.state)

    def __len__(self) -> int:
        return len(self._context.state)


def set_current_session(context: Optional[SessionContext]) -> None:
    if context is None:
        clear_current_session()
        return
    _THREAD_CONTEXT.current_session = context


def get_current_session() -> Optional[SessionContext]:
    return getattr(_THREAD_CONTEXT, "current_session", None)


def clear_current_session() -> None:
    if hasattr(_THREAD_CONTEXT, "current_session"):
        delattr(_THREAD_CONTEXT, "current_session")


__all__ = [
    "SessionContext",
    "SessionContextManager",
    "SessionStateProxy",
    "SessionKey",
    "set_current_session",
    "get_current_session",
    "clear_current_session",
]
