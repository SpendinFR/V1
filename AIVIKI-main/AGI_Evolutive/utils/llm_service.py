"""Service layer to orchestrate repository-wide LLM integrations."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, Tuple

from itertools import islice

from .llm_client import LLMCallError, LLMResult, OllamaLLMClient, OllamaModelConfig
from .llm_specs import LLMIntegrationSpec, get_spec


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return default




def _llm_env_enabled() -> bool:
    """Resolve the current LLM availability from the environment."""

    return _env_flag("AGI_ENABLE_LLM") and not _env_flag("AGI_DISABLE_LLM")


_DEFAULT_ENABLED = _llm_env_enabled()


class LLMIntegrationError(RuntimeError):
    """Raised when the structured LLM integration fails."""


class LLMUnavailableError(LLMIntegrationError):
    """Raised when LLM integration is disabled or unavailable."""


@dataclass
class LLMInvocation:
    """Encapsulate the result of one integration call."""

    spec: LLMIntegrationSpec
    result: LLMResult


@dataclass(frozen=True)
class LLMCallRecord:
    """Trace the outcome of an integration attempt for diagnostics."""

    spec_key: str
    status: str
    timestamp: float
    message: Optional[str] = None


_ACTIVITY_LOG: deque[LLMCallRecord] = deque(maxlen=200)

_URGENT_ACTIVE_CHECK: Optional[Callable[[], bool]] = None
_URGENT_ALLOWANCE_CHECK: Optional[Callable[[], bool]] = None
_BASE_URGENT_STATE = False
_MANUAL_URGENT_COUNT = 0
_GLOBAL_URGENT_STATE = False
_URGENT_STATE_LOCK = threading.Lock()
_URGENT_CLEAR_EVENT = threading.Event()
_URGENT_CLEAR_EVENT.set()
_THREAD_LOCAL = threading.local()

# Gate used to serialize LLM calls and to make urgent callers pre-empt background
# integrations.  The condition protects the ownership bookkeeping so that urgent
# chains can reserve the slot for their entire duration while other threads wait.
_CALL_CONDITION = threading.Condition()
_CALL_OWNER: Optional[int] = None
_CALL_DEPTH = 0
_WAITING_URGENT = 0


class _RecentDuplicateFilter(logging.Filter):
    """Filter to drop immediate duplicate log records for readability."""

    def __init__(self, *, window: float = 0.25) -> None:
        super().__init__()
        self._window = float(window)
        self._lock = threading.Lock()
        self._last_key: Optional[Tuple[str, int]] = None
        self._last_ts: float = 0.0

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logging side effect
        message = record.getMessage()
        key = (message, getattr(record, "thread", 0))
        now = float(getattr(record, "created", time.time()))
        with self._lock:
            if self._last_key == key and (now - self._last_ts) < self._window:
                return False
            self._last_key = key
            self._last_ts = now
        return True


LOGGER = logging.getLogger(__name__)
_has_dup_filter = any(isinstance(flt, _RecentDuplicateFilter) for flt in LOGGER.filters)
if not _has_dup_filter:
    LOGGER.addFilter(_RecentDuplicateFilter())


def _describe_current_thread() -> str:
    thread = threading.current_thread()
    ident = getattr(thread, "ident", None)
    if ident is None:
        return thread.name
    return f"{thread.name}#{ident}"


def _current_thread_is_urgent() -> bool:
    depth = getattr(_THREAD_LOCAL, "manual_urgent_depth", 0)
    if depth > 0:
        return True
    if _URGENT_ALLOWANCE_CHECK is not None:
        try:
            if bool(_URGENT_ALLOWANCE_CHECK()):
                return True
        except Exception:
            return False
    return False


def _notify_call_waiters() -> None:
    with _CALL_CONDITION:
        _CALL_CONDITION.notify_all()


def _acquire_call_slot(spec_key: str, thread_label: str) -> bool:
    """Serialize access to the underlying LLM client.

    Returns ``True`` when the caller holds the slot as part of an urgent chain.
    The return value is only used for diagnostics; releasing the slot is handled
    through :func:`_release_call_slot`.
    """

    global _CALL_OWNER, _CALL_DEPTH, _WAITING_URGENT

    ident = threading.get_ident()
    if ident is None:
        ident = id(threading.current_thread())

    with _CALL_CONDITION:
        if _CALL_OWNER == ident:
            _CALL_DEPTH += 1
            return _current_thread_is_urgent()

        is_urgent = _current_thread_is_urgent()
        wait_started: Optional[float] = None

        if is_urgent:
            _WAITING_URGENT += 1

        try:
            while True:
                owner_free = _CALL_OWNER is None
                manual_guard = _is_global_urgent_active()

                if owner_free:
                    if is_urgent:
                        break
                    if not manual_guard and _WAITING_URGENT == 0:
                        break
                else:
                    if _CALL_OWNER == ident:
                        break

                if wait_started is None:
                    wait_started = time.time()
                    LOGGER.info(
                        "LLM spec '%s' en file d'attente pour le slot (thread %s)",
                        spec_key,
                        thread_label,
                    )
                _CALL_CONDITION.wait(timeout=0.2)
        finally:
            if is_urgent:
                _WAITING_URGENT = max(0, _WAITING_URGENT - 1)

        if wait_started is not None:
            LOGGER.info(
                "LLM spec '%s' obtient le slot après %.2fs (thread %s)",
                spec_key,
                time.time() - wait_started,
                thread_label,
            )

        _CALL_OWNER = ident
        _CALL_DEPTH = 1
        return is_urgent


def _release_call_slot() -> None:
    global _CALL_OWNER, _CALL_DEPTH

    ident = threading.get_ident()
    if ident is None:
        ident = id(threading.current_thread())

    with _CALL_CONDITION:
        if _CALL_OWNER != ident:
            return
        if _CALL_DEPTH > 1:
            _CALL_DEPTH -= 1
            return
        _CALL_OWNER = None
        _CALL_DEPTH = 0
        _CALL_CONDITION.notify_all()


def _record_activity(spec_key: str, status: str, message: Optional[str] = None) -> None:
    try:
        _ACTIVITY_LOG.appendleft(
            LLMCallRecord(
                spec_key=spec_key,
                status=status,
                timestamp=time.time(),
                message=message.strip() if isinstance(message, str) else message,
            )
        )
    except Exception:  # pragma: no cover - defensive guard for diagnostics
        pass


def _await_urgent_clearance(
    spec_key: str,
    *,
    logger: Optional[Any] = None,
    thread_label: Optional[str] = None,
) -> Tuple[float, str]:
    """Block the caller until no urgent chain is active (unless permitted)."""

    if thread_label is None:
        thread_label = _describe_current_thread()

    wait_started: Optional[float] = None

    while True:
        active = _is_global_urgent_active()
        if not active and _URGENT_ACTIVE_CHECK is not None:
            try:
                active = bool(_URGENT_ACTIVE_CHECK())
            except Exception:
                active = False
        if not active:
            break

        allowed = False
        if _URGENT_ALLOWANCE_CHECK is not None:
            try:
                allowed = bool(_URGENT_ALLOWANCE_CHECK())
            except Exception:
                allowed = False
        if allowed:
            break

        if wait_started is None:
            wait_started = time.time()
            LOGGER.info(
                "LLM spec '%s' en attente : chaîne urgente active (thread %s)",
                spec_key,
                thread_label,
            )
            if logger is not None:
                try:
                    logger.debug(
                        "LLM integration '%s' en attente : chaîne urgente active",
                        spec_key,
                    )
                except Exception:
                    pass

        _URGENT_CLEAR_EVENT.wait(timeout=0.2)

    wait_duration = 0.0
    if wait_started is not None:
        wait_duration = time.time() - wait_started
        if logger is not None:
            try:
                logger.debug(
                    "LLM integration '%s' reprise après %.2fs d'attente urgente",
                    spec_key,
                    wait_duration,
                )
            except Exception:
                pass
        LOGGER.info(
            "LLM spec '%s' reprise après %.2fs (thread %s)",
            spec_key,
            wait_duration,
            thread_label,
        )

    return wait_duration, thread_label


def register_urgent_gate(
    *,
    check_active: Optional[Callable[[], bool]] = None,
    allow_current: Optional[Callable[[], bool]] = None,
) -> None:
    """Register callables used to gate background LLM usage during urgent chains."""

    global _URGENT_ACTIVE_CHECK, _URGENT_ALLOWANCE_CHECK
    _URGENT_ACTIVE_CHECK = check_active
    _URGENT_ALLOWANCE_CHECK = allow_current

    # Synchronise the cached urgent flag with the currently registered source so
    # that late registrations (or reconfigurations after a restart) immediately
    # reflect any active urgent window.  This prevents a short race where
    # background threads could observe ``False`` in the cache before the first
    # explicit update emitted by the job manager.
    if check_active is not None:
        try:
            update_urgent_state(bool(check_active()))
        except Exception:  # pragma: no cover - defensive guard
            pass


def _recompute_global_urgent_state() -> None:
    global _GLOBAL_URGENT_STATE
    with _URGENT_STATE_LOCK:
        combined = _BASE_URGENT_STATE or _MANUAL_URGENT_COUNT > 0
        _GLOBAL_URGENT_STATE = combined
    if combined:
        _URGENT_CLEAR_EVENT.clear()
    else:
        _URGENT_CLEAR_EVENT.set()
    _notify_call_waiters()


def update_urgent_state(active: bool) -> None:
    """Cache the urgent flag so background threads can read it without RPC."""

    global _BASE_URGENT_STATE
    with _URGENT_STATE_LOCK:
        _BASE_URGENT_STATE = bool(active)
    _recompute_global_urgent_state()


def manual_urgent_enter() -> None:
    """Enter a manual urgent window outside of the job-manager bookkeeping."""

    global _MANUAL_URGENT_COUNT
    with _URGENT_STATE_LOCK:
        _MANUAL_URGENT_COUNT += 1
    depth = getattr(_THREAD_LOCAL, "manual_urgent_depth", 0) + 1
    _THREAD_LOCAL.manual_urgent_depth = depth
    _recompute_global_urgent_state()


def manual_urgent_exit() -> None:
    """Release a manual urgent window previously entered."""

    global _MANUAL_URGENT_COUNT
    with _URGENT_STATE_LOCK:
        if _MANUAL_URGENT_COUNT > 0:
            _MANUAL_URGENT_COUNT -= 1
    depth = getattr(_THREAD_LOCAL, "manual_urgent_depth", 0)
    if depth > 1:
        _THREAD_LOCAL.manual_urgent_depth = depth - 1
    elif depth == 1:
        try:
            delattr(_THREAD_LOCAL, "manual_urgent_depth")
        except AttributeError:
            pass
    _recompute_global_urgent_state()


def _is_global_urgent_active() -> bool:
    with _URGENT_STATE_LOCK:
        return _GLOBAL_URGENT_STATE


def get_recent_llm_activity(limit: int = 20) -> Sequence[LLMCallRecord]:
    """Return the most recent integration attempts for observability."""

    if limit is None or limit <= 0:
        limit = len(_ACTIVITY_LOG)
    return tuple(islice(_ACTIVITY_LOG, 0, limit))


class LLMIntegrationManager:
    """Coordinates prompt construction and model selection for integrations."""

    def __init__(
        self,
        *,
        client: Optional[OllamaLLMClient] = None,
        model_configs: Optional[Mapping[str, OllamaModelConfig]] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._client = client or OllamaLLMClient()
        self._enabled = _llm_env_enabled() if enabled is None else bool(enabled)
        self._model_configs: MutableMapping[str, OllamaModelConfig] = dict(model_configs or {})
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def call_json(
        self,
        spec_key: str,
        *,
        input_payload: Any | None = None,
        extra_instructions: Optional[Sequence[str]] = None,
        max_retries: int = 1,
        skip_urgent_gate: bool = False,
        logger: Optional[Any] = None,
    ) -> LLMInvocation:
        if not self._enabled:
            raise LLMUnavailableError("LLM integration is disabled")

        thread_label = _describe_current_thread()
        if not skip_urgent_gate:
            _, thread_label = _await_urgent_clearance(
                spec_key,
                logger=logger,
                thread_label=thread_label,
            )

        _acquire_call_slot(spec_key, thread_label)

        try:
            spec = get_spec(spec_key)
            instructions: list[str] = list(spec.extra_instructions)
            if extra_instructions:
                instructions.extend(
                    instr.strip() for instr in extra_instructions if instr and instr.strip()
                )
            instructions.append("Si tu n'es pas certain, explique l'incertitude dans le champ 'notes'.")

            result = self._client.generate_json(
                self._resolve_model(spec.preferred_model),
                spec.prompt_goal,
                input_data=input_payload,
                extra_instructions=instructions,
                example_output=spec.example_output,
                max_retries=max_retries,
            )
        except LLMCallError as exc:  # pragma: no cover - delegated to integration error
            raise LLMIntegrationError(f"LLM call failed for spec '{spec_key}': {exc}") from exc
        finally:
            _release_call_slot()

        return LLMInvocation(spec=spec, result=result)

    def call_dict(
        self,
        spec_key: str,
        *,
        input_payload: Any | None = None,
        extra_instructions: Optional[Sequence[str]] = None,
        max_retries: int = 1,
        skip_urgent_gate: bool = False,
        logger: Optional[Any] = None,
    ) -> Mapping[str, Any]:
        invocation = self.call_json(
            spec_key,
            input_payload=input_payload,
            extra_instructions=extra_instructions,
            max_retries=max_retries,
            skip_urgent_gate=skip_urgent_gate,
            logger=logger,
        )
        parsed = invocation.result.parsed
        if not isinstance(parsed, Mapping):
            raise LLMIntegrationError(
                f"Spec '{spec_key}' returned a non-mapping payload: {type(parsed).__name__}"
            )
        return parsed

    def _resolve_model(self, model_name: str) -> OllamaModelConfig:
        with self._lock:
            if model_name not in self._model_configs:
                self._model_configs[model_name] = OllamaModelConfig(name=model_name)
            return self._model_configs[model_name]


_default_manager: Optional[LLMIntegrationManager] = None
_default_lock = threading.Lock()


def get_llm_manager() -> LLMIntegrationManager:
    global _default_manager
    with _default_lock:
        if _default_manager is None:
            _default_manager = LLMIntegrationManager()
        return _default_manager


def set_llm_manager(manager: Optional[LLMIntegrationManager]) -> None:
    global _default_manager
    with _default_lock:
        _default_manager = manager


def is_llm_enabled() -> bool:
    manager = _default_manager
    if manager is not None:
        return manager.enabled
    return _llm_env_enabled()


def try_call_llm_dict(
    spec_key: str,
    *,
    input_payload: Any | None = None,
    extra_instructions: Optional[Sequence[str]] = None,
    logger: Optional[Any] = None,
    max_retries: int = 1,
) -> Optional[Mapping[str, Any]]:
    """Attempt to call the shared LLM and return a mapping payload.

    The helper centralises the common guard/exception handling logic used by the
    individual modules so they can focus on crafting the structured payload
    without duplicating availability checks.  When the LLM is disabled or the
    integration fails, ``None`` is returned to signal the caller to fall back to
    heuristics.
    """

    thread_label = _describe_current_thread()
    request_ts = time.time()

    if not is_llm_enabled():
        LOGGER.info(
            "LLM spec '%s' ignorée (LLM désactivé) – thread %s",
            spec_key,
            thread_label,
        )
        _record_activity(spec_key, "disabled", "LLM integration désactivée")
        return None

    _, thread_label = _await_urgent_clearance(
        spec_key,
        logger=logger,
        thread_label=thread_label,
    )

    call_started_at = time.time()
    try:
        manager = get_llm_manager()
        payload = manager.call_dict(
            spec_key,
            input_payload=input_payload,
            extra_instructions=extra_instructions,
            max_retries=max_retries,
            skip_urgent_gate=False,
            logger=logger,
        )
        now = time.time()
        total_duration = now - request_ts
        call_duration = now - call_started_at
        LOGGER.info(
            "LLM spec '%s' terminée avec succès en %.2fs (appel %.2fs) – thread %s",
            spec_key,
            total_duration,
            call_duration,
            thread_label,
        )
        _record_activity(spec_key, "success", None)
        return payload
    except (LLMUnavailableError, LLMIntegrationError) as exc:
        now = time.time()
        total_duration = now - request_ts
        LOGGER.warning(
            "LLM spec '%s' indisponible après %.2fs – thread %s : %s",
            spec_key,
            total_duration,
            thread_label,
            exc,
        )
        _record_activity(spec_key, "error", str(exc))
        if logger is not None:
            try:
                logger.debug(
                    "LLM integration '%s' unavailable: %s", spec_key, exc, exc_info=True
                )
            except Exception:  # pragma: no cover - defensive logging guard
                pass
        return None
    except Exception as exc:  # pragma: no cover - unexpected failure safety net
        now = time.time()
        total_duration = now - request_ts
        LOGGER.exception(
            "Erreur inattendue pour la spec LLM '%s' après %.2fs – thread %s",
            spec_key,
            total_duration,
            thread_label,
        )
        _record_activity(spec_key, "error", str(exc))
        if logger is not None:
            try:
                logger.warning(
                    "Unexpected error while calling LLM integration '%s': %s",
                    spec_key,
                    exc,
                    exc_info=True,
                )
            except Exception:
                pass
        return None


__all__ = [
    "LLMIntegrationError",
    "LLMIntegrationManager",
    "LLMInvocation",
    "LLMCallRecord",
    "LLMUnavailableError",
    "manual_urgent_enter",
    "manual_urgent_exit",
    "get_llm_manager",
    "get_recent_llm_activity",
    "is_llm_enabled",
    "register_urgent_gate",
    "set_llm_manager",
    "try_call_llm_dict",
]
