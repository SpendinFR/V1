from __future__ import annotations

import glob
import json
import os
import logging
import threading
import time
import uuid
import weakref
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


_LOGGER = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(json_sanitize(value), ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _unique_keywords(text: str, min_len: int = 4) -> List[str]:
    tokens = []
    for chunk in text.replace("_", " ").split():
        clean = "".join(ch for ch in chunk if ch.isalnum()).lower()
        if len(clean) >= min_len:
            tokens.append(clean)
    seen: Dict[str, bool] = {}
    unique: List[str] = []
    for token in tokens:
        if token not in seen:
            seen[token] = True
            unique.append(token)
    return unique


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {
            "true",
            "yes",
            "y",
            "ok",
            "success",
            "succès",
            "réussi",
            "succeeded",
            "valide",
            "validé",
            "accepte",
            "accepté",
        }:
            return True
        if lowered in {"false", "no", "n", "fail", "failed", "échec", "ko", "invalid", "rejeté", "reject", "refus"}:
            return False
    return None


def _first_text(data: Any, *candidates: str) -> Optional[str]:
    if not data:
        return None
    source_dict = data if isinstance(data, dict) else None
    if source_dict is None and hasattr(data, "__dict"):
        try:
            source_dict = dict(data.__dict__)
        except Exception:
            source_dict = None
    for key in candidates:
        if source_dict and key in source_dict:
            value = source_dict.get(key)
        else:
            value = getattr(data, key, None)
        if value:
            return _normalise_text(value)
    return None


@dataclass
class SkillTrial:
    index: int
    coverage: float
    success: bool
    evidence: List[str] = field(default_factory=list)
    mode: str = "coverage"
    summary: Optional[str] = None
    feedback: Optional[str] = None


@dataclass
class SkillRequest:
    identifier: str
    action_type: str
    description: str
    payload: Dict[str, Any]
    created_at: float
    status: str = "pending"
    attempts: int = 0
    successes: int = 0
    trials: List[SkillTrial] = field(default_factory=list)
    knowledge: List[Dict[str, Any]] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    approval_required: bool = True
    approved_by: Optional[str] = None
    approval_notes: Optional[str] = None
    last_error: Optional[str] = None
    implementation: Optional[Dict[str, Any]] = None
    last_execution: Optional[Dict[str, Any]] = None

    def public_view(self) -> Dict[str, Any]:
        return {
            "id": self.identifier,
            "action_type": self.action_type,
            "description": self.description,
            "status": self.status,
            "attempts": self.attempts,
            "successes": self.successes,
            "requirements": list(self.requirements),
            "approval_required": self.approval_required,
            "approved_by": self.approved_by,
            "approval_notes": self.approval_notes,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "knowledge_items": len(self.knowledge),
            "has_implementation": self.implementation is not None,
            "last_execution": self.last_execution,
        }


@dataclass
class SkillExecutionContext:
    """Context passed to concrete skill operations during execution."""

    manager: "SkillSandboxManager"
    request: SkillRequest
    payload: Mapping[str, Any]
    knowledge: Sequence[Mapping[str, Any]]
    results: Dict[str, Any]

    @property
    def memory(self) -> Optional[Any]:
        return self.manager.memory

    @property
    def interface(self) -> Optional[Any]:
        ref = self.manager.interface_ref
        return ref() if ref else None

    def resolve_value(self, spec: Any) -> Any:
        return self.manager._resolve_implementation_value(
            spec,
            payload=self.payload,
            knowledge=self.knowledge,
            results=self.results,
        )

    def resolve_structure(self, spec: Any) -> Any:
        return self.manager._resolve_implementation_structure(
            spec,
            payload=self.payload,
            knowledge=self.knowledge,
            results=self.results,
        )


OperationFunc = Callable[[SkillExecutionContext, Mapping[str, Any]], Dict[str, Any]]


class SkillSandboxManager:
    """Coordinate autonomous acquisition of new action handlers."""

    def __init__(
        self,
        storage_dir: str = "data/skills",
        *,
        min_trials: int = 3,
        success_threshold: float = 0.75,
        max_attempts: int = 5,
        approval_required: bool = True,
        run_async: bool = True,
        training_interval: float = 1.0,
    ) -> None:
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, "skills_index.json")
        self.min_trials = max(1, int(min_trials))
        self.success_threshold = max(0.0, min(1.0, float(success_threshold)))
        self.max_attempts = max(1, int(max_attempts))
        self.approval_required_default = bool(approval_required)
        self.run_async = bool(run_async)
        self.training_interval = max(0.0, float(training_interval))

        self._lock = threading.Lock()
        self._requests: Dict[str, SkillRequest] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._active_handlers: Dict[str, Any] = {}
        self._operations: Dict[str, OperationFunc] = {}

        self._positive_markers: Tuple[str, ...] = (
            "succès",
            "réussi",
            "réussie",
            "réussite",
            "validé",
            "valide",
            "maîtrisé",
            "maitrise",
            "maîtrise",
            "ready",
            "prêt",
            "complet",
            "acquis",
            "ok",
            "passé",
            "passed",
        )
        self._negative_markers: Tuple[str, ...] = (
            "échec",
            "echec",
            "failed",
            "fail",
            "insuffisant",
            "insufficient",
            "pas prêt",
            "pas pret",
            "not ready",
            "refus",
            "rejet",
            "reject",
            "incomplet",
            "impossible",
            "bloqué",
            "blocked",
        )

        self.memory: Optional[Any] = None
        self.language: Optional[Any] = None
        self.simulator: Optional[Any] = None
        self.jobs: Optional[Any] = None
        self.arch_ref: Optional[weakref.ReferenceType] = None
        self.interface_ref: Optional[weakref.ReferenceType] = None
        self.question_manager_ref: Optional[weakref.ReferenceType] = None
        self.perception_ref: Optional[weakref.ReferenceType] = None
        self._inbox_dir_override: Optional[str] = None

        # Track recently issued clarification requests to avoid spamming the
        # QuestionManager when information is missing.
        self._information_requests: Dict[str, float] = {}

        self._load_state()

    # ------------------------------------------------------------------
    # Binding helpers
    def bind(
        self,
        *,
        memory: Optional[Any] = None,
        language: Optional[Any] = None,
        simulator: Optional[Any] = None,
        jobs: Optional[Any] = None,
        arch: Optional[Any] = None,
        interface: Optional[Any] = None,
        question_manager: Optional[Any] = None,
        perception: Optional[Any] = None,
        inbox_dir: Optional[str] = None,
    ) -> None:
        if memory is not None:
            self.memory = memory
        if language is not None:
            self.language = language
        if simulator is not None:
            self.simulator = simulator
        if jobs is not None:
            self.jobs = jobs
        if arch is not None:
            self.arch_ref = weakref.ref(arch)
        if interface is not None:
            self.interface_ref = weakref.ref(interface)
        if question_manager is not None:
            self.question_manager_ref = weakref.ref(question_manager)
        if perception is not None:
            self.perception_ref = weakref.ref(perception)
        if inbox_dir is not None:
            self._inbox_dir_override = inbox_dir

    # ------------------------------------------------------------------
    def register_intention(
        self,
        *,
        action_type: str,
        description: str,
        payload: Optional[Dict[str, Any]] = None,
        requirements: Optional[Sequence[str]] = None,
        knowledge: Optional[Sequence[Mapping[str, Any]]] = None,
        approval_required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Expose an explicit way to register a new autonomous skill intention.

        This mirrors :meth:`_ensure_request` but allows other modules (meta-
        cognition, evolution manager, etc.) to declaratively seed new skills
        without going through the ActionInterface first.  The method is fully
        idempotent: calling it repeatedly with the same ``action_type`` updates
        the stored request with merged requirements/payload.
        """

        if not action_type:
            raise ValueError("action_type must be provided")
        action_key = str(action_type).strip()
        if not action_key:
            raise ValueError("action_type must not be empty")

        description_text = description.strip() if description else action_key.replace("_", " ")
        payload_map = dict(payload or {})

        with self._lock:
            request = self._requests.get(action_key)
            if request is None:
                request = SkillRequest(
                    identifier=str(uuid.uuid4()),
                    action_type=action_key,
                    description=description_text,
                    payload=dict(payload_map),
                    created_at=_now(),
                    status="pending",
                    approval_required=self.approval_required_default
                    if approval_required is None
                    else bool(approval_required),
                )
                request.requirements = self._extract_requirements(request)
                self._requests[action_key] = request
                self._memorise_event(
                    "skill_intention_registered",
                    {
                        "action_type": action_key,
                        "description": description_text,
                        "requirements": list(request.requirements),
                    },
                )
            else:
                if description_text and description_text != request.description:
                    request.description = description_text
                if approval_required is not None:
                    request.approval_required = bool(approval_required)
                if payload_map:
                    try:
                        request.payload.update(payload_map)
                    except Exception:
                        request.payload = dict(payload_map)

            if requirements:
                merged = list(dict.fromkeys(list(request.requirements) + [str(r) for r in requirements if r]))
                request.requirements = merged[:20]
            else:
                request.requirements = self._extract_requirements(request)

            if knowledge:
                existing = list(request.knowledge)
                for item in knowledge:
                    try:
                        existing.append(dict(item))
                    except Exception:
                        continue
                request.knowledge = existing[:20]

            self._save_state_locked()

        # Trigger training asynchronously if needed
        if request.status in {"pending", "failed"}:
            self._ensure_training(action_key)

        return request.public_view()

    # ------------------------------------------------------------------
    def register_operation(self, name: str, handler: OperationFunc) -> None:
        """Register or override a concrete primitive usable by skill steps."""

        key = str(name or "").strip()
        if not key:
            raise ValueError("operation name must be provided")
        if not callable(handler):
            raise ValueError("operation handler must be callable")
        self._operations[key] = handler

    # ------------------------------------------------------------------
    # Public API used by ActionInterface
    def build_handler(self, act: Any, interface: Any) -> Optional[Any]:
        if interface is not None:
            self.bind(interface=interface)

        request = self._ensure_request(act)
        if request is None:
            return None

        if request.status == "active":
            handler = self._active_handlers.get(request.action_type)
            if handler is None:
                handler = self._build_live_handler(request.action_type)
                if handler is not None:
                    self._active_handlers[request.action_type] = handler
            return handler

        if request.status in {"pending", "training"}:
            self._ensure_training(request.action_type)
            return lambda _: {
                "ok": False,
                "reason": "skill_training_in_progress",
                "skill": request.public_view(),
            }

        if request.status == "awaiting_approval":
            return lambda _: {
                "ok": False,
                "reason": "skill_waiting_user_approval",
                "skill": request.public_view(),
            }

        if request.status == "rejected":
            return lambda _: {
                "ok": False,
                "reason": "skill_rejected",
                "skill": request.public_view(),
            }

        if request.status == "failed":
            self._ensure_training(request.action_type)
            return lambda _: {
                "ok": False,
                "reason": "skill_training_retry",
                "skill": request.public_view(),
            }

        return None

    def handle_simulation(self, act: Any, interface: Any) -> Optional[Dict[str, Any]]:
        if interface is not None:
            self.bind(interface=interface)

        request = self._ensure_request(act)
        if request is None:
            return None

        if request.status == "active":
            try:
                payload = dict(getattr(act, "payload", {}) or {})
            except Exception:
                payload = {}
            return self.execute(request.action_type, payload)

        if request.status in {"pending", "training", "failed"}:
            self._ensure_training(request.action_type)
            status = self.status(request.action_type)
            reason = (
                "skill_training_retry" if request.status == "failed" else "skill_training_in_progress"
            )
            return {"ok": False, "reason": reason, "skill": status}

        if request.status == "awaiting_approval":
            return {
                "ok": False,
                "reason": "skill_waiting_user_approval",
                "skill": self.status(request.action_type),
            }

        if request.status == "rejected":
            return {
                "ok": False,
                "reason": "skill_rejected",
                "skill": self.status(request.action_type),
            }

        return None

    def review(
        self,
        action_type: str,
        decision: str,
        reviewer: Optional[str] = None,
        notes: Optional[str] = None,
        *,
        interface: Optional[Any] = None,
    ) -> Dict[str, Any]:
        decision = (decision or "").strip().lower()
        if not action_type:
            return {"ok": False, "reason": "missing_action_type"}

        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return {"ok": False, "reason": "unknown_skill"}
            if request.status != "awaiting_approval":
                return {"ok": False, "reason": "skill_not_ready", "skill": request.public_view()}

            if decision not in {"approve", "approved", "accept", "reject", "rejected", "deny"}:
                return {"ok": False, "reason": "invalid_decision"}

            if decision in {"reject", "rejected", "deny"}:
                request.status = "rejected"
                request.approved_by = reviewer
                request.approval_notes = notes
                self._save_state_locked()
                return {"ok": True, "status": "rejected", "skill": request.public_view()}

            request.status = "active"
            request.approved_by = reviewer
            request.approval_notes = notes
            self._save_state_locked()

        handler = self._build_live_handler(action_type, interface=interface)
        if handler is not None:
            self._active_handlers[action_type] = handler
            bound_interface = interface
            if bound_interface is None and self.interface_ref is not None:
                bound_interface = self.interface_ref()
            if bound_interface is not None and hasattr(bound_interface, "register_handler"):
                bound_interface.register_handler(action_type, handler)

        self._memorise_event(
            "skill_approved",
            {
                "action_type": action_type,
                "reviewer": reviewer,
                "notes": notes,
            },
        )

        return {"ok": True, "status": "active", "skill": self.status(action_type)}

    def update_implementation(
        self, action_type: str, implementation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if not action_type:
            return {"ok": False, "reason": "missing_action_type"}
        try:
            normalised = self._normalise_implementation(implementation)
        except Exception as exc:
            return {"ok": False, "reason": "invalid_implementation", "error": str(exc)}

        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return {"ok": False, "reason": "unknown_skill"}
            request.implementation = normalised
            self._save_state_locked()
            public = request.public_view()

        return {"ok": True, "implementation": normalised, "skill": public}

    def status(self, action_type: str) -> Dict[str, Any]:
        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return {"action_type": action_type, "status": "missing"}
            payload = request.public_view()
            trials = [
                {
                    "index": t.index,
                    "coverage": t.coverage,
                    "success": t.success,
                    "mode": t.mode,
                    "summary": t.summary,
                    "feedback": t.feedback,
                    "evidence": list(t.evidence),
                }
                for t in request.trials
            ]
            payload.update(
                {
                    "trials": trials,
                    "success_rate": self._success_rate(request),
                    "implementation": request.implementation,
                    "last_execution": request.last_execution,
                }
            )
            return payload

    def list_skills(
        self,
        status: Optional[str] = None,
        *,
        include_trials: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return a snapshot of tracked skills optionally filtered by status."""

        with self._lock:
            requests = list(self._requests.values())

        requests.sort(key=lambda req: req.created_at)

        snapshot: List[Dict[str, Any]] = []
        for request in requests:
            if status and request.status != status:
                continue

            payload = request.public_view()
            payload["success_rate"] = self._success_rate(request)
            if include_trials:
                payload["trials"] = [
                    {
                        "index": t.index,
                        "coverage": t.coverage,
                        "success": t.success,
                        "mode": t.mode,
                        "summary": t.summary,
                        "feedback": t.feedback,
                        "evidence": list(t.evidence),
                    }
                    for t in request.trials
                ]
            else:
                payload["trial_count"] = len(request.trials)

            payload["implementation"] = request.implementation
            payload["last_execution"] = request.last_execution

            snapshot.append(payload)

        return snapshot

    # ------------------------------------------------------------------
    # Execution
    def execute(self, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return {"ok": False, "reason": "unknown_skill"}
            if request.status != "active":
                return {
                    "ok": False,
                    "reason": "skill_not_active",
                    "skill": request.public_view(),
                }
            knowledge = list(request.knowledge)
            implementation = request.implementation

        execution_result: Optional[Dict[str, Any]] = None
        if implementation:
            try:
                execution_result = self._execute_implementation(request, payload, knowledge)
            except Exception as exc:
                execution_result = {"ok": False, "error": str(exc), "steps": [], "outputs": {}}

        summary = self._render_execution_summary(request, payload, knowledge, execution_result)
        response = {
            "ok": True,
            "skill": action_type,
            "summary": summary,
            "knowledge_used": knowledge,
            "trials": [
                {
                    "index": t.index,
                    "coverage": t.coverage,
                    "success": t.success,
                    "mode": t.mode,
                    "summary": t.summary,
                    "feedback": t.feedback,
                    "evidence": list(t.evidence),
                }
                for t in request.trials
            ],
        }
        if execution_result is not None:
            response["result"] = execution_result

        with self._lock:
            request = self._requests.get(action_type)
            if request is not None:
                request.last_execution = {
                    "timestamp": _now(),
                    "payload": json_sanitize(payload),
                    "result": json_sanitize(execution_result),
                }
                self._save_state_locked()

        return response

    # ------------------------------------------------------------------
    # Internal helpers
    def _ensure_request(self, act: Any) -> Optional[SkillRequest]:
        action_type = getattr(act, "type", None)
        payload = getattr(act, "payload", {}) or {}
        context = getattr(act, "context", {}) or {}
        if not action_type:
            return None

        with self._lock:
            if action_type in self._requests:
                request = self._requests[action_type]
            else:
                description = _normalise_text(
                    payload.get("description")
                    or context.get("description")
                    or payload.get("goal")
                    or context.get("goal")
                    or action_type.replace("_", " ")
                )
                request = SkillRequest(
                    identifier=str(uuid.uuid4()),
                    action_type=action_type,
                    description=description,
                    payload=dict(payload),
                    created_at=_now(),
                    status="pending",
                    approval_required=self.approval_required_default,
                )
                request.requirements = self._extract_requirements(request)
                self._requests[action_type] = request
                self._save_state_locked()
                self._memorise_event(
                    "skill_requested",
                    {
                        "action_type": action_type,
                        "description": description,
                        "requirements": request.requirements,
                    },
                )
            return request

    def _ensure_training(self, action_type: str) -> None:
        if not self.run_async:
            self._training_loop(action_type)
            return

        with self._lock:
            thread = self._threads.get(action_type)
            if thread and thread.is_alive():
                return

            thread = threading.Thread(
                target=self._training_loop,
                args=(action_type,),
                name=f"skill-train-{action_type}",
                daemon=True,
            )
            self._threads[action_type] = thread
            thread.start()

    def _training_loop(self, action_type: str) -> None:
        for _ in range(self.max_attempts):
            with self._lock:
                request = self._requests.get(action_type)
                if request is None:
                    return
                if request.status == "active":
                    return
                request.status = "training"
                request.attempts += 1
                self._save_state_locked()

            knowledge = self._gather_knowledge(request)
            trials, successes = self._run_trials(request, knowledge)

            with self._lock:
                request = self._requests.get(action_type)
                if request is None:
                    return
                request.knowledge = knowledge
                request.trials = trials
                request.successes = successes
                success_rate = self._success_rate(request)
                has_impl = self._has_viable_implementation(request.implementation)
                if success_rate >= self.success_threshold and has_impl:
                    request.status = "awaiting_approval" if request.approval_required else "active"
                    self._save_state_locked()
                    self._notify_ready(request)
                    return
                request.status = "failed"
                if not has_impl:
                    request.last_error = "implementation_missing"
                else:
                    request.last_error = "insufficient_success_rate"
                self._save_state_locked()

            if self.training_interval > 0:
                time.sleep(self.training_interval)

        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return
            if request.status not in {"awaiting_approval", "active"}:
                request.status = "failed"
                if not self._has_viable_implementation(request.implementation):
                    request.last_error = "implementation_missing"
                else:
                    request.last_error = "max_attempts_reached"
                self._save_state_locked()

    def _gather_knowledge(self, request: SkillRequest) -> List[Dict[str, Any]]:
        query = request.description or request.action_type.replace("_", " ")
        knowledge: List[Dict[str, Any]] = []
        memory = self.memory
        if memory is not None:
            try:
                if hasattr(memory, "search"):
                    hits = memory.search(query, top_k=8)
                elif hasattr(memory, "get_recent_memories"):
                    hits = memory.get_recent_memories(n=8)
                else:
                    hits = []
                for hit in hits or []:
                    knowledge.append(self._normalise_memory(hit))
            except Exception:
                pass

        inbox_items = self._collect_inbox_context(query)
        if inbox_items:
            knowledge.extend(inbox_items)

        if not knowledge and request.payload:
            knowledge.append({"source": "payload", "content": request.payload})

        simulator = self.simulator
        if simulator is not None and hasattr(simulator, "introspect"):
            try:
                insight = simulator.introspect(query)
                if insight:
                    knowledge.append({"source": "simulator", "content": insight})
            except Exception:
                pass

        self._maybe_request_information(request, knowledge)

        return knowledge

    def _resolve_question_manager(self) -> Optional[Any]:
        if self.question_manager_ref is not None:
            manager = self.question_manager_ref()
            if manager is not None:
                return manager

        arch = self.arch_ref() if self.arch_ref else None
        if arch is not None:
            manager = getattr(arch, "question_manager", None)
            if manager is not None:
                self.question_manager_ref = weakref.ref(manager)
                return manager
        return None

    def _resolve_perception_interface(self) -> Optional[Any]:
        if self.perception_ref is not None:
            perception = self.perception_ref()
            if perception is not None:
                return perception

        arch = self.arch_ref() if self.arch_ref else None
        if arch is not None:
            perception = getattr(arch, "perception_interface", None)
            if perception is not None:
                self.perception_ref = weakref.ref(perception)
                return perception
        return None

    def _resolve_inbox_dir(self) -> Optional[str]:
        if self._inbox_dir_override and os.path.isdir(self._inbox_dir_override):
            return self._inbox_dir_override

        perception = self._resolve_perception_interface()
        if perception is not None:
            inbox_dir = getattr(perception, "inbox_dir", None)
            if isinstance(inbox_dir, str) and os.path.isdir(inbox_dir):
                self._inbox_dir_override = inbox_dir
                return inbox_dir

        arch = self.arch_ref() if self.arch_ref else None
        if arch is not None:
            candidate = getattr(arch, "inbox_dir", None)
            if isinstance(candidate, str) and os.path.isdir(candidate):
                self._inbox_dir_override = candidate
                return candidate

        fallbacks = [os.path.join("data", "inbox"), "inbox"]
        for candidate in fallbacks:
            abs_path = os.path.abspath(candidate)
            if os.path.isdir(abs_path):
                self._inbox_dir_override = abs_path
                return abs_path
        return None

    def _collect_inbox_context(self, query: str) -> List[Dict[str, Any]]:
        inbox_dir = self._resolve_inbox_dir()
        if not inbox_dir:
            return []

        try:
            files = sorted(glob.glob(os.path.join(inbox_dir, "*")))
        except Exception:
            return []

        if not files:
            return []

        tokens = [token.lower() for token in _unique_keywords(query, 4)] if query else []
        gathered: List[Dict[str, Any]] = []

        for path in files:
            if len(gathered) >= 4:
                break
            if not os.path.isfile(path):
                continue
            try:
                text = Path(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if tokens:
                lowered = text.lower()
                if not any(token in lowered for token in tokens):
                    continue

            snippet = text[:800]
            gathered.append(
                {
                    "source": f"inbox:{os.path.basename(path)}",
                    "path": path,
                    "content": snippet,
                    "snippet": snippet,
                    "kind": "inbox",
                }
            )

        return gathered

    def _maybe_request_information(
        self, request: SkillRequest, knowledge: Sequence[Mapping[str, Any]]
    ) -> None:
        manager = self._resolve_question_manager()
        if manager is None:
            return

        meaningful_sources = {
            str(item.get("source"))
            for item in knowledge
            if isinstance(item, Mapping) and item.get("source") not in {None, "", "payload"}
        }
        if meaningful_sources:
            with self._lock:
                self._information_requests.pop(request.action_type, None)
            return

        now = _now()
        with self._lock:
            last_request = self._information_requests.get(request.action_type)
            if last_request is not None and now - last_request < 600.0:
                return
            self._information_requests[request.action_type] = now

        description = request.description or request.action_type.replace("_", " ")
        question = (
            "Peux-tu me fournir les étapes concrètes ou les ressources nécessaires pour «"
            f" {description} » ?"
        )
        metadata = {
            "source": "skill_sandbox",
            "action_type": request.action_type,
            "skill_id": request.identifier,
            "ts": now,
        }
        if request.payload:
            metadata["payload_keys"] = sorted(str(key) for key in request.payload.keys())

        try:
            manager.add_question(
                question,
                qtype="skill_requirement",
                metadata=metadata,
                priority=0.8,
            )
        except Exception:
            pass

    def _run_trials(
        self, request: SkillRequest, knowledge: List[Dict[str, Any]]
    ) -> Tuple[List[SkillTrial], int]:
        trials: List[SkillTrial] = []
        successes = 0
        requirements = request.requirements or _unique_keywords(request.description)

        for index in range(self.min_trials):
            coverage, coverage_evidence = self._coverage(knowledge, requirements)
            practice = self._simulate_practice(
                request,
                knowledge,
                requirements,
                index=index,
                coverage=coverage,
            )
            evidence = list(coverage_evidence)
            evidence.extend(practice.get("evidence", []))
            trial_success = bool(practice.get("success"))
            trials.append(
                SkillTrial(
                    index=index,
                    coverage=coverage,
                    success=trial_success,
                    evidence=evidence[:10],
                    mode=str(practice.get("mode", "coverage")),
                    summary=practice.get("summary"),
                    feedback=practice.get("feedback"),
                )
            )
            if trial_success:
                successes += 1

        return trials, successes

    def _coverage(
        self, knowledge: Iterable[Dict[str, Any]], requirements: Iterable[str]
    ) -> Tuple[float, List[str]]:
        req = [r.lower() for r in requirements if r]
        if not req:
            return 1.0, []
        if not knowledge:
            return 0.0, []

        hits = 0
        evidence: List[str] = []
        for item in knowledge:
            text = _normalise_text(item)
            low = text.lower()
            matched = [r for r in req if r in low]
            if matched:
                evidence.extend(matched)
                hits += len(set(matched))
        coverage = hits / max(1, len(req))
        coverage = max(0.0, min(1.0, coverage))
        return coverage, evidence[:10]

    def _practice_from_knowledge(
        self,
        request: SkillRequest,
        knowledge: Sequence[Mapping[str, Any]],
        requirements: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if self._has_viable_implementation(request.implementation):
            return None

        extracted = self._extract_knowledge_implementation(knowledge)
        if extracted is None:
            return None

        implementation, provenance = extracted
        evidence: List[str] = []
        if provenance:
            evidence.append(provenance)
        evidence.extend(str(item) for item in requirements if item)
        summary = (
            f"Implémentation dérivée de la mémoire ({provenance})."
            if provenance
            else "Implémentation dérivée de la mémoire disponible."
        )

        return {
            "success": True,
            "mode": "knowledge",
            "summary": summary,
            "feedback": summary,
            "evidence": evidence[:10],
            "implementation": implementation,
        }

    def _simulate_practice(
        self,
        request: SkillRequest,
        knowledge: List[Dict[str, Any]],
        requirements: List[str],
        *,
        index: int,
        coverage: float,
    ) -> Dict[str, Any]:
        knowledge_practice = self._practice_from_knowledge(
            request, knowledge, requirements
        )
        if knowledge_practice is not None:
            self._ensure_practice_implementation(knowledge_practice, origin="knowledge")
            self._record_practice_attempt(
                request,
                index,
                knowledge_practice,
                coverage,
                origin="knowledge",
            )
            self._update_implementation_from_practice(
                request.action_type, knowledge_practice
            )
            return knowledge_practice

        payload_snapshot = {
            "action_type": request.action_type,
            "description": request.description,
            "requirements": requirements,
            "knowledge": knowledge,
            "payload": request.payload,
            "attempt": index,
        }

        self._inject_implementation_requirements(request, payload_snapshot)

        simulator = self.simulator
        if simulator is not None and hasattr(simulator, "run"):
            try:
                query = {"mode": "skill_practice", **json_sanitize(payload_snapshot)}
                result = simulator.run(query)
            except Exception:
                result = None
            practice = self._normalise_practice_result(result, coverage)
            if practice is not None:
                self._ensure_practice_implementation(practice, origin="simulator")
                self._record_practice_attempt(request, index, practice, coverage, origin="simulator")
                self._update_implementation_from_practice(request.action_type, practice)
                return practice

        language = self.language
        practice = self._language_practice(language, request, payload_snapshot, coverage)
        if practice is not None:
            self._ensure_practice_implementation(practice, origin="language")
            self._record_practice_attempt(request, index, practice, coverage, origin="language")
            self._update_implementation_from_practice(request.action_type, practice)
            return practice

        fallback = {
            "success": False,
            "mode": "implementation_missing",
            "summary": "Pratique insuffisante : fournir une implémentation complète (opérations + étapes).",
            "feedback": None,
            "evidence": [],
        }
        self._ensure_practice_implementation(fallback, origin="coverage")
        self._record_practice_attempt(request, index, fallback, coverage, origin="coverage")
        return fallback

    def _extract_knowledge_implementation(
        self, knowledge: Sequence[Mapping[str, Any]]
    ) -> Optional[Tuple[Dict[str, Any], Optional[str]]]:
        for item in knowledge:
            if not isinstance(item, Mapping):
                continue

            provenance = self._knowledge_provenance(item)
            candidate = self._find_implementation_in_structure(item)
            if candidate is None:
                continue

            return candidate, provenance

        return None

    def _knowledge_provenance(self, entry: Mapping[str, Any]) -> Optional[str]:
        labels: List[str] = []
        for key in ("title", "name", "label", "source", "origin", "description"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                labels.append(value.strip())

        content = entry.get("content")
        if isinstance(content, Mapping):
            for key in ("title", "name", "label", "source"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    labels.append(value.strip())

        if not labels:
            return None

        ordered = []
        seen: Dict[str, bool] = {}
        for label in labels:
            if label not in seen:
                seen[label] = True
                ordered.append(label)
        return ", ".join(ordered)

    def _find_implementation_in_structure(
        self, root: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        queue: deque[Any] = deque([root])
        seen: Dict[int, bool] = {}

        while queue:
            current = queue.popleft()
            try:
                current_id = id(current)
            except Exception:
                current_id = None

            if current_id is not None:
                if current_id in seen:
                    continue
                seen[current_id] = True

            if isinstance(current, Mapping):
                implementation_candidate = None

                if "implementation" in current:
                    implementation_candidate = current.get("implementation")
                elif {
                    "operations",
                    "steps",
                }.issubset(set(key for key in current.keys())):
                    implementation_candidate = {
                        "operations": current.get("operations"),
                        "steps": current.get("steps"),
                        **{k: current.get(k) for k in ("kind", "description") if k in current},
                    }

                if implementation_candidate is not None:
                    try:
                        normalised = self._normalise_implementation(implementation_candidate)
                    except Exception:
                        normalised = None
                    if normalised and self._implementation_has_required_parts(normalised):
                        return normalised

                for value in current.values():
                    if isinstance(value, str):
                        stripped = value.strip()
                        if stripped.startswith("{") or stripped.startswith("["):
                            try:
                                parsed = json.loads(stripped)
                            except Exception:
                                parsed = None
                            if isinstance(parsed, (Mapping, list, tuple, set)):
                                queue.append(parsed)
                            continue
                    queue.append(value)
                continue

            if isinstance(current, (list, tuple, set)):
                for value in current:
                    queue.append(value)
                continue

            if isinstance(current, str):
                stripped = current.strip()
                if not stripped:
                    continue
                if stripped.startswith("{") or stripped.startswith("["):
                    try:
                        parsed = json.loads(stripped)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, (Mapping, list, tuple, set)):
                        queue.append(parsed)

        return None

    def _normalise_practice_result(self, result: Any, coverage: float) -> Optional[Dict[str, Any]]:
        if result is None:
            return None

        success_value: Optional[bool] = None
        for key in ("success", "ok", "supported", "passed"):
            value = getattr(result, key, None)
            if isinstance(result, dict):
                value = result.get(key, value)
            coerced = _coerce_bool(value)
            if coerced is not None:
                success_value = coerced
                break

        if success_value is None:
            score = getattr(result, "score", None)
            if isinstance(result, dict):
                score = result.get("score", score)
            if isinstance(score, (int, float)):
                success_value = score >= self.success_threshold

        evidence: List[str] = []
        raw_evidence = getattr(result, "evidence", None)
        if isinstance(result, dict):
            raw_evidence = result.get("evidence", raw_evidence)
        if raw_evidence:
            if isinstance(raw_evidence, (list, tuple, set)):
                evidence = [_normalise_text(item) for item in raw_evidence][:10]
            else:
                evidence = [_normalise_text(raw_evidence)]

        summary = _first_text(result, "summary", "description", "result", "message")
        feedback = _first_text(result, "feedback", "notes", "comment", "analysis")
        if summary is None:
            summary = feedback
        if summary is None:
            summary = "Simulation d'entraînement réalisée."

        if success_value is None:
            success_value = coverage >= self.success_threshold or coverage >= 0.6

        payload: Dict[str, Any] = {
            "success": success_value,
            "mode": "simulator",
            "summary": summary,
            "feedback": feedback,
            "evidence": evidence,
        }
        if isinstance(result, Mapping) and result.get("implementation"):
            payload["implementation"] = result.get("implementation")
        return payload

    def _language_practice(
        self,
        language: Any,
        request: SkillRequest,
        payload_snapshot: Dict[str, Any],
        coverage: float,
    ) -> Optional[Dict[str, Any]]:
        if language is None:
            return None

        attempt_summary = json_sanitize(payload_snapshot)
        response: Optional[Any] = None

        try:
            if hasattr(language, "evaluate_skill_attempt"):
                response = language.evaluate_skill_attempt(attempt_summary)
            elif hasattr(language, "practice"):
                response = language.practice(attempt_summary)
            elif hasattr(language, "generate_reflective_reply"):
                arch = self.arch_ref() if self.arch_ref else None
                response = language.generate_reflective_reply(
                    arch,
                    "Évalue objectivement si la compétence peut être exercée avec succès : "
                    + json.dumps(attempt_summary, ensure_ascii=False),
                )
            elif hasattr(language, "reply"):
                response = language.reply(
                    intent="assess_skill_candidate",
                    data={
                        "skill": request.action_type,
                        "description": request.description,
                        "requirements": payload_snapshot.get("requirements", []),
                        "knowledge": payload_snapshot.get("knowledge", []),
                        "attempt": payload_snapshot.get("attempt"),
                    },
                    pragmatic={
                        "speech_act": "assessment",
                        "context": {"channel": "skill_sandbox"},
                    },
                )
        except Exception:
            response = None

        if response is None:
            return None

        parsed: Optional[Dict[str, Any]] = None
        summary_text: Optional[str] = None
        feedback_text: Optional[str] = None
        implementation: Optional[Any] = None

        if isinstance(response, Mapping):
            parsed = dict(response)
        elif isinstance(response, str):
            summary_text = response
            stripped = response.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed_candidate = json.loads(stripped)
                    if isinstance(parsed_candidate, Mapping):
                        parsed = dict(parsed_candidate)
                except Exception:
                    parsed = None
        else:
            summary_text = _normalise_text(response)

        if parsed is not None:
            implementation = parsed.get("implementation")
            summary_text = _first_text(parsed, "summary", "message", "description")
            feedback_text = _first_text(parsed, "feedback", "notes", "comment")
            success_value = parsed.get("success")
            success = _coerce_bool(success_value)
            if success is None:
                score = parsed.get("score")
                if isinstance(score, (int, float)):
                    success = score >= self.success_threshold
            if success is None:
                success = self._interpret_text_success(summary_text or json.dumps(parsed), coverage)
            evidence: List[str] = []
            raw_evidence = parsed.get("evidence")
            if isinstance(raw_evidence, (list, tuple, set)):
                evidence = [_normalise_text(item) for item in raw_evidence][:10]
            elif raw_evidence:
                evidence = [_normalise_text(raw_evidence)]
            if not summary_text:
                summary_text = feedback_text or json.dumps(parsed, ensure_ascii=False)
            return {
                "success": bool(success),
                "mode": "language",
                "summary": summary_text,
                "feedback": feedback_text or summary_text,
                "evidence": evidence,
                "implementation": implementation,
            }

        summary_text = summary_text or _normalise_text(response)
        success = self._interpret_text_success(summary_text, coverage)
        evidence = _unique_keywords(summary_text)[:10]
        return {
            "success": success,
            "mode": "language",
            "summary": summary_text,
            "feedback": summary_text,
            "evidence": evidence,
        }

    def _record_practice_attempt(
        self,
        request: SkillRequest,
        index: int,
        result: Dict[str, Any],
        coverage: float,
        *,
        origin: str,
    ) -> None:
        metadata = {
            "action_type": request.action_type,
            "attempt_index": index,
            "mode": result.get("mode", origin),
            "origin": origin,
            "success": bool(result.get("success")),
            "coverage": float(coverage),
            "summary": (result.get("summary") or "")[:240],
        }
        self._memorise_event("skill_practice_attempt", metadata)

    def _update_implementation_from_practice(
        self, action_type: str, practice: Mapping[str, Any]
    ) -> None:
        if not isinstance(practice, Mapping):
            return
        implementation = practice.get("implementation")
        if not implementation:
            return
        try:
            normalised = self._normalise_implementation(implementation)
        except Exception:
            return

        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return
            request.implementation = normalised
            self._save_state_locked()

    def _ensure_practice_implementation(
        self, practice: Mapping[str, Any], *, origin: str
    ) -> bool:
        if not isinstance(practice, Mapping):
            return False

        implementation = practice.get("implementation")
        if not implementation:
            self._mark_practice_incomplete(practice, "implementation_missing", origin)
            return False

        try:
            normalised = self._normalise_implementation(implementation)
        except Exception as exc:
            self._mark_practice_incomplete(
                practice,
                "implementation_invalid",
                origin,
                error=str(exc),
            )
            return False

        if not self._implementation_has_required_parts(normalised):
            self._mark_practice_incomplete(
                practice,
                "implementation_incomplete",
                origin,
                candidate=normalised,
            )
            return False

        practice["implementation"] = normalised
        return True

    def _mark_practice_incomplete(
        self,
        practice: Mapping[str, Any],
        reason: str,
        origin: str,
        *,
        candidate: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if not isinstance(practice, dict):
            return

        messages = {
            "implementation_missing": "Implémentation absente : fournir des opérations et des étapes complètes.",
            "implementation_invalid": "Implémentation invalide : impossible de normaliser la structure fournie.",
            "implementation_incomplete": "Implémentation incomplète : les opérations et les étapes doivent être définies.",
        }

        note = messages.get(reason, "Implémentation indisponible.")
        if error:
            note = f"{note} ({error})"

        summary = practice.get("summary") or ""
        if summary:
            summary = f"{summary} — {note}"
        else:
            summary = note

        practice["summary"] = summary
        if not practice.get("feedback"):
            practice["feedback"] = summary
        practice["mode"] = practice.get("mode") or origin or "validation"
        practice["success"] = False
        practice["error"] = reason
        evidence = list(practice.get("evidence") or [])
        evidence.append(reason)
        practice["evidence"] = evidence[:10]
        if candidate is not None:
            practice["implementation_candidate"] = candidate
        practice["implementation"] = None
        if error:
            practice["implementation_error"] = error

    def _interpret_text_success(self, text: str, coverage: float) -> bool:
        normalized = text.lower()
        for marker in self._negative_markers:
            if marker in normalized:
                return False
        for marker in self._positive_markers:
            if marker in normalized:
                return True
        return coverage >= self.success_threshold or coverage >= 0.6

    def _extract_requirements(self, request: SkillRequest) -> List[str]:
        payload = request.payload or {}
        req: List[str] = []
        raw = payload.get("requirements") or payload.get("knowledge")
        if isinstance(raw, str):
            req.extend(_unique_keywords(raw))
        elif isinstance(raw, (list, tuple, set)):
            for item in raw:
                req.extend(_unique_keywords(_normalise_text(item)))
        if not req:
            req.extend(_unique_keywords(request.description))
        llm_response = try_call_llm_dict(
            "self_improver_skill_requirements",
            input_payload={
                "action_type": request.action_type,
                "description": request.description,
                "payload": payload,
                "existing_requirements": list(req),
            },
            logger=_LOGGER,
            max_retries=2,
        )
        if llm_response:
            llm_requirements = llm_response.get("requirements")
            if isinstance(llm_requirements, Sequence) and not isinstance(llm_requirements, (str, bytes)):
                for item in llm_requirements:
                    normalized = _normalise_text(item).strip()
                    if normalized:
                        req.append(normalized)
            keywords = llm_response.get("keywords")
            if isinstance(keywords, Sequence) and not isinstance(keywords, (str, bytes)):
                unique_keywords = []
                seen_kw: Dict[str, bool] = {}
                for term in keywords:
                    clean = str(term).strip()
                    if clean and clean.lower() not in seen_kw:
                        seen_kw[clean.lower()] = True
                        unique_keywords.append(clean)
                if unique_keywords:
                    payload.setdefault("llm_keywords", unique_keywords)
        seen_req: Dict[str, bool] = {}
        ordered: List[str] = []
        for item in req:
            clean = str(item).strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen_req:
                continue
            seen_req[key] = True
            ordered.append(clean)
        return ordered[:20]

    def _implementation_requirement_prompts(self) -> List[str]:
        return [
            "Fournir une implémentation exécutable complète : définir les operations disponibles et les steps ordonnés.",
            "Décrire chaque opération avec le code Python ou l'action à appeler ainsi que les entrées nécessaires.",
            "Lister les steps de la séquence avec les conditions, les paramètres et les noms de stockage des résultats.",
        ]

    def _inject_implementation_requirements(
        self, request: SkillRequest, payload_snapshot: Dict[str, Any]
    ) -> None:
        if self._has_viable_implementation(request.implementation):
            return

        existing_req = list(payload_snapshot.get("requirements") or [])
        prompts = self._implementation_requirement_prompts()
        for text in prompts:
            if text not in existing_req:
                existing_req.append(text)

        payload_snapshot["requirements"] = existing_req
        payload_snapshot["implementation_required"] = True
        payload_snapshot["implementation_details"] = {
            "expect_operations": True,
            "expect_steps": True,
            "instructions": prompts,
        }

    def _build_live_handler(
        self, action_type: str, *, interface: Optional[Any] = None
    ) -> Optional[Any]:
        manager_ref = weakref.ref(self)

        def _handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            manager = manager_ref()
            if manager is None:
                return {"ok": False, "reason": "skill_manager_unavailable"}
            return manager.execute(action_type, payload)

        return _handler

    def _normalise_implementation(self, implementation: Any) -> Dict[str, Any]:
        if implementation is None:
            raise ValueError("missing implementation")
        if isinstance(implementation, str):
            try:
                parsed = json.loads(implementation)
            except Exception as exc:  # noqa: F841
                raise ValueError("invalid implementation string") from exc
            implementation = parsed
        if not isinstance(implementation, Mapping):
            raise ValueError("implementation must be a mapping")

        payload = dict(implementation)
        kind = str(payload.get("kind", "sequence") or "sequence")
        steps = payload.get("steps")
        if steps is None:
            steps = []
        if not isinstance(steps, Sequence):
            raise ValueError("implementation steps must be a sequence")

        normalised_steps: List[Dict[str, Any]] = []
        for raw_step in steps:
            if not isinstance(raw_step, Mapping):
                continue
            op = str(
                raw_step.get("op")
                or raw_step.get("operation")
                or raw_step.get("action")
                or raw_step.get("type")
                or ""
            ).strip()
            if not op:
                continue
            step_payload = dict(raw_step)
            step_payload["op"] = op
            normalised_steps.append(json_sanitize(step_payload))

        normalised: Dict[str, Any] = {"kind": kind, "steps": normalised_steps}

        raw_operations = payload.get("operations")
        if isinstance(raw_operations, Mapping):
            normalised_ops: Dict[str, Any] = {}
            for name, spec in raw_operations.items():
                key = str(name or "").strip()
                if not key:
                    continue
                try:
                    normalised_ops[key] = json_sanitize(spec)
                except Exception:
                    normalised_ops[key] = json_sanitize(str(spec))
            if normalised_ops:
                normalised["operations"] = normalised_ops

        for key, value in payload.items():
            if key in {"kind", "steps", "operations"}:
                continue
            normalised[key] = json_sanitize(value)
        return normalised

    def _implementation_has_required_parts(self, implementation: Mapping[str, Any]) -> bool:
        steps = implementation.get("steps") if isinstance(implementation, Mapping) else None
        if not isinstance(steps, Sequence) or not steps:
            return False
        operations = implementation.get("operations")
        if not isinstance(operations, Mapping) or not operations:
            return False
        for step in steps:
            if not isinstance(step, Mapping):
                return False
            op_name = str(
                step.get("op")
                or step.get("use")
                or step.get("operation")
                or step.get("action")
                or ""
            ).strip()
            if not op_name:
                return False
        return True

    def _has_viable_implementation(self, implementation: Optional[Mapping[str, Any]]) -> bool:
        if not isinstance(implementation, Mapping):
            return False
        return self._implementation_has_required_parts(implementation)

    def _deep_get(self, data: Any, path: str) -> Any:
        if not path:
            return data
        current = data
        for segment in path.split("."):
            if isinstance(current, Mapping):
                current = current.get(segment)
            elif isinstance(current, Sequence) and segment.isdigit():
                index = int(segment)
                if index < 0 or index >= len(current):
                    return None
                current = current[index]
            else:
                return None
        return current

    def _resolve_implementation_value(
        self,
        spec: Any,
        *,
        payload: Mapping[str, Any],
        knowledge: Sequence[Mapping[str, Any]],
        results: Mapping[str, Any],
    ) -> Any:
        if isinstance(spec, Mapping):
            if "from_payload" in spec:
                key = spec["from_payload"]
                if isinstance(key, str):
                    return self._deep_get(payload, key)
                return None
            if "from_result" in spec:
                ref = spec["from_result"]
                if isinstance(ref, str):
                    return self._deep_get(results, ref)
                return results.get(ref) if isinstance(ref, str) else None
            if "from_knowledge" in spec:
                ref = spec["from_knowledge"]
                index = 0
                path = ""
                if isinstance(ref, Mapping):
                    index = int(ref.get("index", 0) or 0)
                    path = str(ref.get("path", "") or "")
                else:
                    try:
                        index = int(ref)
                    except Exception:
                        index = 0
                if 0 <= index < len(knowledge):
                    item = knowledge[index]
                    if path:
                        return self._deep_get(item, path)
                    return item
                return None
            if "literal" in spec:
                return spec["literal"]
            if "value" in spec:
                return spec["value"]
        if isinstance(spec, str):
            if spec.startswith("payload."):
                return self._deep_get(payload, spec[len("payload.") :])
            if spec.startswith("result."):
                return self._deep_get(results, spec[len("result.") :])
            if spec.startswith("knowledge[") and spec.endswith("]"):
                inner = spec[len("knowledge[") : -1]
                try:
                    index = int(inner)
                except Exception:
                    index = 0
                if 0 <= index < len(knowledge):
                    return knowledge[index]
        return spec

    def _resolve_implementation_structure(
        self,
        spec: Any,
        *,
        payload: Mapping[str, Any],
        knowledge: Sequence[Mapping[str, Any]],
        results: Mapping[str, Any],
    ) -> Any:
        if isinstance(spec, Mapping):
            pointer_keys = {"from_payload", "from_result", "from_knowledge", "literal"}
            if any(key in spec for key in pointer_keys):
                return self._resolve_implementation_value(
                    spec, payload=payload, knowledge=knowledge, results=results
                )
            return {
                key: self._resolve_implementation_structure(
                    value,
                    payload=payload,
                    knowledge=knowledge,
                    results=results,
                )
                for key, value in spec.items()
            }
        if isinstance(spec, (list, tuple)):
            return [
                self._resolve_implementation_structure(
                    item,
                    payload=payload,
                    knowledge=knowledge,
                    results=results,
                )
                for item in spec
            ]
        return self._resolve_implementation_value(
            spec, payload=payload, knowledge=knowledge, results=results
        )

    def _build_operation_registry(
        self, implementation: Mapping[str, Any]
    ) -> Dict[str, OperationFunc]:
        registry: Dict[str, OperationFunc] = dict(self._operations)
        inline = implementation.get("operations") if isinstance(implementation, Mapping) else None
        if isinstance(inline, Mapping):
            for name, definition in inline.items():
                key = str(name or "").strip()
                if not key:
                    continue
                try:
                    registry[key] = self._compile_operation(key, definition)
                except Exception:
                    continue
        return registry

    def _compile_operation(self, name: str, definition: Any) -> OperationFunc:
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except Exception:
                definition = {"type": "python", "code": str(definition)}
        if not isinstance(definition, Mapping):
            raise ValueError(f"invalid operation definition for {name}")
        op_type = str(
            definition.get("type")
            or definition.get("kind")
            or definition.get("mode")
            or "python"
        ).strip().lower()

        if op_type in {"python", "code"}:
            return self._compile_python_operation(name, definition)

        if op_type in {"action", "skill"}:
            target_spec = definition.get("action") or definition.get("target")

            def _operation(context: SkillExecutionContext, step: Mapping[str, Any]) -> Dict[str, Any]:
                interface = context.interface
                if interface is None or not hasattr(interface, "execute"):
                    raise RuntimeError("action_interface_unavailable")
                action_value = step.get("action") or step.get("target") or target_spec
                action_name = context.resolve_value(action_value)
                if not action_name:
                    raise ValueError("missing action target")
                if str(action_name) == context.request.action_type:
                    raise RuntimeError("recursive_skill_call")
                payload_spec = step.get("payload") or step.get("inputs") or definition.get("payload") or {}
                action_payload = context.resolve_structure(payload_spec)
                result = interface.execute({"type": str(action_name), "payload": action_payload})
                ok = True
                if isinstance(result, Mapping):
                    ok = bool(result.get("ok", True))
                return {"ok": ok, "value": json_sanitize(result)}

            return _operation

        raise ValueError(f"unsupported operation type '{op_type}' for {name}")

    def _compile_python_operation(
        self, name: str, definition: Mapping[str, Any]
    ) -> OperationFunc:
        source = definition.get("code") or definition.get("source") or definition.get("body")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"python operation '{name}' missing code")
        compiled = compile(source, filename=f"<skill-op:{name}>", mode="exec")

        def _operation(context: SkillExecutionContext, step: Mapping[str, Any]) -> Dict[str, Any]:
            inputs = context.resolve_structure(step.get("inputs", {}))
            local_vars = {
                "context": context,
                "inputs": inputs,
                "payload": context.payload,
                "knowledge": context.knowledge,
                "results": context.results,
                "memory": context.memory,
                "interface": context.interface,
                "step": step,
            }
            globals_env = dict(self._python_operation_globals())
            exec(compiled, globals_env, local_vars)
            output = None
            for key in ("result", "results", "output", "outputs"):
                if key in local_vars and local_vars[key] is not None:
                    output = local_vars[key]
                    break
            if output is None:
                output = True
            if isinstance(output, Mapping):
                sanitized = json_sanitize(output)
                if "ok" not in sanitized:
                    sanitized["ok"] = True
                return sanitized
            return {"ok": True, "value": json_sanitize(output)}

        return _operation

    def _python_operation_globals(self) -> Dict[str, Any]:
        return {
            "__builtins__": self._python_operation_builtins(),
            "json": json,
            "time": time,
            "os": os,
            "Path": Path,
        }

    @staticmethod
    def _python_operation_builtins() -> Dict[str, Any]:
        return {
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "reversed": reversed,
            "range": range,
            "enumerate": enumerate,
            "any": any,
            "all": all,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "zip": zip,
            "getattr": getattr,
            "setattr": setattr,
            "hasattr": hasattr,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "print": print,
            "open": open,
            "__import__": __import__,
        }

    def _execute_implementation(
        self,
        request: SkillRequest,
        payload: Mapping[str, Any],
        knowledge: Sequence[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        implementation = request.implementation
        if not implementation:
            return None
        steps = implementation.get("steps") if isinstance(implementation, Mapping) else None
        if not steps:
            return {"ok": True, "steps": [], "outputs": {}}
        if not isinstance(steps, Sequence):
            raise ValueError("invalid implementation steps")

        operation_handlers = self._build_operation_registry(implementation)

        results: Dict[str, Any] = {}
        context = SkillExecutionContext(
            manager=self,
            request=request,
            payload=payload,
            knowledge=knowledge,
            results=results,
        )
        log: List[Dict[str, Any]] = []
        for index, raw_step in enumerate(steps):
            if not isinstance(raw_step, Mapping):
                continue
            op_name = str(
                raw_step.get("op")
                or raw_step.get("use")
                or raw_step.get("operation")
                or raw_step.get("action")
                or raw_step.get("step")
                or ""
            ).strip()
            if not op_name:
                continue
            handler = operation_handlers.get(op_name)
            if handler is None:
                raise ValueError(f"unknown operation '{op_name}'")
            store_as = (
                raw_step.get("store_as")
                or raw_step.get("store")
                or raw_step.get("as")
                or f"step_{index}"
            )
            log_entry: Dict[str, Any] = {
                "index": index,
                "operation": op_name,
                "store_as": str(store_as),
            }
            try:
                should_run = True
                if "when" in raw_step:
                    condition = context.resolve_value(raw_step["when"])
                    should_run = bool(condition)
                    log_entry["when"] = condition
                if not should_run:
                    log_entry["ok"] = True
                    log_entry["skipped"] = True
                    log.append(log_entry)
                    continue

                outcome = handler(context, raw_step)
                if not isinstance(outcome, Mapping):
                    outcome = {"ok": bool(outcome), "value": outcome}
                outcome = dict(outcome)
                outcome.setdefault("ok", True)
                results[str(store_as)] = outcome
                log_entry["ok"] = bool(outcome.get("ok", True))
                log.append(log_entry)
                if not outcome.get("ok", True):
                    return {
                        "ok": False,
                        "error": outcome.get("error", "execution_failed"),
                        "steps": log,
                        "outputs": results,
                    }
            except Exception as exc:
                log_entry["ok"] = False
                log_entry["error"] = str(exc)
                log.append(log_entry)
                return {
                    "ok": False,
                    "error": str(exc),
                    "steps": log,
                    "outputs": results,
                }
        return {"ok": True, "steps": log, "outputs": results}

    def _render_execution_summary(
        self,
        request: SkillRequest,
        payload: Dict[str, Any],
        knowledge: List[Dict[str, Any]],
        result: Optional[Dict[str, Any]],
    ) -> str:
        language = self.language
        description = request.description or request.action_type.replace("_", " ")
        hints = [k.get("content") for k in knowledge if isinstance(k, dict) and "content" in k]
        hints_text = ", ".join(str(h)[:120] for h in hints[:4])
        base_message = {
            "description": description,
            "payload": json_sanitize(payload),
            "result": json_sanitize(result),
            "knowledge": [k.get("content") for k in knowledge[:3] if isinstance(k, dict)],
        }
        message = f"Exécution de {description} terminée."  # fallback
        if result is not None and isinstance(result, Mapping):
            status = "réussie" if result.get("ok", True) else "échouée"
            message = f"Exécution {status} pour {description}."

        if language is not None:
            prompt = {
                "topic": description,
                "summary": hints_text or "Synthèse des connaissances intégrées.",
                "payload": payload,
                "result": result,
            }
            try:
                if hasattr(language, "reply"):
                    message = language.reply(
                        intent="inform",
                        data={
                            "topic": description,
                            "summary": hints_text,
                            "payload": payload,
                            "result": result,
                        },
                        pragmatic={"speech_act": "statement", "context": {"tone": "confident"}},
                    )
                elif hasattr(language, "generate_reflective_reply"):
                    arch = self.arch_ref() if self.arch_ref else None
                    message = language.generate_reflective_reply(
                        arch,
                        "Synthétise l'action"
                        f" {description} avec ces éléments: {json.dumps(json_sanitize(prompt))}",
                    )
            except Exception:
                pass

        if not isinstance(message, str):
            try:
                message = json.dumps(json_sanitize(message), ensure_ascii=False)
            except Exception:
                message = str(message)
        if not message:
            message = json.dumps(json_sanitize(base_message), ensure_ascii=False)
        return message

    def _memorise_event(self, kind: str, metadata: Dict[str, Any]) -> None:
        memory = self.memory
        if memory is None or not hasattr(memory, "add_memory"):
            return
        try:
            memory.add_memory({"kind": kind, "content": metadata.get("action_type", kind), "metadata": metadata})
        except Exception:
            pass

    def _notify_ready(self, request: SkillRequest) -> None:
        if request.status == "awaiting_approval":
            self._memorise_event(
                "skill_ready_for_review",
                {
                    "action_type": request.action_type,
                    "requirements": request.requirements,
                    "attempts": request.attempts,
                    "success_rate": self._success_rate(request),
                },
            )
        elif request.status == "active":
            self._memorise_event(
                "skill_auto_activated",
                {
                    "action_type": request.action_type,
                    "success_rate": self._success_rate(request),
                },
            )

    def _normalise_memory(self, hit: Any) -> Dict[str, Any]:
        if isinstance(hit, dict):
            return hit
        try:
            if hasattr(hit, "to_dict"):
                return hit.to_dict()
        except Exception:
            pass
        try:
            return json_sanitize(hit)  # type: ignore[arg-type]
        except Exception:
            return {"content": _normalise_text(hit)}

    def _success_rate(self, request: SkillRequest) -> float:
        total = len(request.trials)
        if total <= 0:
            return 0.0
        wins = sum(1 for trial in request.trials if trial.success)
        return wins / total

    # ------------------------------------------------------------------
    # Persistence
    def _save_state_locked(self) -> None:
        data = {
            action_type: {
                "identifier": req.identifier,
                "action_type": req.action_type,
                "description": req.description,
                "payload": req.payload,
                "created_at": req.created_at,
                "status": req.status,
                "attempts": req.attempts,
                "successes": req.successes,
                "knowledge": req.knowledge,
                "requirements": req.requirements,
                "approval_required": req.approval_required,
                "approved_by": req.approved_by,
                "approval_notes": req.approval_notes,
                "last_error": req.last_error,
                "implementation": req.implementation,
                "last_execution": req.last_execution,
                "trials": [
                    {
                        "index": t.index,
                        "coverage": t.coverage,
                        "success": t.success,
                        "evidence": list(t.evidence),
                        "mode": t.mode,
                        "summary": t.summary,
                        "feedback": t.feedback,
                    }
                    for t in req.trials
                ],
            }
            for action_type, req in self._requests.items()
        }
        try:
            with open(self.index_path, "w", encoding="utf-8") as handle:
                json.dump(json_sanitize(data), handle, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_state(self) -> None:
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for action_type, data in raw.items():
            try:
                req = SkillRequest(
                    identifier=str(data.get("identifier", uuid.uuid4())),
                    action_type=action_type,
                    description=str(data.get("description", action_type)),
                    payload=dict(data.get("payload", {})),
                    created_at=float(data.get("created_at", _now())),
                    status=str(data.get("status", "pending")),
                    attempts=int(data.get("attempts", 0)),
                    successes=int(data.get("successes", 0)),
                    approval_required=bool(data.get("approval_required", self.approval_required_default)),
                )
                req.knowledge = list(data.get("knowledge", []))
                req.requirements = list(data.get("requirements", []))
                req.approved_by = data.get("approved_by")
                req.approval_notes = data.get("approval_notes")
                req.last_error = data.get("last_error")
                implementation = data.get("implementation")
                if implementation:
                    try:
                        req.implementation = self._normalise_implementation(implementation)
                    except Exception:
                        req.implementation = None
                req.last_execution = data.get("last_execution")
                req.trials = [
                    SkillTrial(
                        index=int(t.get("index", 0)),
                        coverage=float(t.get("coverage", 0.0)),
                        success=bool(t.get("success", False)),
                        evidence=list(t.get("evidence", [])),
                        mode=str(t.get("mode", "coverage")),
                        summary=t.get("summary"),
                        feedback=t.get("feedback"),
                    )
                    for t in data.get("trials", [])
                ]
                if not req.requirements:
                    req.requirements = self._extract_requirements(req)
                self._requests[action_type] = req
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Utility
    def __len__(self) -> int:
        with self._lock:
            return len(self._requests)

