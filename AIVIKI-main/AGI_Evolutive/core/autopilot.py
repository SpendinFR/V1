
"""High level automation loop for the cognitive architecture."""

from __future__ import annotations

import inspect
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .config import cfg
from .document_ingest import DocumentIngest
from .persistence import PersistenceManager
from .question_manager import QuestionManager

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


@dataclass
class StageState:
    """Tracks resilience information for a given execution stage."""

    failures: int = 0
    cooldown_until: float = 0.0

    def reset(self) -> None:
        self.failures = 0
        self.cooldown_until = 0.0


@dataclass
class StageResult:
    """Outcome of a stage execution, with telemetry metadata."""

    name: str
    ok: bool
    duration: float
    failures: int
    skipped: bool = False
    error: Optional[str] = None
    payload: Any = None
    details: Optional[Dict[str, Any]] = None

    def to_metrics(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "ok": self.ok,
            "duration": self.duration,
            "failures": self.failures,
            "skipped": self.skipped,
        }
        if self.error:
            data["error"] = self.error
        if self.details:
            data["details"] = self.details
        return data

    @classmethod
    def skipped(cls, name: str, reason: Optional[str] = None, failures: int = 0) -> "StageResult":
        details = {"reason": reason} if reason else None
        return cls(
            name=name,
            ok=False,
            duration=0.0,
            failures=failures,
            skipped=True,
            details=details,
        )


class StageExecutionError(RuntimeError):
    """Raised when an autopilot stage fails despite error handling."""

    def __init__(self, stage: str, result: StageResult) -> None:
        self.stage = stage
        self.result = result
        error = result.error or "unknown error"
        if result.failures:
            message = f"Stage '{stage}' failed after {result.failures} failure(s): {error}"
        else:
            message = f"Stage '{stage}' failed: {error}"
        details = result.details or {}
        if details:
            message = f"{message} (details: {details})"
        super().__init__(message)


class Autopilot:
    """Coordinates ingestion, cognition cycles and persistence."""

    def __init__(
        self,
        arch,
        project_root: Optional[str] = None,
        orchestrator: Optional[Any] = None,
    ) -> None:
        self.arch = arch
        self.project_root = project_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        data_dir = cfg().get("DATA_DIR", "data")
        self.inbox_dir = os.path.join(data_dir, "inbox")
        self.ingest = DocumentIngest(arch, self.inbox_dir)
        existing_pm = getattr(arch, "persistence", None)
        if existing_pm is not None:
            self.persist = existing_pm
        else:
            self.persist = PersistenceManager(arch)
            setattr(arch, "persistence", self.persist)
        arch_logger = getattr(arch, "logger", None)
        if arch_logger is None:
            self.logger = logging.getLogger(
                f"{arch.__class__.__module__}.{arch.__class__.__name__}.autopilot"
            )
            setattr(arch, "logger", self.logger)
        else:
            self.logger = arch_logger
        self._fallback_logger = logging.getLogger(
            f"{__name__}.{arch.__class__.__name__}"
        )
        try:
            setattr(self.logger, "persistence", self.persist)
        except Exception:
            pass
        existing_qm = getattr(arch, "question_manager", None)
        if existing_qm is not None:
            self.questions = existing_qm
        else:
            self.questions = QuestionManager(arch)
            setattr(arch, "question_manager", self.questions)

        self.orchestrator = orchestrator

        self._stage_states: Dict[str, StageState] = {
            "ingest": StageState(),
            "cycle": StageState(),
            "orchestrator": StageState(),
            "questions": StageState(),
            "persistence": StageState(),
        }
        self.max_stage_failures = 3
        self.stage_cooldown_seconds = 15.0
        self.max_stage_cooldown = 120.0
        self.min_step_interval = float(cfg().get("AUTOPILOT_MIN_STEP_INTERVAL", 0.35))
        self.question_recency_horizon = float(cfg().get("QUESTION_RECENCY_HORIZON", 3600.0))
        self.max_pending_questions = 5
        self.persistence_drift_threshold = float(cfg().get("PERSISTENCE_DRIFT_THRESHOLD", 0.35))
        self._diagnostics_history: deque[Dict[str, Any]] = deque(maxlen=50)
        self._last_cycle_metrics: Dict[str, Any] = {}
        self._step_counter = 0
        self._last_step_ts = 0.0
        self._last_auto_attempt_ts = 0.0

        try:
            self.persist.load()
        except Exception:
            # Persistence is a best-effort facility; failures should not crash boot.
            pass

    def step(self, user_msg: Optional[str] = None):
        """Run a full autopilot iteration."""

        start_ts = time.time()
        since_last = None if not self._last_step_ts else start_ts - self._last_step_ts
        throttled = bool(
            since_last is not None and since_last < max(self.min_step_interval, 0.0)
        )
        if throttled:
            self._log(
                "debug",
                "Autopilot step throttled (Δ=%.3fs < %.3fs)",
                since_last,
                self.min_step_interval,
            )
        self._last_step_ts = start_ts

        metrics: Dict[str, Any] = {
            "step_index": self._step_counter + 1,
            "started_at": start_ts,
            "elapsed_since_last": since_last,
            "throttled": throttled,
            "stages": {},
        }

        # 1) Integrate any freshly dropped documents.
        ingest_result = self._run_stage("ingest", self.ingest.integrate)
        metrics["stages"]["ingest"] = ingest_result.to_metrics()

        # 2) Execute one cognitive cycle.
        cycle_result = self._run_stage(
            "cycle",
            self.arch.cycle,
            kwargs={"user_msg": user_msg, "inbox_docs": None},
        )
        metrics["stages"]["cycle"] = cycle_result.to_metrics()
        out = cycle_result.payload if cycle_result.ok else None

        if cycle_result.ok and not self._cycle_payload_has_text(out):
            details = self._describe_cycle_payload(out)
            error_result = StageResult(
                name="cycle",
                ok=False,
                duration=cycle_result.duration,
                failures=cycle_result.failures,
                error="cycle stage returned no usable reply text",
                details=details,
                payload=out,
            )
            raise StageExecutionError("cycle", error_result)

        # 3) Allow the optional orchestrator to do additional coordination.
        if self.orchestrator is not None:
            orchestrator_result = self._run_stage(
                "orchestrator",
                self._call_orchestrator,
                kwargs={"user_msg": user_msg, "cycle_metrics": metrics},
            )
        else:
            orchestrator_result = StageResult.skipped(
                "orchestrator", reason="not_configured"
            )
        metrics["stages"]["orchestrator"] = orchestrator_result.to_metrics()

        # 4) Maybe create follow-up questions for the user.
        questions_result = self._run_stage(
            "questions", self.questions.maybe_generate_questions
        )
        metrics["stages"]["questions"] = questions_result.to_metrics()
        blocked_channels = self._sync_question_block_state()
        if blocked_channels:
            self._attempt_autonomous_resolution(blocked_channels)

        # 5) Persist the current state regularly.
        persistence_result = self._run_stage("persistence", self.persist.autosave_tick)
        metrics["stages"]["persistence"] = persistence_result.to_metrics()

        # Persistence drift observability.
        drift_summary = self._extract_persistence_drift()
        if drift_summary:
            metrics["persistence_drift"] = drift_summary
            severity = float(drift_summary.get("severity", 0.0) or 0.0)
            if severity >= self.persistence_drift_threshold:
                self._log("info", "Persistence drift detected: %s", drift_summary)

        metrics["duration"] = time.time() - start_ts
        metrics["cycle_output_present"] = out is not None

        self._diagnostics_history.append(metrics)
        self._last_cycle_metrics = metrics
        self._step_counter += 1

        if not cycle_result.ok:
            raise StageExecutionError("cycle", cycle_result)

        return out

    def _cycle_payload_has_text(self, payload: Any) -> bool:
        if payload is None:
            return False
        if isinstance(payload, str):
            return bool(payload.strip())
        if isinstance(payload, dict):
            text = payload.get("text") or payload.get("raw")
            if isinstance(text, str):
                return bool(text.strip())
            return False
        text_attr = getattr(payload, "text", None)
        if isinstance(text_attr, str):
            return bool(text_attr.strip())
        return False

    def _describe_cycle_payload(self, payload: Any) -> Dict[str, Any]:
        details: Dict[str, Any] = {"payload_type": type(payload).__name__}
        if isinstance(payload, dict):
            keys = list(payload.keys())
            if len(keys) > 8:
                details["payload_keys"] = keys[:8] + ["…"]
            else:
                details["payload_keys"] = keys
            text = payload.get("text") or payload.get("raw")
            if isinstance(text, str):
                preview = text.strip()
                if preview:
                    details["text_preview"] = preview[:120]
        elif isinstance(payload, str):
            details["text_length"] = len(payload)
            preview = payload.strip()
            if preview:
                details["text_preview"] = preview[:120]
        else:
            text_attr = getattr(payload, "text", None)
            if isinstance(text_attr, str):
                preview = text_attr.strip()
                if preview:
                    details["text_preview"] = preview[:120]
        return details

    def pending_questions(self):
        """Retourne les questions auto-générées, + celles de validation d'apprentissage."""
        # Celles déjà générées par policies/métacog
        now = time.time()
        candidates: Dict[str, Dict[str, Any]] = {}

        try:
            self_model = getattr(self.arch, "self_model", None)
            if self_model is not None and hasattr(self_model, "ensure_awakened"):
                self_model.ensure_awakened(question_manager=self.questions, now=now)
        except Exception:
            pass

        try:
            generated = self.questions.pop_questions()
        except Exception:
            generated = []

        for raw in generated:
            normalized = self._normalize_question(raw, source="question_manager")
            if not normalized:
                continue
            normalized["score"] = self._score_question(normalized, now)
            key = normalized["text"]
            if key not in candidates or normalized["score"] > candidates[key]["score"]:
                candidates[key] = normalized

        try:
            mem = getattr(self.arch, "memory", None)
            if mem and hasattr(mem, "get_recent_memories"):
                recents = mem.get_recent_memories(50)
                for item in recents:
                    normalized = self._question_from_memory(item, now)
                    if not normalized:
                        continue
                    key = normalized["text"]
                    if key not in candidates or normalized["score"] > candidates[key]["score"]:
                        candidates[key] = normalized
        except Exception:
            pass

        base_ranked: List[Dict[str, Any]] = sorted(
            candidates.values(), key=lambda q: q.get("score", 0.0), reverse=True
        )
        llm_ranked = self._rank_questions_with_llm(base_ranked, now)
        ranked = llm_ranked if llm_ranked is not None else base_ranked
        return ranked[: self.max_pending_questions]

    def _sync_question_block_state(self) -> List[str]:
        qm = getattr(self, "questions", None)
        if not qm or not hasattr(qm, "blocked_channels"):
            return []
        blocked = list(qm.blocked_channels())
        primary_blocked = bool(getattr(qm, "is_channel_blocked", lambda _c: False)("primary"))
        immediate_blocked = bool(getattr(qm, "is_channel_blocked", lambda _c: False)("immediate"))

        goals = getattr(self.arch, "goals", None)
        if goals and hasattr(goals, "set_question_block"):
            try:
                goals.set_question_block(primary_blocked)
            except Exception:
                pass

        if self.orchestrator and hasattr(self.orchestrator, "set_immediate_question_block"):
            try:
                self.orchestrator.set_immediate_question_block(immediate_blocked)
            except Exception:
                pass

        return blocked

    def _attempt_autonomous_resolution(self, blocked_channels: Iterable[str]) -> None:
        now = time.time()
        if now - self._last_auto_attempt_ts < 20.0:
            return
        self._last_auto_attempt_ts = now
        channels_list = ",".join(sorted(set(blocked_channels)))
        self._log("info", "Autonomous resolution attempt for blocked channels: %s", channels_list)

        perception = getattr(self.arch, "perception_interface", None) or getattr(
            self.arch, "perception", None
        )
        if perception and hasattr(perception, "scan_inbox"):
            try:
                perception.scan_inbox(force=False)
            except TypeError:
                try:
                    perception.scan_inbox()
                except Exception:
                    pass
            except Exception:
                pass

        goals = getattr(self.arch, "goals", None)
        if goals and hasattr(goals, "step"):
            try:
                goals.step()
            except Exception:
                pass

        if hasattr(self.questions, "attempt_auto_answers"):
            try:
                self.questions.attempt_auto_answers(channels=blocked_channels)
            except Exception:
                pass

    def save_now(self):
        """Force a persistence checkpoint."""

        try:
            path = self.persist.save()
            self._log("info", "Manual persistence checkpoint stored at %s", path)
            return path
        except Exception as exc:
            self._log("exception", "Manual persistence checkpoint failed", exc_info=exc)
            return False

    # ------------------------------------------------------------------
    # Diagnostics helpers
    def _run_stage(
        self,
        name: str,
        func: Any,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        state = self._stage_states.setdefault(name, StageState())
        now = time.time()
        if state.cooldown_until and now < state.cooldown_until:
            remaining = max(0.0, state.cooldown_until - now)
            self._log(
                "warning",
                "Stage %s skipped due to cooldown (%.2fs remaining)",
                name,
                remaining,
            )
            return StageResult(
                name=name,
                ok=False,
                duration=0.0,
                failures=state.failures,
                skipped=True,
                details={"cooldown_remaining": remaining},
            )

        call_args = tuple(args or [])
        call_kwargs = dict(kwargs or {})
        started = time.perf_counter()
        try:
            payload = func(*call_args, **call_kwargs)
            duration = time.perf_counter() - started
            if state.failures:
                self._log(
                    "info",
                    "Stage %s recovered after %d failure(s)",
                    name,
                    state.failures,
                )
            state.reset()
            return StageResult(
                name=name,
                ok=True,
                duration=duration,
                failures=state.failures,
                payload=payload,
            )
        except Exception as exc:
            duration = time.perf_counter() - started
            state.failures += 1
            cooldown_until = 0.0
            if state.failures >= self.max_stage_failures:
                multiplier = state.failures - self.max_stage_failures + 1
                cooldown = min(
                    self.stage_cooldown_seconds * max(multiplier, 1),
                    self.max_stage_cooldown,
                )
                state.cooldown_until = now + cooldown
                cooldown_until = state.cooldown_until
            self._log(
                "exception",
                "Stage %s failed (%d consecutive)",
                name,
                state.failures,
                exc_info=exc,
            )
            details = {"cooldown_until": cooldown_until} if cooldown_until else None
            return StageResult(
                name=name,
                ok=False,
                duration=duration,
                failures=state.failures,
                error=str(exc),
                details=details,
            )

    def _call_orchestrator(
        self, *, user_msg: Optional[str], cycle_metrics: Dict[str, Any]
    ) -> Any:
        orchestrator = self.orchestrator
        if orchestrator is None:
            return None
        runner = getattr(orchestrator, "run_once_cycle", None)
        if runner is None:
            return None

        try:
            sig = inspect.signature(runner)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            has_cycle_metrics = "cycle_metrics" in sig.parameters
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
        else:
            has_cycle_metrics = True
            accepts_kwargs = True

        if has_cycle_metrics or accepts_kwargs:
            return runner(user_msg=user_msg, cycle_metrics=cycle_metrics)
        return runner(user_msg=user_msg)

    def _extract_persistence_drift(self) -> Optional[Dict[str, Any]]:
        drift_accessor = getattr(self.persist, "get_last_drift", None)
        if callable(drift_accessor):
            return drift_accessor()
        legacy = getattr(self.persist, "last_drift_summary", None)
        if isinstance(legacy, dict):
            return legacy
        return None

    def recent_cycle_diagnostics(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return the most recent cycle metrics snapshots."""

        if limit <= 0:
            return []
        tail = list(self._diagnostics_history)[-limit:]
        return [dict(item) for item in tail]

    def last_cycle_diagnostics(self) -> Dict[str, Any]:
        """Return metrics for the latest completed cycle."""

        return dict(self._last_cycle_metrics)

    # ------------------------------------------------------------------
    # Question scoring helpers
    def _normalize_question(
        self, question: Dict[str, Any], *, source: str
    ) -> Optional[Dict[str, Any]]:
        text = (question.get("text") or "").strip()
        if not text:
            return None
        meta = dict(question.get("meta") or {})
        meta.setdefault("source", source)
        ts = question.get("ts") or meta.get("ts") or time.time()
        try:
            meta["ts"] = float(ts)
        except (TypeError, ValueError):
            meta["ts"] = time.time()
        payload = {
            "id": question.get("id")
            or meta.get("id")
            or str(uuid.uuid4()),
            "type": question.get("type") or "question",
            "text": text,
            "meta": meta,
        }
        if "score" in question:
            try:
                payload["score"] = float(question["score"])
            except (TypeError, ValueError):
                payload["score"] = 0.0
        for field in ("importance", "confidence", "urgency"):
            if field in question and field not in payload:
                payload[field] = question[field]
        return payload

    def _score_question(self, question: Dict[str, Any], now: float) -> float:
        base = float(question.get("score", 0.0) or 0.0)
        meta = question.get("meta", {}) or {}
        recency_ts = meta.get("ts") or question.get("ts")
        if recency_ts:
            try:
                age = max(0.0, now - float(recency_ts))
                horizon = max(self.question_recency_horizon, 1.0)
                recency_boost = max(0.0, 1.0 - min(age, horizon) / horizon)
                base += 0.25 * recency_boost
            except (TypeError, ValueError):
                pass
        importance = question.get("importance") or meta.get("importance")
        if importance is not None:
            try:
                base += 0.35 * float(importance)
            except (TypeError, ValueError):
                pass
        urgency = question.get("urgency") or meta.get("urgency")
        if urgency is not None:
            try:
                base += 0.2 * float(urgency)
            except (TypeError, ValueError):
                pass
        confidence = question.get("confidence") or meta.get("confidence")
        if confidence is not None:
            try:
                base += 0.2 * (1.0 - float(confidence))
            except (TypeError, ValueError):
                pass
        qtype = question.get("type")
        if qtype == "validation":
            base += 0.1
        elif qtype == "active":
            base += 0.05
        return max(0.0, min(1.0, base))

    def _question_from_memory(
        self, item: Dict[str, Any], now: float
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        kind = item.get("kind") or ""
        text = None
        qtype = None
        base_score = 0.0
        meta = dict(item.get("metadata") or {})
        meta.setdefault("memory_ref", item.get("id"))
        if "ts" not in meta and item.get("ts"):
            meta["ts"] = item.get("ts")
        if kind == "validation_request":
            text = (
                item.get("content")
                or item.get("text")
                or "Valider un apprentissage"
            )
            qtype = "validation"
            base_score = 0.7
            meta.setdefault("importance", meta.get("importance", 0.85))
            meta.setdefault("confidence", meta.get("confidence", 0.4))
        elif kind == "question_active":
            text = item.get("content") or item.get("text") or ""
            qtype = "active"
            base_score = 0.55
            meta.setdefault("urgency", meta.get("urgency", 0.6))
        else:
            return None
        if not text:
            return None
        payload = self._normalize_question(
            {
                "type": qtype,
                "text": text,
                "score": base_score,
                "meta": meta,
            },
            source=f"memory.{kind}",
        )
        if not payload:
            return None
        payload["score"] = self._score_question(payload, now)
        return payload

    def _rank_questions_with_llm(
        self, candidates: List[Dict[str, Any]], now: float
    ) -> Optional[List[Dict[str, Any]]]:
        if not candidates:
            return None

        questions_payload: List[Dict[str, Any]] = []
        for item in candidates:
            meta = dict(item.get("meta") or {})
            questions_payload.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "type": item.get("type"),
                    "heuristic_score": item.get("score"),
                    "importance": item.get("importance") or meta.get("importance"),
                    "urgency": item.get("urgency") or meta.get("urgency"),
                    "confidence": item.get("confidence") or meta.get("confidence"),
                    "metadata": meta,
                }
            )

        payload = {
            "questions": questions_payload,
            "max_to_select": self.max_pending_questions,
            "recent_cycle_metrics": self._last_cycle_metrics,
            "now": now,
        }

        response = try_call_llm_dict(
            "autopilot_question_prioritization",
            input_payload=json_sanitize(payload),
            logger=getattr(self, "logger", None),
            max_retries=2,
        )
        if not response:
            return None

        prioritized = response.get("prioritized_questions")
        if not isinstance(prioritized, list):
            return None

        by_id = {}
        for item in candidates:
            qid = item.get("id")
            if isinstance(qid, str):
                by_id[qid] = item

        ranked: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for entry in prioritized:
            if not isinstance(entry, dict):
                continue
            qid = entry.get("id")
            base_item = None
            if isinstance(qid, str) and qid in by_id:
                base_item = by_id[qid]
            else:
                text = entry.get("text")
                if isinstance(text, str):
                    base_item = next(
                        (cand for cand in candidates if cand.get("text") == text),
                        None,
                    )
            if not base_item:
                continue

            cloned = dict(base_item)
            cloned_meta = dict(cloned.get("meta") or {})

            reason = entry.get("reason")
            if isinstance(reason, str) and reason.strip():
                cloned_meta["llm_reason"] = reason.strip()

            notes = entry.get("notes")
            if isinstance(notes, str) and notes.strip():
                cloned_meta["llm_notes"] = notes.strip()

            priority = entry.get("priority")
            if isinstance(priority, (int, float)):
                cloned["score"] = max(0.0, min(1.0, float(priority)))

            cloned["meta"] = cloned_meta
            ranked.append(cloned)

            if isinstance(qid, str):
                seen.add(qid)
            elif isinstance(cloned.get("id"), str):
                seen.add(cloned["id"])

        if not ranked:
            return None

        for item in candidates:
            qid = item.get("id")
            if isinstance(qid, str) and qid in seen:
                continue
            ranked.append(item)

        return ranked

    # ------------------------------------------------------------------
    # Logging helpers
    def _log_level_value(self, level: str) -> int:
        mapping = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "exception": logging.ERROR,
        }
        return mapping.get(level, logging.INFO)

    def _log(
        self,
        level: str,
        message: str,
        *args: Any,
        exc_info: Optional[BaseException] = None,
        **fields: Any,
    ) -> None:
        logger = getattr(self, "logger", None)
        log_method = getattr(logger, level, None) if logger is not None else None
        if exc_info is not None and not isinstance(exc_info, (tuple, bool)):
            exc_details = (
                exc_info.__class__,
                exc_info,
                getattr(exc_info, "__traceback__", None),
            )
        else:
            exc_details = exc_info
        if callable(log_method):
            if level == "exception":
                log_method(message, *args, exc_info=exc_details or True)
            else:
                log_method(message, *args)
            return

        formatted = message % args if args else message
        write_method = getattr(logger, "write", None) if logger is not None else None
        if callable(write_method):
            payload = {"message": formatted, **fields}
            if exc_info:
                payload["error"] = repr(exc_info)
            try:
                write_method(f"autopilot.{level}", **payload)
            except Exception:
                pass
            return

        fallback = getattr(self, "_fallback_logger", None)
        if fallback is None:
            fallback = logging.getLogger(__name__)
            self._fallback_logger = fallback
        if level == "exception":
            fallback.log(self._log_level_value(level), formatted, exc_info=exc_details or True)
        else:
            fallback.log(self._log_level_value(level), formatted)
