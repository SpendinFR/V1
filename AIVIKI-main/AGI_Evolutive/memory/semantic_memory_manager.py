"""Semantic memory maintenance pipeline with concept, episodic and summarization tasks."""
from __future__ import annotations


from collections import deque
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Deque, Dict, Mapping, Optional
import random
import time

from .summarizer import ProgressiveSummarizer, SummarizerConfig
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


@dataclass
class TaskStats:
    """Lightweight container for reporting task executions."""

    last_run: float
    iterations: int = 0
    last_result: Optional[Any] = None
    history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))


_TASK_ALIASES: Dict[str, str] = {
    "concept": "concept",
    "concepts": "concept",
    "concept_update": "concept",
    "episodic": "episodic",
    "episodes": "episodic",
    "episodique": "episodic",
    "summarize": "summarize",
    "summary": "summarize",
    "synthese": "summarize",
}


class AdaptiveEMA:
    """Adaptive EMA that selects smoothing via Thompson Sampling over betas."""

    def __init__(self, betas: Optional[tuple[float, ...]] = None) -> None:
        self._betas = betas or (0.2, 0.4, 0.6, 0.8)
        self._values: Dict[float, Optional[float]] = {beta: None for beta in self._betas}
        self._success: Dict[float, float] = {beta: 1.0 for beta in self._betas}
        self._fail: Dict[float, float] = {beta: 1.0 for beta in self._betas}
        self.current_beta: float = self._betas[0] if self._betas else 1.0
        self.value: float = 0.0

    def update(self, value: float) -> float:
        if not self._betas:
            self.value = value
            return value

        scale = max(abs(value), 1.0)
        for beta in self._betas:
            previous = self._values[beta]
            if previous is None:
                self._values[beta] = value
                continue
            error = abs(previous - value)
            # convert error into pseudo reward in [0, 1]
            success = max(0.0, min(1.0, 1.0 - min(error / (scale * 2.0), 1.0)))
            self._success[beta] += success
            self._fail[beta] += 1.0 - success
            self._values[beta] = beta * value + (1.0 - beta) * previous

        samples = {
            beta: random.betavariate(self._success[beta], self._fail[beta]) for beta in self._betas
        }
        self.current_beta = max(samples, key=samples.get)
        selected = self._values[self.current_beta]
        self.value = selected if selected is not None else value
        return self.value


class AdaptivePeriodController:
    """Adaptive scheduler that modulates task periods based on observed load."""

    def __init__(
        self,
        period: float,
        *,
        name: str,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
    ) -> None:
        self.name = name
        self.period = float(period) if period else 0.0
        if self.period <= 0:
            self.min_period = float(min_period or 0.0)
            self.max_period = float(max_period or self.min_period)
        else:
            computed_min = self.period * 0.25
            self.min_period = max(1.0, float(min_period) if min_period is not None else computed_min)
            computed_max = self.period * 4.0
            candidate_max = float(max_period) if max_period is not None else computed_max
            self.max_period = max(self.min_period, candidate_max)
        self.load = AdaptiveEMA()
        self.rate = AdaptiveEMA()
        self.drift_log: Deque[Dict[str, float]] = deque(maxlen=64)
        self.last_adjusted: Optional[float] = None

    def update(
        self,
        period: float,
        *,
        processed: int,
        duration: float,
        limit: Optional[int] = None,
        wall_ts: Optional[float] = None,
    ) -> float:
        if period <= 0:
            self.period = 0.0
            return 0.0

        ts = float(wall_ts) if wall_ts is not None else time.time()
        period = float(period)
        processed = max(0, int(processed))
        duration = max(duration, 1e-3)

        smoothed_load = self.load.update(float(processed))
        instantaneous_rate = float(processed) / duration
        smoothed_rate = self.rate.update(instantaneous_rate)

        new_period = period
        if limit:
            limit_f = float(limit)
            target = 0.4 * limit_f
            if smoothed_load > 0.85 * limit_f:
                new_period = period * 0.5
            elif smoothed_load > target:
                new_period = period * 0.75
            elif smoothed_load < target * 0.3:
                new_period = period * 1.35
            elif smoothed_load < target * 0.6:
                new_period = period * 1.15
        else:
            baseline = self.rate.value if self.rate.value else smoothed_rate
            if baseline:
                ratio = smoothed_rate / max(baseline, 1e-6)
                if ratio > 1.5:
                    new_period = period * 0.7
                elif ratio < 0.7:
                    new_period = period * 1.2

        new_period = max(self.min_period, min(self.max_period, new_period))

        if abs(new_period - period) > max(1.0, 0.1 * period):
            self.drift_log.append(
                {
                    "ts": ts,
                    "from": period,
                    "to": new_period,
                    "load": smoothed_load,
                    "rate": smoothed_rate,
                }
            )

        self.period = new_period
        self.last_adjusted = ts
        return new_period


class SemanticMemoryManager:
    """Coordinate concept updates, episodic linking and summarization."""

    def __init__(
        self,
        *,
        memory_store: Any,
        concept_store: Optional[Any] = None,
        episodic_linker: Optional[Any] = None,
        consolidator: Optional[Any] = None,
        summarizer: Optional[ProgressiveSummarizer] = None,
        summarize_period_s: int = 60 * 30,
        concept_period_s: int = 10 * 60,
        episodic_period_s: int = 15 * 60,
        jitter_frac: float = 0.25,
        summarizer_config: Optional[SummarizerConfig] = None,
        llm_summarize_fn: Optional[Callable[..., str]] = None,
    ) -> None:
        self.m = memory_store
        self.c = concept_store
        self.e = episodic_linker
        self.consolidator = consolidator
        self.concept_period = max(0, concept_period_s)
        self.episodic_period = max(0, episodic_period_s)
        self.summarize_period = max(0, summarize_period_s)
        self.jitter_frac = max(0.0, min(1.0, jitter_frac))
        self.summarizer = summarizer or ProgressiveSummarizer(
            memory_store,
            concept_store=concept_store,
            config=summarizer_config,
            llm_summarize_fn=llm_summarize_fn,
        )

        now = self._now()
        self.next_concept = now
        self.next_episodic = now
        self.next_summarize = now

        self.concept_task = TaskStats(last_run=now)
        self.episodic_task = TaskStats(last_run=now)
        self.summary_task = TaskStats(last_run=now)

        self._task_meta: Dict[str, Dict[str, Any]] = {
            "concept": {
                "task_attr": "concept_task",
                "period_attr": "concept_period",
                "next_attr": "next_concept",
                "limit": 500,
                "min_period": 60.0,
                "max_factor": 6.0,
            },
            "episodic": {
                "task_attr": "episodic_task",
                "period_attr": "episodic_period",
                "next_attr": "next_episodic",
                "limit": 800,
                "min_period": 120.0,
                "max_factor": 6.0,
            },
            "summarize": {
                "task_attr": "summary_task",
                "period_attr": "summarize_period",
                "next_attr": "next_summarize",
                "limit": None,
                "min_period": 180.0,
                "max_factor": 6.0,
            },
        }

        self._controllers: Dict[str, AdaptivePeriodController] = {}
        for name, meta in self._task_meta.items():
            period = float(getattr(self, meta["period_attr"]))
            min_period = meta.get("min_period")
            max_factor = meta.get("max_factor") or 4.0
            max_period = None if period <= 0 else period * max_factor
            self._controllers[name] = AdaptivePeriodController(
                period,
                name=name,
                min_period=min_period,
                max_period=max_period,
            )

        self.drift_events: Deque[Dict[str, Any]] = deque(maxlen=128)
        self._drift_cursors: Dict[str, int] = {name: 0 for name in self._controllers}
        self.last_llm_guidance: Optional[Dict[str, Any]] = None
        self.llm_guidance_history: Deque[Dict[str, Any]] = deque(maxlen=50)

    # ------------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Run maintenance tasks according to their schedule."""

        now = self._now(now)
        stats: Dict[str, Any] = {"ts": now}

        if self.concept_period and now >= self.next_concept:
            processed_concepts = 0
            started_at = self._now()
            try:
                processed_concepts = self._run_concept_update(now)
                stats["concepts"] = processed_concepts
            finally:
                self._finalize_task_run(
                    "concept",
                    now=now,
                    started_at=started_at,
                    processed=processed_concepts,
                    result=stats.get("concepts"),
                )

        if self.episodic_period and now >= self.next_episodic:
            processed_links = 0
            started_at = self._now()
            try:
                processed_links = self._run_episodic_linking(now)
                stats["episodic"] = processed_links
            finally:
                self._finalize_task_run(
                    "episodic",
                    now=now,
                    started_at=started_at,
                    processed=processed_links,
                    result=stats.get("episodic"),
                )

        if self.summarize_period and now >= self.next_summarize:
            step_stats: Any = None
            started_at = self._now()
            try:
                step_stats = self.summarizer.step(now)
                stats["summaries"] = step_stats
            finally:
                self._finalize_task_run(
                    "summarize",
                    now=now,
                    started_at=started_at,
                    processed=None,
                    result=stats.get("summaries"),
                )

        guidance = self._llm_prioritize(now, stats)
        if guidance:
            stats["llm_guidance"] = guidance

        return stats

    # ------------------------------------------------------------------
    def on_new_items(self, urgency: float = 0.5) -> None:
        """Nudge schedules so new items are processed sooner."""

        now = self._now()
        urgency = max(0.0, min(1.0, urgency))
        if self.concept_period:
            self.next_concept = min(self.next_concept, now + urgency * self.concept_period * 0.5)
        if self.episodic_period:
            self.next_episodic = min(self.next_episodic, now + urgency * self.episodic_period * 0.5)
        if self.summarize_period:
            self.next_summarize = min(self.next_summarize, now + urgency * self.summarize_period * 0.5)

    # ------------------------------------------------------------------
    def get_task_metrics(self, task: str) -> Dict[str, Any]:
        """Expose a snapshot of adaptive statistics for observability."""

        meta = self._task_meta.get(task)
        if not meta:
            return {}
        stats: TaskStats = getattr(self, meta["task_attr"], None)
        controller = self._controllers.get(task)
        if not stats:
            return {}
        snapshot: Dict[str, Any] = {
            "period": getattr(self, meta["period_attr"]),
            "next_run": getattr(self, meta["next_attr"]),
            "iterations": stats.iterations,
            "history": list(stats.history),
        }
        if controller:
            snapshot.update(
                {
                    "ema_load": controller.load.value,
                    "ema_rate": controller.rate.value,
                    "drifts": list(controller.drift_log),
                }
            )
        return snapshot

    def get_drift_events(self, limit: int = 20) -> Deque[Dict[str, Any]]:
        """Return recent drift events capped by *limit*."""

        if limit <= 0:
            return deque()
        sliced = list(self.drift_events)[-limit:]
        return deque(sliced)

    def _llm_prioritize(self, now: float, stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = self._build_llm_payload(now, stats)
        if not payload:
            return None

        try:
            response = try_call_llm_dict("semantic_memory_manager", input_payload=payload)
        except Exception:  # pragma: no cover - prudence
            LOGGER.debug("Guidance LLM pour la mémoire sémantique indisponible", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        guidance = dict(response)
        applied = self._apply_llm_prioritization(guidance.get("tasks"), now)
        if applied:
            guidance["applied_adjustments"] = applied

        self.last_llm_guidance = guidance
        self.llm_guidance_history.append({"ts": now, "guidance": guidance})
        return guidance

    def _build_llm_payload(self, now: float, stats: Dict[str, Any]) -> Dict[str, Any]:
        tasks_info: list[Dict[str, Any]] = []
        for name, meta in self._task_meta.items():
            snapshot = self.get_task_metrics(name)
            controller = self._controllers.get(name)
            task_stats: TaskStats = getattr(self, meta["task_attr"])
            next_run = float(snapshot.get("next_run", now) or now)
            history = list(snapshot.get("history", []))
            if history:
                history = history[-3:]
            entry = {
                "name": name,
                "period": float(snapshot.get("period", getattr(self, meta["period_attr"])) or 0.0),
                "next_run_in": max(0.0, next_run - now),
                "iterations": int(task_stats.iterations),
                "last_processed": history[-1]["processed"] if history else None,
                "ema_load": controller.load.value if controller else None,
                "ema_rate": controller.rate.value if controller else None,
                "last_result": task_stats.last_result,
            }
            if history:
                entry["history"] = history
            tasks_info.append(entry)

        concept_trails: list[Dict[str, Any]] = []
        if self.c and hasattr(self.c, "get_top_concepts"):
            try:
                top_concepts = self.c.get_top_concepts(3)
            except Exception:
                top_concepts = []
            if top_concepts and hasattr(self.c, "walk_associations"):
                for concept in top_concepts:
                    label = getattr(concept, "label", None)
                    if not label:
                        continue
                    try:
                        associations = self.c.walk_associations(label, max_depth=2, limit=12)
                    except Exception:
                        associations = []
                    if associations:
                        concept_trails.append({"concept": label, "associations": associations})

        payload: Dict[str, Any] = {
            "timestamp": now,
            "tasks": tasks_info,
            "drift_events": list(self.drift_events)[-5:],
            "recent_stats": stats,
        }
        if concept_trails:
            payload["concept_trails"] = concept_trails
        return payload

    def _resolve_task_name(self, raw: Any) -> Optional[str]:
        if not raw:
            return None
        name = str(raw).strip().lower()
        if name in self._task_meta:
            return name
        return _TASK_ALIASES.get(name)

    def _apply_llm_prioritization(
        self, tasks: Any, now: float
    ) -> Dict[str, Dict[str, float]]:
        if not isinstance(tasks, list):
            return {}

        applied: Dict[str, Dict[str, float]] = {}
        for entry in tasks:
            if not isinstance(entry, Mapping):
                continue
            resolved = self._resolve_task_name(entry.get("task"))
            if not resolved or resolved not in self._task_meta:
                continue
            controller = self._controllers.get(resolved)
            meta = self._task_meta[resolved]
            period_attr = meta["period_attr"]
            next_attr = meta["next_attr"]
            current_period = float(getattr(self, period_attr) or 0.0)
            if current_period <= 0:
                continue

            category = str(entry.get("category") or "").lower().replace(" ", "_")
            if category == "urgent":
                target_period = max(controller.min_period, current_period * 0.5) if controller else current_period * 0.5
                next_base = controller.min_period if controller else target_period * 0.5
            elif category in {"court_terme", "court-terme", "court_term", "court"}:
                target_period = max(controller.min_period, current_period * 0.75) if controller else current_period * 0.75
                next_base = target_period * 0.5
            elif category in {"long_terme", "long-terme", "long_term", "long"}:
                candidate = current_period * 1.25
                max_period = controller.max_period if controller else None
                target_period = min(max_period, candidate) if max_period else candidate
                next_base = target_period
            else:
                continue

            if controller:
                controller.period = target_period
            setattr(self, period_attr, target_period)
            next_run = now + self._jitter(next_base if next_base > 0 else target_period)
            setattr(self, next_attr, next_run)
            applied[resolved] = {
                "period": target_period,
                "previous_period": current_period,
                "next_run": next_run,
                "category": category,
            }

        return applied

    # ------------------------------------------------------------------
    def _run_concept_update(self, now: float) -> int:
        # Decay + update from recent items
        if hasattr(self.c, "decay_tick"):
            try:
                self.c.decay_tick(now)
            except Exception:
                pass

        # pull recent items since last window
        window = now - 3600  # last hour
        try:
            recent = list(self.m.list_items({"newer_than_ts": window, "limit": 500}) or [])
        except Exception:
            recent = []
        if hasattr(self.c, "update_from_items"):
            try:
                self.c.update_from_items(recent)
            except Exception:
                pass
        return len(recent)

    def _run_episodic_linking(self, now: float) -> int:
        if not self.e:
            return 0
        window = now - 4 * 3600  # last 4 hours
        try:
            recent = list(self.m.list_items({"newer_than_ts": window, "limit": 800}) or [])
        except Exception:
            recent = []
        try:
            created = self.e.link(recent) or []
        except Exception:
            created = []
        return len(created)

    def _jitter(self, period: float) -> float:
        if period <= 0:
            return 0.0
        j = self.jitter_frac
        spread = period * j
        return period + random.uniform(-spread, +spread)

    def _finalize_task_run(
        self,
        task: str,
        *,
        now: float,
        started_at: float,
        processed: Optional[int],
        result: Any,
    ) -> None:
        meta = self._task_meta.get(task)
        if not meta:
            return

        stats: TaskStats = getattr(self, meta["task_attr"])
        period_attr = meta["period_attr"]
        next_attr = meta["next_attr"]
        period = getattr(self, period_attr)
        limit = meta.get("limit")

        duration = max(self._now() - started_at, 0.0)
        if processed is None:
            processed = self._infer_processed_count(result)
        processed = max(0, int(processed or 0))

        stats.last_run = now
        stats.iterations += 1
        stats.last_result = result

        history_entry = {
            "ts": now,
            "period_before": period,
            "processed": processed,
            "duration": duration,
            "limit": limit,
        }
        stats.history.append(history_entry)

        controller = self._controllers.get(task)
        if controller and period:
            new_period = controller.update(
                period,
                processed=processed,
                duration=duration,
                limit=limit,
                wall_ts=now,
            )
            setattr(self, period_attr, new_period)
            next_run = now + self._jitter(new_period) if new_period else now
            history_entry["period_after"] = new_period
            setattr(self, next_attr, next_run)
            history_entry["next_run"] = next_run
            self._capture_drift_events(task)
        else:
            next_run = now + self._jitter(period) if period else now
            history_entry["period_after"] = period
            history_entry["next_run"] = next_run
            setattr(self, next_attr, next_run)

    def _capture_drift_events(self, task: str) -> None:
        controller = self._controllers.get(task)
        if not controller:
            return
        cursor = self._drift_cursors.get(task, 0)
        if len(controller.drift_log) <= cursor:
            self._drift_cursors[task] = len(controller.drift_log)
            return
        for drift in list(controller.drift_log)[cursor:]:
            event = dict(drift)
            event["task"] = task
            self.drift_events.append(event)
        self._drift_cursors[task] = len(controller.drift_log)

    def _infer_processed_count(self, result: Any) -> int:
        if isinstance(result, (int, float)):
            return int(result)
        if isinstance(result, dict):
            for key in ("processed", "items", "count", "written", "created"):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    return int(value)
        return 0

    def _now(self, override: Optional[float] = None) -> float:
        if override is not None:
            return float(override)
        if hasattr(self.m, "now"):
            try:
                return float(self.m.now())
            except Exception:
                pass
        return time.time()


__all__ = ["SemanticMemoryManager", "SummarizerConfig", "ProgressiveSummarizer"]
