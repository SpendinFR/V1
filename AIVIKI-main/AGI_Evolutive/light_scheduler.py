"""Lightweight, tick-driven scheduler used by the orchestrator layer."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Dict, Mapping, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)

__all__ = ["LightScheduler"]


class LightScheduler:
    """Minimal scheduler that runs jobs when :meth:`tick` is called.

    This helper is deliberately synchronous and state-less between ticks.  It
    targets integration tests and tight control loops where the caller already
    drives the execution.  The background, thread-based scheduler remains
    available under :mod:`AGI_Evolutive.runtime.scheduler`.
    """

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._llm_notes: Dict[str, Dict[str, Any]] = {}

    def register_job(
        self,
        name: str,
        interval_sec: float,
        func: Callable[[], None],
        *,
        jitter_sec: float = 0.0,
    ) -> None:
        """Register a job to be executed on subsequent :meth:`tick` calls.

        Args:
            name: Identifier used for stats retrieval.
            interval_sec: Base interval between executions in seconds.
            func: Callable to execute.
            jitter_sec: Optional absolute jitter applied after each run.

        Raises:
            ValueError: If the provided arguments are not valid.
        """

        if not callable(func):
            raise ValueError("Job func must be callable")
        if interval_sec <= 0:
            raise ValueError("interval_sec must be positive")
        if jitter_sec < 0:
            raise ValueError("jitter_sec must be non-negative")

        interval = max(5.0, float(interval_sec))
        jitter = float(jitter_sec)

        self.jobs[name] = {
            "interval": interval,
            "func": func,
            "last": 0.0,
            "jitter": jitter,
            "pending_jitter": 0.0,
            "exec_count": 0,
            "total_duration": 0.0,
            "name": name,
        }

    def tick(self) -> None:
        now = time.time()
        for job in self.jobs.values():
            wait_interval = job["interval"] + job["pending_jitter"]
            if wait_interval < 0:
                wait_interval = 0
            if now - job["last"] >= wait_interval:
                start = time.perf_counter()
                try:
                    job["func"]()
                finally:
                    job["last"] = time.time()
                    elapsed = time.perf_counter() - start
                    job["exec_count"] += 1
                    job["total_duration"] += elapsed
                    if job["jitter"]:
                        job["pending_jitter"] = random.uniform(-job["jitter"], job["jitter"])
                    else:
                        job["pending_jitter"] = 0.0
                    self._llm_adjust_interval(job, elapsed)

    def stats(self) -> Mapping[str, Dict[str, Any]]:
        """Return execution statistics for registered jobs."""

        return {
            name: {
                "interval": job["interval"],
                "last": job["last"],
                "exec_count": job["exec_count"],
                "total_duration": job["total_duration"],
                "jitter": job["jitter"],
                "llm": job.get("llm_recommendation"),
            }
            for name, job in self.jobs.items()
        }

    def _llm_adjust_interval(self, job: Dict[str, Any], elapsed: float) -> None:
        payload = {
            "job_id": job.get("name"),
            "interval": job.get("interval"),
            "elapsed": elapsed,
            "exec_count": job.get("exec_count"),
            "total_duration": job.get("total_duration"),
        }
        response = try_call_llm_dict(
            "light_scheduler",
            input_payload=payload,
            logger=logger,
        )
        if not response:
            job.pop("llm_recommendation", None)
            return

        recommended = response.get("interval_suggere")
        try:
            if recommended is not None:
                job["interval"] = max(1.0, float(recommended))
        except (TypeError, ValueError):
            logger.debug("Invalid interval suggested by LLM: %r", recommended)
        job["llm_recommendation"] = dict(response)
