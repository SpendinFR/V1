from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


@dataclass
class Premise:
    statement: str
    source: str
    confidence: float = 0.5


@dataclass
class Option:
    option_id: str
    description: str
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class CounterExample:
    option_id: str
    description: str
    severity: float = 0.5


@dataclass
class Trace:
    trace_id: str
    topic: Optional[str] = None
    started_ts: float = field(default_factory=time.time)
    ended_ts: Optional[float] = None
    premises: List[Premise] = field(default_factory=list)
    options: List[Option] = field(default_factory=list)
    counterexamples: List[CounterExample] = field(default_factory=list)
    chosen_id: Optional[str] = None
    justification: Optional[str] = None
    stop_rules_hit: bool = False
    outcome: Optional[Dict[str, float]] = None


class ReasoningLedger:
    """Captures reasoning traces and persists them in memory."""

    def __init__(self, memory_store=None) -> None:
        self.memory = memory_store
        self._active: Dict[str, Trace] = {}

    def start_trace(self, topic: str | None = None) -> str:
        trace_id = f"rt:{uuid.uuid4().hex[:12]}"
        self._active[trace_id] = Trace(trace_id=trace_id, topic=topic, started_ts=time.time())
        return trace_id

    def log_premise(self, trace_id: str, source: str, statement: str, confidence: float = 0.5) -> None:
        trace = self._active.get(trace_id)
        if trace:
            trace.premises.append(Premise(statement=statement, source=source, confidence=confidence))

    def log_option(self, trace_id: str, option_id: str, description: str, scores: Dict[str, float]) -> None:
        trace = self._active.get(trace_id)
        if trace:
            trace.options.append(Option(option_id=option_id, description=description, scores=scores))

    def log_counterexample(self, trace_id: str, option_id: str, description: str, severity: float = 0.5) -> None:
        trace = self._active.get(trace_id)
        if trace:
            trace.counterexamples.append(
                CounterExample(option_id=option_id, description=description, severity=severity)
            )

    def select_option(
        self,
        trace_id: str,
        chosen_id: str,
        justification_text: str,
        stop_rules_hit: bool = False,
    ) -> None:
        trace = self._active.get(trace_id)
        if trace:
            trace.chosen_id = chosen_id
            trace.justification = justification_text
            trace.stop_rules_hit = stop_rules_hit

    def end_trace(self, trace_id: str, expected: float, obtained: float) -> None:
        trace = self._active.get(trace_id)
        if not trace:
            return
        trace.ended_ts = time.time()
        error = abs(float(obtained) - float(expected))
        trace.outcome = {"expected": float(expected), "obtained": float(obtained), "error": error}
        llm_response = try_call_llm_dict(
            "reasoning_ledger",
            input_payload={
                "trace_id": trace.trace_id,
                "topic": trace.topic,
                "premises": [p.__dict__ for p in trace.premises],
                "options": [o.__dict__ for o in trace.options],
                "counterexamples": [c.__dict__ for c in trace.counterexamples],
                "chosen": {
                    "id": trace.chosen_id,
                    "justification": trace.justification,
                    "stop_rules_hit": trace.stop_rules_hit,
                },
                "outcome": trace.outcome,
            },
            logger=LOGGER,
            max_retries=2,
        )
        if self.memory is not None:
            self.memory.add(
                {
                    "kind": "reasoning_trace",
                    "trace_id": trace.trace_id,
                    "topic": trace.topic,
                    "premises": [p.__dict__ for p in trace.premises],
                    "options": [o.__dict__ for o in trace.options],
                    "counterexamples": [c.__dict__ for c in trace.counterexamples],
                    "chosen": {
                        "id": trace.chosen_id,
                        "justification": trace.justification,
                        "stop_rules_hit": trace.stop_rules_hit,
                    },
                    "outcome": trace.outcome,
                    "ts": trace.ended_ts,
                    "llm_entry": llm_response.get("entry") if llm_response else None,
                }
            )
        del self._active[trace_id]

    def why(self, trace_id: str) -> str:
        trace = self._active.get(trace_id)
        if not trace:
            return "(trace closed or not found)"
        parts: List[str] = [f"We considered {len(trace.options)} option(s) and chose {trace.chosen_id}."]
        if trace.justification:
            parts.append(f"Reason: {trace.justification}.")
        if trace.counterexamples:
            tested = ", ".join({c.option_id for c in trace.counterexamples})
            parts.append(f"Counterexamples were tested against: {tested}.")
        return " ".join(parts)
