import json
import logging
import os
import threading
import time
from collections import deque
from typing import Optional

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict
from .errors import ErrorReport
from .trace import current_trace_id


LOGGER = logging.getLogger(__name__)


class Telemetry:
    def __init__(self, maxlen=2000):
        self.events = deque(maxlen=maxlen)
        self._jsonl_path = None
        self._console = False
        self._lock = threading.RLock()

    def enable_jsonl(self, path="logs/events.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._jsonl_path = path

    def enable_console(self, on=True):
        self._console = bool(on)

    def log(
        self,
        event_type,
        subsystem,
        data=None,
        level="info",
        *,
        trace_id: Optional[str] = None,
        error: Optional[ErrorReport] = None,
    ):
        payload = data or {}
        event = {
            "schema_version": 1,
            "ts": time.time(),
            "type": event_type,
            "subsystem": subsystem,
            "level": level,
            "data": payload,
        }
        if trace_id is None:
            trace_id = current_trace_id()
        if trace_id:
            event["trace_id"] = trace_id
        if error:
            event["error"] = {
                "code": error.code,
                "message": error.message,
                "details": error.details or {},
            }
        try:
            annotation = self._llm_annotate(event)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug("LLM annotation failed: %s", exc, exc_info=True)
            annotation = None
        if annotation:
            event["llm_annotation"] = annotation
        with self._lock:
            self.events.append(event)
            if self._jsonl_path:
                try:
                    with open(self._jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(json_sanitize(event), ensure_ascii=False) + "\n")
                except Exception:
                    pass
        if self._console and level in ("info", "warn", "error"):
            ts = time.strftime("%H:%M:%S", time.localtime(event["ts"]))
            print(f"[{ts}] {subsystem}/{level} {event_type} :: {event['data']}")

    def tail(self, n=50):
        return list(self.events)[-max(0, n):]

    def snapshot(self):
        by_sub = {}
        for e in self.events:
            by_sub[e["subsystem"]] = by_sub.get(e["subsystem"], 0) + 1
        return {"events_count": len(self.events), "events_by_subsystem": by_sub}

    def _llm_annotate(self, event: dict) -> Optional[dict]:
        response = try_call_llm_dict(
            "telemetry_annotation",
            input_payload=event,
            logger=LOGGER,
            max_retries=0,
        )
        if not response:
            return None
        summary = response.get("summary")
        severity = response.get("severity")
        if not isinstance(summary, str):
            return None
        if severity is not None and not isinstance(severity, str):
            severity = None
        return {
            "event_id": response.get("event_id", ""),
            "summary": summary,
            "severity": severity,
            "routine": bool(response.get("routine", False)),
            "notes": response.get("notes", ""),
        }
