"""High-level orchestrator for long-horizon memory access.

This module introduces :class:`LongTermMemoryHub`, a lightweight façade that
coordinates the persistent :class:`~AGI_Evolutive.memory.memory_store.MemoryStore`,
progressive digests produced by the summariser, the belief graph and the
``SelfModel``.  The goal is to expose an always-on view of the agent history,
aggregating raw traces, hierarchical summaries and structured knowledge so the
agent can answer questions about very old interactions without bespoke glue
code in every caller.

The hub deliberately keeps dependencies optional: every source can be absent and
callers still receive a coherent (possibly empty) snapshot.  When the
summariser is available the hub can request a refresh pass so the digest levels
stay in sync with recent raw memories.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set
import time

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


DAY_SECONDS = 24 * 3600


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except Exception:
        return default


def _to_utc(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return ""


@dataclass
class DigestDetails:
    """Structured payload returned by :meth:`LongTermMemoryHub.describe_period`."""

    level: str
    start_ts: float
    end_ts: float
    summary: str
    digest_id: Optional[str]
    lineage: Sequence[str]
    coverage: Sequence[Dict[str, Any]]
    llm_analysis: Optional[Mapping[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "start_iso": _to_utc(self.start_ts),
            "end_iso": _to_utc(self.end_ts),
            "summary": self.summary,
            "digest_id": self.digest_id,
            "lineage": list(self.lineage),
            "coverage": list(self.coverage),
            "llm_analysis": dict(self.llm_analysis) if isinstance(self.llm_analysis, Mapping) else None,
        }


class LongTermMemoryHub:
    """Aggregate views across persistent memories and knowledge bases."""

    DIGEST_KINDS: Sequence[str] = ("digest.daily", "digest.weekly", "digest.monthly")
    KNOWLEDGE_KINDS: Sequence[str] = ("lesson", "insight", "knowledge")

    def __init__(
        self,
        memory_store: Optional[Any],
        *,
        summarizer: Optional[Any] = None,
        belief_graph: Optional[Any] = None,
        self_model: Optional[Any] = None,
        goals: Optional[Any] = None,
        auto_refresh: bool = False,
    ) -> None:
        self.memory_store = memory_store
        self.summarizer = summarizer
        self.belief_graph = belief_graph
        self.self_model = self_model
        self.goals = goals
        self.auto_refresh = bool(auto_refresh)

    # ------------------------------------------------------------------
    # Binding helpers
    def rebind(
        self,
        *,
        memory_store: Optional[Any] = None,
        summarizer: Optional[Any] = None,
        belief_graph: Optional[Any] = None,
        self_model: Optional[Any] = None,
        goals: Optional[Any] = None,
    ) -> None:
        if memory_store is not None:
            self.memory_store = memory_store
        if summarizer is not None:
            self.summarizer = summarizer
        if belief_graph is not None:
            self.belief_graph = belief_graph
        if self_model is not None:
            self.self_model = self_model
        if goals is not None:
            self.goals = goals

    # ------------------------------------------------------------------
    def refresh(self, *, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Trigger a summariser maintenance step if available."""

        if not self.summarizer:
            return None
        try:
            return self.summarizer.step(now)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _ensure_fresh(self) -> None:
        if self.auto_refresh:
            self.refresh()

    # ------------------------------------------------------------------
    def _store_items(
        self,
        *,
        kind: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        filters: Optional[MutableMapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.memory_store:
            return []
        query = dict(filters or {})
        if kind is not None:
            kinds = list(kind)
            if len(kinds) == 1:
                query["kind"] = kinds[0]
            else:
                query["kind"] = kinds
        if limit is not None:
            query["limit"] = limit
        try:
            results = self.memory_store.list_items(query)
        except Exception:
            return []
        if not isinstance(results, list):
            return []
        return [dict(item) for item in results]

    # ------------------------------------------------------------------
    def _normalize_entry(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(item)
        payload.setdefault("kind", str(item.get("kind") or "generic"))
        ts = item.get("ts") or item.get("timestamp") or item.get("t")
        payload["ts"] = _safe_float(ts, time.time())
        payload.setdefault("id", item.get("id"))
        payload.setdefault("text", item.get("text") or item.get("content"))
        return payload

    # ------------------------------------------------------------------
    def timeline(
        self,
        *,
        limit_recent: int = 256,
        limit_digests: int = 120,
        include_raw: bool = True,
        include_digests: bool = True,
        since_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return a chronologically sorted view mixing raw traces and digests."""

        self._ensure_fresh()
        combined: List[Dict[str, Any]] = []
        recent: List[Dict[str, Any]] = []
        digests_by_level: Dict[str, List[Dict[str, Any]]] = {
            "daily": [],
            "weekly": [],
            "monthly": [],
        }

        if include_raw and limit_recent:
            raw_items = self._store_items(limit=limit_recent)
            if since_ts is not None:
                raw_items = [item for item in raw_items if _safe_float(item.get("ts"), 0.0) >= since_ts]
            recent = [self._normalize_entry(item) for item in raw_items]
            combined.extend(recent)

        if include_digests and limit_digests:
            for kind in self.DIGEST_KINDS:
                entries = self._store_items(kind=[kind], limit=limit_digests)
                if since_ts is not None:
                    entries = [item for item in entries if _safe_float(item.get("ts"), 0.0) >= since_ts]
                normalized = [self._normalize_entry(item) for item in entries]
                level = kind.split(".", 1)[-1]
                digests_by_level[level].extend(normalized)
                combined.extend(normalized)

        combined.sort(key=lambda item: item.get("ts", 0.0))

        return {
            "combined": combined,
            "recent": recent,
            "digests": {level: sorted(entries, key=lambda item: item.get("ts", 0.0)) for level, entries in digests_by_level.items()},
        }

    # ------------------------------------------------------------------
    def expand_digest(self, digest_id: str) -> List[Dict[str, Any]]:
        if not self.memory_store or not digest_id:
            return []
        try:
            digest = self.memory_store.get_item(digest_id)
        except Exception:
            digest = None
        if not digest:
            return []
        lineage = digest.get("lineage") or []
        expanded: List[Dict[str, Any]] = []
        for child_id in lineage:
            try:
                child = self.memory_store.get_item(child_id)
            except Exception:
                child = None
            if child:
                expanded.append(self._normalize_entry(child))
        return expanded

    # ------------------------------------------------------------------
    def describe_period(
        self,
        *,
        days_ago: int,
        level: str = "daily",
    ) -> Optional[DigestDetails]:
        """Locate a digest covering the requested period and expand its lineage."""

        if days_ago < 0:
            return None
        self._ensure_fresh()
        now = time.time()
        start = now - (days_ago + 1) * DAY_SECONDS
        end = now - days_ago * DAY_SECONDS
        lookup_level = f"digest.{level}" if not level.startswith("digest.") else level
        items = self._store_items(kind=[lookup_level], limit=400)
        target: Optional[Dict[str, Any]] = None
        for item in items:
            metadata = item.get("metadata") or {}
            span_start = _safe_float(metadata.get("start_ts"), _safe_float(item.get("ts")))
            span_end = _safe_float(metadata.get("end_ts"), span_start)
            if span_start <= end and span_end >= start:
                target = item
                break
        if target is None:
            return None

        lineage = list(target.get("lineage") or [])
        coverage = self.expand_digest(str(target.get("id"))) if lineage else []
        metadata = target.get("metadata") or {}
        details = DigestDetails(
            level=level,
            start_ts=_safe_float(metadata.get("start_ts"), start),
            end_ts=_safe_float(metadata.get("end_ts"), end),
            summary=str(target.get("text") or ""),
            digest_id=str(target.get("id")) if target.get("id") else None,
            lineage=lineage,
            coverage=coverage,
        )
        llm_payload = {
            "level": level,
            "summary": details.summary,
            "start_ts": details.start_ts,
            "end_ts": details.end_ts,
            "coverage": [
                {
                    "id": item.get("id"),
                    "kind": item.get("kind"),
                    "ts": item.get("ts"),
                    "text": item.get("text"),
                }
                for item in coverage[:12]
            ],
        }
        llm_response = try_call_llm_dict(
            "memory_long_term_digest",
            input_payload=llm_payload,
            logger=LOGGER,
        )
        if llm_response:
            details.llm_analysis = llm_response
        return details

    # ------------------------------------------------------------------
    def _belief_entries(self, top_n: int) -> List[Dict[str, Any]]:
        graph = self.belief_graph
        if not graph or top_n <= 0:
            return []
        try:
            if hasattr(graph, "iter_beliefs"):
                beliefs: Iterable[Any] = graph.iter_beliefs()
            elif hasattr(graph, "all"):
                beliefs = graph.all()
            else:
                beliefs = []
        except Exception:
            return []
        scored: List[Dict[str, Any]] = []
        for belief in beliefs:
            try:
                confidence = float(getattr(belief, "confidence", 0.0))
            except Exception:
                confidence = 0.0
            entry = {
                "id": getattr(belief, "id", None),
                "subject": getattr(belief, "subject_label", getattr(belief, "subject", "")),
                "relation": getattr(belief, "relation", ""),
                "value": getattr(belief, "value_label", getattr(belief, "value", "")),
                "confidence": confidence,
                "updated_at": _safe_float(getattr(belief, "updated_at", 0.0)),
                "polarity": getattr(belief, "polarity", 1),
            }
            scored.append(entry)
        scored.sort(key=lambda item: (item["confidence"], item["updated_at"]), reverse=True)
        return scored[:top_n]

    # ------------------------------------------------------------------
    def _completed_goals(self, top_n: int) -> List[Dict[str, Any]]:
        records = self._store_items(kind=["goal_completion"], limit=top_n * 4)
        if not records:
            return []
        entries = []
        for item in records:
            entries.append(
                {
                    "goal_id": item.get("goal_id"),
                    "description": item.get("description"),
                    "completed_at": _safe_float(item.get("completed_at"), _safe_float(item.get("ts"))),
                    "criteria": list(item.get("criteria", [])),
                    "metadata": item.get("metadata"),
                }
            )
        entries.sort(key=lambda e: e.get("completed_at", 0.0), reverse=True)
        return entries[:top_n]

    # ------------------------------------------------------------------
    def _lessons(self, top_n: int) -> List[Dict[str, Any]]:
        records = self._store_items(kind=self.KNOWLEDGE_KINDS, limit=top_n * 3)
        lessons: List[Dict[str, Any]] = []
        for item in records:
            lessons.append(
                {
                    "kind": item.get("kind"),
                    "ts": _safe_float(item.get("ts")),
                    "text": item.get("text") or item.get("content"),
                    "tags": list(item.get("tags", [])),
                }
            )
        lessons.sort(key=lambda e: e.get("ts", 0.0), reverse=True)
        return lessons[:top_n]

    # ------------------------------------------------------------------
    def knowledge_snapshot(
        self,
        *,
        top_beliefs: int = 20,
        top_completed_goals: int = 12,
        top_lessons: int = 12,
    ) -> Dict[str, Any]:
        return {
            "beliefs": self._belief_entries(top_beliefs),
            "completed_goals": self._completed_goals(top_completed_goals),
            "lessons": self._lessons(top_lessons),
        }

    # ------------------------------------------------------------------
    def full_history(
        self,
        *,
        include_raw: bool = True,
        include_digests: bool = True,
        include_expanded: bool = True,
        include_knowledge: bool = True,
        include_self_model: bool = True,
        limit_recent: int = 512,
        limit_digests: int = 180,
        since_ts: Optional[float] = None,
        top_beliefs: int = 20,
        top_completed_goals: int = 12,
        top_lessons: int = 12,
        self_model_max_items: int = 6,
    ) -> Dict[str, Any]:
        """Return an all-time bundle mixing timeline, knowledge and self lenses."""

        timeline = self.timeline(
            include_raw=include_raw,
            include_digests=include_digests,
            limit_recent=limit_recent,
            limit_digests=limit_digests,
            since_ts=since_ts,
        )

        combined = list(timeline.get("combined", []))
        stats: Dict[str, Any] = {
            "total_entries": len(combined),
            "raw_count": 0,
            "digest_count": 0,
            "coverage_entries": 0,
            "oldest_ts": combined[0].get("ts") if combined else None,
            "newest_ts": combined[-1].get("ts") if combined else None,
        }

        raw_seen: Set[str] = set()
        for item in combined:
            kind = str(item.get("kind", ""))
            if kind.startswith("digest."):
                continue
            item_id = item.get("id")
            if item_id:
                raw_seen.add(str(item_id))
            else:
                raw_seen.add(f"{kind}@{item.get('ts')}")
        stats["raw_count"] = len(raw_seen)

        digest_sections = timeline.get("digests", {})
        stats["digest_count"] = sum(len(entries) for entries in digest_sections.values())

        expanded_payload: Dict[str, List[Dict[str, Any]]] = {}
        coverage_total = 0
        if include_expanded and include_digests:
            for level, entries in digest_sections.items():
                level_expanded: List[Dict[str, Any]] = []
                for entry in entries:
                    digest_id = entry.get("id")
                    coverage = self.expand_digest(str(digest_id)) if digest_id else []
                    coverage_total += len(coverage)
                    level_expanded.append(
                        {
                            "digest": dict(entry),
                            "entries": [dict(item) for item in coverage],
                        }
                    )
                if level_expanded:
                    expanded_payload[level] = level_expanded
        stats["coverage_entries"] = coverage_total

        result: Dict[str, Any] = {"timeline": timeline, "stats": stats}
        if include_expanded:
            result["expanded"] = expanded_payload
        if include_knowledge:
            result["knowledge"] = self.knowledge_snapshot(
                top_beliefs=top_beliefs,
                top_completed_goals=top_completed_goals,
                top_lessons=top_lessons,
            )
        if include_self_model:
            result["self_model"] = self.self_model_snapshot(max_items=self_model_max_items)
        return result

    # ------------------------------------------------------------------
    def self_model_snapshot(self, *, max_items: int = 6) -> Dict[str, Any]:
        model = self.self_model
        if not model:
            return {}
        builder = getattr(model, "build_synthesis", None)
        if callable(builder):
            try:
                return builder(max_items=max_items)
            except TypeError:
                return builder()
            except Exception:
                return {}
        desc = getattr(model, "describe", None)
        if callable(desc):
            try:
                payload = desc()
            except Exception:
                payload = {}
            return payload or {}
        return {}

    # ------------------------------------------------------------------
    def build_snapshot(self) -> Dict[str, Any]:
        snapshot = self.full_history(
            include_expanded=False,
            include_knowledge=True,
            include_self_model=True,
        )
        return {
            "timeline": snapshot.get("timeline", {}),
            "knowledge": snapshot.get("knowledge", {}),
            "self_model": snapshot.get("self_model", {}),
            "stats": snapshot.get("stats", {}),
        }

