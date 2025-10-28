from __future__ import annotations

import logging
import time
from collections import defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


class TimelineManager:
    """Belief snapshots & deltas per topic, persisted in memory."""

    def __init__(self, memory_store=None) -> None:
        self.memory = memory_store
        # Local event journal per topic to enable replay/time-travel inspection.
        self._events = defaultdict(list)
        self._last_state = {}
        self._last_delta = {}

    def snapshot(self, topic: str, beliefs: List[Dict]) -> str:
        event_id = f"belief_snapshot:{int(time.time() * 1000)}"
        event = {
            "kind": "belief_snapshot",
            "id": event_id,
            "topic": topic,
            "beliefs": deepcopy(beliefs),
            "ts": time.time(),
        }
        self._record_event(topic, event)
        return event_id

    def delta(self, topic: str, b1: List[Dict], b2: List[Dict]) -> Dict:
        s1 = {b.get("stmt"): b for b in b1}
        s2 = {b.get("stmt"): b for b in b2}
        added = [deepcopy(s2[k]) for k in s2.keys() - s1.keys()]
        removed = [deepcopy(s1[k]) for k in s1.keys() - s2.keys()]

        common_keys = s1.keys() & s2.keys()
        updated = [deepcopy(s2[k]) for k in common_keys if s1[k].get("conf") != s2[k].get("conf")]

        attribute_updates = self._extract_attribute_updates(s1, s2, common_keys)
        conf_drift = sum(
            abs(s2.get(k, {}).get("conf", 0.0) - s1.get(k, {}).get("conf", 0.0))
            for k in common_keys
        )

        semantic_shifts = self._detect_semantic_shifts(removed, added)

        event = {
            "kind": "belief_delta",
            "id": f"belief_delta:{int(time.time() * 1000)}",
            "topic": topic,
            "added": added,
            "removed": removed,
            "updated": updated,
            "attribute_updates": attribute_updates,
            "confidence_drift": conf_drift,
            "semantic_shifts": semantic_shifts,
            "ts": time.time(),
        }

        enrichment = try_call_llm_dict(
            "timeline_manager",
            input_payload={
                "topic": topic,
                "delta": event,
                "previous_state": self._last_state.get(topic, {}),
            },
            logger=LOGGER,
            max_retries=2,
        )
        if enrichment:
            event["llm"] = dict(enrichment)

        self._last_delta[topic] = event
        self._record_event(topic, event)
        return event

    def project(self, topic: str, gaps: List[str]) -> List[Dict]:
        delta = self._last_delta.get(topic)
        severity = self._score_delta(delta) if delta else 0.0
        prioritized = self._prioritize_gaps(gaps, delta)
        llm_info = delta.get("llm") if isinstance(delta, dict) else None
        missing = []
        if isinstance(llm_info, dict):
            missing = [str(item) for item in llm_info.get("missing_information", []) if item]
            milestones = llm_info.get("milestones")
        else:
            milestones = None
        if missing:
            prioritized = missing + [gap for gap in prioritized if gap not in missing]
        return [
            {
                "goal_kind": "LearnConcept",
                "topic": topic,
                "concept": gap,
                "priority": idx + 1,
                "severity": severity,
                "milestones": milestones if idx == 0 else None,
            }
            for idx, gap in enumerate(prioritized)
        ]

    # ------------------------------------------------------------------
    # Introspection helpers
    def replay(self, topic: str, start_ts: Optional[float] = None, end_ts: Optional[float] = None) -> List[Dict]:
        """Reconstruct successive belief states between start and end timestamps."""

        events = self._events.get(topic, [])
        if not events:
            return []

        state = {}
        timeline = []

        for event in events:
            ts = event["ts"]
            if end_ts is not None and ts > end_ts:
                break

            if event["kind"] == "belief_snapshot":
                state = {b.get("stmt"): deepcopy(b) for b in event.get("beliefs", [])}
            elif event["kind"] == "belief_delta":
                for removed in event.get("removed", []):
                    state.pop(removed.get("stmt"), None)
                for updated in event.get("updated", []):
                    stmt = updated.get("stmt")
                    if stmt in state:
                        state[stmt] = deepcopy(updated)
                for attribute_update in event.get("attribute_updates", []):
                    stmt = attribute_update.get("stmt")
                    if stmt in state:
                        state[stmt].update(attribute_update.get("changes", {}))
                for added in event.get("added", []):
                    state[added.get("stmt")] = deepcopy(added)

            if start_ts is not None and ts < start_ts:
                continue

            timeline.append({
                "ts": ts,
                "beliefs": [deepcopy(belief) for belief in state.values()],
                "source_event": event.get("id"),
            })

        return timeline

    # ------------------------------------------------------------------
    # Internal utilities
    def _record_event(self, topic: str, event: Dict) -> None:
        self._events[topic].append(event)
        if self.memory is not None:
            self.memory.add(event)
        if event["kind"] == "belief_snapshot":
            self._last_state[topic] = {b.get("stmt"): deepcopy(b) for b in event.get("beliefs", [])}
        elif event["kind"] == "belief_delta":
            self._last_state.setdefault(topic, {})
            state = self._last_state[topic]
            for removed in event.get("removed", []):
                state.pop(removed.get("stmt"), None)
            for updated in event.get("updated", []):
                stmt = updated.get("stmt")
                if stmt is not None:
                    state[stmt] = deepcopy(updated)
            for attribute_update in event.get("attribute_updates", []):
                stmt = attribute_update.get("stmt")
                if stmt in state:
                    state[stmt].update(attribute_update.get("changes", {}))
            for added in event.get("added", []):
                stmt = added.get("stmt")
                if stmt is not None:
                    state[stmt] = deepcopy(added)

    def _extract_attribute_updates(
        self, previous: Dict[str, Dict], current: Dict[str, Dict], keys: Iterable[str]
    ) -> List[Dict]:
        updates = []
        for stmt in keys:
            old = previous.get(stmt, {})
            new = current.get(stmt, {})
            changes = {
                k: new.get(k)
                for k in set(new.keys()) | set(old.keys())
                if k not in {"stmt", "conf"} and old.get(k) != new.get(k)
            }
            if changes:
                updates.append({"stmt": stmt, "changes": changes})
        return updates

    def _detect_semantic_shifts(self, removed: List[Dict], added: List[Dict]) -> List[Dict]:
        shifts = []
        for old in removed:
            stmt_old = old.get("stmt", "")
            best_similarity = 0.0
            best_match = None
            for new in added:
                stmt_new = new.get("stmt", "")
                similarity = SequenceMatcher(None, stmt_old, stmt_new).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = new
            if best_match is not None and best_similarity >= 0.6:
                shifts.append(
                    {
                        "from": stmt_old,
                        "to": best_match.get("stmt", ""),
                        "similarity": best_similarity,
                    }
                )
        return shifts

    def _score_delta(self, delta: Dict) -> float:
        if not delta:
            return 0.0
        base = float(delta.get("confidence_drift", 0.0))
        attribute_weight = 0.5 * len(delta.get("attribute_updates", []))
        semantic_weight = sum(shift.get("similarity", 0.0) for shift in delta.get("semantic_shifts", []))
        return base + attribute_weight + semantic_weight

    def _prioritize_gaps(self, gaps: List[str], delta: Optional[Dict]) -> List[str]:
        if not delta:
            return gaps

        severity_map = {gap: 0.0 for gap in gaps}
        for shift in delta.get("semantic_shifts", []):
            for gap in gaps:
                if gap.lower() in (shift.get("to", "") or "").lower():
                    severity_map[gap] += shift.get("similarity", 0.0)

        drift_bonus = float(delta.get("confidence_drift", 0.0))
        if drift_bonus:
            for gap in gaps:
                severity_map[gap] += drift_bonus / max(len(gaps), 1)

        prioritized = sorted(gaps, key=lambda g: severity_map.get(g, 0.0), reverse=True)
        return prioritized
