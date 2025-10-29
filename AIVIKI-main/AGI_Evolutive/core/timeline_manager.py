from __future__ import annotations

import logging
import re
from time import monotonic, monotonic_ns
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict
from AGI_Evolutive.memory.embedding_adapters import AdaptiveSemanticEmbedder


STOPWORDS = {
    "et",
    "ou",
    "de",
    "des",
    "le",
    "la",
    "les",
    "un",
    "une",
    "du",
    "au",
    "aux",
    "en",
    "à",
    "pour",
    "par",
    "sur",
    "dans",
    "que",
    "qui",
    "se",
    "ce",
    "ces",
    "ne",
    "pas",
}


LOGGER = logging.getLogger(__name__)


class TimelineManager:
    """Belief snapshots & deltas per topic, persisted in memory."""

    def __init__(self, memory_store=None) -> None:
        self.memory = memory_store
        # Local event journal per topic to enable replay/time-travel inspection.
        self._max_events_per_topic = 1000
        self._compaction_tail = 200
        self._events = defaultdict(self._new_topic_journal)
        self._last_state = {}
        self._last_delta = {}
        self._embedder: Optional[AdaptiveSemanticEmbedder] = None

    # ------------------------------------------------------------------
    # Event helpers
    def _new_topic_journal(self) -> deque:
        return deque(maxlen=self._max_events_per_topic)

    @staticmethod
    def _monotonic_now() -> float:
        return monotonic()

    @staticmethod
    def _monotonic_ns() -> int:
        return monotonic_ns()

    def _new_event_meta(self, topic: str, kind: str) -> Tuple[str, float, str]:
        event_id = f"{kind}:{topic}:{self._monotonic_ns()}"
        ts_monotonic = self._monotonic_now()
        wall_clock = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return event_id, ts_monotonic, wall_clock

    def _get_embedder(self) -> AdaptiveSemanticEmbedder:
        if self._embedder is None:
            self._embedder = AdaptiveSemanticEmbedder()
        return self._embedder

    def snapshot(self, topic: str, beliefs: List[Dict]) -> str:
        event_id, ts_monotonic, wall_clock = self._new_event_meta(topic, "snapshot")
        event = {
            "kind": "belief_snapshot",
            "id": event_id,
            "topic": topic,
            "beliefs": deepcopy(beliefs),
            "ts": ts_monotonic,
            "wall_time": wall_clock,
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
        drift_by_stmt = {
            k: abs(s2.get(k, {}).get("conf", 0.0) - s1.get(k, {}).get("conf", 0.0))
            for k in common_keys
        }
        drift_text_map = {
            k: (s2.get(k) or s1.get(k) or {}).get("stmt") or str(k)
            for k in common_keys
        }
        total_drift = sum(drift_by_stmt.values())
        mean_drift = total_drift / max(1, len(common_keys))

        semantic_shifts = self._detect_semantic_shifts(topic, removed, added)

        event_id, ts_monotonic, wall_clock = self._new_event_meta(topic, "belief_delta")

        event = {
            "kind": "belief_delta",
            "id": event_id,
            "topic": topic,
            "added": added,
            "removed": removed,
            "updated": updated,
            "attribute_updates": attribute_updates,
            "confidence_drift": mean_drift,
            "confidence_drift_total": total_drift,
            "confidence_drift_map": drift_by_stmt,
            "confidence_drift_text": drift_text_map,
            "semantic_shifts": semantic_shifts,
            "ts": ts_monotonic,
            "wall_time": wall_clock,
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
        severity_map = self._score_gaps(gaps, delta)
        llm_info = delta.get("llm") if isinstance(delta, dict) else None
        scores, milestones_map = self._combine_with_llm(gaps, severity_map, llm_info)

        prioritized = sorted(
            scores.items(), key=lambda item: item[1]["score"], reverse=True
        )

        goals = []
        for idx, (gap, payload) in enumerate(prioritized):
            goals.append(
                {
                    "goal_kind": "LearnConcept",
                    "topic": topic,
                    "concept": gap,
                    "priority": idx + 1,
                    "severity": payload["heuristic"],
                    "milestones": milestones_map.get(gap) or None,
                }
            )
        return goals

    # ------------------------------------------------------------------
    # Introspection helpers
    def replay(self, topic: str, start_ts: Optional[float] = None, end_ts: Optional[float] = None) -> List[Dict]:
        """Reconstruct successive belief states between start and end timestamps."""

        events = list(self._events.get(topic, []))
        if not events:
            return []

        events.sort(key=lambda evt: float(evt.get("ts", 0.0)))
        state = {}
        timeline = []

        for event in events:
            ts = event["ts"]
            if end_ts is not None and ts > end_ts:
                break

            if event["kind"] == "belief_snapshot":
                state = {b.get("stmt"): deepcopy(b) for b in event.get("beliefs", [])}
            elif event["kind"] == "belief_delta":
                self._apply_delta(state, event)

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
        journal = self._events[topic]
        journal.append(event)
        if len(journal) >= journal.maxlen:
            self._compact_topic_journal(topic)
        if self.memory is not None and event["kind"] == "belief_snapshot":
            self.memory.add(event)
        if event["kind"] == "belief_snapshot":
            self._last_state[topic] = {b.get("stmt"): deepcopy(b) for b in event.get("beliefs", [])}
        elif event["kind"] == "belief_delta":
            self._last_state.setdefault(topic, {})
            self._apply_delta(self._last_state[topic], event)

    def _compact_topic_journal(self, topic: str) -> None:
        state_snapshot = [deepcopy(b) for b in self._last_state.get(topic, {}).values()]
        snapshot_id, ts_monotonic, wall_clock = self._new_event_meta(topic, "compacted_snapshot")
        snapshot_event = {
            "kind": "belief_snapshot",
            "id": snapshot_id,
            "topic": topic,
            "beliefs": state_snapshot,
            "ts": ts_monotonic,
            "wall_time": wall_clock,
            "meta": {"compacted": True},
        }
        tail = list(self._events[topic])[-self._compaction_tail :]
        self._events[topic] = self._new_topic_journal()
        self._events[topic].append(snapshot_event)
        for evt in tail:
            self._events[topic].append(evt)

    def _apply_delta(self, state: Dict[str, Dict], delta: Dict) -> None:
        for removed in delta.get("removed", []) or []:
            state.pop(removed.get("stmt"), None)
        for added in delta.get("added", []) or []:
            stmt = added.get("stmt")
            if stmt is None:
                continue
            state[stmt] = {**deepcopy(state.get(stmt, {})), **deepcopy(added)}
        for updated in delta.get("updated", []) or []:
            stmt = updated.get("stmt")
            if stmt is None:
                continue
            base = deepcopy(state.get(stmt, {}))
            base.update(deepcopy(updated))
            state[stmt] = base
        for attribute_update in delta.get("attribute_updates", []) or []:
            stmt = attribute_update.get("stmt")
            if stmt is None:
                continue
            base = deepcopy(state.get(stmt, {}))
            base.update(deepcopy(attribute_update.get("changes", {})))
            state[stmt] = base

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

    def _detect_semantic_shifts(self, topic: str, removed: List[Dict], added: List[Dict]) -> List[Dict]:
        if not removed or not added:
            return []

        embedder = self._get_embedder()
        added_by_topic: List[Tuple[Dict, Dict[str, float]]] = []
        for new in added:
            stmt_new = self._normalize_text(new.get("stmt", ""))
            if not stmt_new:
                continue
            stmt_topic = new.get("topic") or topic
            if stmt_topic != topic:
                continue
            features = embedder(stmt_new)
            if features:
                added_by_topic.append((new, features))

        if not added_by_topic:
            return []

        shifts = []
        used_targets: set = set()
        for old in removed:
            stmt_old_raw = old.get("stmt", "")
            stmt_old = self._normalize_text(stmt_old_raw)
            if not stmt_old:
                continue
            stmt_topic = old.get("topic") or topic
            if stmt_topic != topic:
                continue
            old_features = embedder(stmt_old)
            if not old_features:
                continue
            best_score = 0.0
            best_target: Optional[Dict] = None
            for candidate, feat in added_by_topic:
                candidate_id = id(candidate)
                if candidate_id in used_targets:
                    continue
                score = self._cosine_similarity(old_features, feat)
                if score > best_score:
                    best_score = score
                    best_target = candidate
            if best_target is not None and best_score >= 0.85:
                used_targets.add(id(best_target))
                shifts.append(
                    {
                        "from": stmt_old_raw,
                        "to": best_target.get("stmt", ""),
                        "similarity": best_score,
                        "topic": topic,
                    }
                )
        return shifts

    def _score_delta(self, delta: Dict) -> float:
        if not delta:
            return 0.0
        base = float(delta.get("confidence_drift", 0.0))
        attribute_weight = 0.2 * sum(
            len((item or {}).get("changes", {})) for item in delta.get("attribute_updates", []) or []
        )
        semantic_weight = sum(shift.get("similarity", 0.0) for shift in delta.get("semantic_shifts", []))
        return base + attribute_weight + semantic_weight

    def _score_gaps(self, gaps: List[str], delta: Optional[Dict]) -> Dict[str, float]:
        severity_map = {gap: 0.0 for gap in gaps}
        if not delta:
            return severity_map

        for shift in delta.get("semantic_shifts", []) or []:
            target_stmt = shift.get("to", "")
            for gap in gaps:
                match = self._match_gap(gap, target_stmt)
                if match > 0.0:
                    severity_map[gap] += match * float(shift.get("similarity", 0.0))

        drift_map = delta.get("confidence_drift_map", {}) or {}
        drift_texts = delta.get("confidence_drift_text", {}) or {}
        for stmt, drift in drift_map.items():
            stmt_text = drift_texts.get(stmt, stmt)
            for gap in gaps:
                match = self._match_gap(gap, stmt_text)
                if match > 0.0:
                    severity_map[gap] += match * float(drift)

        for update in delta.get("attribute_updates", []) or []:
            stmt = update.get("stmt", "")
            changes = (update or {}).get("changes", {})
            weight = 0.15 * len(changes)
            if not weight:
                continue
            change_text = " ".join(
                f"{key} {value}" for key, value in changes.items() if value is not None
            )
            for gap in gaps:
                match = max(
                    self._match_gap(gap, stmt),
                    self._match_gap(gap, change_text),
                )
                if match > 0.0:
                    severity_map[gap] += match * weight

        return severity_map

    def _combine_with_llm(
        self, gaps: List[str], severity_map: Dict[str, float], llm_info: Optional[Dict]
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[Dict]]]:
        scores: Dict[str, Dict[str, float]] = {
            gap: {"heuristic": float(severity_map.get(gap, 0.0)), "score": float(severity_map.get(gap, 0.0))}
            for gap in gaps
        }
        milestones_map: Dict[str, List[Dict]] = {}
        if not isinstance(llm_info, dict):
            return scores, milestones_map

        llm_conf = float(llm_info.get("confidence", 0.7) or 0.0)
        llm_weight = min(0.3, float(llm_info.get("weight", 0.25) or 0.25))
        missing_items = [str(item) for item in llm_info.get("missing_information", []) if item]

        if llm_conf >= 0.6 and missing_items:
            for gap in gaps:
                best = 0.0
                for missing in missing_items:
                    best = max(best, self._match_gap(gap, missing))
                if best > 0.0:
                    scores[gap]["score"] = scores[gap]["heuristic"] + llm_weight * best
                else:
                    scores[gap]["score"] = scores[gap]["heuristic"]
        else:
            for gap in gaps:
                scores[gap]["score"] = scores[gap]["heuristic"]

        milestones_payload = llm_info.get("milestones")
        if isinstance(milestones_payload, list):
            for milestone in milestones_payload:
                if not isinstance(milestone, dict):
                    continue
                milestone_text = " ".join(
                    str(milestone.get(key, "")) for key in ("title", "description", "category")
                )
                if not milestone_text:
                    continue
                for gap in gaps:
                    if self._match_gap(gap, milestone_text) > 0.0:
                        milestones_map.setdefault(gap, []).append(dict(milestone))
        return scores, milestones_map

    @staticmethod
    def _match_gap(gap: str, text: str) -> float:
        gap_tokens = {
            tok
            for tok in re.findall(r"\w+", (gap or "").lower())
            if len(tok) > 2 and tok not in STOPWORDS
        }
        if not gap_tokens:
            return 0.0
        text_tokens = {
            tok for tok in re.findall(r"\w+", (text or "").lower()) if len(tok) > 1
        }
        if not text_tokens:
            return 0.0
        overlap = gap_tokens & text_tokens
        if not overlap:
            return 0.0
        return len(overlap) / float(len(gap_tokens))

    @staticmethod
    def _normalize_text(text: str) -> str:
        tokens = [tok for tok in re.findall(r"\w+", (text or "").lower()) if tok not in STOPWORDS]
        return " ".join(tokens)

    @staticmethod
    def _cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        numerator = 0.0
        for key, value in vec1.items():
            if key in vec2:
                numerator += float(value) * float(vec2[key])
        if numerator <= 0.0:
            return 0.0
        norm1 = sum(value * value for value in vec1.values()) ** 0.5
        norm2 = sum(value * value for value in vec2.values()) ** 0.5
        if not norm1 or not norm2:
            return 0.0
        return numerator / (norm1 * norm2)
