"""Persistent storage for goal DAG structures."""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


@dataclass
class GoalNode:
    id: str
    description: str
    criteria: List[str] = field(default_factory=list)
    progress: float = 0.0
    value: float = 0.5
    competence: float = 0.5
    curiosity: float = 0.2
    urgency: float = 0.3
    priority: float = 0.0
    status: str = "pending"
    created_by: str = "system"
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OnlinePriorityModel:
    """Simple online logistic model for adaptive priority scoring."""

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        learning_rate: float = 0.1,
        weight_bounds: Optional[Dict[str, float]] = None,
        warmup_updates: int = 20,
    ):
        self.feature_names = feature_names or []
        self.learning_rate = learning_rate
        self.weight_bounds = weight_bounds or {"min": 0.0, "max": 1.0}
        self.warmup_updates = max(1, warmup_updates)
        self.bias = 0.0
        self.weights: Dict[str, float] = {name: 0.5 for name in self.feature_names}
        self._updates = 0

    def predict(self, features: Dict[str, float]) -> float:
        z = self.bias
        for name, value in features.items():
            weight = self.weights.get(name)
            if weight is None:
                continue
            z += weight * value
        # Logistic link for stability
        try:
            score = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            score = 1.0 if z > 0 else 0.0
        return float(max(0.0, min(1.0, score)))

    def update(self, features: Dict[str, float], target: float, importance: float = 1.0) -> None:
        target = max(0.0, min(1.0, target))
        importance = max(0.0, importance)
        if not importance:
            return
        self.ensure_features(list(features.keys()))
        prediction = self.predict(features)
        error = (prediction - target) * importance
        lr = self.learning_rate
        for name, value in features.items():
            if name not in self.weights:
                self.weights[name] = 0.5
            update = lr * error * value
            self.weights[name] = self._clip_weight(self.weights[name] - update)
        self.bias = self._clip_bias(self.bias - lr * error)
        self._updates += 1

    def confidence(self) -> float:
        if self._updates <= 0:
            return 0.0
        return max(0.0, min(1.0, self._updates / float(self.warmup_updates)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "learning_rate": self.learning_rate,
            "weight_bounds": self.weight_bounds,
            "warmup_updates": self.warmup_updates,
            "bias": self.bias,
            "weights": self.weights,
            "updates": self._updates,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OnlinePriorityModel":
        model = cls(
            feature_names=data.get("feature_names"),
            learning_rate=data.get("learning_rate", 0.1),
            weight_bounds=data.get("weight_bounds"),
            warmup_updates=data.get("warmup_updates", 20),
        )
        model.bias = float(data.get("bias", 0.0))
        model.weights = {k: float(v) for k, v in data.get("weights", {}).items()}
        model._updates = int(data.get("updates", 0))
        return model

    def ensure_features(self, feature_names: List[str]) -> None:
        for name in feature_names:
            if name not in self.weights:
                self.weights[name] = 0.5
        self.feature_names = sorted(set(self.feature_names).union(feature_names))

    def _clip_weight(self, value: float) -> float:
        return float(
            max(
                self.weight_bounds.get("min", 0.0),
                min(self.weight_bounds.get("max", 1.0), value),
            )
        )

    @staticmethod
    def _clip_bias(value: float) -> float:
        return float(max(-5.0, min(5.0, value)))


class DagStore:
    """Minimal persistent DAG store for long-lived goals."""

    def __init__(self, persist_path: str, dashboard_path: str):
        self.persist_path = persist_path
        self.dashboard_path = dashboard_path
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        self.nodes: Dict[str, GoalNode] = {}
        self.active_goal_id: Optional[str] = None
        self.priority_model = OnlinePriorityModel(
            feature_names=[
                "value",
                "urgency",
                "curiosity",
                "competence_balanced",
                "value_x_urgency",
                "curiosity_x_gap",
                "progress",
                "recency",
                "status_active",
                "status_pending",
            ],
            learning_rate=0.1,
            weight_bounds={"min": 0.0, "max": 1.0},
            warmup_updates=20,
        )
        self._load()

    # ------------------------------------------------------------------
    def _llm_review_priority(
        self,
        node: GoalNode,
        base_priority: float,
        adaptive_score: float,
        fallback_priority: float,
    ) -> Optional[Dict[str, Any]]:
        goal_snapshot = {
            "id": node.id,
            "description": node.description,
            "status": node.status,
            "value": node.value,
            "urgency": node.urgency,
            "curiosity": node.curiosity,
            "competence": node.competence,
            "progress": node.progress,
            "created_by": node.created_by,
            "parent_ids": list(node.parent_ids),
            "child_ids": list(node.child_ids),
        }
        if node.evidence:
            goal_snapshot["recent_evidence"] = node.evidence[-3:]

        payload = {
            "goal": goal_snapshot,
            "base_priority": float(base_priority),
            "adaptive_score": float(adaptive_score),
            "fallback_priority": float(fallback_priority),
            "feature_vector": self._feature_vector(node),
        }

        response = try_call_llm_dict(
            "goal_priority_review",
            input_payload=payload,
            logger=LOGGER,
        )
        if not isinstance(response, dict):
            return None

        recommended = response.get("priority")
        if recommended is None and "priority_delta" in response:
            try:
                recommended = fallback_priority + float(response["priority_delta"])
            except (TypeError, ValueError):
                recommended = None

        try:
            priority_value = float(recommended)
        except (TypeError, ValueError):
            return None

        priority_value = max(0.0, min(1.0, priority_value))
        confidence = response.get("confidence")
        try:
            confidence_value = max(0.0, min(1.0, float(confidence))) if confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence_value = 0.0

        if confidence_value:
            blended = (1.0 - confidence_value) * fallback_priority + confidence_value * priority_value
        else:
            blended = priority_value

        review: Dict[str, Any] = {
            "priority": max(0.0, min(1.0, blended)),
            "confidence": confidence_value,
        }

        reason = response.get("reason") or response.get("rationale")
        if isinstance(reason, str) and reason.strip():
            review["reason"] = reason.strip()
        notes = response.get("notes")
        if isinstance(notes, str) and notes.strip():
            review["notes"] = notes.strip()
        elif isinstance(notes, (list, tuple)):
            extracted = [n.strip() for n in notes if isinstance(n, str) and n.strip()]
            if extracted:
                review["notes"] = extracted

        adjustments = response.get("adjustments")
        if isinstance(adjustments, dict) and adjustments:
            review["adjustments"] = adjustments

        return review

    # ------------------------------------------------------------------
    # CRUD
    def add_goal(
        self,
        description: str,
        criteria: Optional[List[str]] = None,
        created_by: str = "system",
        value: float = 0.5,
        competence: float = 0.5,
        curiosity: float = 0.2,
        urgency: float = 0.3,
        parent_ids: Optional[List[str]] = None,
    ) -> GoalNode:
        gid = str(uuid.uuid4())[:8]
        node = GoalNode(
            id=gid,
            description=description,
            criteria=list(criteria or []),
            value=float(max(0.0, min(1.0, value))),
            competence=float(max(0.0, min(1.0, competence))),
            curiosity=float(max(0.0, min(1.0, curiosity))),
            urgency=float(max(0.0, min(1.0, urgency))),
            created_by=created_by,
            parent_ids=list(parent_ids or []),
        )
        self.nodes[gid] = node
        for pid in node.parent_ids:
            parent = self.nodes.get(pid)
            if parent and gid not in parent.child_ids:
                parent.child_ids.append(gid)
                parent.updated_at = _now()
        self._recompute_priority(node)
        self._propagate_progress_from(node)
        self._persist()
        return node

    def link(self, parent_id: str, child_id: str) -> None:
        parent = self.nodes.get(parent_id)
        child = self.nodes.get(child_id)
        if not parent or not child:
            return
        if child_id not in parent.child_ids:
            parent.child_ids.append(child_id)
        if parent_id not in child.parent_ids:
            child.parent_ids.append(parent_id)
        parent.updated_at = _now()
        child.updated_at = _now()
        self._persist()

    def update_goal(self, goal_id: str, updates: Dict[str, Any]) -> Optional[GoalNode]:
        node = self.nodes.get(goal_id)
        if not node:
            return None
        progress_before = node.progress
        status_before = node.status
        progress_changed = False
        completion_event = False
        for key, value in updates.items():
            if not hasattr(node, key):
                continue
            if key == "progress":
                try:
                    progress_value = float(value)
                except (TypeError, ValueError):
                    progress_value = node.progress
                progress_value = max(0.0, min(1.0, progress_value))
                node.progress = progress_value
                node.evidence.append(
                    {
                        "t": _now(),
                        "event": "progress_update",
                        "progress": progress_value,
                    }
                )
                progress_changed = True
            else:
                setattr(node, key, value)
        if progress_changed and node.progress >= 0.999 and node.status not in {"done", "abandoned"}:
            node.status = "done"
            completion_event = True
            node.evidence.append(
                {
                    "t": _now(),
                    "event": "status_auto_complete",
                    "source": "progress_threshold",
                }
            )
        node.updated_at = _now()
        self._recompute_priority(node)
        if progress_changed:
            self._propagate_progress_from(node)
        if progress_changed and progress_before < 1.0 <= node.progress:
            # If progress reached completion through update, treat it as success feedback.
            self._record_feedback(node, success=True, note="progress_completion")
        elif completion_event and status_before != "done":
            self._record_feedback(node, success=True, note="status_completion")
        self._persist()
        return node

    def get_goal(self, goal_id: str) -> Optional[GoalNode]:
        return self.nodes.get(goal_id)

    def set_active(self, goal_id: Optional[str]) -> Optional[GoalNode]:
        if goal_id is None:
            self.active_goal_id = None
            self._export_dashboard()
            return None
        node = self.nodes.get(goal_id)
        if not node:
            return None
        self.active_goal_id = goal_id
        if node.status == "pending":
            node.status = "active"
        node.updated_at = _now()
        self._recompute_priority(node)
        self._persist()
        return node

    def get_active(self) -> Optional[GoalNode]:
        if self.active_goal_id:
            return self.nodes.get(self.active_goal_id)
        return None

    def complete_goal(self, goal_id: str, success: bool = True, note: str = "") -> None:
        node = self.nodes.get(goal_id)
        if not node:
            return
        completion_time = _now()
        duration = completion_time - node.created_at
        features = self._feature_vector(node)
        priority_before = node.priority
        self.priority_model.update(
            features,
            target=1.0 if success else 0.0,
            importance=self._completion_importance(duration),
        )
        node.status = "done" if success else "abandoned"
        node.progress = 1.0 if success else node.progress
        node.updated_at = completion_time
        node.evidence.append(
            {
                "t": completion_time,
                "note": note,
                "success": success,
                "duration_sec": max(0.0, duration),
                "priority_before": priority_before,
            }
        )
        if self.active_goal_id == goal_id:
            self.active_goal_id = None
        self._recompute_priority(node)
        self._propagate_progress_from(node)
        self._persist()

    # ------------------------------------------------------------------
    # Queries
    def topk(self, k: int = 5, only_pending: bool = True) -> List[GoalNode]:
        pool = [
            node
            for node in self.nodes.values()
            if (not only_pending) or node.status in {"pending", "active"}
        ]
        pool.sort(key=lambda n: n.priority, reverse=True)
        return pool[:k]

    # ------------------------------------------------------------------
    # Internal helpers
    def _recompute_priority(self, node: GoalNode) -> None:
        value = max(0.0, min(1.0, node.value))
        urgency = max(0.0, min(1.0, node.urgency))
        curiosity = max(0.0, min(1.0, node.curiosity))
        competence = max(0.0, min(1.0, node.competence))
        base = 0.4 * value + 0.3 * urgency + 0.2 * curiosity + 0.1 * (1.0 - abs(0.5 - competence) * 2.0)
        adaptive_features = self._feature_vector(node)
        self.priority_model.ensure_features(list(adaptive_features.keys()))
        adaptive_score = self.priority_model.predict(adaptive_features)
        blend = self.priority_model.confidence()
        fallback_priority = (1.0 - blend) * base + blend * adaptive_score
        review = self._llm_review_priority(node, base, adaptive_score, fallback_priority)
        if review:
            priority = review.get("priority", fallback_priority)
            if review.get("reason") or review.get("notes"):
                node.evidence.append(
                    {
                        "t": _now(),
                        "event": "llm_priority_review",
                        "reason": review.get("reason"),
                        "notes": review.get("notes"),
                        "confidence": review.get("confidence"),
                    }
                )
        else:
            priority = fallback_priority
        if node.status not in {"pending", "active"}:
            priority *= 0.2
        node.priority = float(max(0.0, min(1.0, priority)))

    def _feature_vector(self, node: GoalNode) -> Dict[str, float]:
        now = _now()
        value = max(0.0, min(1.0, node.value))
        urgency = max(0.0, min(1.0, node.urgency))
        curiosity = max(0.0, min(1.0, node.curiosity))
        competence = max(0.0, min(1.0, node.competence))
        competence_balanced = 1.0 - abs(0.5 - competence) * 2.0
        progress = max(0.0, min(1.0, node.progress))
        age_seconds = max(0.0, now - node.created_at)
        # recency decays over 72 hours
        recency = 1.0 - min(1.0, age_seconds / (72.0 * 3600.0))
        status_active = 1.0 if node.status == "active" else 0.0
        status_pending = 1.0 if node.status == "pending" else 0.0
        curiosity_gap = curiosity * (1.0 - competence)
        features = {
            "value": value,
            "urgency": urgency,
            "curiosity": curiosity,
            "competence_balanced": competence_balanced,
            "value_x_urgency": value * urgency,
            "curiosity_x_gap": curiosity_gap,
            "progress": progress,
            "recency": max(0.0, recency),
            "status_active": status_active,
            "status_pending": status_pending,
        }
        return features

    @staticmethod
    def _completion_importance(duration: float) -> float:
        if duration <= 0:
            return 1.0
        horizon = 3600.0  # one hour reference
        score = 1.0 / (1.0 + duration / horizon)
        return max(0.1, min(1.0, score))

    def _record_feedback(self, node: GoalNode, success: bool, note: str = "") -> None:
        duration = max(0.0, _now() - node.created_at)
        features = self._feature_vector(node)
        self.priority_model.update(
            features,
            target=1.0 if success else 0.0,
            importance=self._completion_importance(duration),
        )
        node.evidence.append(
            {
                "t": _now(),
                "event": "auto_feedback",
                "note": note,
                "success": success,
                "duration_sec": duration,
            }
        )
        self._recompute_priority(node)

    def _persist(self) -> None:
        payload = {
            "active_goal_id": self.active_goal_id,
            "nodes": {gid: node.to_dict() for gid, node in self.nodes.items()},
            "priority_model": self.priority_model.to_dict(),
        }
        try:
            with open(self.persist_path, "w", encoding="utf-8") as fh:
                json.dump(json_sanitize(payload), fh, ensure_ascii=False, indent=2)
        finally:
            self._export_dashboard()

    def _load(self) -> None:
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return
        nodes = {}
        for gid, node_data in data.get("nodes", {}).items():
            try:
                nodes[gid] = GoalNode(**node_data)
            except TypeError:
                continue
        self.nodes = nodes
        self.active_goal_id = data.get("active_goal_id")
        if "priority_model" in data:
            try:
                self.priority_model = OnlinePriorityModel.from_dict(data["priority_model"])
            except Exception:
                pass

    def _export_dashboard(self) -> None:
        snapshot = {
            "t": _now(),
            "active_goal": self.get_active().to_dict() if self.get_active() else None,
            "top5": [node.to_dict() for node in self.topk(5, only_pending=False)],
            "counts": {
                "total": len(self.nodes),
                "pending": sum(1 for n in self.nodes.values() if n.status == "pending"),
                "active": sum(1 for n in self.nodes.values() if n.status == "active"),
                "done": sum(1 for n in self.nodes.values() if n.status == "done"),
            },
        }
        try:
            with open(self.dashboard_path, "w", encoding="utf-8") as fh:
                json.dump(json_sanitize(snapshot), fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _propagate_progress_from(self, node: GoalNode) -> None:
        queue: List[str] = list(node.parent_ids)
        visited: Set[str] = set()
        while queue:
            pid = queue.pop()
            if pid in visited:
                continue
            visited.add(pid)
            parent = self.nodes.get(pid)
            if not parent:
                continue
            child_progress: List[float] = []
            for cid in parent.child_ids:
                child = self.nodes.get(cid)
                if child:
                    child_progress.append(max(0.0, min(1.0, child.progress)))
            if not child_progress:
                continue
            new_progress = max(0.0, min(1.0, sum(child_progress) / len(child_progress)))
            if abs(parent.progress - new_progress) <= 1e-3:
                continue
            parent.progress = new_progress
            parent.updated_at = _now()
            parent.evidence.append(
                {
                    "t": _now(),
                    "event": "progress_propagation",
                    "source_child": node.id,
                    "progress": new_progress,
                }
            )
            self._recompute_priority(parent)
            queue.extend(parent.parent_ids)


class GoalDAG:
    """Lightweight view for quick goal sampling."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._state = {"id": "maintain_loop", "evi": 0.5, "progress": 0.1}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return
        if isinstance(data, dict):
            self._state.update(data)

    def choose_next_goal(self) -> Dict[str, Any]:
        return dict(self._state)

    def update_goal(self, goal_id: str, evi: float, progress: float) -> None:
        self._state.update({"id": goal_id, "evi": float(evi), "progress": float(progress)})
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(json_sanitize(self._state), fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def bump_progress(self, delta: float = 0.01) -> float:
        cur = float(self._state.get("progress", 0.0))
        cur = min(1.0, max(0.0, cur + float(delta)))
        self._state["progress"] = cur
        try:
            self._save()  # si tu as déjà _save(); sinon ignore
        except Exception:
            try:
                with open(self.path, "w", encoding="utf-8") as fh:
                    json.dump(json_sanitize(self._state), fh, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return cur
