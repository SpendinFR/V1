"""Global workspace coordination utilities."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from AGI_Evolutive.cognition.meta_cognition import OnlineLinear
from AGI_Evolutive.core.structures.mai import Bid
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


def _heuristic_info_gain(signals: Dict[str, float]) -> float:
    """Legacy heuristic for information gain produced by RAG signals."""
    if not signals:
        return 0.0
    top1 = float(signals.get("rag_top1", 0.0))
    mean = float(signals.get("rag_mean", 0.0))
    div = float(signals.get("rag_diversity", 0.0))
    return max(0.0, min(1.0, 0.5 * top1 + 0.3 * mean + 0.2 * div))


def _rag_features(signals: Dict[str, float]) -> Dict[str, float]:
    """Build a bounded feature map from RAG telemetry."""
    top1 = max(0.0, min(1.0, float(signals.get("rag_top1", 0.0))))
    mean = max(0.0, min(1.0, float(signals.get("rag_mean", 0.0))))
    div = max(0.0, min(1.0, float(signals.get("rag_diversity", 0.0))))
    n_docs = max(0.0, float(signals.get("rag_docs", 0.0)))
    novelty = max(0.0, min(1.0, float(signals.get("rag_novelty", 0.0))))
    spread = max(0.0, min(1.0, float(signals.get("rag_spread", 0.0))))
    norm_docs = min(1.0, n_docs / 5.0)
    interaction = min(1.0, top1 * div)
    return {
        "rag_top1": top1,
        "rag_mean": mean,
        "rag_diversity": div,
        "rag_docs_norm": norm_docs,
        "rag_top1_x_div": interaction,
        "rag_mean_x_div": min(1.0, mean * div),
        "rag_novelty": novelty,
        "rag_spread": spread,
    }


def _score_features(
    avg_gain: float,
    avg_urg: float,
    avg_aff: float,
    avg_cost: float,
) -> Dict[str, float]:
    """Build quadratic interaction features for the scoring model."""
    gain = max(0.0, avg_gain)
    urg = max(0.0, avg_urg)
    aff = max(0.0, avg_aff)
    cost = max(0.0, avg_cost)
    gain_cost = max(0.0, gain - cost)
    urg_cost = max(0.0, urg - cost)
    aff_cost = max(0.0, aff - cost)
    return {
        "gain": min(1.0, gain),
        "urgency": min(1.0, urg),
        "affect": min(1.0, aff),
        "cost": min(1.0, cost),
        "gain_x_urg": min(1.0, gain * urg),
        "gain_x_aff": min(1.0, gain * aff),
        "urg_x_aff": min(1.0, urg * aff),
        "gain_minus_cost": min(1.0, gain_cost),
        "urg_minus_cost": min(1.0, urg_cost),
        "aff_minus_cost": min(1.0, aff_cost),
        "cost_sq": min(1.0, cost * cost),
    }


@dataclass
class ThompsonBandit:
    """Discrete Thompson Sampling helper over a finite set of parameter tuples."""

    candidates: List[Tuple[float, float, float]]
    state: Optional[Dict[str, Dict[str, float]]] = None

    def __post_init__(self) -> None:
        if not self.candidates:
            # fallback on legacy weights
            self.candidates = [(0.5, 0.2, 0.3)]
        base = {self._key(c): {"success": 1.0, "failure": 1.0} for c in self.candidates}
        if isinstance(self.state, dict):
            for key, value in self.state.items():
                if key in base and isinstance(value, dict):
                    success = float(value.get("success", 1.0))
                    failure = float(value.get("failure", 1.0))
                    base[key] = {
                        "success": max(1e-3, success),
                        "failure": max(1e-3, failure),
                    }
        self._dist = base

    @staticmethod
    def _key(candidate: Tuple[float, float, float]) -> str:
        return ",".join(f"{c:.3f}" for c in candidate)

    def sample(self) -> Tuple[str, Tuple[float, float, float]]:
        """Sample a candidate tuple according to Thompson sampling."""
        best_key = None
        best_score = -1.0
        for candidate in self.candidates:
            key = self._key(candidate)
            stats = self._dist[key]
            draw = random.betavariate(stats["success"], stats["failure"])
            if draw > best_score:
                best_key = key
                best_score = draw
        assert best_key is not None
        parts = best_key.split(",")
        selected = tuple(float(p) for p in parts)
        return best_key, selected  # type: ignore[return-value]

    def update(self, key: str, reward: float) -> None:
        reward = max(0.0, min(1.0, float(reward)))
        stats = self._dist.setdefault(key, {"success": 1.0, "failure": 1.0})
        stats["success"] = max(1e-3, stats.get("success", 1.0) + reward)
        stats["failure"] = max(1e-3, stats.get("failure", 1.0) + (1.0 - reward))

    def to_state(self) -> Dict[str, Dict[str, float]]:
        return {key: dict(value) for key, value in self._dist.items()}


def _urgency_from_frame(frame: Any) -> float:
    """Compute an urgency score based on frame metadata."""
    u = 0.5
    try:
        if getattr(frame, "blocking", False):
            u += 0.3
        if getattr(frame, "deadline_ts", None):
            dt = frame.deadline_ts - time.time()
            if dt < 0:
                u += 0.2
            elif dt < 3600:
                u += 0.15
    except Exception:  # pragma: no cover - defensive guard
        pass
    return max(0.0, min(1.0, u))


class GlobalWorkspace:
    """Simple global workspace capable of collecting broadcasts and bids."""

    def __init__(
        self,
        policy: Optional[Any] = None,
        planner: Optional[Any] = None,
        data_dir: str = "data",
    ) -> None:
        self.broadcasts: List[Any] = []
        self._bids: List[Tuple[str, str, float, Dict[str, Any]]] = []
        self.policy = policy
        self.planner = planner
        self._pending_bids: List[Bid] = []
        self._trace_last_winners: List[Bid] = []
        self._decision_trace: Dict[str, Dict[str, Any]] = {}
        self._rag_trace: Dict[str, Dict[str, float]] = {}

        if data_dir:
            self._state_path = os.path.join(data_dir, "global_workspace.json")
        else:
            self._state_path = ""
        self._state = self._load_state()
        rag_state = self._state.get("rag_model") if isinstance(self._state, dict) else None
        score_state = self._state.get("score_model") if isinstance(self._state, dict) else None
        self._rag_model = self._init_rag_model(rag_state)
        self._score_model = self._init_score_model(score_state)
        bandit_state = self._state.get("bandit") if isinstance(self._state, dict) else None
        self._bandit = ThompsonBandit(
            candidates=[
                (0.3, 0.15, 0.2),
                (0.4, 0.25, 0.25),
                (0.5, 0.2, 0.3),
                (0.6, 0.25, 0.35),
            ],
            state=bandit_state if isinstance(bandit_state, dict) else None,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path:
            return {}
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
        except Exception:
            pass
        return {}

    def _save_state(self) -> None:
        state = {
            "rag_model": self._rag_model.to_state(),
            "score_model": self._score_model.to_state(),
            "bandit": self._bandit.to_state(),
        }
        if not self._state_path:
            return
        directory = os.path.dirname(self._state_path) or "."
        os.makedirs(directory, exist_ok=True)
        try:
            with open(self._state_path, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _init_rag_model(self, payload: Optional[Dict[str, Any]]) -> OnlineLinear:
        model = OnlineLinear.from_state(
            payload,
            bounds=(0.0, 1.0),
            lr=0.035,
            l2=0.0015,
            max_grad=0.2,
            warmup=10,
            init_weight=0.12,
        )
        if not model.feature_names:
            base = list(_rag_features({}).keys())
            model.feature_names = base
            model.index = {name: idx for idx, name in enumerate(base)}
            model.weights = [model.init_weight for _ in base]
        return model

    def _init_score_model(self, payload: Optional[Dict[str, Any]]) -> OnlineLinear:
        model = OnlineLinear.from_state(
            payload,
            bounds=(-0.5, 0.5),
            lr=0.04,
            l2=0.0015,
            max_grad=0.25,
            warmup=14,
            init_weight=0.05,
        )
        if not model.feature_names:
            base = list(_score_features(0.0, 0.0, 0.0, 0.0).keys())
            model.feature_names = base
            model.index = {name: idx for idx, name in enumerate(base)}
            model.weights = [model.init_weight for _ in base]
        return model

    def broadcast(self, item: Any) -> None:
        self.broadcasts.append(item)
        if len(self.broadcasts) > 1000:
            self.broadcasts.pop(0)

    def submit_bid(self, channel: str, bid_type: str, attention: float, payload: Dict[str, Any]) -> None:
        """Register a bid emitted by a module."""
        attention = max(0.0, min(1.0, attention))
        self._bids.append((channel, bid_type, attention, payload))

    def _policy_score_safe(self, frame: Any, option: Optional[str] = None) -> float:
        """Return a policy score using the best available API or a RAG-based fallback."""

        pol = getattr(self, "policy", None)
        try:
            if pol and hasattr(pol, "evaluate"):
                return float(pol.evaluate(frame, option=option))
            if pol and hasattr(pol, "confidence_for"):
                return float(pol.confidence_for(frame, option=option))
            if pol and hasattr(pol, "confidence"):
                return float(pol.confidence(frame))
        except Exception:
            pass

        if option == "no_rag_use":
            return 0.0

        signals = getattr(frame, "signals", {}) or {}
        top1 = float(signals.get("rag_top1", 0.0))
        mean = float(signals.get("rag_mean", 0.0))
        div = float(signals.get("rag_diversity", 0.0))
        n_docs = float(signals.get("rag_docs", 0.0))
        fallback = 0.45 * top1 + 0.35 * mean + 0.15 * div + 0.05 * min(n_docs / 5.0, 1.0)
        return max(0.0, min(1.0, fallback))

    def _submit_rag_bid(self, frame: Any, utility_delta: float) -> None:
        signals = getattr(frame, "signals", {}) or {}
        features = _rag_features(signals)
        baseline = _heuristic_info_gain(signals)
        learned = self._rag_model.predict(features)
        confidence = self._rag_model.confidence()
        ig = (1.0 - confidence) * baseline + confidence * learned
        urg = _urgency_from_frame(frame)
        attention = ig * max(0.0, utility_delta) * urg

        payload = {
            "type": "RAGEvidence",
            "signals": signals,
            "grounded_context": (getattr(frame, "context", {}) or {}).get("grounded_evidence", []),
        }
        self.submit_bid("evidence", "RAGEvidence", attention, payload)
        frame_id = getattr(frame, "uid", None) or getattr(frame, "id", None)
        if frame_id is not None:
            self._rag_trace[str(frame_id)] = {
                "features": features,
                "baseline": baseline,
                "attention": attention,
            }

    # ------------------------------------------------------------------
    # MAI integration helpers
    def submit(self, bid: Bid) -> None:
        self._pending_bids.append(bid)

    def step(self, state: Dict, timebox_iters: int = 2) -> None:
        groups: Dict[str, List[Bid]] = {}
        now = time.time()
        self._pending_bids = [
            b for b in self._pending_bids if (b.expires_at or now + 1e9) > now
        ]
        for bid in self._pending_bids:
            groups.setdefault(bid.action_hint, []).append(bid)

        scored: List[Tuple[float, Bid]] = []
        self._decision_trace.clear()
        llm_candidates: List[Dict[str, Any]] = []
        scored_candidates: List[Dict[str, Any]] = []
        for hint, bids in groups.items():
            avg_gain = sum(max(0.0, x.expected_info_gain) for x in bids) / max(1, len(bids))
            avg_urg = sum(max(0.0, x.urgency) for x in bids) / max(1, len(bids))
            avg_aff = sum(max(0.0, x.affect_value) for x in bids) / max(1, len(bids))
            avg_cost = sum(max(0.0, x.cost) for x in bids) / max(1, len(bids))
            features = _score_features(avg_gain, avg_urg, avg_aff, avg_cost)
            bandit_key, (urg_w, aff_w, cost_w) = self._bandit.sample()
            base_score = (avg_gain + urg_w * avg_urg + aff_w * avg_aff) - cost_w * avg_cost
            model_pred = self._score_model.predict(features)
            confidence = self._score_model.confidence()
            correction = (model_pred - 0.5) * (0.5 + 0.5 * confidence)
            score = base_score + correction
            primary_bid = bids[0]
            bid_id = primary_bid.payload.get("id") if isinstance(primary_bid.payload, Mapping) else None
            if not bid_id:
                bid_id = f"{primary_bid.source}:{primary_bid.action_hint}:{id(primary_bid)}"
            scored.append((score, primary_bid))
            candidate_payload = {
                "id": bid_id,
                "hint": hint,
                "source": primary_bid.source,
                "action_hint": primary_bid.action_hint,
                "score": score,
                "base_score": base_score,
                "model_pred": model_pred,
                "bandit_key": bandit_key,
                "confidence": confidence,
                "payload": _truncate_payload(primary_bid.payload),
            }
            llm_candidates.append(candidate_payload)
            scored_candidates.append(
                {
                    "id": bid_id,
                    "hint": hint,
                    "score": score,
                    "bid": primary_bid,
                }
            )
            self._decision_trace[hint] = {
                "features": features,
                "bandit_key": bandit_key,
                "base_score": base_score,
                "model_pred": model_pred,
                "correction": correction,
                "final_score": score,
                "candidate_id": bid_id,
            }

        ordered = self._apply_llm_ranking(llm_candidates, scored_candidates)
        scored = [(entry["score"], entry["bid"]) for entry in ordered]
        scored.sort(key=lambda t: t[0], reverse=True)
        K = min(5, len(scored))
        self._trace_last_winners = [bid for _, bid in scored[:K]]
        self._save_state()

    def winners(self) -> List[Bid]:
        return list(self._trace_last_winners)

    def last_trace(self) -> List[Bid]:
        return list(self._trace_last_winners)

    def process_frame(self, frame: Any) -> None:
        """Run planner/policy pipeline for a frame and publish RAG evidence bids."""
        if self.planner is not None:
            self.planner.plan(frame)
        if self.policy is None:
            return

        current_u = self._policy_score_safe(frame, option="no_rag_use")
        rag_u = self._policy_score_safe(frame, option="use_rag")
        utility_delta = rag_u - current_u
        self._submit_rag_bid(frame, utility_delta)

    # ------------------------------------------------------------------
    # Feedback integration
    def register_feedback(
        self,
        action_hint: str,
        reward: float,
        rag_feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update adaptive models with delayed feedback.

        Args:
            action_hint: Identifier of the MAI hint that was executed.
            reward: Success metric in [0, 1].
            rag_feedback: Optional signals dict to adjust the RAG model.
        """

        reward = max(0.0, min(1.0, float(reward)))
        trace = self._decision_trace.get(action_hint)
        if trace:
            self._score_model.update(trace["features"], reward)
            self._bandit.update(trace["bandit_key"], reward)

        if rag_feedback:
            rag_features = _rag_features(rag_feedback)
            self._rag_model.update(rag_features, reward)
        self._save_state()

    def register_rag_outcome(self, frame_id: Any, reward: float) -> None:
        """Shortcut to update the RAG model when the frame outcome is known."""

        reward = max(0.0, min(1.0, float(reward)))
        trace = self._rag_trace.get(str(frame_id))
        if trace:
            self._rag_model.update(trace.get("features", {}), reward)
        self._rag_trace.pop(str(frame_id), None)
        self._save_state()

    def _apply_llm_ranking(
        self,
        candidates: Sequence[Mapping[str, Any]],
        scored_candidates: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ask the LLM to rank bids and merge with heuristic scores."""

        llm_payload = {
            "candidates": [dict(candidate) for candidate in candidates],
            "recent_winners": [
                {
                    "id": getattr(bid, "action_hint", None),
                    "source": getattr(bid, "source", None),
                }
                for bid in self._trace_last_winners
            ],
        }
        response = try_call_llm_dict(
            "global_workspace",
            input_payload=llm_payload,
            logger=LOGGER,
            max_retries=2,
        )
        candidate_by_id = {entry["id"]: dict(entry) for entry in scored_candidates if entry.get("id")}
        if response:
            self._decision_trace["llm_ranking"] = dict(response)
        if not response:
            return [dict(entry) for entry in scored_candidates]

        ranking = response.get("ranking")
        ordered: List[Dict[str, Any]] = []
        if isinstance(ranking, Sequence):
            for position, item in enumerate(ranking):
                if not isinstance(item, Mapping):
                    continue
                candidate_id = item.get("id") or item.get("hint")
                candidate = candidate_by_id.get(candidate_id)
                if not candidate:
                    continue
                candidate = dict(candidate)
                try:
                    candidate["llm_score"] = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    candidate["llm_score"] = 0.0
                candidate["llm_rank"] = position
                candidate["llm_explanation"] = item.get("explanation")
                ordered.append(candidate)
        remaining = [
            dict(entry)
            for entry in scored_candidates
            if entry.get("id") not in {item.get("id") for item in ordered}
        ]
        remaining.sort(key=lambda entry: entry.get("score", 0.0), reverse=True)
        ordered.extend(remaining)
        return ordered


def _truncate_payload(payload: Mapping[str, Any] | None, *, max_items: int = 8) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    result: Dict[str, Any] = {}
    for idx, (key, value) in enumerate(payload.items()):
        if idx >= max_items:
            break
        try:
            result[str(key)] = value if isinstance(value, (str, int, float, bool)) else str(value)
        except Exception:
            result[str(key)] = "(unserializable)"
    return result
