"""Batch summarizer producing hierarchical belief summaries."""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

from .graph import Belief, BeliefGraph


LOGGER = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    timeframe: str  # "daily" | "weekly"
    window_seconds: float
    output_path: str
    max_anchors: int = 20
    max_episodes: int = 50
    diversification: int = 3


@dataclass
class AdaptiveWeights:
    """Lightweight online linear model with bounded coefficients."""

    weights: Dict[str, float]
    bounds: Dict[str, Tuple[float, float]]
    learning_rate: float = 0.05
    forget: float = 0.02

    def score(self, features: Dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in features.items())

    def update(self, features: Dict[str, float], target: float) -> None:
        """Perceptron-style update with forgetting to track drift."""

        if not features:
            return
        prediction = self.score(features)
        error = target - prediction
        for name, value in features.items():
            if value == 0:
                continue
            lo, hi = self.bounds.get(name, (-1.0, 1.0))
            delta = self.learning_rate * error * value
            updated = self.weights.get(name, 0.0) * (1.0 - self.forget) + delta
            self.weights[name] = float(min(hi, max(lo, updated)))


@dataclass
class SummaryVariant:
    id: str
    anchors: List[Belief] = field(default_factory=list)
    episodes: List[Belief] = field(default_factory=list)
    score: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)


DEFAULT_CONFIGS = {
    "daily": SummaryConfig(
        timeframe="daily",
        window_seconds=60 * 60 * 24,
        output_path="data/summaries/daily.jsonl",
        max_anchors=20,
        max_episodes=40,
        diversification=4,
    ),
    "weekly": SummaryConfig(
        timeframe="weekly",
        window_seconds=60 * 60 * 24 * 7,
        output_path="data/summaries/weekly.jsonl",
        max_anchors=30,
        max_episodes=60,
        diversification=5,
    ),
}


class BeliefSummarizer:
    """Aggregates beliefs into anchors and episodic summaries."""

    def __init__(self, graph: BeliefGraph, *, weights_path: str = "data/summaries/weights.json") -> None:
        self.graph = graph
        self.weights_path = weights_path
        self._weights = self._load_weights()
        self._feedback_buffer: List[Tuple[str, float, Dict[str, float]]] = []
        self._last_variant: Optional[SummaryVariant] = None
        self._last_timeframe: Optional[str] = None

    # ------------------------------------------------------------------
    # Adaptive scoring helpers
    def _default_weights(self) -> AdaptiveWeights:
        base_weights = {
            "confidence": 1.2,
            "recency": 0.8,
            "anchor_bonus": 0.6,
            "support_strength": 0.5,
            "novelty": 0.4,
            "centrality": 0.3,
            "polarity_consistency": 0.2,
            "stability_penalty": -0.3,
        }
        bounds = {name: (-2.0, 2.0) for name in base_weights}
        bounds.update({"anchor_bonus": (0.0, 2.5), "stability_penalty": (-2.0, 0.0)})
        return AdaptiveWeights(base_weights, bounds)

    def _load_weights(self) -> AdaptiveWeights:
        if not os.path.exists(self.weights_path):
            return self._default_weights()
        try:
            with open(self.weights_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return self._default_weights()
        weights = payload.get("weights", {})
        raw_bounds = payload.get("bounds", {})
        bounds = {name: tuple(value) for name, value in raw_bounds.items()}
        if not bounds:
            bounds = {name: (-2.0, 2.0) for name in weights}
        model = AdaptiveWeights(dict(weights), bounds)
        model.learning_rate = float(payload.get("learning_rate", model.learning_rate))
        model.forget = float(payload.get("forget", model.forget))
        return model

    def _save_weights(self) -> None:
        os.makedirs(os.path.dirname(self.weights_path) or ".", exist_ok=True)
        payload = {
            "weights": self._weights.weights,
            "bounds": {k: list(v) for k, v in self._weights.bounds.items()},
            "learning_rate": self._weights.learning_rate,
            "forget": self._weights.forget,
        }
        with open(self.weights_path, "w", encoding="utf-8") as fh:
            json.dump(json_sanitize(payload), fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Feature engineering
    def _graph_metrics(self) -> Dict[str, Dict[str, float]]:
        subject_counts: Dict[str, int] = {}
        relation_counts: Dict[str, int] = {}
        polarity_balance: Dict[str, float] = {}
        for belief in self.graph.all():
            subject_counts[belief.subject] = subject_counts.get(belief.subject, 0) + 1
            relation_counts[belief.relation] = relation_counts.get(belief.relation, 0) + 1
            key = f"{belief.subject}|{belief.relation}"
            balance = polarity_balance.get(key, 0.0)
            polarity_balance[key] = balance + belief.polarity
        return {
            "subject_counts": {k: float(v) for k, v in subject_counts.items()},
            "relation_counts": {k: float(v) for k, v in relation_counts.items()},
            "polarity_balance": {k: float(v) for k, v in polarity_balance.items()},
        }

    def _features_for(self, belief: Belief, metrics: Dict[str, Dict[str, float]], now: float) -> Dict[str, float]:
        subject_count = metrics["subject_counts"].get(belief.subject, 1.0)
        relation_count = metrics["relation_counts"].get(belief.relation, 1.0)
        key = f"{belief.subject}|{belief.relation}"
        polarity_balance = metrics["polarity_balance"].get(key, belief.polarity)
        recency = math.exp(-max(0.0, now - belief.updated_at) / (60 * 60 * 24))
        support_strength = sum(e.weight for e in belief.justifications) / max(1.0, len(belief.justifications))
        novelty = 1.0 / math.sqrt(subject_count)
        centrality = math.log1p(subject_count + relation_count)
        stability_penalty = 1.0 if belief.stability != "anchor" else 0.0
        anchor_bonus = 1.0 if belief.stability == "anchor" else 0.0
        polarity_consistency = 1.0 - abs(polarity_balance) / max(1.0, subject_count)
        return {
            "confidence": belief.confidence,
            "recency": recency,
            "support_strength": support_strength,
            "novelty": novelty,
            "centrality": centrality,
            "stability_penalty": stability_penalty,
            "anchor_bonus": anchor_bonus,
            "polarity_consistency": polarity_consistency,
        }

    # ------------------------------------------------------------------
    # Selection & diversification
    def _select_beliefs(
        self,
        beliefs: Iterable[Belief],
        *,
        now: Optional[float] = None,
        window: float = 0.0,
        cfg: SummaryConfig,
    ) -> List[SummaryVariant]:
        now = now or time.time()
        effective_window = self._adaptive_window(window, now=now)
        metrics = self._graph_metrics()
        eligible: List[Tuple[Belief, float, Dict[str, float]]] = []
        for belief in beliefs:
            if belief.stability != "anchor" and belief.updated_at < now - effective_window:
                continue
            features = self._features_for(belief, metrics, now)
            base_score = self._weights.score(features)
            eligible.append((belief, base_score, features))

        eligible.sort(key=lambda item: item[1], reverse=True)
        if not eligible:
            return [SummaryVariant(id="baseline", anchors=[], episodes=[], score=0.0)]

        variants: List[SummaryVariant] = []
        baseline = self._materialize_variant("baseline", eligible, cfg)
        variants.append(baseline)

        # Diversification by sampling perturbations of weights
        for idx in range(1, max(1, cfg.diversification)):
            perturbed = self._perturb_scores(eligible, strength=0.1 * idx)
            variant = self._materialize_variant(f"mutant_{idx}", perturbed, cfg)
            variants.append(variant)
        variants.sort(key=lambda v: v.score, reverse=True)
        return variants

    def _materialize_variant(
        self, variant_id: str, scored: List[Tuple[Belief, float, Dict[str, float]]], cfg: SummaryConfig
    ) -> SummaryVariant:
        anchors: List[Belief] = []
        episodes: List[Belief] = []
        aggregated: Dict[str, float] = {}
        total = 0
        for belief, _score, features in scored:
            bucket = anchors if belief.stability == "anchor" else episodes
            limit = cfg.max_anchors if bucket is anchors else cfg.max_episodes
            if len(bucket) >= limit:
                continue
            bucket.append(belief)
            for name, value in features.items():
                aggregated[name] = aggregated.get(name, 0.0) + float(value)
            total += 1
            if len(anchors) >= cfg.max_anchors and len(episodes) >= cfg.max_episodes:
                break
        averaged = {name: value / total for name, value in aggregated.items()} if total else {}
        return SummaryVariant(
            id=variant_id,
            anchors=anchors[: cfg.max_anchors],
            episodes=episodes[: cfg.max_episodes],
            score=self._variant_score(anchors, episodes),
            features=averaged,
        )

    def _variant_score(self, anchors: List[Belief], episodes: List[Belief]) -> float:
        if not anchors and not episodes:
            return 0.0
        scores = [b.confidence for b in anchors + episodes]
        diversity = len({(b.subject, b.relation) for b in anchors + episodes})
        return float(sum(scores) / max(1, len(scores)) + 0.05 * diversity)

    def _perturb_scores(
        self, eligible: List[Tuple[Belief, float, Dict[str, float]]], *, strength: float
    ) -> List[Tuple[Belief, float, Dict[str, float]]]:
        mutated: List[Tuple[Belief, float, Dict[str, float]]] = []
        for belief, score, features in eligible:
            noise = random.uniform(-strength, strength)
            mutated.append((belief, score + noise, features))
        mutated.sort(key=lambda item: item[1], reverse=True)
        return mutated

    def _adaptive_window(self, base_window: float, *, now: float) -> float:
        recent = [b for b in self.graph.all() if now - b.updated_at <= base_window]
        if not recent:
            return base_window
        avg_conf = sum(b.confidence for b in recent) / len(recent)
        activity = len(recent) / max(1.0, base_window / (60 * 60))  # beliefs per hour
        if activity > 10:
            return max(base_window * 0.5, base_window - 6 * 60 * 60)
        if avg_conf < 0.4:
            return min(base_window * 1.5, base_window + 12 * 60 * 60)
        return base_window

    # ------------------------------------------------------------------
    def build_summary(self, timeframe: str, *, now: Optional[float] = None) -> Dict[str, List[Dict[str, object]]]:
        cfg = DEFAULT_CONFIGS.get(timeframe)
        if not cfg:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        now = now or time.time()
        beliefs = self.graph.all()
        variants = self._select_beliefs(beliefs, now=now, window=cfg.window_seconds, cfg=cfg)
        best = variants[0]
        self._last_variant = best
        self._last_timeframe = timeframe
        anchors = [self._serialize_belief(b) for b in best.anchors]
        episodes = [self._serialize_belief(b) for b in best.episodes]
        summary = {"anchors": anchors, "episodes": episodes}
        enrichment = self._llm_enrich_summary(timeframe, summary, features=best.features)
        if enrichment:
            summary["llm_enrichment"] = enrichment
        return summary

    def _llm_enrich_summary(
        self,
        timeframe: str,
        summary: Dict[str, List[Dict[str, object]]],
        *,
        features: Dict[str, float],
    ) -> Optional[Dict[str, object]]:
        payload = {
            "timeframe": timeframe,
            "anchors": summary.get("anchors", []),
            "episodes": summary.get("episodes", []),
            "aggregated_features": features,
        }
        response = try_call_llm_dict(
            "belief_summarizer",
            input_payload=payload,
            logger=LOGGER,
        )
        if not response:
            return None
        narrative = response.get("narrative")
        anchors = response.get("anchors")
        coherence = response.get("coherence_score")
        if not isinstance(narrative, str):
            return None
        if anchors is not None and not isinstance(anchors, list):
            return None
        if coherence is not None and not isinstance(coherence, (int, float)):
            return None
        cleaned = {
            "narrative": narrative,
            "anchors": anchors or [],
            "coherence_score": float(coherence) if coherence is not None else None,
            "notes": response.get("notes", ""),
        }
        return cleaned

    def _serialize_belief(self, belief: Belief) -> Dict[str, object]:
        return {
            "subject": belief.subject,
            "relation": belief.relation,
            "value": belief.value,
            "confidence": round(belief.confidence, 3),
            "polarity": belief.polarity,
            "evidence": [e.source for e in belief.justifications],
            "support": [json_sanitize(e.__dict__) for e in belief.justifications],
            "updated_at": belief.updated_at,
            "temporal_segments": [seg.to_dict() for seg in belief.temporal_segments],
            "stability": belief.stability,
        }

    # ------------------------------------------------------------------
    def record_feedback(
        self, score: float, timeframe: Optional[str] = None, features: Optional[Dict[str, float]] = None
    ) -> None:
        """Store quality feedback (0..1) for online updates."""

        if not 0.0 <= score <= 1.0:
            return
        if timeframe is None:
            timeframe = self._last_timeframe or "daily"
        if features is None and self._last_variant is not None:
            features = dict(self._last_variant.features)
        payload = features or {"baseline": 1.0}
        self._feedback_buffer.append((timeframe, score, payload))
        if len(self._feedback_buffer) >= 10:
            self._flush_feedback()

    def _flush_feedback(self) -> None:
        for _, target, features in self._feedback_buffer:
            self._weights.update(features, target)
        self._feedback_buffer.clear()
        self._save_weights()

    def write_summary(self, timeframe: str, *, now: Optional[float] = None) -> Dict[str, List[Dict[str, object]]]:
        cfg = DEFAULT_CONFIGS[timeframe]
        summary = self.build_summary(timeframe, now=now)
        os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
        with open(cfg.output_path, "a", encoding="utf-8") as fh:
            row = {"time": now or time.time(), "summary": summary}
            fh.write(json.dumps(json_sanitize(row), ensure_ascii=False) + "\n")
        if self._feedback_buffer:
            self._flush_feedback()
        return summary


def run_batch(graph: BeliefGraph, timeframes: Iterable[str] = ("daily", "weekly")) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    summarizer = BeliefSummarizer(graph)
    now = time.time()
    return {tf: summarizer.write_summary(tf, now=now) for tf in timeframes}
