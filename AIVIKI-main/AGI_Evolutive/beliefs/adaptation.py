"""Adaptive feedback utilities for the belief graph.

The goal of this module is to keep the rest of the codebase lightweight while
adding instrumentation and online-learning helpers.  The objects here are
designed to be simple Python data-structures (no heavy ML dependencies) so the
graph can progressively adapt without breaking existing behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple
import json
import logging
import math
import os
import random
import time

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic feedback/statistics handling


@dataclass
class FeedbackStats:
    """Aggregated feedback for a particular object (belief, rule, relation).

    The agent mainly relies on binary signals (success/failure).  We also track
    the recency of the signal so that adaptive mechanisms can react to drifts.
    """

    success: float = 0.0
    failure: float = 0.0
    last_seen: float = 0.0
    last_success: float = 0.0
    evidence: float = 0.0

    def register(self, outcome: Optional[bool], *, weight: float = 1.0) -> None:
        now = time.time()
        self.last_seen = now
        if outcome is True:
            self.success += weight
            self.last_success = now
        elif outcome is False:
            self.failure += weight

    def register_evidence(self, weight: float) -> None:
        self.evidence += weight

    def total(self) -> float:
        return self.success + self.failure

    def ratio(self, default: float = 0.5) -> float:
        total = self.total()
        if total <= 0.0:
            return default
        return self.success / total

    def decay_modifier(self, *, half_life: float = 3600.0) -> float:
        """Return a multiplicative factor (0.5-2.0) based on recency of success.

        If the belief was successful very recently the modifier is < 1 (slower
        decay).  If it has not been successful for a long time the modifier is
        > 1 (faster decay).
        """

        if self.last_seen <= 0.0:
            return 1.0

        elapsed = max(0.0, time.time() - self.last_success)
        if elapsed <= 0.0:
            return 0.75

        decay_steps = elapsed / half_life
        # Cap the modifier to avoid brutal swings.
        return float(min(2.0, max(0.5, math.exp(-decay_steps * 0.2))))


@dataclass
class ThompsonParameter:
    """Simple Thompson Sampling helper using a Beta distribution."""

    alpha: float = 1.0
    beta: float = 1.0
    minimum: float = 0.0
    maximum: float = 1.0

    def sample(self) -> float:
        draw = random.betavariate(max(self.alpha, 1e-3), max(self.beta, 1e-3))
        return self.minimum + (self.maximum - self.minimum) * draw

    def update(self, outcome: Optional[bool], weight: float = 1.0) -> None:
        if outcome is True:
            self.alpha += weight
        elif outcome is False:
            self.beta += weight


@dataclass
class RuleCandidate:
    """Potential rule inferred from repeated co-occurrences."""

    if_relation: str
    then_relation: str
    polarity: int
    support: float
    confidence: float
    subject_type: Optional[str] = None
    value_type: Optional[str] = None


class FeedbackTracker:
    """Centralised persistence of feedback signals.

    The tracker keeps the file format intentionally very small: we append a
    single JSON document that contains plain numbers.  It is flushed by the
    caller (``BeliefGraph``) when the main datastore is written.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.belief_stats: Dict[str, FeedbackStats] = {}
        self.rule_stats: Dict[str, FeedbackStats] = {}
        self.relation_stats: Dict[Tuple[str, str, int], FeedbackStats] = {}
        self.decay_bandits: Dict[str, ThompsonParameter] = {}
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    # Persistence

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return

        for belief_id, stats in payload.get("beliefs", {}).items():
            self.belief_stats[belief_id] = FeedbackStats(**stats)
        for rule_id, stats in payload.get("rules", {}).items():
            self.rule_stats[rule_id] = FeedbackStats(**stats)
        for key, stats in payload.get("relations", {}).items():
            if isinstance(key, str):
                try:
                    rel_if, rel_then, pol = key.split("|", 2)
                    key_tuple = (rel_if, rel_then, int(pol))
                except Exception:
                    continue
            else:
                key_tuple = tuple(key)
            self.relation_stats[key_tuple] = FeedbackStats(**stats)
        for stability, params in payload.get("decay", {}).items():
            self.decay_bandits[stability] = ThompsonParameter(**params)

    def flush(self) -> None:
        if not self._dirty:
            return
        payload = {
            "beliefs": {k: asdict(v) for k, v in self.belief_stats.items()},
            "rules": {k: asdict(v) for k, v in self.rule_stats.items()},
            "relations": {
                "|".join([k[0], k[1], str(k[2])]): asdict(v)
                for k, v in self.relation_stats.items()
            },
            "decay": {k: asdict(v) for k, v in self.decay_bandits.items()},
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self._dirty = False

    # ------------------------------------------------------------------
    # Registration helpers

    def ensure_belief(self, belief_id: str) -> FeedbackStats:
        stats = self.belief_stats.get(belief_id)
        if not stats:
            stats = FeedbackStats()
            self.belief_stats[belief_id] = stats
            self._dirty = True
        return stats

    def ensure_decay_bandit(self, stability: str) -> ThompsonParameter:
        tp = self.decay_bandits.get(stability)
        if not tp:
            # Default prior encourages values close to 1.0 × base rate.
            tp = ThompsonParameter(alpha=2.0, beta=2.0, minimum=0.5, maximum=1.5)
            self.decay_bandits[stability] = tp
            self._dirty = True
        return tp

    # ------------------------------------------------------------------
    # Feedback ingestion

    def record_belief_outcome(
        self,
        belief_id: str,
        *,
        outcome: Optional[bool],
        weight: float = 1.0,
        stability: Optional[str] = None,
    ) -> None:
        stats = self.ensure_belief(belief_id)
        stats.register(outcome, weight=weight)
        self._dirty = True

        if stability:
            tp = self.ensure_decay_bandit(stability)
            tp.update(outcome, weight=weight)

    def record_evidence(self, belief_id: str, weight: float) -> None:
        stats = self.ensure_belief(belief_id)
        stats.register_evidence(weight)
        self._dirty = True

    def record_rule_outcome(
        self, rule_id: str, *, outcome: Optional[bool], weight: float = 1.0
    ) -> None:
        stats = self.rule_stats.setdefault(rule_id, FeedbackStats())
        stats.register(outcome, weight=weight)
        self._dirty = True

    def record_relation_pair(
        self,
        if_relation: str,
        then_relation: str,
        polarity: int,
        *,
        outcome: Optional[bool] = None,
        weight: float = 1.0,
    ) -> None:
        key = (if_relation, then_relation, polarity)
        stats = self.relation_stats.setdefault(key, FeedbackStats())
        stats.register(outcome, weight=weight)
        self._dirty = True

    # ------------------------------------------------------------------
    # Query helpers

    def get_belief_stats(self, belief_id: str) -> FeedbackStats:
        return self.belief_stats.get(belief_id, FeedbackStats())

    def decay_modifier(self, belief_id: str, stability: str) -> float:
        base = self.get_belief_stats(belief_id).decay_modifier()
        bandit = self.ensure_decay_bandit(stability)
        sampled = bandit.sample()
        return float(min(2.0, max(0.25, base * sampled)))

    def candidate_rules(
        self,
        *,
        min_support: float = 5.0,
        min_confidence: float = 0.65,
    ) -> Iterator[RuleCandidate]:
        for (rel_if, rel_then, pol), stats in self.relation_stats.items():
            support = stats.total()
            if support < min_support:
                continue
            confidence = stats.ratio()
            if confidence < min_confidence:
                continue
            yield RuleCandidate(
                if_relation=rel_if,
                then_relation=rel_then,
                polarity=pol,
                support=support,
                confidence=confidence,
            )

    def iter_rule_stats(self) -> Iterable[Tuple[str, FeedbackStats]]:
        return list(self.rule_stats.items())

    # ------------------------------------------------------------------
    # LLM integration

    def _llm_payload(
        self,
        belief_id: str,
        belief: Mapping[str, Any] | None,
        stats: FeedbackStats,
        *,
        suggested_delta: float,
    ) -> Dict[str, Any]:
        payload = {
            "belief_id": belief_id,
            "statistics": {
                "success": stats.success,
                "failure": stats.failure,
                "evidence": stats.evidence,
                "last_seen": stats.last_seen,
                "last_success": stats.last_success,
            },
            "suggested_delta": suggested_delta,
        }
        if belief:
            payload["belief"] = dict(belief)
        return payload

    def _fallback_adjustment(
        self,
        belief_id: str,
        stats: FeedbackStats,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        ratio = stats.ratio()
        total = max(1.0, stats.total())
        confidence = min(1.0, total / (total + 5.0))
        delta = max(-0.45, min(0.45, (ratio - 0.5) * (0.8 + 0.2 * confidence)))
        justification = (
            "Renforcer" if delta > 0 else "Atténuer" if delta < 0 else "Stabiliser"
        )
        return {
            "belief": belief_id,
            "delta": float(delta),
            "confidence": float(confidence),
            "justification": f"{justification} la croyance (ratio succès={ratio:.2f}).",
            "notes": "",
            "context": dict(context or {}),
        }

    def suggest_adjustment(
        self,
        belief_id: str,
        belief_snapshot: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return an adjustment proposal for ``belief_id`` using the shared LLM.

        The method keeps the historical heuristic behaviour when the LLM is not
        available.  When the integration succeeds, the returned mapping mirrors
        :mod:`AGI_Evolutive.utils.llm_specs` expectations and is suitable for
        direct persistence in the belief graph.
        """

        stats = self.get_belief_stats(belief_id)
        fallback = self._fallback_adjustment(belief_id, stats, context=belief_snapshot)

        response = try_call_llm_dict(
            "belief_adaptation",
            input_payload=self._llm_payload(
                belief_id,
                belief_snapshot,
                stats,
                suggested_delta=fallback["delta"],
            ),
            logger=LOGGER,
        )

        if isinstance(response, Mapping):
            payload = dict(response)
            payload.setdefault("belief", belief_id)
            payload.setdefault("delta", fallback["delta"])
            payload.setdefault("justification", fallback["justification"])
            payload.setdefault("confidence", fallback.get("confidence", 0.5))
            payload.setdefault("notes", fallback.get("notes", ""))
            return payload

        return fallback

