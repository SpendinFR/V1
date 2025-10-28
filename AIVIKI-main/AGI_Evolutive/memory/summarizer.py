"""
Progressive summarization and compaction pipeline for long-term memory.


Design goals
------------
1) Turn old raw memories into hierarchical digests: daily → weekly → monthly
2) Keep salient facts (concepts, preferences, decisions) while dropping noise
3) Maintain lineage for drill-down, and mark raw items as compressed
4) TTL policies by memory kind, with pin/keep overrides
5) Idempotent runs: safe to call repeatedly (e.g., on a cron or manager tick)


Integration contract (MemoryStore protocol)
-------------------------------------------
The summarizer expects a `memory_store` object exposing:


- list_items(filter: dict) -> Iterable[dict]
  Supported filters (all optional):
    kind: str | list[str]
    older_than_ts: float
    newer_than_ts: float
    not_compressed: bool  # only items where item.get("compressed_into") is falsy
    limit: int


- add_item(item: dict) -> str
  Returns new item id. The item must at least define: kind, ts, text.


- update_item(id: str, patch: dict) -> None


- get_item(id: str) -> dict


- now() -> float


The item schema is flexible but these fields are leveraged if present:
    id: str
    ts: float (epoch seconds)
    kind: str (e.g., "interaction", "thought", "episode", "digest.daily", ...)
    text: str
    concepts: list[str]
    salience: float (0..1)
    pinned: bool
    tags: list[str]
    metadata: dict (e.g., {"emotion": "happy", "context": "chat", ...})
    compressed_into: str | None
    lineage: list[str] (children item ids consolidated into this digest)
    expiry_ts: float | None (optional TTL)


The summarizer optionally uses a ConceptStore-like object:
    - get_concept_weight(concept: str) -> float  # salience/support (0..1+)
  If not provided, a neutral weight of 1.0 is assumed.


No external ML dependency: we implement deterministic summarization with a small
scoring heuristic. You can plug an LLM by passing `llm_summarize_fn` if desired.
"""
from __future__ import annotations


import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import math
import random
from typing import Callable, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import time

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


DAY_SECONDS = 24 * 3600


# ----------------------
# Configuration defaults
# ----------------------
@dataclass
class AdaptiveWeightConfig:
    """Hyper-parameters for the online scoring weights."""

    initial_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)
    learning_rate: float = 0.05
    min_weight: float = 0.0
    max_weight: float = 1.5
    drift: float = 0.01
    max_score: float = 1.5


@dataclass
class AdaptiveTTLConfig:
    """Thompson-sampling configuration for TTL selection."""

    daily_options: Tuple[Optional[int], ...] = (180, 365, 540)
    weekly_options: Tuple[Optional[int], ...] = (365, 540, 720)
    monthly_options: Tuple[Optional[int], ...] = (720, None)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    exploration: float = 0.05


class OnlineWeightModel:
    """Simple online linear model with bounded weights."""

    def __init__(self, config: AdaptiveWeightConfig) -> None:
        self.config = config
        self.weights = list(config.initial_weights)
        self._normalize()

    def predict(self, features: Sequence[float]) -> float:
        if not features:
            return 0.0
        value = 0.0
        for weight, feature in zip(self.weights, features):
            value += weight * feature
        value = max(0.0, min(self.config.max_score, value))
        return value

    def observe(self, features: Sequence[float], reward: Optional[float]) -> None:
        if reward is None or not features:
            return
        reward = max(0.0, min(self.config.max_score, float(reward)))
        prediction = self.predict(features)
        error = reward - prediction
        lr = self.config.learning_rate
        for index, feature in enumerate(features):
            if index >= len(self.weights):
                break
            update = lr * error * float(feature)
            self.weights[index] += update
        self._apply_constraints()
        self._apply_drift()
        self._normalize()

    def export_state(self) -> Mapping[str, object]:
        return {
            "weights": list(self.weights),
            "max_score": self.config.max_score,
        }

    def _apply_constraints(self) -> None:
        for index, weight in enumerate(self.weights):
            bounded = max(self.config.min_weight, min(self.config.max_weight, weight))
            self.weights[index] = bounded

    def _apply_drift(self) -> None:
        drift = self.config.drift
        if drift <= 0.0:
            return
        baseline = self.config.initial_weights
        for index, base in enumerate(baseline):
            anchor = float(base)
            current = self.weights[index]
            self.weights[index] = (1.0 - drift) * current + drift * anchor

    def _normalize(self) -> None:
        total = sum(self.weights)
        if total <= 0.0:
            self.weights = list(self.config.initial_weights)
            total = sum(self.weights)
        if total > 0.0:
            self.weights = [weight / total for weight in self.weights]


class AdaptiveTTLStrategy:
    """Discrete Thompson Sampling for TTL selection per level."""

    def __init__(self, config: AdaptiveTTLConfig) -> None:
        self.config = config
        self.state: MutableMapping[str, MutableMapping[Optional[int], List[float]]] = defaultdict(dict)

    def select(self, level: str, default: Optional[int]) -> tuple[Optional[int], Mapping[str, object]]:
        options = self._options_for(level, default)
        if not options:
            return default, {"strategy": "static", "days": default}
        best_option: Optional[int] = None
        best_score = float("-inf")
        sampled: MutableMapping[Optional[int], float] = {}
        for option in options:
            alpha, beta = self._posterior(level, option)
            if random.random() < self.config.exploration:
                sample = random.random()
            else:
                sample = random.betavariate(alpha, beta)
            sampled[option] = sample
            if sample > best_score:
                best_score = sample
                best_option = option
        return best_option, {
            "strategy": "thompson",
            "days": best_option,
            "posterior": {
                option: {"alpha": self._posterior(level, option)[0], "beta": self._posterior(level, option)[1]}
                for option in options
            },
            "samples": sampled,
        }

    def observe(self, level: str, option: Optional[int], reward: Optional[float]) -> None:
        if option is None or reward is None:
            return
        reward = max(0.0, min(1.0, float(reward)))
        posterior = self.state[level].setdefault(
            option,
            [self.config.prior_alpha, self.config.prior_beta],
        )
        posterior[0] += reward
        posterior[1] += 1.0 - reward

    def export_state(self) -> Mapping[str, object]:
        return {
            level: {
                option: {"alpha": params[0], "beta": params[1]}
                for option, params in options.items()
            }
            for level, options in self.state.items()
        }

    def _options_for(self, level: str, default: Optional[int]) -> Tuple[Optional[int], ...]:
        if level == "daily":
            options = self.config.daily_options
        elif level == "weekly":
            options = self.config.weekly_options
        elif level == "monthly":
            options = self.config.monthly_options
        else:
            options = tuple()
        option_set = list(options)
        if default not in option_set:
            option_set.append(default)
        # Remove duplicates while preserving order
        seen: set[Optional[int]] = set()
        deduped: List[Optional[int]] = []
        for option in option_set:
            if option in seen:
                continue
            seen.add(option)
            deduped.append(option)
        return tuple(deduped)

    def _posterior(self, level: str, option: Optional[int]) -> Tuple[float, float]:
        params = self.state[level].setdefault(
            option,
            [self.config.prior_alpha, self.config.prior_beta],
        )
        return params[0], params[1]


@dataclass
class SummarizerConfig:
    """Configuration for :class:`ProgressiveSummarizer`."""

    # Age thresholds (days)
    daily_after_days: int = 7  # raw → daily after 7 days
    weekly_after_days: int = 21  # aggregate dailies older than 21 days
    monthly_after_days: int = 90  # aggregate weeklies older than 90 days

    # Limits
    max_raw_per_day: int = 400  # cap for daily window
    max_daily_per_week: int = 14  # collect up to N daily digests per weekly
    max_weekly_per_month: int = 8  # collect up to N weekly digests per monthly

    # Minimum number of source items before emitting a digest
    min_items_daily: int = 1
    min_items_weekly: int = 1
    min_items_monthly: int = 1

    # TTL policies (days) – ``None`` disables TTL for that level
    ttl_daily_days: Optional[int] = 365
    ttl_weekly_days: Optional[int] = 730
    ttl_monthly_days: Optional[int] = None

    # Fallback summary formatting
    max_summary_bullets: int = 6
    max_bullet_chars: int = 280

    # Feedback ingestion
    feedback_scan_limit: int = 400

    # Adaptive heuristics configuration
    adaptive_weight: AdaptiveWeightConfig = field(default_factory=AdaptiveWeightConfig)
    adaptive_ttl: AdaptiveTTLConfig = field(default_factory=AdaptiveTTLConfig)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class ProgressiveSummarizer:
    """Hierarchical summarization pipeline for long-term memories."""

    def __init__(
        self,
        memory_store: object,
        *,
        concept_store: Optional[object] = None,
        config: Optional[SummarizerConfig] = None,
        llm_summarize_fn: Optional[Callable[[str, Sequence[Mapping[str, object]]], str]] = None,
    ) -> None:
        self.m = memory_store
        self.c = concept_store
        self.config = config or SummarizerConfig()
        self.llm_summarize_fn = llm_summarize_fn
        self._known_fingerprints: set[str] = set()
        self.weight_model = OnlineWeightModel(self.config.adaptive_weight)
        self.ttl_strategy = AdaptiveTTLStrategy(self.config.adaptive_ttl)
        self._hydrate_fingerprint_cache()
        self._ingest_feedback_history()
        self.last_llm_summary: Optional[Mapping[str, object]] = None

    # ------------------------------------------------------------------
    def step(self, now: Optional[float] = None) -> MutableMapping[str, object]:
        """Run one maintenance pass of the summarization pipeline."""

        now = self._now(now)
        stats: MutableMapping[str, object] = {
            "ts": now,
            "daily": self._promote_raw_to_daily(now),
            "weekly": self._promote_daily_to_weekly(now),
            "monthly": self._promote_weekly_to_monthly(now),
        }
        return stats

    # ------------------------------------------------------------------
    def _hydrate_fingerprint_cache(self) -> None:
        levels = ("digest.daily", "digest.weekly", "digest.monthly")
        for kind in levels:
            try:
                existing = self.m.list_items({"kind": kind, "limit": 500})
            except Exception:
                existing = []
            for item in existing or []:
                fp = self._extract_fingerprint(item)
                if fp:
                    self._known_fingerprints.add(fp)

    def _ingest_feedback_history(self) -> None:
        limit = max(0, int(self.config.feedback_scan_limit))
        kinds = {
            "digest.daily": "daily",
            "digest.weekly": "weekly",
            "digest.monthly": "monthly",
        }
        for kind, level in kinds.items():
            try:
                history = list(self.m.list_items({"kind": kind, "limit": limit}) or [])
            except Exception:
                history = []
            for item in history:
                reward = self._extract_feedback_score(item)
                ttl_choice = self._infer_ttl_choice(item)
                if ttl_choice is not None and reward is not None:
                    self.ttl_strategy.observe(level, ttl_choice, reward)
                features = self._extract_features(item)
                if reward is not None:
                    self.weight_model.observe(features, reward)

    # ------------------------------------------------------------------
    def _promote_raw_to_daily(self, now: float) -> MutableMapping[str, object]:
        threshold = now - self.config.daily_after_days * DAY_SECONDS
        try:
            raw_items = list(
                self.m.list_items(
                    {
                        "older_than_ts": threshold,
                        "not_compressed": True,
                        "limit": self.config.max_raw_per_day,
                    }
                )
                or []
            )
        except Exception:
            raw_items = []
        candidates = [item for item in raw_items if self._is_raw_item(item)]
        buckets = self._bucket_by_period(candidates, period="daily")
        created = 0
        skipped = 0
        for (start_ts, end_ts), bucket in buckets.items():
            if len(bucket) < self.config.min_items_daily:
                skipped += 1
                continue
            digest_id = self._materialize_digest(
                level="daily",
                items=bucket,
                start_ts=start_ts,
                end_ts=end_ts,
                now=now,
            )
            if digest_id:
                created += 1
            else:
                skipped += 1
        return {
            "candidates": len(candidates),
            "created": created,
            "skipped": skipped,
        }

    def _promote_daily_to_weekly(self, now: float) -> MutableMapping[str, object]:
        threshold = now - self.config.weekly_after_days * DAY_SECONDS
        try:
            daily_items = list(
                self.m.list_items(
                    {
                        "kind": "digest.daily",
                        "older_than_ts": threshold,
                        "not_compressed": True,
                        "limit": self.config.max_daily_per_week,
                    }
                )
                or []
            )
        except Exception:
            daily_items = []
        buckets = self._bucket_by_period(daily_items, period="weekly")
        created = 0
        skipped = 0
        for (start_ts, end_ts), bucket in buckets.items():
            if len(bucket) < self.config.min_items_weekly:
                skipped += 1
                continue
            digest_id = self._materialize_digest(
                level="weekly",
                items=bucket,
                start_ts=start_ts,
                end_ts=end_ts,
                now=now,
            )
            if digest_id:
                created += 1
            else:
                skipped += 1
        return {
            "candidates": len(daily_items),
            "created": created,
            "skipped": skipped,
        }

    def _promote_weekly_to_monthly(self, now: float) -> MutableMapping[str, object]:
        threshold = now - self.config.monthly_after_days * DAY_SECONDS
        try:
            weekly_items = list(
                self.m.list_items(
                    {
                        "kind": "digest.weekly",
                        "older_than_ts": threshold,
                        "not_compressed": True,
                        "limit": self.config.max_weekly_per_month,
                    }
                )
                or []
            )
        except Exception:
            weekly_items = []
        buckets = self._bucket_by_period(weekly_items, period="monthly")
        created = 0
        skipped = 0
        for (start_ts, end_ts), bucket in buckets.items():
            if len(bucket) < self.config.min_items_monthly:
                skipped += 1
                continue
            digest_id = self._materialize_digest(
                level="monthly",
                items=bucket,
                start_ts=start_ts,
                end_ts=end_ts,
                now=now,
            )
            if digest_id:
                created += 1
            else:
                skipped += 1
        return {
            "candidates": len(weekly_items),
            "created": created,
            "skipped": skipped,
        }

    # ------------------------------------------------------------------
    def _materialize_digest(
        self,
        *,
        level: str,
        items: Sequence[Mapping[str, object]],
        start_ts: float,
        end_ts: float,
        now: float,
    ) -> Optional[str]:
        child_ids = sorted(
            str(item.get("id"))
            for item in items
            if isinstance(item, Mapping) and item.get("id")
        )
        if not child_ids:
            return None
        fingerprint = self._fingerprint(child_ids)
        if fingerprint in self._known_fingerprints:
            return None
        summary = self._compose_summary(level, items, start_ts, end_ts)
        if not summary.strip():
            return None
        digest_ts = min(end_ts - 1.0, now)
        ttl_days, ttl_meta = self._ttl_days_for(level, items)
        metadata = {
            "digest_level": level,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "source_count": len(child_ids),
            "fingerprint": fingerprint,
            "bridges": {"sources": child_ids},
            "adaptive": {
                "ttl": dict(ttl_meta),
                "weights": self.weight_model.export_state(),
                "strategy_state": self.ttl_strategy.export_state(),
            },
            "usage_metrics": {
                "views": 0,
                "positive": 0,
                "negative": 0,
                "last_view_ts": None,
            },
        }
        payload = {
            "kind": f"digest.{level}",
            "ts": digest_ts,
            "text": summary,
            "concepts": self._merge_list_field(items, "concepts"),
            "tags": self._merge_list_field(items, "tags"),
            "salience": self._aggregate_salience(items),
            "lineage": child_ids,
            "metadata": metadata,
        }
        if ttl_days is not None:
            payload["expiry_ts"] = digest_ts + ttl_days * DAY_SECONDS
        try:
            digest_id = self.m.add_item(payload)
        except Exception:
            return None
        self._known_fingerprints.add(fingerprint)
        for item in items:
            item_id = item.get("id") if isinstance(item, Mapping) else None
            if not item_id:
                continue
            patch = {"compressed_into": digest_id}
            try:
                self.m.update_item(str(item_id), patch)
            except Exception:
                continue
        return str(digest_id)

    # ------------------------------------------------------------------
    def _compose_summary(
        self,
        level: str,
        items: Sequence[Mapping[str, object]],
        start_ts: float,
        end_ts: float,
    ) -> str:
        label = self._format_period(level, start_ts, end_ts)
        structured_items = [self._prepare_summary_item(item, level) for item in items]
        structured_items = [item for item in structured_items if item["text"]]
        if not structured_items:
            return ""
        llm_payload = {
            "level": level,
            "period": {"start_ts": start_ts, "end_ts": end_ts},
            "items": [
                {
                    "id": entry.get("id"),
                    "text": entry["text"],
                    "score": entry["score"],
                    "kind": entry.get("kind"),
                }
                for entry in structured_items[:20]
            ],
        }
        llm_response = try_call_llm_dict(
            "memory_summarizer_guidance",
            input_payload=llm_payload,
            logger=LOGGER,
        )
        if llm_response:
            self.last_llm_summary = llm_response
            summary_text = llm_response.get("summary")
            if summary_text:
                return str(summary_text)
        if self.llm_summarize_fn is not None:
            try:
                return str(self.llm_summarize_fn(level, structured_items))
            except Exception:
                pass
        structured_items.sort(key=lambda payload: payload["score"], reverse=True)
        bullets = []
        limit = max(1, self.config.max_summary_bullets)
        for payload in structured_items[:limit]:
            text = payload["text"]
            if len(text) > self.config.max_bullet_chars:
                text = text[: self.config.max_bullet_chars - 1].rstrip() + "…"
            bullets.append(f"- {text}")
        concepts = self._merge_list_field(items, "concepts")
        concept_fragment = ""
        if concepts:
            concept_fragment = f"\nConcepts marquants: {', '.join(concepts[:8])}"
        return f"{label}\n" + "\n".join(bullets) + concept_fragment

    def _prepare_summary_item(self, item: Mapping[str, object], level: str) -> Mapping[str, object]:
        text = str(item.get("text", "")).strip()
        score = self._score_item(item)
        return {
            "id": item.get("id"),
            "text": text,
            "score": score,
            "level": level,
            "kind": item.get("kind"),
        }

    # ------------------------------------------------------------------
    def _bucket_by_period(
        self,
        items: Sequence[Mapping[str, object]],
        *,
        period: str,
    ) -> MutableMapping[tuple[float, float], List[Mapping[str, object]]]:
        buckets: MutableMapping[tuple[float, float], List[Mapping[str, object]]] = defaultdict(list)
        for item in items:
            if not isinstance(item, Mapping):
                continue
            ts = _safe_float(item.get("ts"), default=0.0)
            if ts <= 0.0:
                continue
            start_ts, end_ts = self._period_bounds(ts, period)
            buckets[(start_ts, end_ts)].append(item)
        return buckets

    def _period_bounds(self, ts: float, period: str) -> tuple[float, float]:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if period == "daily":
            start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == "weekly":
            start = dt - timedelta(days=dt.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
        elif period == "monthly":
            start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        else:
            raise ValueError(f"Unknown period: {period}")
        return (start.timestamp(), end.timestamp())

    def _format_period(self, level: str, start_ts: float, end_ts: float) -> str:
        start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        if level == "daily":
            return f"Synthèse du {start.strftime('%Y-%m-%d')}"
        if level == "weekly":
            iso_year, iso_week, _ = start.isocalendar()
            return f"Synthèse hebdomadaire {iso_year}-W{iso_week:02d}"
        if level == "monthly":
            return f"Synthèse mensuelle {start.strftime('%Y-%m')}"
        return f"Synthèse ({level}) {start.strftime('%Y-%m-%d')}"  # fallback

    # ------------------------------------------------------------------
    def _merge_list_field(self, items: Sequence[Mapping[str, object]], field: str) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for item in items:
            values = item.get(field) if isinstance(item, Mapping) else None
            if not isinstance(values, Iterable):
                continue
            for value in values:
                if not value:
                    continue
                key = str(value)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(key)
        return merged

    def _aggregate_salience(self, items: Sequence[Mapping[str, object]]) -> float:
        if not items:
            return 0.0
        scores = [self._score_item(item) for item in items]
        return max(0.0, min(1.0, sum(scores) / max(1, len(scores))))

    def _extract_features(self, item: Mapping[str, object]) -> Tuple[float, float, float]:
        if not isinstance(item, Mapping):
            return (0.3, 1.0, 1.0)
        base = max(0.0, min(1.0, _safe_float(item.get("salience"), default=0.3)))
        concept_strength = self._concept_strength(item)
        recency = self._recency_feature(item)
        return (base, concept_strength, recency)

    def _concept_strength(self, item: Mapping[str, object]) -> float:
        concepts = item.get("concepts") if isinstance(item, Mapping) else None
        if not concepts:
            return 1.0
        if isinstance(concepts, (str, bytes)):
            concepts_iter = [concepts]
        elif isinstance(concepts, Iterable):
            concepts_iter = concepts
        else:
            return 1.0
        weights = []
        for concept in concepts_iter:
            weight = self._concept_weight(str(concept))
            if weight is not None:
                weights.append(float(weight))
        if not weights:
            return 1.0
        return max(0.0, sum(weights) / len(weights))

    def _recency_feature(self, item: Mapping[str, object]) -> float:
        ts = _safe_float(item.get("ts"), default=self._now())
        age_days = max(0.0, (self._now() - ts) / DAY_SECONDS)
        return math.exp(-age_days / 60.0)

    def _score_item(self, item: Mapping[str, object]) -> float:
        features = self._extract_features(item)
        score = self.weight_model.predict(features)
        reward = self._extract_feedback_score(item)
        if reward is not None:
            self.weight_model.observe(features, reward)
            score = self.weight_model.predict(features)
        return max(0.0, min(self.config.adaptive_weight.max_score, score))

    def _extract_feedback_score(self, item: Mapping[str, object]) -> Optional[float]:
        if not isinstance(item, Mapping):
            return None
        metadata = item.get("metadata")
        reward: Optional[float] = None
        if isinstance(metadata, Mapping):
            if "feedback_score" in metadata:
                reward = _safe_float(metadata.get("feedback_score"), default=0.0)
            elif "reward" in metadata:
                reward = _safe_float(metadata.get("reward"), default=0.0)
            else:
                feedback = metadata.get("feedback")
                if isinstance(feedback, Mapping):
                    if "score" in feedback:
                        reward = _safe_float(feedback.get("score"), default=0.0)
                    elif "reward" in feedback:
                        reward = _safe_float(feedback.get("reward"), default=0.0)
                usage = metadata.get("usage_metrics")
                if isinstance(usage, Mapping):
                    positive = max(0.0, _safe_float(usage.get("positive"), default=0.0))
                    negative = max(0.0, _safe_float(usage.get("negative"), default=0.0))
                    total = positive + negative
                    if total > 0.0:
                        reward = positive / total
                    else:
                        views = max(0.0, _safe_float(usage.get("views"), default=0.0))
                        if views > 0.0:
                            reward = min(1.0, positive / views)
        if reward is None and "reward" in item:
            reward = _safe_float(item.get("reward"), default=0.0)
        if reward is None:
            return None
        return max(0.0, min(1.0, float(reward)))

    def _estimate_bucket_reward(self, items: Sequence[Mapping[str, object]]) -> Optional[float]:
        rewards = [r for r in (self._extract_feedback_score(item) for item in items) if r is not None]
        if not rewards:
            return None
        return sum(rewards) / len(rewards)

    def _concept_weight(self, concept: str) -> float:
        if not self.c:
            return 1.0
        if hasattr(self.c, "get_concept_weight"):
            try:
                weight = self.c.get_concept_weight(concept)  # type: ignore[attr-defined]
                return _safe_float(weight, default=1.0)
            except Exception:
                return 1.0
        concepts = getattr(self.c, "concepts", None)
        if isinstance(concepts, Mapping):
            payload = concepts.get(concept)
            if payload is None and hasattr(self.c, "_find_by_label"):
                try:
                    cid = self.c._find_by_label(concept)  # type: ignore[attr-defined]
                    if cid:
                        payload = concepts.get(cid)
                except Exception:
                    payload = None
            if payload is not None:
                if hasattr(payload, "salience"):
                    return _safe_float(getattr(payload, "salience"), default=1.0)
                if isinstance(payload, Mapping) and "salience" in payload:
                    return _safe_float(payload.get("salience"), default=1.0)
        return 1.0

    # ------------------------------------------------------------------
    def _is_raw_item(self, item: Mapping[str, object]) -> bool:
        if not isinstance(item, Mapping):
            return False
        if item.get("pinned"):
            return False
        if item.get("compressed_into"):
            return False
        kind = str(item.get("kind", ""))
        return not kind.startswith("digest.")

    def _extract_fingerprint(self, item: Mapping[str, object]) -> Optional[str]:
        if not isinstance(item, Mapping):
            return None
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            fp = metadata.get("fingerprint")
            if fp:
                return str(fp)
        lineage = item.get("lineage")
        if isinstance(lineage, Iterable):
            child_ids = [str(cid) for cid in lineage if cid]
            if child_ids:
                return self._fingerprint(sorted(child_ids))
        return None

    def _infer_ttl_choice(self, item: Mapping[str, object]) -> Optional[int]:
        if not isinstance(item, Mapping):
            return None
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            adaptive = metadata.get("adaptive")
            if isinstance(adaptive, Mapping):
                ttl_payload = adaptive.get("ttl")
                if isinstance(ttl_payload, Mapping) and "days" in ttl_payload:
                    raw = ttl_payload.get("days")
                    if raw is None:
                        return None
                    return int(_safe_float(raw, default=0.0)) or None
        expiry_ts = item.get("expiry_ts")
        if expiry_ts is None:
            return None
        expiry_ts = _safe_float(expiry_ts, default=0.0)
        ts = _safe_float(item.get("ts"), default=0.0)
        if expiry_ts <= 0.0 or ts <= 0.0:
            return None
        days = int(max(0.0, (expiry_ts - ts) / DAY_SECONDS))
        return days or None

    def _ttl_days_for(
        self, level: str, items: Optional[Sequence[Mapping[str, object]]] = None
    ) -> tuple[Optional[int], Mapping[str, object]]:
        base = self._static_ttl(level)
        ttl_days, ttl_meta = self.ttl_strategy.select(level, base)
        reward = self._estimate_bucket_reward(items or [])
        meta = dict(ttl_meta)
        meta.setdefault("default_days", base)
        if reward is not None:
            self.ttl_strategy.observe(level, ttl_days, reward)
            meta["pre_reward"] = reward
        return ttl_days, meta

    def _static_ttl(self, level: str) -> Optional[int]:
        if level == "daily":
            return self.config.ttl_daily_days
        if level == "weekly":
            return self.config.ttl_weekly_days
        if level == "monthly":
            return self.config.ttl_monthly_days
        return None

    def _fingerprint(self, texts: Sequence[str]) -> str:
        h = 0
        for text in texts:
            data = str(text).encode("utf-8", errors="ignore")
            digest = hashlib.blake2b(data, digest_size=8).digest()
            h ^= int.from_bytes(digest, "big", signed=False)
        return f"fp{h:016x}"

    def _now(self, override: Optional[float] = None) -> float:
        if override is not None:
            return float(override)
        if hasattr(self.m, "now"):
            try:
                return float(self.m.now())
            except Exception:
                pass
        return time.time()


__all__ = [
    "AdaptiveTTLConfig",
    "AdaptiveWeightConfig",
    "SummarizerConfig",
    "ProgressiveSummarizer",
]
