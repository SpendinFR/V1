from __future__ import annotations

import logging
import math
import re
import time
import unicodedata
import datetime as dt
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)


logger = logging.getLogger(__name__)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()


def _fmt_date(ts: float) -> str:
    try:
        return dt.datetime.fromtimestamp(ts).strftime("%d/%m/%Y")
    except Exception:
        return "?"


def _now() -> float:
    return time.time()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return text.strip()


@dataclass
class _TTLCacheEntry:
    ttl_index: int
    ttl_seconds: float
    updated_at: float


class OnlineTopicClassifier:
    """Very small online learner leveraging n-grams and punctuation.

    It behaves as a permissive fallback when heuristic keyword extraction
    misses salient content.  The implementation purposefully remains
    lightweight to avoid introducing heavy ML dependencies while still
    honouring the "online" requirement (continuous updates at inference
    time).
    """

    def __init__(self, learning_rate: float = 0.12, decay: float = 0.997) -> None:
        self.learning_rate = float(max(0.01, learning_rate))
        self.decay = float(min(0.999, max(0.9, decay)))
        self.bias: float = 0.0
        self.feature_weights: Dict[str, float] = defaultdict(float)
        self.token_memory: Counter[str] = Counter()
        self._last_decay_ts: float = _now()

    # ---- Feature extraction -------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        cleaned = _normalize_text(text).lower()
        words = re.findall(r"[\w'’]{2,}", cleaned, flags=re.UNICODE)
        return words

    def _extract_features(self, text: str) -> List[str]:
        tokens = self._tokenize(text)
        feats: List[str] = []
        feats.extend(f"token::{tok}" for tok in tokens)
        feats.extend(f"bigram::{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        if "?" in text:
            feats.append("punct::question")
        if "!" in text:
            feats.append("punct::exclaim")
        if re.search(r"[\U0001F300-\U0001FAFF]", text):
            feats.append("has_emoji")
        return feats

    # ---- Learning -----------------------------------------------------------
    def _apply_decay(self) -> None:
        now = _now()
        elapsed = max(0.0, now - self._last_decay_ts)
        if elapsed < 1.0:
            return
        factor = self.decay ** (elapsed / 30.0)
        if factor >= 0.9995:
            return
        for feat in list(self.feature_weights.keys()):
            self.feature_weights[feat] *= factor
            if abs(self.feature_weights[feat]) < 1e-6:
                self.feature_weights.pop(feat, None)
        for token in list(self.token_memory.keys()):
            self.token_memory[token] = int(self.token_memory[token] * factor)
            if self.token_memory[token] <= 0:
                self.token_memory.pop(token, None)
        self.bias *= factor
        self._last_decay_ts = now

    def observe(self, text: str, positive_tokens: Sequence[str]) -> None:
        self._apply_decay()
        feats = self._extract_features(text)
        label = 1.0 if positive_tokens else 0.0
        score = self.score(text)
        error = label - score
        step = self.learning_rate * error
        if abs(step) < 1e-5:
            step = 0.0
        if step:
            for feat in feats:
                self.feature_weights[feat] += step
            self.bias += step
        for token in positive_tokens:
            norm = token.lower().strip()
            if norm:
                self.token_memory[norm] += 1

    # ---- Inference ----------------------------------------------------------
    def score(self, text: str) -> float:
        feats = self._extract_features(text)
        activation = self.bias
        for feat in feats:
            activation += self.feature_weights.get(feat, 0.0)
        try:
            return 1.0 / (1.0 + math.exp(-activation))
        except OverflowError:
            return 0.0 if activation < 0 else 1.0

    def fallback_topics(
        self,
        text: str,
        exclude: Sequence[str],
        top_k: int = 3,
        min_score: float = 0.5,
    ) -> List[str]:
        exclude_set = {e.lower() for e in exclude}
        feats = self._extract_features(text)
        scored: List[Tuple[float, str]] = []
        for feat in feats:
            if not feat.startswith("token::"):
                continue
            token = feat.split("::", 1)[1]
            if token in exclude_set:
                continue
            weight = self.feature_weights.get(feat, 0.0) + 0.15 * self.token_memory.get(token, 0)
            if weight <= 0:
                continue
            scored.append((weight, token))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for weight, token in scored:
            if len(results) >= top_k:
                break
            # logistic style check for the token context
            if self.score(token) < min_score:
                continue
            results.append(token)
        return results

    def diagnostics(self) -> Dict[str, Any]:
        top_tokens = Counter({k.split("::", 1)[1]: v for k, v in self.feature_weights.items() if k.startswith("token::")})
        return {
            "bias": round(self.bias, 5),
            "known_tokens": len(self.token_memory),
            "top_features": [tok for tok, _ in top_tokens.most_common(5)],
        }


class ContextBuilder:
    """Build a conversational snapshot from the memory buffers.

    The conversation module uses this builder to feed UI/prompts with
    recent exchanges, salient milestones and lightweight user style
    heuristics.  It deliberately focuses on human-facing summaries and is
    distinct from the rule-engine context builder located in
    :mod:`AGI_Evolutive.social.interaction_rule`, which prepares symbolic
    predicates for social tactics selection.
    """

    TTL_OPTIONS_DAYS: Tuple[int, ...] = (3, 7, 14, 30)
    _TTL_STEP_MAX: int = 1

    def __init__(self, arch):
        self.arch = arch
        self.mem = arch.memory
        self._ttl_cache: Dict[str, _TTLCacheEntry] = {}
        self._hazard_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=64))
        self._drift_log: Deque[Tuple[float, str, int, int]] = deque(maxlen=50)
        self._topic_classifier = OnlineTopicClassifier()
        self._feedback_history: Deque[Dict[str, Any]] = deque(maxlen=128)
        self._sequence_buffer: Deque[str] = deque(maxlen=128)
        self._persona_overrides: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------ TTL --
    def _ttl_seconds_for(self, kind: str, timestamps: Sequence[float]) -> float:
        kind = (kind or "unknown").lower()
        ttl_entry = self._ttl_cache.get(kind)
        now = _now()

        def select_index_from_half_life(half_life_days: float) -> int:
            closest_idx = min(
                range(len(self.TTL_OPTIONS_DAYS)),
                key=lambda idx: abs(self.TTL_OPTIONS_DAYS[idx] - half_life_days),
            )
            if ttl_entry is None:
                return closest_idx
            prev_idx = ttl_entry.ttl_index
            if closest_idx == prev_idx:
                return prev_idx
            # bound the update speed (step max)
            direction = 1 if closest_idx > prev_idx else -1
            step = max(-self._TTL_STEP_MAX, min(self._TTL_STEP_MAX, direction))
            bounded_idx = prev_idx + step
            if bounded_idx != prev_idx:
                self._drift_log.append((now, kind, prev_idx, bounded_idx))
            return bounded_idx

        if timestamps:
            sorted_ts = sorted(set(_safe_float(ts, now) for ts in timestamps if ts))
            if len(sorted_ts) > 1:
                intervals = [b - a for a, b in zip(sorted_ts, sorted_ts[1:]) if b >= a]
                if intervals:
                    mean_interval = sum(intervals) / len(intervals)
                    hazard = 1.0 / max(mean_interval, 60.0)
                    half_life_sec = math.log(2.0) / hazard
                    half_life_days = max(1.0, min(45.0, half_life_sec / 86400.0))
                    idx = select_index_from_half_life(half_life_days)
                    ttl_days = self.TTL_OPTIONS_DAYS[idx]
                    ttl_seconds = float(ttl_days * 86400)
                    self._ttl_cache[kind] = _TTLCacheEntry(idx, ttl_seconds, now)
                    return ttl_seconds
        if ttl_entry is not None:
            return ttl_entry.ttl_seconds
        # default selection leaning on frequency (no timestamps or single point)
        default_idx = select_index_from_half_life(7.0)
        ttl_seconds = float(self.TTL_OPTIONS_DAYS[default_idx] * 86400)
        self._ttl_cache[kind] = _TTLCacheEntry(default_idx, ttl_seconds, now)
        return ttl_seconds

    def _group_timestamps_by_kind(
        self, memories: Sequence[Mapping[str, Any]]
    ) -> Dict[str, List[float]]:
        grouped: Dict[str, List[float]] = defaultdict(list)
        for memory in memories:
            kind = (memory.get("memory_type") or memory.get("kind") or "unknown").lower()
            ts = memory.get("ts") or memory.get("timestamp") or memory.get("t")
            grouped[kind].append(_safe_float(ts, _now()))
        return grouped

    # -------------------------------------------------------------- Persona --
    def _infer_user_id(self) -> str:
        user_model = getattr(self.arch, "user_model", None)
        if user_model is None:
            return "anonymous"
        try:
            desc = user_model.describe()
        except Exception:
            desc = {}
        for key in ("id", "user_id", "uuid"):
            value = desc.get(key)
            if value:
                return str(value)
        return "anonymous"

    def _persona_settings(self, user_id: str) -> Dict[str, float]:
        if user_id in self._persona_overrides:
            return self._persona_overrides[user_id]
        persona: Dict[str, Any] = {}
        user_model = getattr(self.arch, "user_model", None)
        if user_model is not None:
            try:
                persona = (user_model.describe() or {}).get("persona", {}) or {}
            except Exception:
                persona = {}
        tone = (persona.get("tone") or "").lower()
        overrides = {
            "long_threshold": 160.0,
            "question_ratio": 0.2,
            "exclaim_ratio": 0.1,
        }
        if "verbose" in tone or "déta" in tone:
            overrides["long_threshold"] = 120.0
        if "introvert" in tone:
            overrides["question_ratio"] = 0.12
        values = persona.get("values") or {}
        if isinstance(values, Mapping) and values.get("curiosity", 0) > 0.6:
            overrides["question_ratio"] = 0.15
        self._persona_overrides[user_id] = overrides
        return overrides

    # ------------------------------------------------------------- Feedback --
    def _pull_feedback(self) -> Iterable[Dict[str, Any]]:
        sources = []
        for attr in ("conversation_feedback", "feedback_bus", "telemetry"):
            payload = getattr(self.arch, attr, None)
            if payload is None:
                continue
            if hasattr(payload, "pop"):
                try:
                    item = payload.pop("conversation", None)
                    if item:
                        sources.append(item)
                except Exception:
                    continue
            elif callable(getattr(payload, "get", None)):
                try:
                    entries = payload.get("conversation_feedback")
                except Exception:
                    entries = None
                if entries:
                    sources.extend(entries)
        return sources

    def _update_feedback_loop(self, ctx: Dict[str, Any]) -> None:
        for feedback in self._pull_feedback():
            try:
                payload = {
                    "ts": _safe_float(feedback.get("ts"), _now()),
                    "kind": feedback.get("kind") or "unknown",
                    "delta_topics": feedback.get("delta_topics"),
                }
                self._feedback_history.append(payload)
            except Exception:
                continue
        if ctx.get("topics"):
            self._sequence_buffer.extend(ctx["topics"])

    # -------------------------------------------------------- Seq. motifs ---
    def _sequence_motifs(self, recents: Sequence[Mapping[str, Any]]) -> List[str]:
        sequences: Counter[Tuple[str, str]] = Counter()
        normalized_msgs: List[List[str]] = []
        for memo in recents:
            txt = _normalize_text(memo.get("text") or memo.get("content") or "")
            if not txt:
                continue
            tokens = [t for t in re.findall(r"[\w'’]{3,}", txt.lower()) if len(t) >= 3]
            if tokens:
                normalized_msgs.append(tokens)
        for tokens in normalized_msgs:
            for pair in zip(tokens, tokens[1:]):
                sequences[pair] += 1
        motifs: List[str] = []
        for (a, b), count in sequences.most_common(5):
            if count < 2:
                continue
            motifs.append(f"{a} → {b} (x{count})")
        return motifs

    # --------------------------------------------------------- Recent msgs --
    def _fetch_recent_memories(self, limit: int) -> List[Dict[str, Any]]:
        try:
            return list(self.mem.get_recent_memories(n=limit))
        except TypeError:
            # legacy signature using limit
            return list(self.mem.get_recent_memories(limit=limit))
        except Exception:
            return []

    def _recent_msgs(self, k: int = 8) -> List[Dict[str, Any]]:
        rec = self._fetch_recent_memories(240)
        grouped = self._group_timestamps_by_kind(rec)
        ttl_per_kind = {kind: self._ttl_seconds_for(kind, ts_list) for kind, ts_list in grouped.items()}
        now = _now()
        chats: List[Dict[str, Any]] = []
        for memory in rec:
            kind = (memory.get("kind") or memory.get("memory_type") or "").lower()
            if kind not in {"interaction", "chat", "message"}:
                continue
            ttl_seconds = ttl_per_kind.get(kind, ttl_per_kind.get("interaction", 7 * 86400.0))
            ts = _safe_float(
                memory.get("ts") or memory.get("timestamp") or memory.get("t"),
                now,
            )
            if now - ts > ttl_seconds:
                continue
            chats.append(memory)
        return chats[-k:]

    # ------------------------------------------------------------ Moments ---
    def _key_moments(self, horizon: int = 2000) -> List[str]:
        rec = self._fetch_recent_memories(horizon)
        grouped = self._group_timestamps_by_kind(rec)
        ttl_per_kind = {kind: self._ttl_seconds_for(kind, ts_list) for kind, ts_list in grouped.items()}
        now = _now()
        marks: List[str] = []
        watch_tags = {"milestone", "decision", "preference", "pinned"}
        for memory in rec:
            if not watch_tags.intersection(set(memory.get("tags") or [])):
                continue
            kind = (memory.get("memory_type") or memory.get("kind") or "milestone").lower()
            ttl_seconds = ttl_per_kind.get(kind, 14 * 86400.0)
            ts = _safe_float(memory.get("ts") or memory.get("timestamp"), now)
            if now - ts > ttl_seconds:
                continue
            txt = (memory.get("text") or memory.get("content") or "")
            marks.append(f"- {_fmt_date(ts)} : {txt[:80]}")
        return marks[-8:]

    # -------------------------------------------------------------- Topics ---
    _STOPWORDS = {
        "bonjour",
        "merci",
        "avec",
        "alors",
        "aussi",
        "cela",
        "comme",
        "avoir",
        "faire",
        "cette",
        "c'est",
    }

    _STRUCTURE_PATTERNS = (
        re.compile(r"\best\s+(?:un|une|le|la|l['’])\s+([\wà-ÿ]{3,})", re.IGNORECASE),
        re.compile(r"\bc['’]est\s+(?:quoi\s+)?([\wà-ÿ]{3,})", re.IGNORECASE),
    )

    def _topics(self, recents: List[Dict[str, Any]]) -> List[str]:
        c = Counter()
        candidate_tokens: List[str] = []
        for memo in recents:
            txt_raw = memo.get("text") or memo.get("content") or ""
            txt = _normalize_text(txt_raw)
            if not txt:
                continue
            lowered = txt.lower()
            tokens = re.findall(r"[a-zà-ÿ]{3,}", lowered)
            for pattern in self._STRUCTURE_PATTERNS:
                for match in pattern.findall(txt):
                    cleaned = match.lower()
                    if cleaned:
                        c[cleaned] += 1.5
            for w in tokens:
                if w in self._STOPWORDS:
                    continue
                c[w] += 1
            candidate_tokens.extend(tokens)
            if not tokens:
                self._topic_classifier.observe(txt, [])
            else:
                top = [tok for tok in tokens if tok not in self._STOPWORDS][:3]
                self._topic_classifier.observe(txt, top)
        top_topics = [w for w, _ in c.most_common(10)]
        if len(top_topics) < 5 and candidate_tokens:
            fallback = []
            for memo in recents:
                txt = memo.get("text") or memo.get("content") or ""
                fallback.extend(self._topic_classifier.fallback_topics(txt, top_topics, top_k=5))
            for token in fallback:
                if token not in top_topics:
                    top_topics.append(token)
                if len(top_topics) >= 10:
                    break
        return top_topics[:10]

    # ------------------------------------------------------------ User style --
    def _user_style(self, recents: List[Dict[str, Any]], persona_overrides: Dict[str, float]) -> Dict[str, Any]:
        long_threshold = persona_overrides.get("long_threshold", 160.0)
        question_ratio = persona_overrides.get("question_ratio", 0.2)
        exclaim_ratio = persona_overrides.get("exclaim_ratio", 0.1)
        count = len(recents) or 1
        long_msgs = sum(1 for m in recents if len(_normalize_text(m.get("text") or "")) > long_threshold)
        questions = sum(1 for m in recents if "?" in (m.get("text") or ""))
        exclam = sum(1 for m in recents if "!" in (m.get("text") or ""))
        expressive_score = exclam / count
        return {
            "prefers_long": (long_msgs / count) > 0.3,
            "asks_questions": (questions / count) > question_ratio,
            "expressive": expressive_score > exclaim_ratio,
            "expressive_score": round(expressive_score, 3),
        }

    # ----------------------------------------------------------- Related doc --
    def _related_inbox(self, user_msg: str) -> List[str]:
        # si tu as un index de docs, branche-le ici ; sinon laisse vide
        related_provider = getattr(self.arch, "related_index", None)
        if related_provider and hasattr(related_provider, "search"):
            try:
                results = related_provider.search(user_msg, top_k=3)
                return [str(item) for item in results]
            except Exception:
                logger.debug("related_inbox search failed", exc_info=True)
        return []

    # ------------------------------------------------------------- Monitoring --
    def _monitoring_snapshot(self) -> Dict[str, Any]:
        ttl_snapshot = {
            kind: {
                "days": self.TTL_OPTIONS_DAYS[entry.ttl_index],
                "seconds": entry.ttl_seconds,
                "updated_at": entry.updated_at,
            }
            for kind, entry in self._ttl_cache.items()
        }
        recent_drifts = list(self._drift_log)[-10:]
        drift_entries = [
            {
                "ts": ts,
                "kind": kind,
                "from": self.TTL_OPTIONS_DAYS[old_idx],
                "to": self.TTL_OPTIONS_DAYS[new_idx],
            }
            for ts, kind, old_idx, new_idx in recent_drifts
        ]
        return {
            "ttl": ttl_snapshot,
            "drift_log": drift_entries,
            "classifier": self._topic_classifier.diagnostics(),
            "feedback_buffer": list(self._feedback_history)[-5:],
        }

    def _llm_enrichment(
        self,
        user_msg: str,
        recent: List[Dict[str, Any]],
        long_summary: List[Any],
        topics: List[str],
        persona_overrides: Mapping[str, float],
    ) -> Optional[Mapping[str, Any]]:
        if not _llm_enabled():
            return None

        recent_payload: List[Dict[str, Any]] = []
        for memo in recent[-6:]:
            text = memo.get("text") or memo.get("content") or ""
            recent_payload.append(
                {
                    "speaker": memo.get("speaker") or memo.get("role"),
                    "text": _normalize_text(text)[:500],
                    "ts": memo.get("ts") or memo.get("t"),
                }
            )

        payload = {
            "last_message": user_msg,
            "recent_messages": recent_payload,
            "topics": topics,
            "lessons": long_summary,
            "user_style": self._user_style(recent, persona_overrides),
        }

        try:
            response = _llm_manager().call_dict(
                "conversation_context",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            logger.debug("LLM conversation context unavailable", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        return dict(response)

    # ---------------------------------------------------------------- Build ---
    def build(self, user_msg: str) -> Dict[str, Any]:
        user_id = self._infer_user_id()
        persona_overrides = self._persona_settings(user_id)
        recent = self._recent_msgs(8)
        long_summary: List[Any]
        try:
            long_summary = (self.arch.consolidator.state.get("lessons", []) or [])[-5:]
        except Exception:
            long_summary = []
        topics = self._topics(recent)
        ctx = {
            "last_message": user_msg,
            "active_thread": recent,
            "summary": long_summary,              # bullets
            "key_moments": self._key_moments(),   # avec dates
            "topics": topics,
            "user_style": self._user_style(recent, persona_overrides),
            "related_inbox": self._related_inbox(user_msg),
            "sequence_motifs": self._sequence_motifs(recent),
            "persona_overrides": persona_overrides,
        }
        self._update_feedback_loop(ctx)
        ctx["monitoring"] = self._monitoring_snapshot()

        llm_bundle = self._llm_enrichment(user_msg, recent, long_summary, topics, persona_overrides)
        if llm_bundle:
            ctx.setdefault("llm_summary", llm_bundle)
            summary_text = llm_bundle.get("summary")
            if isinstance(summary_text, str) and summary_text:
                ctx.setdefault("llm_summary_text", summary_text)
            llm_topics = llm_bundle.get("topics")
            if isinstance(llm_topics, list) and llm_topics:
                ctx.setdefault("llm_topics", llm_topics)
            tone = llm_bundle.get("tone")
            if isinstance(tone, str) and tone:
                ctx.setdefault("tone", tone)
            alerts = llm_bundle.get("alerts")
            if isinstance(alerts, list) and alerts:
                ctx.setdefault("alerts", alerts)

        return ctx
