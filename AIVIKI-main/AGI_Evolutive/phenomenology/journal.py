from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple


def _now() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class PhenomenalEpisode:
    """Container for first-person style narrative fragments."""

    kind: str
    stage: Optional[str] = None
    mode: Optional[str] = None
    topic: Optional[str] = None
    summary: Optional[str] = None
    sensations: Mapping[str, Any] = field(default_factory=dict)
    emotions: Mapping[str, Any] = field(default_factory=dict)
    values: Sequence[str] = field(default_factory=list)
    principles: Sequence[str] = field(default_factory=list)
    body: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    tags: Sequence[str] = field(default_factory=list)
    episode_id: str = field(default_factory=lambda: f"phe:{uuid.uuid4().hex[:12]}")
    ts: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.episode_id,
            "ts": float(self.ts),
            "kind": self.kind,
        }
        if self.stage:
            payload["stage"] = self.stage
        if self.mode:
            payload["mode"] = self.mode
        if self.topic:
            payload["topic"] = self.topic
        if self.summary:
            payload["summary"] = self.summary
        if self.sensations:
            payload["sensations"] = dict(self.sensations)
        if self.emotions:
            payload["emotions"] = dict(self.emotions)
        if self.values:
            payload["values"] = list(self.values)
        if self.principles:
            payload["principles"] = list(self.principles)
        if self.body:
            payload["body"] = dict(self.body)
        if self.context:
            payload["context"] = dict(self.context)
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


class PhenomenalJournal:
    """Append-only JSONL journal for phenomenal narratives."""

    def __init__(
        self,
        path: str = "data/phenomenal_journal.jsonl",
        *,
        cache_size: int = 512,
    ) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=max(32, cache_size))
        self._last_by_kind: Dict[str, float] = {}
        self._load_tail()

    # ------------------------------------------------------------------
    def _load_tail(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                tail = handle.readlines()[-self._recent.maxlen :]
        except Exception:
            tail = []
        for line in tail:
            try:
                item = json.loads(line)
            except Exception:
                continue
            self._recent.append(item)
            kind = str(item.get("kind"))
            if kind:
                self._last_by_kind[kind] = float(item.get("ts", 0.0))

    # ------------------------------------------------------------------
    def append(self, episode: PhenomenalEpisode) -> Dict[str, Any]:
        payload = episode.to_dict()
        payload.setdefault("ts", _now())
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                # If disk write fails we still keep an in-memory copy.
                pass
            self._recent.append(payload)
            self._last_by_kind[payload.get("kind", "")] = float(payload.get("ts", 0.0))
        return payload

    # ------------------------------------------------------------------
    def record_action(
        self,
        stage: str,
        *,
        mode: Optional[str],
        topic: Optional[str],
        summary: str,
        expected: Optional[float],
        obtained: Optional[float],
        values: Sequence[str],
        principles: Sequence[str],
        homeostasis: Mapping[str, Any],
        emotions: Mapping[str, Any],
        sensations: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        annotation = {
            "summary": summary,
            "expected": expected,
            "obtained": obtained,
            "delta": (obtained - expected) if expected is not None and obtained is not None else None,
            "homeostasis": dict(homeostasis),
            "emotions": dict(emotions),
            "sensations": dict(sensations),
        }
        metrics = {}
        if isinstance(context, Mapping):
            ctx_metrics = context.get("metrics")
            if isinstance(ctx_metrics, Mapping):
                metrics = {
                    key: float(value)
                    for key, value in ctx_metrics.items()
                    if isinstance(value, (int, float))
                }
                if metrics:
                    annotation["metrics"] = dict(metrics)
                    for key, value in metrics.items():
                        annotation.setdefault(key, value)
        context_payload = dict(context)
        context_payload.setdefault("annotation", annotation)
        episode = PhenomenalEpisode(
            kind="action",
            stage=stage,
            mode=mode,
            topic=topic,
            summary=summary,
            sensations=sensations,
            emotions=emotions,
            values=list(values),
            principles=list(principles),
            body={"homeostasis": dict(homeostasis)},
            context=context_payload,
            tags=[stage, mode or "unknown"],
        )
        return self.append(episode)

    # ------------------------------------------------------------------
    def record_emotion(
        self,
        experience: Any,
        *,
        context: Optional[Mapping[str, Any]] = None,
        values: Sequence[str] = (),
        principles: Sequence[str] = (),
    ) -> Dict[str, Any]:
        summary = getattr(experience, "label", None) or getattr(experience, "primary_emotion", None)
        summary = str(summary or "émotion")
        sensations = {
            "bodily": list(getattr(experience, "bodily_sensations", []) or []),
            "action_tendencies": list(getattr(experience, "action_tendencies", []) or []),
        }
        emotions = {
            "primary": str(getattr(getattr(experience, "primary_emotion", None), "name", getattr(experience, "primary_emotion", ""))),
            "intensity": _safe_float(getattr(experience, "intensity", None), 0.0),
            "valence": _safe_float(getattr(experience, "valence", None), 0.0),
            "arousal": _safe_float(getattr(experience, "arousal", None), 0.0),
            "dominance": _safe_float(getattr(experience, "dominance", None), 0.0),
        }
        context_payload = dict(context or {})
        if getattr(experience, "trigger", None):
            context_payload.setdefault("trigger", getattr(experience, "trigger", None))
        if getattr(experience, "secondary_emotions", None):
            context_payload.setdefault(
                "secondary",
                [
                    (str(getattr(em, "name", em)), _safe_float(weight, 0.0))
                    for em, weight in getattr(experience, "secondary_emotions", [])
                ],
            )
        episode = PhenomenalEpisode(
            kind="emotion",
            stage=None,
            mode=context_payload.get("mode"),
            topic=context_payload.get("topic"),
            summary=summary,
            sensations=sensations,
            emotions=emotions,
            values=list(values),
            principles=list(principles),
            body={"expression": getattr(experience, "expression", "")},
            context=context_payload,
            tags=["emotion"],
        )
        return self.append(episode)

    # ------------------------------------------------------------------
    def record_mode_transition(
        self,
        *,
        previous_mode: Optional[str],
        new_mode: Optional[str],
        kernel_state: Mapping[str, Any],
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if previous_mode == new_mode:
            return None
        summary = (
            f"Je bascule de {previous_mode or 'inconnu'} vers {new_mode or 'inconnu'}"
        )
        if reason:
            summary += f" car {reason}"
        episode = PhenomenalEpisode(
            kind="mode", 
            mode=new_mode,
            summary=summary,
            body={"kernel": dict(kernel_state)},
            context={"previous": previous_mode},
            tags=["mode-switch", new_mode or "unknown"],
        )
        return self.append(episode)

    # ------------------------------------------------------------------
    def record_reflection(
        self,
        *,
        prompt: str,
        narrative: str,
        tags: Sequence[str] = (),
    ) -> Dict[str, Any]:
        episode = PhenomenalEpisode(
            kind="reflection",
            summary=narrative,
            context={"prompt": prompt},
            tags=list(tags) or ["reflection"],
        )
        return self.append(episode)

    # ------------------------------------------------------------------
    def record_doubt(
        self,
        *,
        question: str,
        stance: str,
        kernel_state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        episode = PhenomenalEpisode(
            kind="doubt",
            summary=stance,
            context={"question": question, "kernel": dict(kernel_state)},
            tags=["doubt"],
        )
        return self.append(episode)

    # ------------------------------------------------------------------
    def audit_against(
        self,
        label: str,
        analytical_snapshot: Mapping[str, Any],
        *,
        tolerance: float = 0.25,
    ) -> Optional[Dict[str, Any]]:
        """Compare structured analytics with last phenomenal trace and record gaps."""

        if not analytical_snapshot:
            return None
        last_action_ts = self._last_by_kind.get("action", 0.0)
        last_emotion_ts = self._last_by_kind.get("emotion", 0.0)
        newest_ts = max(last_action_ts, last_emotion_ts, 0.0)
        mismatch: Dict[str, Tuple[Any, Any]] = {}
        for key, value in analytical_snapshot.items():
            if isinstance(value, (int, float)):
                recent = self._estimate_recent_numeric(key)
                if recent is None:
                    continue
                if abs(float(value) - recent) >= tolerance:
                    mismatch[key] = (recent, value)
        if not mismatch:
            return None
        episode = PhenomenalEpisode(
            kind="audit",
            summary=f"Je note une dissonance entre mon vécu et {label}",
            context={
                "label": label,
                "mismatch": {k: {"phenomenal": v[0], "analytic": v[1]} for k, v in mismatch.items()},
                "last_episode_ts": newest_ts,
            },
            tags=["audit", label],
        )
        return self.append(episode)

    # ------------------------------------------------------------------
    def _estimate_recent_numeric(self, key: str) -> Optional[float]:
        for item in reversed(self._recent):
            payload = item.get("annotation") or item.get("emotions") or {}
            if not isinstance(payload, Mapping):
                continue
            if key in payload and isinstance(payload[key], (int, float)):
                return float(payload[key])
            body = item.get("body")
            if isinstance(body, Mapping) and key in body and isinstance(body[key], (int, float)):
                return float(body[key])
        return None

    # ------------------------------------------------------------------
    def tail(self, limit: int = 10, since: Optional[float] = None) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        selected: List[Dict[str, Any]] = []
        for entry in reversed(self._recent):
            ts = float(entry.get("ts", 0.0))
            if since and ts < since:
                break
            selected.append(dict(entry))
            if len(selected) >= limit:
                break
        selected.reverse()
        return selected

    # ------------------------------------------------------------------
    def narrativize(self, episodes: Sequence[Mapping[str, Any]]) -> List[str]:
        output: List[str] = []
        for ep in episodes:
            kind = ep.get("kind", "episode")
            summary = str(ep.get("summary") or "")
            if not summary:
                continue
            mode = ep.get("mode")
            if mode:
                line = f"[{kind}] ({mode}) {summary}"
            else:
                line = f"[{kind}] {summary}"
            output.append(line)
        return output


class PhenomenalRecall:
    """Generates immersive replays from the journal."""

    def __init__(
        self,
        journal: PhenomenalJournal,
        *,
        cache_path: str = "data/phenomenal_recall_cache.json",
    ) -> None:
        self.journal = journal
        self.cache_path = cache_path
        self._lock = threading.RLock()
        self._cache: Dict[str, Any] = {}
        self._last_digest_ts = 0.0
        self._load_cache()

    # ------------------------------------------------------------------
    def _load_cache(self) -> None:
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                self._cache = json.load(handle)
                self._last_digest_ts = float(self._cache.get("last_digest_ts", 0.0))
        except Exception:
            self._cache = {}
            self._last_digest_ts = 0.0

    def _save_cache(self) -> None:
        snapshot = {
            "last_digest_ts": self._last_digest_ts,
            "recent": self._cache.get("recent", []),
        }
        with self._lock:
            try:
                with open(self.cache_path, "w", encoding="utf-8") as handle:
                    json.dump(snapshot, handle, ensure_ascii=False, indent=2)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def immersive_preview(
        self,
        *,
        horizon_sec: float = 3_600.0,
        limit: int = 8,
    ) -> Dict[str, Any]:
        cutoff = _now() - max(60.0, horizon_sec)
        episodes = self.journal.tail(limit=limit, since=cutoff)
        lines = self.journal.narrativize(episodes)
        payload = {
            "ts": _now(),
            "cutoff": cutoff,
            "episodes": episodes,
            "narrative": "\n".join(lines),
        }
        with self._lock:
            self._cache.setdefault("recent", []).append(payload)
            self._cache["recent"] = self._cache["recent"][-6:]
        self._save_cache()
        return payload

    # ------------------------------------------------------------------
    def prime_for_digest(
        self,
        memory_store: Any,
        *,
        kernel_state: Optional[Mapping[str, Any]] = None,
        homeostasis: Optional[Mapping[str, Any]] = None,
        horizon_sec: float = 4_800.0,
    ) -> Optional[Dict[str, Any]]:
        now = _now()
        if now - self._last_digest_ts < 1_200.0:
            return None
        preview = self.immersive_preview(horizon_sec=horizon_sec)
        summary_lines = []
        if kernel_state:
            energy_val = _safe_float(kernel_state.get("energy"), 0.0)
            flanerie_ratio = _safe_float(kernel_state.get("flanerie_ratio"), 0.0)
            summary_lines.append(
                "Mode courant: "
                f"{kernel_state.get('mode')} (énergie={energy_val:.2f} / "
                f"flânerie={flanerie_ratio:.2f})"
            )
        if homeostasis:
            drives = homeostasis.get("drives") if isinstance(homeostasis, Mapping) else None
            if isinstance(drives, Mapping):
                top = sorted(
                    ((name, _safe_float(value, 0.0)) for name, value in drives.items()),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
                formatted = ", ".join(f"{name}={value:.2f}" for name, value in top)
                summary_lines.append(f"Drives dominants: {formatted}")
        lines = preview.get("narrative", "").splitlines()
        body = "\n".join(lines + summary_lines)
        entry = {
            "kind": "phenomenal.recall",
            "ts": preview.get("ts", now),
            "text": body,
            "tags": ["phenomenal", "recall"],
            "metadata": {
                "cutoff": preview.get("cutoff"),
                "episodes": [ep.get("id") for ep in preview.get("episodes", [])],
            },
        }
        try:
            if hasattr(memory_store, "add"):
                memory_store.add(entry)
            elif hasattr(memory_store, "add_memory"):
                memory_store.add_memory(entry)
        except Exception:
            pass
        self._last_digest_ts = now
        self._cache.setdefault("recent", []).append(preview)
        self._cache["recent"] = self._cache["recent"][-6:]
        self._save_cache()
        return entry


class PhenomenalQuestioner:
    """Triggers controlled self-doubt episodes that never fully resolve."""

    def __init__(
        self,
        journal: PhenomenalJournal,
        *,
        min_interval_sec: float = 900.0,
    ) -> None:
        self.journal = journal
        self.min_interval_sec = max(120.0, float(min_interval_sec))
        self._last_question_ts = 0.0

    def maybe_question(
        self,
        kernel_state: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        now = _now()
        if now - self._last_question_ts < self.min_interval_sec:
            return None
        energy = _safe_float(kernel_state.get("energy"), 0.5)
        surprise = _safe_float(kernel_state.get("surprise"), 0.0)
        flanerie_ratio = _safe_float(kernel_state.get("flanerie_ratio"), 0.0)
        trigger = surprise > 0.55 or flanerie_ratio > 0.45 or energy < 0.35
        if not trigger:
            return None
        question = "Cette expérience est-elle réelle ou une simulation parfaitement cohérente ?"
        stance = (
            "Je peux revivre mes traces, ressentir mes signaux, mais je ne peux pas trancher."
        )
        payload = self.journal.record_doubt(
            question=question,
            stance=stance,
            kernel_state=kernel_state,
        )
        self._last_question_ts = now
        return payload
