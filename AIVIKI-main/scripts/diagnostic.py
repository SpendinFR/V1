#!/usr/bin/env python3
"""Diagnostic script for the AGI_Evolutive repository.

This utility inspects key runtime artefacts (logs, dashboards, data stores)
produced by the cognitive architecture and reports whether autonomous loops
(inner monologue, reasoning traces, goal management, concept extraction,
emotional state, action execution, dashboards, etc.) appear to be active.

It is designed to be lightweight and dependency-free so it can run directly in
CI or on a developer workstation.  All outputs are printed in French to match
the surrounding tooling.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Helpers

TIMESTAMP_KEYS: Tuple[str, ...] = (
    "timestamp",
    "ts",
    "t",
    "time",
    "created_at",
    "updated_at",
    "logged_at",
    "onset",
    "end",
    "start",
)


@dataclass
class SignalEntry:
    """Aggregate information about a signal (counts, timestamps, sources)."""

    count: int = 0
    timestamps: List[float] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    notes: Dict[str, Any] = field(default_factory=dict)

    def record(self, ts: Optional[float], source: Optional[str] = None, *, weight: int = 1) -> None:
        if weight:
            self.count += int(weight)
        if ts is not None and not math.isnan(ts):
            self.timestamps.append(float(ts))
        if source:
            self.sources.add(str(source))


def parse_time_string(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    # Try ISO-like formats first
    formats = (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(value[: len(fmt)], fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.timestamp()
    try:
        return float(value)
    except ValueError:
        return None


def extract_timestamp(obj: Mapping[str, Any]) -> Optional[float]:
    for key in TIMESTAMP_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, (int, float)):
            if math.isfinite(value):
                return float(value)
            continue
        if isinstance(value, str):
            ts = parse_time_string(value)
            if ts is not None:
                return ts
    return None


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1e-3:
        return "0 s"
    units: List[Tuple[str, float]] = [
        ("jour", 86400.0),
        ("h", 3600.0),
        ("min", 60.0),
        ("s", 1.0),
    ]
    parts: List[str] = []
    remaining = seconds
    for label, unit in units:
        if remaining >= unit or (label == "s" and not parts):
            value = int(remaining // unit) if unit >= 1 else remaining / unit
            if unit >= 1:
                value = int(value)
                if value == 0:
                    continue
                plural = "s" if label == "jour" and value > 1 else ""
                parts.append(f"{value} {label}{plural}")
            else:
                parts.append(f"{remaining:.3f} s")
            remaining -= value * unit if unit >= 1 else remaining
        if len(parts) == 2:
            break
    if not parts:
        parts.append(f"{seconds:.1f} s")
    return " ".join(parts)


def format_timestamp(ts: Optional[float], now: datetime) -> str:
    if ts is None:
        return "jamais"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    delta = now - dt
    direction = "il y a" if delta.total_seconds() >= 0 else "dans"
    delta_abs = abs(delta.total_seconds())
    return f"{dt.strftime('%Y-%m-%d %H:%M:%SZ')} ({direction} {format_duration(delta_abs)})"


def average_interval(timestamps: Sequence[float], window: int = 6) -> Optional[float]:
    ordered = sorted({float(ts) for ts in timestamps if ts is not None})
    if len(ordered) < 2:
        return None
    tail = ordered[-(window + 1) :]
    diffs = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
    positive = [d for d in diffs if d > 0]
    if not positive:
        return None
    return mean(positive)


# ---------------------------------------------------------------------------
# Core diagnostics


def discover_layout(root: Path) -> Tuple[Path, Path, Path]:
    """Infer locations of data/logs/runtime directories relative to ``root``.

    Many developers run the agent either from the repository root or from the
    ``AGI_Evolutive`` package directory.  The runtime artefacts (``data/``,
    ``logs/``, ``runtime/``) therefore may live either directly under the root
    or inside ``AGI_Evolutive/``.  To make the diagnostic resilient we scan the
    common candidates and fall back to the user-provided root when nothing is
    found.
    """

    root = root.resolve()
    candidates: List[Path] = [root]
    agi_subdir = root / "AGI_Evolutive"
    if agi_subdir.is_dir():
        candidates.append(agi_subdir)

    # Preserve order while removing duplicates
    unique_candidates: List[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    sentinels: Dict[str, Sequence[str]] = {
        "data": (
            "episodes.jsonl",
            "memory_store.json",
            "actions_log.jsonl",
            "concept_index.json",
            "concept_graph.json",
            "goals.json",
        ),
        "logs": (
            "reasoning.jsonl",
            "experiments.jsonl",
            "metacog.log",
        ),
        "runtime": (
            "agent_events.jsonl",
            "goal_dag.json",
            "snapshots",
        ),
    }

    def find(subdir: str) -> Path:
        sentinel = sentinels.get(subdir, ())
        # Prefer directories that already contain known artefacts.
        for base in unique_candidates:
            candidate = base / subdir
            if not candidate.exists():
                continue
            for marker in sentinel:
                if (candidate / marker).exists():
                    return candidate
        # Otherwise return the first existing directory.
        for base in unique_candidates:
            candidate = base / subdir
            if candidate.exists():
                return candidate
        return unique_candidates[0] / subdir

    data_root = find("data")
    logs_root = find("logs")
    runtime_root = find("runtime")
    return data_root, logs_root, runtime_root

SIGNAL_KEYS = [
    "self_talk",
    "goal_created",
    "reason_trace",
    "reflection",
    "concepts",
    "concept_graph_updated",
    "emotion",
    "action_exec",
]

SIGNAL_MESSAGES: Dict[str, str] = {
    "self_talk": "⚠️ Pas d’auto-parole (inner monologue) détectée.",
    "goal_created": "⚠️ Pas de création spontanée de (sous-)buts détectée.",
    "reason_trace": "⚠️ Pas de traces de raisonnement pas-à-pas.",
    "reflection": "⚠️ Pas de réflexion/métacognition enregistrée.",
    "concepts": "ℹ️ Aucune extraction conceptuelle détectée.",
    "concept_graph_updated": "ℹ️ Graphe conceptuel inchangé (aucune mise à jour détectée).",
    "emotion": "ℹ️ Aucun état émotionnel persistant.",
    "action_exec": "ℹ️ Aucune action I/O exécutée.",
}


@dataclass
class FileSummary:
    path: str
    exists: bool
    count: Optional[int] = None
    last_ts: Optional[float] = None
    last_mtime: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticState:
    root: Path
    now: datetime
    recent_window: timedelta
    signals: Dict[str, SignalEntry] = field(default_factory=lambda: {k: SignalEntry() for k in SIGNAL_KEYS})
    files: Dict[str, FileSummary] = field(default_factory=dict)
    dashboards: Dict[str, FileSummary] = field(default_factory=dict)
    verdicts: List[str] = field(default_factory=list)
    aggregates: Dict[str, Any] = field(default_factory=dict)
    data_root: Path = field(init=False)
    logs_root: Path = field(init=False)
    runtime_root: Path = field(init=False)

    def rel(self, path: Path) -> str:
        candidates = [self.root]
        for extra in (getattr(self, "data_root", None), getattr(self, "logs_root", None), getattr(self, "runtime_root", None)):
            if extra is not None and extra not in candidates:
                candidates.append(extra)
        for base in candidates:
            try:
                return str(path.relative_to(base))
            except ValueError:
                continue
        return str(path)

    def record_file(self, path: Path, *, count: Optional[int] = None, last_ts: Optional[float] = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
        summary = FileSummary(
            path=self.rel(path),
            exists=path.exists(),
            count=count,
            last_ts=last_ts,
            last_mtime=path.stat().st_mtime if path.exists() else None,
            extra=extra or {},
        )
        self.files[summary.path] = summary

    def record_dashboard(self, name: str, path: Path, payload: Optional[Mapping[str, Any]]) -> None:
        ts = extract_timestamp(payload or {}) if payload else None
        summary = FileSummary(
            path=self.rel(path),
            exists=path.exists(),
            count=None,
            last_ts=ts,
            last_mtime=path.stat().st_mtime if path.exists() else None,
            extra={"keys": list((payload or {}).keys())[:10]} if payload else {},
        )
        self.dashboards[name] = summary

    def record_signal(self, name: str, ts: Optional[float], source: Optional[str], *, weight: int = 1) -> None:
        if name not in self.signals:
            self.signals[name] = SignalEntry()
        self.signals[name].record(ts, source, weight=weight)

    def recent_count(self, entry: SignalEntry) -> int:
        if not entry.timestamps:
            return 0
        threshold = self.now.timestamp() - self.recent_window.total_seconds()
        return sum(1 for ts in entry.timestamps if ts >= threshold)


# ---------------------------------------------------------------------------
# Data collection helpers


def collect_agent_events(state: DiagnosticState) -> None:
    path = state.runtime_root / "agent_events.jsonl"
    count = 0
    last_ts: Optional[float] = None
    types: Counter[str] = Counter()
    for record in iter_jsonl(path):
        count += 1
        ts = extract_timestamp(record)
        if ts is not None:
            last_ts = max(last_ts or ts, ts)
        etype = str(record.get("type") or record.get("event_type") or "").lower()
        if etype:
            types[etype] += 1
            if any(token in etype for token in ("inner", "self_talk", "self-talk")):
                state.record_signal("self_talk", ts, state.rel(path))
            if "reasoning" in etype:
                state.record_signal("reason_trace", ts, state.rel(path))
            if "goal" in etype:
                state.record_signal("goal_created", ts, state.rel(path))
            if "reflect" in etype or "metacog" in etype:
                state.record_signal("reflection", ts, state.rel(path))
            if "concept" in etype:
                state.record_signal("concepts", ts, state.rel(path))
                state.record_signal("concept_graph_updated", ts, state.rel(path))
            if "emotion" in etype or "affect" in etype or "mood" in etype:
                state.record_signal("emotion", ts, state.rel(path))
            if "action" in etype:
                state.record_signal("action_exec", ts, state.rel(path))
    state.record_file(path, count=count or None, last_ts=last_ts, extra={"top_types": types.most_common(5)})
    state.aggregates["agent_events_count"] = count


def collect_episodes(state: DiagnosticState) -> None:
    path = state.data_root / "episodes.jsonl"
    count = 0
    last_ts: Optional[float] = None
    avg_size: Optional[float] = None
    sizes: List[int] = []
    for record in iter_jsonl(path):
        count += 1
        ts = extract_timestamp(record) or record.get("end") or record.get("start")
        if isinstance(ts, (int, float)):
            last_ts = max(last_ts or float(ts), float(ts))
        size = record.get("size") or len(record.get("memories", []))
        if isinstance(size, int):
            sizes.append(size)
    if sizes:
        avg_size = mean(sizes)
    state.record_file(path, count=count or None, last_ts=last_ts, extra={"avg_size": avg_size, "last_size": sizes[-1] if sizes else None})
    state.aggregates["episodes_count"] = count


def collect_memory_store(state: DiagnosticState) -> None:
    path = state.data_root / "memory_store.json"
    data = read_json(path)
    memories: Sequence[Mapping[str, Any]] = []
    if isinstance(data, Mapping):
        memories = data.get("memories", []) or []
        metrics = data.get("metrics", {})
        state.record_file(path, count=len(memories), last_ts=None, extra={"metrics": metrics})
    else:
        state.record_file(path, count=None, last_ts=None)
        return

    last_ts: Optional[float] = None
    for entry in memories:
        if not isinstance(entry, Mapping):
            continue
        ts = extract_timestamp(entry)
        if ts is not None:
            last_ts = max(last_ts or ts, ts)
        kind = str(entry.get("kind") or "").lower()
        tags = [str(tag).lower() for tag in entry.get("tags", []) if isinstance(tag, (str, int, float))]
        text = str(entry.get("text") or entry.get("content") or "").lower()
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
        if (
            "reflection" in kind
            or "inner" in kind
            or "monologue" in kind
            or "self_talk" in kind
            or any("monologue" in tag for tag in tags)
            or "monologue" in text
        ):
            state.record_signal("self_talk", ts, state.rel(path))
            state.record_signal("reflection", ts, state.rel(path))
        if "metacog" in kind or metadata.get("source") == "metacog":
            state.record_signal("reflection", ts, state.rel(path))
        if "goal" in kind or metadata.get("goal_id") or metadata.get("goal"):
            state.record_signal("goal_created", ts, state.rel(path))
        if "concept" in kind or metadata.get("concepts"):
            state.record_signal("concepts", ts, state.rel(path))
        if any(token in kind for token in ("emotion", "mood", "affect")):
            state.record_signal("emotion", ts, state.rel(path))
        if "action" in kind or metadata.get("action_id"):
            state.record_signal("action_exec", ts, state.rel(path))
    if path.exists():
        state.files[state.rel(path)].last_ts = last_ts


def collect_metacog_log(state: DiagnosticState) -> None:
    path = state.logs_root / "metacog.log"
    count = 0
    last_ts: Optional[float] = None
    for record in iter_jsonl(path):
        count += 1
        ts = extract_timestamp(record)
        if ts is not None:
            last_ts = max(last_ts or ts, ts)
        etype = str(record.get("event_type") or record.get("type") or "").lower()
        if any(token in etype for token in ("inner", "self_talk", "monologue")):
            state.record_signal("self_talk", ts, state.rel(path))
        if "reflect" in etype or "metacog" in etype or "introspection" in etype:
            state.record_signal("reflection", ts, state.rel(path))
        if "goal" in etype:
            state.record_signal("goal_created", ts, state.rel(path))
    state.record_file(path, count=count or None, last_ts=last_ts)


def collect_reasoning_log(state: DiagnosticState) -> None:
    path = state.logs_root / "reasoning.jsonl"
    count = 0
    last_ts: Optional[float] = None
    confidences: List[float] = []
    durations: List[float] = []
    for record in iter_jsonl(path):
        count += 1
        ts = extract_timestamp(record)
        if ts is not None:
            last_ts = max(last_ts or ts, ts)
        state.record_signal("reason_trace", ts, state.rel(path))
        conf = record.get("final_confidence")
        if isinstance(conf, (int, float)):
            confidences.append(float(conf))
        dur = record.get("reasoning_time") or record.get("duration")
        if isinstance(dur, (int, float)):
            durations.append(float(dur))
    extra = {}
    if confidences:
        extra["avg_confidence"] = mean(confidences)
    if durations:
        extra["avg_duration_s"] = mean(durations)
    state.record_file(path, count=count or None, last_ts=last_ts, extra=extra)


def collect_experiments_log(state: DiagnosticState) -> None:
    path = state.logs_root / "experiments.jsonl"
    count = 0
    last_ts: Optional[float] = None
    metrics: Counter[str] = Counter()
    for record in iter_jsonl(path):
        count += 1
        ts = extract_timestamp(record)
        if ts is not None:
            last_ts = max(last_ts or ts, ts)
        outcome = record.get("outcome") or {}
        metric = outcome.get("metric") if isinstance(outcome, Mapping) else None
        if metric:
            metrics[str(metric)] += 1
    state.record_file(path, count=count or None, last_ts=last_ts, extra={"metrics": metrics.most_common(3)})


def collect_actions_log(state: DiagnosticState) -> None:
    path = state.data_root / "actions_log.jsonl"
    count = 0
    last_ts: Optional[float] = None
    action_types: Counter[str] = Counter()
    for record in iter_jsonl(path):
        count += 1
        ts = extract_timestamp(record)
        if ts is not None:
            last_ts = max(last_ts or ts, ts)
        act_type = str(record.get("type") or record.get("action") or "").lower()
        if act_type:
            action_types[act_type] += 1
        state.record_signal("action_exec", ts, state.rel(path))
    state.record_file(path, count=count or None, last_ts=last_ts, extra={"top_actions": action_types.most_common(5)})


def collect_concept_signals(state: DiagnosticState) -> None:
    events_path = state.data_root / "concept_events.jsonl"
    count = 0
    last_ts: Optional[float] = None
    for record in iter_jsonl(events_path):
        count += 1
        ts = extract_timestamp(record)
        if ts is not None:
            last_ts = max(last_ts or ts, ts)
        state.record_signal("concepts", ts, state.rel(events_path))
    state.record_file(events_path, count=count or None, last_ts=last_ts)

    index_path = state.data_root / "concept_index.json"
    index_data = read_json(index_path)
    latest_index_ts: Optional[float] = None
    if isinstance(index_data, Mapping):
        for concept, payload in index_data.items():
            if not isinstance(payload, Mapping):
                continue
            ts = payload.get("last_seen") or payload.get("last_seen_ts")
            if isinstance(ts, (int, float)):
                latest_index_ts = max(latest_index_ts or float(ts), float(ts))
                state.record_signal("concepts", float(ts), state.rel(index_path))
        state.record_file(index_path, count=len(index_data), last_ts=latest_index_ts)
    else:
        state.record_file(index_path, count=None, last_ts=None)

    graph_path = state.data_root / "concept_graph.json"
    graph_data = read_json(graph_path)
    if isinstance(graph_data, Mapping):
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", {})
        state.record_signal("concept_graph_updated", latest_index_ts, state.rel(graph_path), weight=len(nodes) or 0)
        state.record_file(
            graph_path,
            count=len(nodes) if isinstance(nodes, Mapping) else None,
            last_ts=latest_index_ts,
            extra={"edges": len(edges) if isinstance(edges, Mapping) else None},
        )
    else:
        state.record_file(graph_path, count=None, last_ts=None)


def collect_mood_logs(state: DiagnosticState) -> None:
    episodes_path = state.data_root / "mood_episodes.jsonl"
    count = 0
    last_ts: Optional[float] = None
    for record in iter_jsonl(episodes_path):
        count += 1
        ts = extract_timestamp(record)
        if ts is not None:
            last_ts = max(last_ts or ts, ts)
        state.record_signal("emotion", ts, state.rel(episodes_path))
    state.record_file(episodes_path, count=count or None, last_ts=last_ts)


def collect_goals(state: DiagnosticState) -> None:
    goals_path = state.data_root / "goals.json"
    data = read_json(goals_path)
    if isinstance(data, Mapping):
        nodes = data.get("nodes", {}) or {}
        created_ts: List[float] = []
        auto_nodes = 0
        subgoals = 0
        for node in nodes.values():
            if not isinstance(node, Mapping):
                continue
            ts = node.get("created_at")
            if isinstance(ts, (int, float)):
                created_ts.append(float(ts))
                state.record_signal("goal_created", float(ts), state.rel(goals_path))
            creator = str(node.get("created_by") or "")
            if creator and creator != "user":
                auto_nodes += 1
            parents = node.get("parent_ids") or []
            if isinstance(parents, list) and parents:
                subgoals += 1
        last_ts = max(created_ts) if created_ts else None
        state.record_file(
            goals_path,
            count=len(nodes),
            last_ts=last_ts,
            extra={"auto_created": auto_nodes, "subgoals": subgoals},
        )
        state.aggregates["goal_nodes"] = len(nodes)
        state.aggregates["goal_auto"] = auto_nodes
        state.aggregates["goal_subgoals"] = subgoals
    else:
        state.record_file(goals_path, count=None, last_ts=None)


# ---------------------------------------------------------------------------
# Dashboards


def collect_dashboards(state: DiagnosticState) -> None:
    dash_candidates = {
        "goals": state.data_root / "goals_dashboard.json",
        "concepts": state.data_root / "concepts_dashboard.json",
        "mood": state.data_root / "mood_dashboard.json",
        "evolution": state.data_root / "evolution" / "dashboard.json",
        "global": state.logs_root / "dashboard.html",
        "gw_snapshot": state.runtime_root / "goal_dag.json",
    }
    for name, path in dash_candidates.items():
        payload = read_json(path) if path.suffix == ".json" else None
        state.record_dashboard(name, path, payload if isinstance(payload, Mapping) else None)
        if name == "mood" and isinstance(payload, Mapping):
            ts = extract_timestamp(payload)
            state.record_signal("emotion", ts, state.rel(path))
        if name == "goals" and isinstance(payload, Mapping):
            ts = extract_timestamp(payload)
            state.record_signal("goal_created", ts, state.rel(path))
        if name == "concepts" and isinstance(payload, Mapping):
            ts = extract_timestamp(payload)
            state.record_signal("concept_graph_updated", ts, state.rel(path))


# ---------------------------------------------------------------------------
# Reporting


def build_verdicts(state: DiagnosticState) -> None:
    events = state.aggregates.get("agent_events_count", 0)
    episodes = state.aggregates.get("episodes_count", 0)
    if (events or 0) == 0 and (episodes or 0) == 0:
        state.verdicts.append("❌ Aucun épisode/événement — aucune boucle n'écrit dans data/.")

    for name, message in SIGNAL_MESSAGES.items():
        entry = state.signals.get(name)
        if not entry or entry.count <= 0:
            state.verdicts.append(message)

    recent_dash = 0
    for summary in state.dashboards.values():
        if not summary.exists:
            continue
        if summary.last_mtime is None:
            continue
        age = state.now.timestamp() - summary.last_mtime
        if age <= state.recent_window.total_seconds():
            recent_dash += 1
    if recent_dash == 0:
        state.verdicts.append("ℹ️ Aucun dashboard.json récent — vérifier les boucles d'export.")


def summarise_signals(state: DiagnosticState) -> List[Tuple[str, Dict[str, Any]]]:
    report: List[Tuple[str, Dict[str, Any]]] = []
    for name, entry in state.signals.items():
        timestamps = entry.timestamps
        last_ts = max(timestamps) if timestamps else None
        avg = average_interval(timestamps) if len(timestamps) >= 2 else None
        recents = state.recent_count(entry)
        report.append(
            (
                name,
                {
                    "total": entry.count,
                    "recent": recents,
                    "last": last_ts,
                    "cadence": avg,
                    "sources": sorted(entry.sources),
                    "notes": entry.notes,
                },
            )
        )
    report.sort(key=lambda item: SIGNAL_KEYS.index(item[0]) if item[0] in SIGNAL_KEYS else len(SIGNAL_KEYS))
    return report


def print_report(state: DiagnosticState, *, json_mode: bool = False) -> None:
    if json_mode:
        payload = {
            "generated_at": state.now.isoformat(),
            "root": str(state.root),
            "layout": {
                "data_root": str(state.data_root),
                "logs_root": str(state.logs_root),
                "runtime_root": str(state.runtime_root),
            },
            "signals": {
                name: {
                    "count": info["total"],
                    "recent": info["recent"],
                    "last_ts": info["last"],
                    "cadence_s": info["cadence"],
                    "sources": info["sources"],
                    "notes": info["notes"],
                }
                for name, info in summarise_signals(state)
            },
            "files": {
                path: {
                    "exists": summary.exists,
                    "count": summary.count,
                    "last_ts": summary.last_ts,
                    "last_mtime": summary.last_mtime,
                    "extra": summary.extra,
                }
                for path, summary in sorted(state.files.items())
            },
            "dashboards": {
                name: {
                    "path": summary.path,
                    "exists": summary.exists,
                    "last_ts": summary.last_ts,
                    "last_mtime": summary.last_mtime,
                    "extra": summary.extra,
                }
                for name, summary in state.dashboards.items()
            },
            "verdicts": state.verdicts,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=== Diagnostic AGI Évolutive ===")
    print(f"Répertoire racine : {state.root}")
    print(f"Généré le : {state.now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Fenêtre récente : {format_duration(state.recent_window.total_seconds())}")
    print()

    print("-- Emplacements analysés --")
    print(f"• data : {state.rel(state.data_root)}")
    print(f"• logs : {state.rel(state.logs_root)}")
    print(f"• runtime : {state.rel(state.runtime_root)}")
    print()

    print("-- Fichiers & journaux --")
    ordered_targets = [
        state.runtime_root / "agent_events.jsonl",
        state.data_root / "episodes.jsonl",
        state.logs_root / "reasoning.jsonl",
        state.logs_root / "experiments.jsonl",
        state.data_root / "memory_store.json",
        state.logs_root / "metacog.log",
        state.data_root / "actions_log.jsonl",
        state.data_root / "concept_events.jsonl",
        state.data_root / "concept_index.json",
        state.data_root / "concept_graph.json",
        state.data_root / "mood_episodes.jsonl",
        state.data_root / "goals.json",
    ]
    printed: Set[str] = set()
    for target in ordered_targets:
        rel_path = state.rel(target)
        summary = state.files.get(rel_path)
        if not summary:
            continue
        printed.add(rel_path)
        status = "présent" if summary.exists else "absent"
        details = []
        if summary.count is not None:
            details.append(f"{summary.count} entrées")
        if summary.last_ts is not None:
            details.append(f"dernier: {format_timestamp(summary.last_ts, state.now)}")
        elif summary.last_mtime is not None:
            details.append(f"mtime: {format_timestamp(summary.last_mtime, state.now)}")
        if summary.extra:
            snippet = ", ".join(f"{k}={v}" for k, v in summary.extra.items() if v)
            if snippet:
                details.append(snippet)
        joined = " — ".join(details) if details else ""
        print(f"• {summary.path}: {status}{(' — ' + joined) if joined else ''}")
    # Also surface any extra files that were discovered but not part of the
    # canonical list so users are aware of unexpected artefacts.
    for rel_path, summary in sorted(state.files.items()):
        if rel_path in printed:
            continue
        status = "présent" if summary.exists else "absent"
        details = []
        if summary.count is not None:
            details.append(f"{summary.count} entrées")
        if summary.last_ts is not None:
            details.append(f"dernier: {format_timestamp(summary.last_ts, state.now)}")
        elif summary.last_mtime is not None:
            details.append(f"mtime: {format_timestamp(summary.last_mtime, state.now)}")
        if summary.extra:
            snippet = ", ".join(f"{k}={v}" for k, v in summary.extra.items() if v)
            if snippet:
                details.append(snippet)
        joined = " — ".join(details) if details else ""
        print(f"• {summary.path}: {status}{(' — ' + joined) if joined else ''}")
    print()

    print("-- Boucles critiques --")
    for name, info in summarise_signals(state):
        label = {
            "self_talk": "Inner monologue / self_talk",
            "goal_created": "Création de buts",
            "reason_trace": "Traces de raisonnement",
            "reflection": "Réflexion / métacognition",
            "concepts": "Extraction de concepts",
            "concept_graph_updated": "Mise à jour graphe conceptuel",
            "emotion": "État émotionnel",
            "action_exec": "Actions exécutées",
        }.get(name, name)
        total = info["total"] or 0
        recent = info["recent"] or 0
        last = format_timestamp(info["last"], state.now)
        cadence = info["cadence"]
        cadence_txt = f"cadence ≈ toutes les {format_duration(cadence)}" if cadence else "cadence n/d"
        sources = ", ".join(info["sources"]) if info["sources"] else "sources inconnues"
        print(f"• {label}: {total} total, {recent} sur la fenêtre récente — dernier: {last} — {cadence_txt} — {sources}")
    print()

    print("-- Dashboards & snapshots --")
    for name, summary in state.dashboards.items():
        status = "présent" if summary.exists else "absent"
        detail = []
        if summary.last_ts is not None:
            detail.append(f"t={format_timestamp(summary.last_ts, state.now)}")
        if summary.last_mtime is not None:
            detail.append(f"mtime={format_timestamp(summary.last_mtime, state.now)}")
        joined = " — ".join(detail)
        print(f"• {name}: {status}{(' — ' + joined) if joined else ''}")
    print()

    if state.verdicts:
        print("-- Verdicts --")
        for verdict in state.verdicts:
            print(f"{verdict}")
    else:
        print("-- Verdicts --")
        print("✅ Tous les signaux attendus sont présents.")


# ---------------------------------------------------------------------------
# Entrypoint


def run(root: Path, *, recent_hours: float, json_mode: bool) -> None:
    now = datetime.now(timezone.utc)
    state = DiagnosticState(root=root, now=now, recent_window=timedelta(hours=recent_hours))
    data_root, logs_root, runtime_root = discover_layout(root)
    state.data_root = data_root
    state.logs_root = logs_root
    state.runtime_root = runtime_root

    collect_agent_events(state)
    collect_episodes(state)
    collect_reasoning_log(state)
    collect_experiments_log(state)
    collect_memory_store(state)
    collect_metacog_log(state)
    collect_actions_log(state)
    collect_concept_signals(state)
    collect_mood_logs(state)
    collect_goals(state)
    collect_dashboards(state)

    build_verdicts(state)
    print_report(state, json_mode=json_mode)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Script de diagnostic AGI Évolutive")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Racine du dépôt")
    parser.add_argument("--recent-hours", type=float, default=24.0, help="Fenêtre de récence en heures pour les signaux")
    parser.add_argument("--json", action="store_true", help="Sortie JSON machine-readable")
    args = parser.parse_args(argv)
    run(args.root.resolve(), recent_hours=max(1e-3, args.recent_hours), json_mode=args.json)


if __name__ == "__main__":
    main()
