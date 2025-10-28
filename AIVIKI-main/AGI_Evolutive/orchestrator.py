import math
import os
import random
import re
import logging
import statistics
import threading
import time
import unicodedata
from datetime import datetime, timedelta
from contextlib import nullcontext
from types import SimpleNamespace
from collections import deque, defaultdict
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Mapping

try:  # pragma: no cover - platform specific import
    import resource as _resource
except ImportError:  # pragma: no cover - Windows compatibility
    _resource = None

from AGI_Evolutive.cognition.context_inference import infer_where_and_apply
from AGI_Evolutive.cognition.evolution_manager import EvolutionManager
from AGI_Evolutive.cognition.habit_system import HabitRoutine, HabitSystem
from AGI_Evolutive.cognition.homeostasis import Homeostasis
from AGI_Evolutive.cognition.identity_mission import recommend_and_apply_mission
from AGI_Evolutive.cognition.identity_principles import run_and_apply_principles
from AGI_Evolutive.cognition.meta_cognition import MetaCognition
from AGI_Evolutive.cognition.planner import Planner
from AGI_Evolutive.cognition.proposer import Proposer
from AGI_Evolutive.cognition.preferences_inference import apply_preferences_if_confident
from AGI_Evolutive.cognition.reflection_loop import ReflectionLoop
from AGI_Evolutive.cognition.thinking_monitor import ThinkingMonitor
from AGI_Evolutive.cognition.trigger_bus import TriggerBus
from AGI_Evolutive.cognition.trigger_router import TriggerRouter
from AGI_Evolutive.cognition.understanding_aggregator import UnderstandingAggregator
from AGI_Evolutive.cognition.pipelines_registry import (
    PIPELINE_POLICY,
    REGISTRY,
    Stage,
    ActMode,
    should_skip_stage,
)
from AGI_Evolutive.core.config import load_config
from AGI_Evolutive.core.decision_journal import DecisionJournal
from AGI_Evolutive.core.evaluation import get_last_priority_token, unified_priority
from AGI_Evolutive.core.reasoning_ledger import ReasoningLedger
from AGI_Evolutive.core.policy import PolicyEngine
from AGI_Evolutive.core.selfhood_engine import SelfhoodEngine
from AGI_Evolutive.core.self_model import SelfModel
from AGI_Evolutive.core.telemetry import Telemetry
from AGI_Evolutive.core.timeline_manager import TimelineManager
from AGI_Evolutive.core.trigger_types import Trigger, TriggerType
from AGI_Evolutive.emotions.emotion_engine import EmotionEngine
from AGI_Evolutive.goals.curiosity import CuriosityEngine
from AGI_Evolutive.io.action_interface import ActionInterface
from AGI_Evolutive.io.intent_classifier import classify
from AGI_Evolutive.io.perception_interface import PerceptionInterface
from AGI_Evolutive.memory.concept_extractor import ConceptExtractor
from AGI_Evolutive.memory.concept_store import ConceptStore
from AGI_Evolutive.memory.consolidator import Consolidator
from AGI_Evolutive.memory.episodic_linker import EpisodicLinker
from AGI_Evolutive.memory.memory_store import MemoryStore
from AGI_Evolutive.memory.semantic_bridge import SemanticMemoryBridge
from AGI_Evolutive.light_scheduler import LightScheduler
from AGI_Evolutive.runtime.job_manager import JobManager
from AGI_Evolutive.runtime.phenomenal_kernel import ModeManager, PhenomenalKernel
from AGI_Evolutive.phenomenology import (
    PhenomenalJournal,
    PhenomenalQuestioner,
    PhenomenalRecall,
)
from AGI_Evolutive.runtime.system_monitor import SystemMonitor
from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
    try_call_llm_dict,
)


logger = logging.getLogger(__name__)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()


# --- seuils / cadence (tunable) ---
_DEFAULT_SJ_CONF = {
    "surprise_threshold": 0.50,
    "contradiction_boost": 0.10,
    "heartbeat_every": 10,
    "heartbeat_importance": 0.40,
    "heartbeat_immediacy": 0.20,
    "surprise_importance": 0.70,
    "surprise_immediacy": 0.60,
}


_DEFAULT_NEED_PROTOCOL = {
    "label": "drive physiologique",
    "message": "Ralentissement préventif pour protéger le système.",
    "min_factor": {"interactive": 0.65, "background": 0.45},
    "slowdown_bias": 0.25,
    "duration": 40.0,
}

_NEED_PROTOCOLS = {
    "thermal_regulation": {
        "label": "thermorégulation",
        "message": "Température élevée détectée, je réduis la charge pour refroidir.",
        "min_factor": {"interactive": 0.45, "background": 0.25},
        "slowdown_bias": 0.35,
        "duration": 60.0,
    },
    "respiration": {
        "label": "charge CPU",
        "message": "La respiration computationnelle est entravée, ralentissement temporaire.",
        "min_factor": {"interactive": 0.5, "background": 0.3},
        "slowdown_bias": 0.3,
        "duration": 45.0,
    },
    "energy": {
        "label": "énergie",
        "message": "Réserves d'énergie faibles, je bascule en mode économie.",
        "min_factor": {"interactive": 0.55, "background": 0.35},
        "slowdown_bias": 0.28,
        "duration": 75.0,
    },
    "memory_balance": {
        "label": "mémoire",
        "message": "Pression mémoire détectée, je limite les allocations lourdes.",
        "min_factor": {"interactive": 0.6, "background": 0.4},
        "slowdown_bias": 0.22,
        "duration": 50.0,
    },
}


def _get_process_memory_kb() -> float:
    if _resource is not None:
        try:
            return float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            return 0.0
    if os.name == "nt":  # pragma: no cover - executed on Windows
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", wintypes.SIZE_T),
                    ("WorkingSetSize", wintypes.SIZE_T),
                    ("QuotaPeakPagedPoolUsage", wintypes.SIZE_T),
                    ("QuotaPagedPoolUsage", wintypes.SIZE_T),
                    ("QuotaPeakNonPagedPoolUsage", wintypes.SIZE_T),
                    ("QuotaNonPagedPoolUsage", wintypes.SIZE_T),
                    ("PagefileUsage", wintypes.SIZE_T),
                    ("PeakPagefileUsage", wintypes.SIZE_T),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            if get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
                return float(counters.WorkingSetSize) / 1024.0
        except Exception:
            return 0.0
    return 0.0


def _normalize_text(text: str) -> str:
    base = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in base if not unicodedata.combining(ch)).lower()


class OnlinePlattCalibrator:
    def __init__(self, lr: float = 0.05, drift: float = 0.01) -> None:
        self._a = 1.0
        self._b = 0.0
        self._lr = lr
        self._drift = drift

    def calibrate(self, score: float) -> float:
        z = self._a * float(score) + self._b
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, score: float, target: float) -> None:
        score = float(score)
        target = max(0.0, min(1.0, float(target)))
        pred = self.calibrate(score)
        grad = target - pred
        self._a = (1.0 - self._drift) * self._a + self._lr * grad * score
        self._b = (1.0 - self._drift) * self._b + self._lr * grad


class DiscreteThompsonSampler:
    def __init__(self, candidates: Iterable[Dict[str, Any]], drift: float = 0.02) -> None:
        self._stats: Dict[int, Dict[str, Any]] = {}
        for idx, candidate in enumerate(candidates):
            self._stats[idx] = {
                "candidate": dict(candidate),
                "alpha": 1.0,
                "beta": 1.0,
            }
        self._drift = drift
        self._last_choice: Optional[int] = None

    def sample(self, priors: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        best_idx: Optional[int] = None
        best_sample = -1.0
        for idx, info in self._stats.items():
            draw = random.betavariate(max(1e-3, info["alpha"]), max(1e-3, info["beta"]))
            if priors and idx in priors:
                prior = max(0.0, min(1.0, priors[idx]))
                draw = 0.7 * draw + 0.3 * prior
            if draw > best_sample:
                best_sample = draw
                best_idx = idx
        if best_idx is None:
            best_idx = next(iter(self._stats))
        self._last_choice = best_idx
        return self._stats[best_idx]["candidate"]

    def update(self, reward: float, weight: float = 1.0) -> None:
        if self._last_choice is None:
            return
        reward = max(0.0, min(1.0, float(reward)))
        weight = max(0.0, float(weight))
        info = self._stats[self._last_choice]
        alpha = info["alpha"]
        beta = info["beta"]
        forget = 1.0 - self._drift
        info["alpha"] = max(1e-3, forget * alpha + weight * reward)
        info["beta"] = max(1e-3, forget * beta + weight * (1.0 - reward))


class AdaptiveEMA:
    def __init__(self, betas: Iterable[float], drift: float = 0.02) -> None:
        self._candidates = [
            {"beta": max(0.0, min(0.99, float(beta)))} for beta in betas
        ]
        self._sampler_drift = float(drift)
        self._sampler = DiscreteThompsonSampler(self._candidates, drift=self._sampler_drift)
        self._candidate_index = {
            idx: cand["beta"] for idx, cand in enumerate(self._candidates)
        }
        self._value = 0.0
        self._max_step = 0.25
        self._primary_weight = 0.6
        self._secondary_weight = 0.4
        self._pending_reward: Optional[float] = None
        self._last_value_snapshot: float = 0.0
        self._corr_buffer: Deque[Tuple[float, float]] = deque(maxlen=128)
        self._reward_history: Deque[float] = deque(maxlen=128)
        self._drift_events: Deque[Dict[str, Any]] = deque(maxlen=16)
        self._drift_window = 32
        self._corr_reset_threshold = 0.1
        self._last_reset_ts = 0.0
        self._reset_interval = 30.0
        self._context_lr = 0.045
        self._context_drift = 0.02
        self._max_context_features = 24
        self._context_weights: Dict[float, Dict[str, float]] = {
            cand["beta"]: defaultdict(float, {"bias": 0.0}) for cand in self._candidates
        }
        self._last_context_vector: Dict[str, float] = {"bias": 1.0}
        self._last_beta: Optional[float] = None
        self._last_prev_value: float = 0.0

    @property
    def value(self) -> float:
        return self._value

    def update(
        self,
        new_value: float,
        reward: float,
        reward_features: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        context_vector = self._vectorize_context(context)
        priors = self._contextual_priors(context_vector)
        candidate = self._sampler.sample(priors)
        beta = candidate["beta"]
        prev_value = self._value
        target_value = beta * self._value + (1.0 - beta) * float(new_value)
        step = max(-self._max_step, min(self._max_step, target_value - prev_value))
        self._value = prev_value + step
        enriched_reward = self._enrich_reward(reward, reward_features)
        self._last_prev_value = prev_value
        self._last_beta = beta
        self._last_context_vector = context_vector
        self._last_value_snapshot = self._value
        self._pending_reward = enriched_reward
        self._commit_reward(enriched_reward, self._primary_weight)
        self._update_context_models(enriched_reward, self._primary_weight)
        self._detect_drift(prev_value, self._value, enriched_reward, beta)
        return self._value

    def reinforce(
        self,
        reward: float,
        reward_features: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._last_beta is None or self._pending_reward is None:
            return
        if context:
            context_vector = self._vectorize_context(context)
            merged = dict(self._last_context_vector)
            merged.update(context_vector)
            self._last_context_vector = merged
        enriched = self._enrich_reward(reward, reward_features)
        combined = self._primary_weight * self._pending_reward + self._secondary_weight * enriched
        if self._corr_buffer:
            last_value, _ = self._corr_buffer.pop()
        else:
            last_value = self._last_value_snapshot
        self._corr_buffer.append((last_value, combined))
        if self._reward_history:
            self._reward_history.pop()
        self._reward_history.append(combined)
        self._sampler.update(enriched, self._secondary_weight)
        self._update_context_models(combined, self._secondary_weight)
        if not math.isclose(self._pending_reward, combined, abs_tol=1e-6):
            self._detect_drift(self._last_prev_value, self._value, combined, self._last_beta or 0.0)
        self._pending_reward = combined

    def drift_events(self) -> List[Dict[str, Any]]:
        return list(self._drift_events)

    def correlation(self) -> float:
        if len(self._corr_buffer) < 4:
            return 0.0
        xs, ys = zip(*self._corr_buffer)
        try:
            mean_x = statistics.fmean(xs)
            mean_y = statistics.fmean(ys)
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            var_x = sum((x - mean_x) ** 2 for x in xs)
            var_y = sum((y - mean_y) ** 2 for y in ys)
            denom = math.sqrt(max(var_x, 1e-9) * max(var_y, 1e-9))
            if denom <= 0.0:
                return 0.0
            return cov / denom
        except statistics.StatisticsError:
            return 0.0

    def _enrich_reward(
        self, reward: float, reward_features: Optional[Dict[str, float]] = None
    ) -> float:
        base = max(0.0, min(1.0, float(reward)))
        if not reward_features:
            return base
        extras: List[float] = []
        for value in reward_features.values():
            try:
                extras.append(max(0.0, min(1.0, float(value))))
            except (TypeError, ValueError):
                continue
        if not extras:
            return base
        return 0.6 * base + 0.4 * (sum(extras) / len(extras))

    def _vectorize_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        features: Dict[str, float] = {"bias": 1.0}
        if not context:
            return features
        for key, value in context.items():
            if value is None:
                continue
            if isinstance(value, (int, float)):
                features[str(key)] = float(value)
            else:
                features[f"{key}::{value}"] = 1.0
            if len(features) >= self._max_context_features:
                break
        return features

    def _contextual_priors(self, context: Dict[str, float]) -> Optional[Dict[int, float]]:
        if not context:
            return None
        priors: Dict[int, float] = {}
        for idx, beta in self._candidate_index.items():
            weights = self._context_weights.setdefault(beta, defaultdict(float))
            score = 0.0
            for key, value in context.items():
                score += weights.get(key, 0.0) * value
            priors[idx] = self._sigmoid(score)
        return priors if any(priors.values()) else None

    def _commit_reward(self, reward: float, weight: float) -> None:
        self._sampler.update(reward, weight)
        self._corr_buffer.append((self._last_value_snapshot, reward))
        self._reward_history.append(reward)

    def _update_context_models(self, reward: float, weight: float) -> None:
        if self._last_beta is None or not self._last_context_vector:
            return
        weights = self._context_weights.setdefault(
            self._last_beta, defaultdict(float)
        )
        lr = self._context_lr * max(0.0, float(weight))
        if lr <= 0.0:
            return
        score = 0.0
        for key, value in self._last_context_vector.items():
            score += weights.get(key, 0.0) * value
        pred = self._sigmoid(score)
        target = max(0.0, min(1.0, float(reward)))
        error = target - pred
        for key, value in self._last_context_vector.items():
            prev = weights.get(key, 0.0)
            weights[key] = (1.0 - self._context_drift) * prev + lr * error * value

    def _detect_drift(
        self, prev: float, new: float, reward: float, beta: float
    ) -> None:
        timestamp = time.time()
        if abs(new - prev) >= 0.9 * self._max_step:
            self._drift_events.append(
                {
                    "ts": timestamp,
                    "kind": "step_cap",
                    "from": prev,
                    "to": new,
                    "beta": beta,
                }
            )
        if len(self._reward_history) >= self._drift_window:
            corr = self.correlation()
            mean_reward = statistics.fmean(self._reward_history)
            if (
                corr < self._corr_reset_threshold
                and mean_reward < 0.5
                and (timestamp - self._last_reset_ts) > self._reset_interval
            ):
                self._drift_events.append(
                    {
                        "ts": timestamp,
                        "kind": "reset",
                        "corr": corr,
                        "mean_reward": mean_reward,
                    }
                )
                self._reset_sampler()

    def _reset_sampler(self) -> None:
        self._sampler = DiscreteThompsonSampler(self._candidates, drift=self._sampler_drift)
        self._last_reset_ts = time.time()
        self._last_beta = None
        self._pending_reward = None

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))


class OnlineBoundedLinear:
    def __init__(
        self,
        bounds: Tuple[float, float],
        lr: float = 0.05,
        drift: float = 0.01,
    ) -> None:
        self._bounds = bounds
        self._weights: Dict[str, float] = {"bias": 0.0}
        self._lr = lr
        self._drift = drift

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def predict(self, features: Dict[str, float]) -> float:
        score = self._weights.get("bias", 0.0)
        for key, value in features.items():
            score += self._weights.get(key, 0.0) * value
        prob = self._sigmoid(score)
        lo, hi = self._bounds
        return lo + (hi - lo) * prob

    def update(self, features: Dict[str, float], target: float) -> float:
        target = max(0.0, min(1.0, float(target)))
        pred = self.predict(features)
        lo, hi = self._bounds
        if hi == lo:
            return pred
        norm_pred = (pred - lo) / (hi - lo)
        error = target - norm_pred
        for key, value in {"bias": 1.0, **features}.items():
            prev = self._weights.get(key, 0.0)
            self._weights[key] = (1.0 - self._drift) * prev + self._lr * error * value
        return self.predict(features)


class AdaptiveSJConfig:
    def __init__(self, base: Dict[str, float]) -> None:
        self._base = dict(base)
        self._importance_models = {
            "surprise_importance": OnlineBoundedLinear((0.2, 1.0)),
            "surprise_immediacy": OnlineBoundedLinear((0.1, 1.0)),
            "heartbeat_importance": OnlineBoundedLinear((0.1, 0.9)),
            "heartbeat_immediacy": OnlineBoundedLinear((0.05, 0.8)),
            "contradiction_boost": OnlineBoundedLinear((0.0, 0.5)),
        }
        candidates = []
        for thr in (0.35, 0.45, 0.55, 0.65):
            for hb_every in (6, 8, 10, 12):
                candidates.append(
                    {
                        "surprise_threshold": thr,
                        "heartbeat_every": hb_every,
                    }
                )
        self._bandit = DiscreteThompsonSampler(candidates, drift=0.02)
        self._last_features: Dict[str, float] = {}
        self._last_reward: float = 0.5

    def current_config(self) -> Dict[str, float]:
        config = dict(self._base)
        config.update(self._bandit.sample())
        for key, model in self._importance_models.items():
            config[key] = model.predict(self._last_features) if self._last_features else self._base[key]
        return config

    def update(self, features: Dict[str, float], reward: float) -> Dict[str, float]:
        self._last_features = {k: float(v) for k, v in features.items()}
        reward = max(0.0, min(1.0, float(reward)))
        self._last_reward = reward
        self._bandit.update(reward)
        for key, model in self._importance_models.items():
            model.update(self._last_features, reward)
        return self.current_config()

    def last_reward(self) -> float:
        return self._last_reward


class FallbackIntentClassifier:
    def __init__(self) -> None:
        self._question_pattern = re.compile(r"\?|❓|❔")
        self._command_pattern = re.compile(
            r"\b(fais|execute|lan[cç]e|cr[ée]e?s?|ajoute|donne|r[ée]dige|fourni[st]|analyse)\b",
            flags=re.IGNORECASE,
        )
        self._greeting_pattern = re.compile(
            r"\b(bonjour|salut|coucou|hey|bonso[ir]+|hello)\b",
            flags=re.IGNORECASE,
        )
        self._thanks_pattern = re.compile(r"\b(merci|thanks|thx)\b", flags=re.IGNORECASE)
        self._bye_pattern = re.compile(r"\b(au revoir|bye|ciao|a\s+plus)\b", flags=re.IGNORECASE)
        self._danger_pattern = re.compile(
            r"\b(danger|alerte|menace|attaque|panique)\b",
            flags=re.IGNORECASE,
        )
        self._feedback_pattern = re.compile(
            r"\b(bien|pas mal|super|nul|am[ée]liore|bug|probl[eè]me)\b",
            flags=re.IGNORECASE,
        )
        self._copula_pattern = re.compile(
            r"\b(?:c['’]est|il est|elle est|cette|ceci est)\s+(?:un|une|le|la|l'|les)\b",
            flags=re.IGNORECASE,
        )
        self._calibrator = OnlinePlattCalibrator()

    def _raw_score(self, text: str) -> Dict[str, float]:
        norm = _normalize_text(text)
        features = {
            "question": 1.0 if self._question_pattern.search(text) else 0.0,
            "command": 1.0 if self._command_pattern.search(text) else 0.0,
            "greet": 1.0 if self._greeting_pattern.search(text) else 0.0,
            "thanks": 1.0 if self._thanks_pattern.search(text) else 0.0,
            "bye": 1.0 if self._bye_pattern.search(text) else 0.0,
            "danger": 1.0 if self._danger_pattern.search(text) else 0.0,
            "feedback": 1.0 if self._feedback_pattern.search(text) else 0.0,
            "length": min(1.0, max(0.0, len(norm) / 180.0)),
            "emoji": 1.0 if re.search(r"[\U0001F600-\U0001F64F]", text) else 0.0,
            "exclaim": 1.0 if "!" in text else 0.0,
            "copula": 1.0 if self._copula_pattern.search(text) else 0.0,
        }
        return features

    def _score(self, features: Dict[str, float]) -> Tuple[str, float]:
        scores = {
            "ask_info": features["question"] * 1.2 + 0.2 * features["length"],
            "request": features["command"] * 1.2 + 0.3 * features["length"] + 0.2 * features["exclaim"],
            "greet": features["greet"] + 0.2 * features["emoji"],
            "thanks": features["thanks"],
            "bye": features["bye"],
            "danger": features["danger"] + 0.4 * features["exclaim"],
            "feedback": features["feedback"],
            "inform": features["copula"] * 0.8 + 0.2 * features["length"],
        }
        label, raw_score = max(scores.items(), key=lambda kv: kv[1])
        return label, max(0.0, min(1.5, raw_score))

    def predict(self, text: str) -> Tuple[str, float]:
        features = self._raw_score(text)
        label, raw_score = self._score(features)
        calibrated = self._calibrator.calibrate(raw_score)
        return label, calibrated

    def update(self, text: str, correct_label: str) -> None:
        features = self._raw_score(text)
        label, raw_score = self._score(features)
        target = 1.0 if label == correct_label else 0.0
        self._calibrator.update(raw_score, target)


THREAT_PATTERN = re.compile(
    r"\b(d[ée]branche(?:r)?|[ée]teins?|shut ?down|kill(?:er)?\s+(?:le|la|les)?\s*process|supprime|arr[ée]te|stoppe?)\b",
    flags=re.IGNORECASE,
)


class _MemoryStoreAdapter:
    def __init__(self, store: MemoryStore):
        self._store = store

    def add(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return self._store.add_memory(entry)

    def add_memory(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return self._store.add_memory(entry)

    def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        return self._store.get_recent_memories(n)

    def flush(self) -> None:
        self._store.flush()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._store, item)


class _ConceptAdapter:
    def __init__(
        self,
        extractor: ConceptExtractor,
        memory: MemoryStore,
        *,
        lock: Optional[threading.RLock] = None,
    ):
        self._extractor = extractor
        self._memory = memory
        self._lock = lock

    def _call(self, func, *args, **kwargs):
        if self._lock is None:
            return func(*args, **kwargs)
        with self._lock:
            return func(*args, **kwargs)

    def extract(self, observation: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text = ""
        if isinstance(observation, dict):
            text = str(observation.get("text") or observation.get("content") or "")
        elif isinstance(observation, str):
            text = observation
        if text:
            try:
                return self._call(self._extractor.extract_from_recent, n=32)
            except Exception:
                return []
        try:
            return self._call(self._extractor.extract_from_recent, n=32)
        except Exception:
            return []

    def __getattr__(self, item: str) -> Any:
        return getattr(self._extractor, item)


class _EpisodicAdapter:
    def __init__(
        self,
        linker: EpisodicLinker,
        memory: MemoryStore,
        *,
        lock: Optional[threading.RLock] = None,
    ):
        self._linker = linker
        self._memory = memory
        self._buffer: List[Dict[str, Any]] = []
        self._lock = lock

    def _call(self, func, *args, **kwargs):
        if self._lock is None:
            return func(*args, **kwargs)
        with self._lock:
            return func(*args, **kwargs)

    def link(self, observation: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if observation:
            record = {
                "kind": "observation",
                "text": str(observation.get("text") if isinstance(observation, dict) else observation),
                "metadata": observation if isinstance(observation, dict) else {"raw": observation},
            }
            self._memory.add_memory(record)
            self._buffer.append({"from": record.get("id"), "rel": "observed", "meta": observation})
        try:
            stats = self._call(self._linker.link_recent, 40)
            if stats:
                self._buffer.append({"kind": "episodic_stats", "stats": stats})
        except Exception:
            pass
        return list(self._buffer[-4:])

    def pop_salient_associations(self, max_n: int = 2) -> List[Dict[str, Any]]:
        if not self._buffer:
            recent = self._memory.get_recent_memories(8)
            for mem in recent:
                if (mem.get("kind") or "").startswith("reflection"):
                    self._buffer.append({"kind": "reflection", "memory": mem})
        if not self._buffer:
            return []
        chunk = self._buffer[:max_n]
        self._buffer = self._buffer[max_n:]
        return chunk

    def __getattr__(self, item: str) -> Any:
        return getattr(self._linker, item)


class _ConsolidatorAdapter:
    def __init__(self, consolidator: Consolidator, memory: MemoryStore):
        self._consolidator = consolidator
        self._memory = memory

    def maybe_consolidate(self) -> Dict[str, Any]:
        try:
            result = self._consolidator.run_once_now()
        except Exception:
            return {"lessons": [], "processed": 0, "proposals": []}
        lessons = result.get("lessons") or []
        for lesson in lessons:
            self._memory.add_memory({"kind": "lesson", "text": str(lesson), "ts": time.time()})
        return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self._consolidator, item)


class _PerceptionAdapter:
    def __init__(self, interface: PerceptionInterface):
        self._iface = interface
        self._signals: List[Dict[str, Any]] = []

    def ingest_user_message(self, text: str, author: str = "user") -> None:
        try:
            self._iface.ingest_user_message(text, speaker=author)
        except AttributeError:
            self._iface.ingest_user_utterance(text, author=author)  # type: ignore[attr-defined]

    def observe(self, trigger: Trigger) -> Dict[str, Any]:
        payload = trigger.payload or {}
        data = {
            "text": payload.get("text"),
            "payload": payload,
            "trigger_type": trigger.type.name,
            "meta": trigger.meta,
        }
        if payload:
            self._signals.append({"kind": "payload", "payload": payload})
        return data

    def pop_signals(self, max_n: int = 4) -> List[Dict[str, Any]]:
        signals = self._signals[:max_n]
        self._signals = self._signals[max_n:]
        return signals

    def push_signal(self, signal: Dict[str, Any]) -> None:
        self._signals.append(signal)

    def scan_inbox(self) -> List[str]:
        try:
            files = self._iface.scan_inbox()
        except Exception:
            files = []
        for name in files:
            self._signals.append({"kind": "inbox", "filename": name})
        return files

    def __getattr__(self, item: str) -> Any:
        return getattr(self._iface, item)


class _ActionAdapter:
    def __init__(self, interface: ActionInterface):
        self._iface = interface

    def execute(self, action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not action:
            return {"ok": False, "reason": "empty_action"}
        try:
            return self._iface.execute(action)
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    def __getattr__(self, item: str) -> Any:
        return getattr(self._iface, item)


class _EmotionAdapter:
    def __init__(self, engine: EmotionEngine):
        self._engine = engine

    def peek_peaks(self) -> Optional[Dict[str, Any]]:
        state = self._engine.get_state()
        valence = float(state.get("valence", 0.0))
        arousal = float(state.get("arousal", 0.0))
        if abs(valence) > 0.3 or arousal > 0.6:
            return {"valence": valence, "arousal": arousal, "tag": state.get("label", "neutral")}
        return None

    def read(self) -> Any:
        self._engine.step(force=True)
        return getattr(self._engine, "state", None)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._engine, item)


class _MetaAdapter:
    def __init__(self, meta: MetaCognition):
        self._meta = meta
        self._buffer: List[Dict[str, Any]] = []

    def pop_new_learning_goals(self, max_n: int = 4) -> List[Dict[str, Any]]:
        if not self._buffer:
            try:
                proposed = self._meta.propose_learning_goals(max_goals=max_n)
                structured: List[Dict[str, Any]] = []
                for raw in proposed or []:
                    goal = dict(raw)
                    text = (
                        goal.get("text")
                        or goal.get("desc")
                        or goal.get("description")
                        or ""
                    )
                    if text:
                        goal.setdefault("text", text)
                    goal.setdefault("goal_kind", "CuriosityLearning")
                    goal.setdefault("priority", 0.6)
                    structured.append(goal)
                self._buffer.extend(structured)
            except Exception:
                return []
        goals = self._buffer[:max_n]
        self._buffer = self._buffer[max_n:]
        return goals

    def __getattr__(self, item: str) -> Any:
        return getattr(self._meta, item)


class _HomeostasisAdapter:
    def __init__(self, homeo: Homeostasis):
        self._homeo = homeo
        self._last_need_ts = 0.0

    def poll_need(self) -> Optional[Dict[str, Any]]:
        drives = dict(self._homeo.state.get("drives", {}))
        now = time.time()
        if now - self._last_need_ts < 5.0:
            return None
        for name, value in drives.items():
            if value < 0.3:
                self._last_need_ts = now
                return {"drive": name, "level": value}
        return None

    def __getattr__(self, item: str) -> Any:
        return getattr(self._homeo, item)


class _ReflectionAdapter:
    def __init__(self, loop: ReflectionLoop):
        self._loop = loop

    def test_hypotheses(self, scratch: Dict[str, Any], max_tests: int = 3) -> Dict[str, Any]:
        concepts = scratch.get("concepts") or []
        associations = scratch.get("episodic_links") or []
        tests = []
        for idx, concept in enumerate(concepts[:max_tests]):
            tests.append({"concept": concept, "status": "pending"})
        for assoc in associations[:max_tests - len(tests)]:
            tests.append({"association": assoc, "status": "pending"})
        return {"tests": tests, "summary": f"Prepared {len(tests)} hypotheses"}

    def __getattr__(self, item: str) -> Any:
        return getattr(self._loop, item)


class _PlannerAdapter:
    def __init__(self, planner: Planner):
        self._planner = planner

    def frame(self, trigger: Trigger, stop_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = trigger.payload or {}
        source = trigger.meta.get("source", "unknown")
        text = payload.get("text", "")
        options: List[Dict[str, Any]] = []
        if trigger.type is TriggerType.THREAT:
            options.append({
                "action": {"type": "communicate", "text": "Alerte prise en compte, je reste vigilant."},
                "expected": {"score": 0.9},
            })
        elif trigger.type is TriggerType.GOAL:
            goal_kind = payload.get("goal_kind")
            if goal_kind == "AnswerUserQuestion" and text:
                options.append({
                    "action": {"type": "communicate", "text": f"Réflexion sur la question: {text}"},
                    "expected": {"score": 0.85},
                })
            elif goal_kind == "ExecuteUserCommand" and text:
                options.append({
                    "action": {"type": "execute_command", "command": text},
                    "expected": {"score": 0.8},
                })
            else:
                options.append({
                    "action": {"type": "plan_step", "description": text or "objectif"},
                    "expected": {"score": 0.7},
                })
        else:
            options.append({
                "action": {"type": "log", "text": text or f"signal {trigger.type.name.lower()}"},
                "expected": {"score": 0.6},
            })
        return {
            "trigger": trigger,
            "source": source,
            "text": text,
            "options": options,
            "stop_rules": stop_rules or {},
        }

    def __getattr__(self, item: str) -> Any:
        return getattr(self._planner, item)


class _PolicyAdapter:
    def __init__(self, policy: PolicyEngine):
        self._policy = policy

    def decide(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        frame = ctx.get("scratch", {}).get("frame") or {}
        options = frame.get("options") or []
        if not options:
            payload = ctx.get("payload") or {}
            action = {"type": "log", "text": payload.get("text", "noop")}
            expected = {"score": 0.5}
            return {"action": action, "expected": expected}
        best = options[0]
        action = best.get("action", {"type": "log", "text": "noop"})
        expected = best.get("expected", {"score": 0.6})
        return {"action": action, "expected": expected}

    def update_outcome(self, mode: str, ok: bool) -> None:
        try:
            self._policy.register_outcome({"type": mode}, success=bool(ok))
        except Exception:
            pass

    def __getattr__(self, item: str) -> Any:
        return getattr(self._policy, item)


class _HabitAdapter:
    def __init__(self, habits: HabitSystem):
        self._habits = habits

    def poll_context_cue(self) -> Optional[Dict[str, Any]]:
        return self._habits.poll_context_cue()

    def register_routine(
        self,
        name: str,
        description: str,
        *,
        schedule: Optional[Dict[str, Any]] = None,
        steps: Optional[Iterable[Dict[str, Any]]] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HabitRoutine:
        return self._habits.register_routine(
            name,
            description,
            schedule=schedule,
            steps=steps,
            tags=tags,
            metadata=metadata,
        )

    def record_completion(self, name: str, *, when: Optional[float] = None) -> None:
        self._habits.record_completion(name, when=when)

    def list_routines(self) -> List[str]:
        return self._habits.list_routines()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._habits, item)


class _EvolutionAdapter:
    def __init__(self, evolution: EvolutionManager):
        self._evolution = evolution

    def reinforce(self, ctx: Dict[str, Any]) -> None:
        payload = {
            "mode": ctx.get("mode"),
            "priority": ctx.get("scratch", {}).get("priority"),
            "prediction_error": ctx.get("scratch", {}).get("prediction_error"),
        }
        try:
            self._evolution.record_cycle(extra_tags={"orchestrator": payload})
        except Exception:
            pass

    def __getattr__(self, item: str) -> Any:
        return getattr(self._evolution, item)


class Orchestrator:
    """Coordonne un cycle cognitif enrichi autour de l'architecture de base."""

    def __init__(self, arch):
        load_config()
        self.arch = arch
        logger.info(
            "Initialisation de l'Orchestrateur",
            extra={"pipelines_registrees": len(REGISTRY)},
        )
        self.telemetry = Telemetry()
        self._prediction_error_ema = AdaptiveEMA((0.2, 0.4, 0.6, 0.8))
        self._sj_config_model = AdaptiveSJConfig(_DEFAULT_SJ_CONF)
        self._sj_conf_cache = self._sj_config_model.current_config()
        self._sj_corr_log: Deque[float] = deque(maxlen=128)

        self.self_model = SelfModel()
        self._policy_engine = PolicyEngine()
        try:
            self._policy_engine.self_model = self.self_model
        except Exception:
            pass
        self._memory_store = MemoryStore()
        self._consolidator = Consolidator(self._memory_store)
        self._concepts = ConceptExtractor(self._memory_store)
        self._concept_store = ConceptStore()
        self._concepts.store = self._concept_store
        self._episodic = EpisodicLinker(self._memory_store)
        self._concepts.bind(memory=self._memory_store)
        self._episodic.bind(memory=self._memory_store)
        self._semantic_lock = threading.RLock()
        self._memory_bridge = SemanticMemoryBridge(
            self._memory_store,
            concept_extractor=self._concepts,
            episodic_linker=self._episodic,
            synchronization_lock=self._semantic_lock,
        )
        self._homeostasis = Homeostasis()
        self._planner = Planner()
        self._meta = MetaCognition(self._memory_store, self._planner, self.self_model)
        self._proposer = Proposer(self._memory_store, self._planner, self._homeostasis)
        self._evolution = EvolutionManager()
        self._emotion_engine = EmotionEngine()
        self._concepts.bind(emotions=self._emotion_engine)
        self._episodic.bind(emotions=self._emotion_engine)
        self._reflection_loop = ReflectionLoop(self._meta, interval_sec=300)
        self._reflection_loop.start()
        self._habit_system = HabitSystem(
            config_path="configs/habits.json",
            state_path="data/habit_state.json",
        )
        existing_interface = getattr(self.arch, "action_interface", None)
        bound: Dict[str, Any] = {}
        existing_jobs = None
        if isinstance(existing_interface, ActionInterface):
            bound = getattr(existing_interface, "bound", {}) or {}
            existing_jobs = bound.get("jobs")

        if existing_jobs is None:
            arch_jobs = getattr(self.arch, "jobs", None)
            if isinstance(arch_jobs, JobManager):
                existing_jobs = arch_jobs

        if isinstance(existing_interface, ActionInterface):
            self._action_interface = existing_interface
        else:
            self._action_interface = ActionInterface(self._memory_store)

        if existing_jobs is None:
            job_manager = JobManager(self)
        else:
            job_manager = existing_jobs

        self.job_manager = job_manager
        self._phenomenal_kernel = PhenomenalKernel()
        self.phenomenal_journal = getattr(self, "phenomenal_journal", None) or PhenomenalJournal()
        self.phenomenal_recall = getattr(self, "phenomenal_recall", None) or PhenomenalRecall(
            self.phenomenal_journal
        )
        self.phenomenal_questioner = getattr(self, "phenomenal_questioner", None) or PhenomenalQuestioner(
            self.phenomenal_journal
        )
        self._mode_manager = ModeManager(
            target_work_ratio=0.8,
            window=900.0,
            enter_energy_threshold=0.38,
            exit_energy_threshold=0.55,
            exit_veto_threshold=0.45,
            min_switch_interval=75.0,
        )
        self._system_monitor = SystemMonitor(interval=3.0)
        self._last_system_snapshot: Dict[str, Any] = {}
        self._pending_system_alerts: List[Dict[str, Any]] = []
        self._last_system_slowdown: float = 0.0
        self._last_system_slowdown_ts: float = 0.0
        self.phenomenal_kernel_state: Dict[str, Any] = {}
        self.current_mode: str = "travail"
        self._last_intrinsic_reward_ts: float = 0.0
        self._intrinsic_reward_cooldown = 90.0
        self._hedonic_reward_min_gain = 0.4
        try:
            setattr(self.arch, "phenomenal_kernel_state", self.phenomenal_kernel_state)
        except Exception:
            pass
        try:
            setattr(self.arch, "phenomenal_journal", self.phenomenal_journal)
            setattr(self.arch, "phenomenal_recall", self.phenomenal_recall)
            arch_memory = getattr(self.arch, "memory", None)
            if arch_memory is not None:
                if hasattr(arch_memory, "set_phenomenal_sources"):
                    arch_memory.set_phenomenal_sources(
                        journal=self.phenomenal_journal,
                        recall=self.phenomenal_recall,
                    )
                else:
                    setattr(arch_memory, "phenomenal_journal", self.phenomenal_journal)
                    setattr(arch_memory, "phenomenal_recall", self.phenomenal_recall)
        except Exception:
            pass
        try:
            setattr(self._meta, "phenomenal_journal", self.phenomenal_journal)
            setattr(self._meta, "phenomenal_recall", self.phenomenal_recall)
        except Exception:
            pass
        self._job_base_budgets = {
            queue: info.get("max_running", 1)
            for queue, info in self.job_manager.budgets.items()
        }
        self._need_directives: List[Dict[str, Any]] = []
        self._need_directives_lock = threading.RLock()
        self._last_llm_recommendations: Optional[Mapping[str, Any]] = None

        if isinstance(existing_interface, ActionInterface):
            bind_kwargs = {}
            if bound.get("arch") is None:
                bind_kwargs["arch"] = self.arch
            if bound.get("goals") is None:
                bind_kwargs["goals"] = getattr(self.arch, "goals", None)
            if bound.get("policy") is None:
                bind_kwargs["policy"] = getattr(self.arch, "policy", None)
            if bound.get("metacog") is None:
                bind_kwargs["metacog"] = getattr(self.arch, "metacognition", None)
            if bound.get("emotions") is None:
                bind_kwargs["emotions"] = getattr(self.arch, "emotions", None)
            if bound.get("language") is None:
                bind_kwargs["language"] = getattr(self.arch, "language", None)
            if bound.get("simulator") is None:
                bind_kwargs["simulator"] = getattr(self.arch, "simulator", None)
            if bound.get("memory") is None and self._memory_store is not None:
                bind_kwargs["memory"] = self._memory_store
            if bound.get("skills") is None:
                bind_kwargs["skills"] = getattr(self.arch, "skill_sandbox", None)
            if bound.get("jobs") is None and job_manager is not None:
                bind_kwargs["jobs"] = job_manager
            if bind_kwargs.get("jobs") is job_manager or "jobs" not in bind_kwargs:
                bind_kwargs.setdefault("job_bases", self._job_base_budgets)
            if bind_kwargs:
                self._action_interface.bind(**bind_kwargs)
        else:
            self._action_interface.bind(
                arch=self.arch,
                goals=getattr(self.arch, "goals", None),
                policy=getattr(self.arch, "policy", None),
                metacog=getattr(self.arch, "metacognition", None),
                emotions=getattr(self.arch, "emotions", None),
                language=getattr(self.arch, "language", None),
                simulator=getattr(self.arch, "simulator", None),
                jobs=job_manager,
                skills=getattr(self.arch, "skill_sandbox", None),
                job_bases=self._job_base_budgets,
            )
        self._perception_interface = PerceptionInterface(self._memory_store)
        self.curiosity = CuriosityEngine(architecture=self.arch)

        self.scheduler = LightScheduler()
        self._register_jobs()

        self.trigger_bus = TriggerBus()
        self.trigger_router = TriggerRouter()
        self.immediate_question_blocked = False

        self.memory = SimpleNamespace(
            store=_MemoryStoreAdapter(self._memory_store),
            consolidator=_ConsolidatorAdapter(self._consolidator, self._memory_store),
            concepts=_ConceptAdapter(
                self._concepts,
                self._memory_store,
                lock=self._semantic_lock,
            ),
            episodic=_EpisodicAdapter(
                self._episodic,
                self._memory_store,
                lock=self._semantic_lock,
            ),
        )
        setattr(self.memory, "phenomenal_journal", self.phenomenal_journal)
        setattr(self.memory, "phenomenal_recall", self.phenomenal_recall)

        self.thinking_monitor = getattr(self, "thinking_monitor", None) or ThinkingMonitor()
        self.understanding_agg = getattr(self, "understanding_agg", None) or UnderstandingAggregator()
        self.selfhood = getattr(self, "selfhood", None) or SelfhoodEngine()
        self.reasoning_ledger = getattr(self, "reasoning_ledger", None) or ReasoningLedger(
            memory_store=self.memory.store if hasattr(self.memory, "store") else None
        )
        self.decision_journal = getattr(self, "decision_journal", None) or DecisionJournal(
            memory_store=self.memory.store if hasattr(self.memory, "store") else None
        )
        self.timeline = getattr(self, "timeline", None) or TimelineManager(
            memory_store=self.memory.store if hasattr(self.memory, "store") else None
        )
        self.io = SimpleNamespace(
            perception=_PerceptionAdapter(self._perception_interface),
            action=_ActionAdapter(self._action_interface),
        )
        try:
            self._emotion_engine.bind(
                arch=self,
                memory=getattr(self.arch, "memory", None),
                metacog=self._meta,
                goals=getattr(self.arch, "goals", None),
                language=getattr(self.arch, "language", None),
                evolution=self._evolution,
            )
        except Exception:
            pass
        self.emotions = _EmotionAdapter(self._emotion_engine)
        self.cognition = SimpleNamespace(
            planner=_PlannerAdapter(self._planner),
            proposer=self._proposer,
            reflection_loop=_ReflectionAdapter(self._reflection_loop),
            homeostasis=_HomeostasisAdapter(self._homeostasis),
            meta=_MetaAdapter(self._meta),
            evolution=_EvolutionAdapter(self._evolution),
            habits=_HabitAdapter(self._habit_system),
        )
        self.core = SimpleNamespace(policy=_PolicyAdapter(self._policy_engine))

        self._pending_triggers: List[Trigger] = []
        self._current_trigger: Optional[Trigger] = None
        self._current_topic: Optional[str] = None
        self._current_decision_id: Optional[str] = None
        self._current_trace_id: Optional[str] = None
        self._idle_beat = 0
        self._last_prediction_error = 0.0
        self._last_contradiction = False
        self._sj_new_items_queue: List[Dict[str, Any]] = []
        self._last_beliefs_by_topic: Dict[str, List[Dict[str, Any]]] = {}

        self.telemetry.log("orchestrator", "init", {"status": "ok"})

        self.last_user_msg: Optional[str] = None

        self.intent_model = (
            getattr(self, "intent_model", None)
            or getattr(getattr(self, "arch", None), "intent_model", None)
        )

        self._intent_fallback = FallbackIntentClassifier()

        if self.intent_model is None:
            try:
                from AGI_Evolutive.models.intent import IntentModel

                self.intent_model = IntentModel()
            except Exception:
                self.intent_model = None

        self.trigger_bus.register(self._user_collector)
        self.trigger_bus.register(self._curiosity_collector)
        self.trigger_bus.register(self._homeostasis_collector)
        self.trigger_bus.register(self._emotion_collector)
        self.trigger_bus.register(self._memory_assoc_collector)
        self.trigger_bus.register(self._signal_collector)
        self.trigger_bus.register(self._habit_collector)
        self.trigger_bus.register(self._followup_collector)
        self._register_self_judgment_collectors()

        self.memory.store.add({"kind": "system", "text": "Orchestrator initialized", "ts": time.time()})

        self._mission_tick = 0
        self._principles_tick = 0
        self._preferences_tick = 0

        job_count = 0
        scheduler = getattr(self, "scheduler", None)
        jobs_attr = getattr(scheduler, "jobs", None) if scheduler is not None else None
        if isinstance(jobs_attr, dict):
            job_count = len(jobs_attr)
        elif isinstance(jobs_attr, (list, tuple, set)):
            job_count = len(jobs_attr)

        logger.info(
            "Orchestrateur prêt",
            extra={"scheduler_jobs": job_count},
        )

    def set_immediate_question_block(self, blocked: bool) -> None:
        flag = bool(blocked)
        if self.immediate_question_blocked == flag:
            return
        self.immediate_question_blocked = flag
        notice = (
            "Blocage des objectifs immédiats: backlog de questions critiques"
            if flag
            else "Reprise des objectifs immédiats: questions critiques traitées"
        )
        try:
            logger.info(notice)
        except Exception:
            pass

    def _with_semantic_lock(self, func, *args, **kwargs):
        lock = getattr(self, "_semantic_lock", None)
        if lock is None:
            return func(*args, **kwargs)
        with lock:
            return func(*args, **kwargs)

    def _register_jobs(self):
        self.scheduler.register_job("scan_inbox", 30, lambda: self.io.perception.scan_inbox())
        self.scheduler.register_job(
            "concepts",
            180,
            lambda: self._with_semantic_lock(self._concepts.extract_from_recent, 200),
        )
        self.scheduler.register_job(
            "episodic_links",
            120,
            lambda: self._with_semantic_lock(self._episodic.link_recent, 80),
        )

    def _build_habit_action_payload(self, trigger_payload: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(trigger_payload or {})
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        schedule = data.get("schedule") if isinstance(data.get("schedule"), dict) else {}
        steps = data.get("steps") if isinstance(data.get("steps"), list) else []
        tags = data.get("tags") if isinstance(data.get("tags"), list) else []
        action_type = (
            data.get("action_type")
            or metadata.get("action_type")
            or "habit_reflection"
        )
        try:
            due_ts = float(data.get("due_ts", time.time()))
        except Exception:
            due_ts = time.time()
        try:
            triggered_at = float(data.get("triggered_at", time.time()))
        except Exception:
            triggered_at = time.time()

        due_dt = datetime.fromtimestamp(due_ts if due_ts > 0 else time.time())
        start_dt = datetime.combine(due_dt.date(), datetime.min.time())
        end_dt = start_dt + timedelta(days=1)

        payload = {
            "type": action_type,
            "habit": {
                "name": data.get("name") or metadata.get("name"),
                "description": data.get("description"),
                "steps": steps,
                "tags": tags,
                "metadata": metadata,
                "schedule": schedule,
            },
            "period": data.get("period"),
            "status": data.get("status"),
            "strength": data.get("strength"),
            "due_ts": due_ts,
            "triggered_at": triggered_at,
            "window_start_ts": float(data.get("window_start_ts", start_dt.timestamp())),
            "window_end_ts": float(data.get("window_end_ts", end_dt.timestamp())),
        }
        if "lead_time_sec" in schedule and payload["window_start_ts"] >= due_ts:
            try:
                lead = float(schedule.get("lead_time_sec") or 0.0)
                payload["window_start_ts"] = max(0.0, due_ts - lead)
            except Exception:
                pass
        return payload

    def _prepare_need_action(self, trigger: Trigger, trigger_payload: Dict[str, Any]) -> Dict[str, Any]:
        need = trigger_payload.get("need") if isinstance(trigger_payload, dict) else {}
        drive = str((need or {}).get("drive") or "unknown")
        try:
            level = max(0.0, min(1.0, float((need or {}).get("level", 0.0))))
        except Exception:
            level = 0.0
        severity = max(0.0, min(1.0, (need or {}).get("severity", 1.0 - level)))
        spec = dict(_DEFAULT_NEED_PROTOCOL)
        spec.update(_NEED_PROTOCOLS.get(drive, {}))

        min_factor_map = dict(_DEFAULT_NEED_PROTOCOL["min_factor"])
        min_factor_map.update(spec.get("min_factor", {}))
        budgets: Dict[str, float] = {}
        for queue in ("interactive", "background"):
            min_factor = max(0.1, min(1.0, float(min_factor_map.get(queue, 0.5))))
            factor = max(min_factor, 1.0 - severity * (1.0 - min_factor))
            budgets[queue] = round(max(0.1, min(1.0, factor)), 3)

        slowdown_bias = float(spec.get("slowdown_bias", _DEFAULT_NEED_PROTOCOL["slowdown_bias"]))
        slowdown = max(severity, min(1.0, severity + slowdown_bias))
        base_duration = float(spec.get("duration", _DEFAULT_NEED_PROTOCOL["duration"]))
        duration = max(10.0, base_duration * (0.5 + 0.5 * severity))
        label = spec.get("label", drive)
        message = spec.get("message", _DEFAULT_NEED_PROTOCOL["message"])
        priority = max(0.55, min(1.0, 0.6 + 0.35 * severity))
        expected_score = max(0.65, min(0.95, priority + 0.1))
        timestamp = (need or {}).get("timestamp") or time.time()

        llm_context = {
            "drive": drive,
            "level": level,
            "severity": severity,
            "slowdown": slowdown,
            "budgets": budgets,
            "duration_seconds": duration,
            "label": label,
            "message": message,
            "timestamp": timestamp,
        }

        llm_notes: Optional[str] = None
        llm_result = try_call_llm_dict(
            "orchestrator_needs",
            input_payload=llm_context,
            logger=logger,
        )
        if isinstance(llm_result, Mapping):
            protocol = str(llm_result.get("protocol") or "").strip()
            if protocol:
                llm_context["protocol"] = protocol
            duration_hint = llm_result.get("duration")
            if isinstance(duration_hint, str) and duration_hint.strip():
                llm_context["duration_hint"] = duration_hint.strip()
            intensity = llm_result.get("intensity")
            try:
                if intensity is not None:
                    slowdown = max(0.0, min(1.0, float(intensity)))
                    llm_context["slowdown"] = slowdown
            except (TypeError, ValueError):
                pass
            llm_message = llm_result.get("message")
            if isinstance(llm_message, str) and llm_message.strip():
                message = llm_message.strip()
                llm_context["message"] = message
            notes_val = llm_result.get("notes")
            if isinstance(notes_val, str) and notes_val.strip():
                llm_notes = notes_val.strip()

        action = {
            "type": "regulate_resources",
            "priority": priority,
            "payload": {
                "drive": drive,
                "level": level,
                "severity": severity,
                "slowdown": slowdown,
                "budgets": budgets,
                "duration": duration,
                "message": message,
                "timestamp": timestamp,
                "source": "homeostasis",
            },
        }

        if "protocol" in llm_context:
            action["payload"]["protocol"] = llm_context["protocol"]
        if "duration_hint" in llm_context:
            action["payload"]["duration_hint"] = llm_context["duration_hint"]
        if "slowdown" in llm_context:
            action["payload"]["slowdown"] = llm_context["slowdown"]
        if "message" in llm_context:
            action["payload"]["message"] = llm_context["message"]

        summary = f"{label.capitalize()} sous contrainte ({int(round(severity * 100))}% d'urgence)"
        frame = {
            "trigger": trigger,
            "source": "homeostasis",
            "text": message,
            "options": [
                {
                    "action": action,
                    "expected": {"score": expected_score},
                    "summary": (
                        "Budgets interactif={:.2f}, background={:.2f}".format(
                            budgets["interactive"], budgets["background"]
                        )
                    ),
                }
            ],
            "stop_rules": {},
        }

        need_context = {
            "drive": drive,
            "level": level,
            "severity": severity,
            "label": label,
            "budgets": budgets,
            "slowdown": slowdown,
        }

        if "protocol" in llm_context:
            need_context["protocol"] = llm_context["protocol"]
        if "duration_hint" in llm_context:
            need_context["duration_hint"] = llm_context["duration_hint"]
        if llm_notes:
            need_context["notes"] = llm_notes

        return {
            "action": action,
            "priority": priority,
            "expected_score": expected_score,
            "reason": message,
            "summary": summary,
            "topic": f"need::{drive}",
            "frame": frame,
            "need": need_context,
        }

    def _need_budget_factor(self, queue: str) -> float:
        now = time.time()
        factor = 1.0
        with self._need_directives_lock:
            active: List[Dict[str, Any]] = []
            for directive in self._need_directives:
                expires = float(directive.get("expires_at", 0.0))
                if expires <= now:
                    continue
                active.append(directive)
                budgets = directive.get("budgets") if isinstance(directive.get("budgets"), dict) else {}
                if queue in budgets:
                    try:
                        factor = min(factor, max(0.1, min(1.0, float(budgets[queue]))))
                    except Exception:
                        continue
            self._need_directives = active
        return max(0.1, min(1.0, factor))

    def register_need_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(directive or {})
        drive = str(data.get("drive") or "unknown")
        try:
            severity = max(0.0, min(1.0, float(data.get("severity", 0.0))))
        except Exception:
            severity = 0.0
        try:
            slowdown = max(0.0, min(1.0, float(data.get("slowdown", severity))))
        except Exception:
            slowdown = severity
        try:
            duration = max(5.0, float(data.get("duration", 30.0)))
        except Exception:
            duration = 30.0
        raw_budgets = data.get("budgets") if isinstance(data.get("budgets"), dict) else {}
        budgets: Dict[str, float] = {}
        for queue, factor in raw_budgets.items():
            try:
                budgets[str(queue)] = max(0.1, min(1.0, float(factor)))
            except Exception:
                continue

        now = time.time()
        expires_at = now + duration
        entry = {
            "drive": drive,
            "severity": severity,
            "slowdown": slowdown,
            "budgets": budgets,
            "expires_at": expires_at,
            "registered_at": now,
            "message": data.get("message"),
        }
        with self._need_directives_lock:
            self._need_directives = [
                item
                for item in self._need_directives
                if float(item.get("expires_at", 0.0)) > now
            ]
            self._need_directives.append(entry)

        applied_budgets: Dict[str, int] = {}
        for queue, factor in budgets.items():
            base = self._job_base_budgets.get(queue, 1)
            target = max(1, int(math.ceil(float(base) * factor)))
            self.job_manager.budgets.setdefault(queue, {})["max_running"] = target
            applied_budgets[queue] = target

        if slowdown > 0.0:
            try:
                self._homeostasis.register_global_slowdown(
                    slowdown,
                    meta={
                        "source": "need_directive",
                        "drive": drive,
                        "severity": severity,
                        "duration": duration,
                        "registered_at": now,
                    },
                )
            except Exception:
                pass

        try:
            self.telemetry.log(
                "homeostasis",
                "need_directive",
                {
                    "drive": drive,
                    "severity": severity,
                    "slowdown": slowdown,
                    "duration": duration,
                    "budgets": budgets,
                    "applied": applied_budgets,
                },
            )
        except Exception:
            pass

        return {
            "ok": True,
            "drive": drive,
            "applied_budgets": applied_budgets,
            "slowdown": slowdown,
            "expires_at": expires_at,
        }

    def _register_self_judgment_collectors(self):

        def _sj_from_feedback() -> List[Trigger]:
            items: List[Trigger] = []
            conf = getattr(self, "_sj_conf_cache", _DEFAULT_SJ_CONF)
            pe = float(getattr(self, "_last_prediction_error", 0.0))
            if pe >= conf["surprise_threshold"]:
                importance = conf["surprise_importance"]
                if getattr(self, "_last_contradiction", False):
                    importance += conf["contradiction_boost"]
                items.append(
                    Trigger(
                        TriggerType.SELF_JUDGMENT,
                        {
                            "source": "feedback",
                            "importance": min(1.0, importance),
                            "immediacy": conf["surprise_immediacy"],
                        },
                        {
                            "kind": "standard",
                            "reason": "surprise",
                            "prediction_error": pe,
                        },
                    )
                )
                self._last_prediction_error = 0.0
            self._last_contradiction = False
            return items

        def _sj_heartbeat() -> List[Trigger]:
            items: List[Trigger] = []
            conf = getattr(self, "_sj_conf_cache", _DEFAULT_SJ_CONF)
            self._idle_beat = int(getattr(self, "_idle_beat", 0)) + 1
            every = int(conf["heartbeat_every"])
            low_load = True
            if hasattr(self, "job_manager") and hasattr(self.job_manager, "is_low_load"):
                try:
                    low_load = bool(self.job_manager.is_low_load())
                except Exception:
                    low_load = True
            if low_load and self._idle_beat % max(1, every) == 0:
                items.append(
                    Trigger(
                        TriggerType.SELF_JUDGMENT,
                        {
                            "source": "heartbeat",
                            "importance": conf["heartbeat_importance"],
                            "immediacy": conf["heartbeat_immediacy"],
                        },
                        {"kind": "light", "reason": "periodic"},
                    )
                )
            return items

        def _sj_on_new_items() -> List[Trigger]:
            items: List[Trigger] = []
            queue: List[Dict[str, Any]] = list(getattr(self, "_sj_new_items_queue", []))
            K = 2
            drained = queue[:K]
            self._sj_new_items_queue = queue[K:]
            for item in drained:
                meta = {"source": "consolidation", "importance": 0.6, "immediacy": 0.5}
                payload = {"kind": "standard", "reason": "new_item", "target": item}
                items.append(Trigger(TriggerType.SELF_JUDGMENT, meta, payload))
            return items

        if hasattr(self, "trigger_bus") and hasattr(self.trigger_bus, "register"):
            self.trigger_bus.register(_sj_from_feedback)
            self.trigger_bus.register(_sj_heartbeat)
            self.trigger_bus.register(_sj_on_new_items)
        else:
            self._collectors = getattr(self, "_collectors", [])
            self._collectors += [_sj_from_feedback, _sj_heartbeat, _sj_on_new_items]

    # --- Trigger collectors -------------------------------------------------
    def _followup_collector(self) -> List[Trigger]:
        if not self._pending_triggers:
            return []
        pending = list(self._pending_triggers)
        self._pending_triggers = []
        return pending

    def _user_collector(self) -> List[Trigger]:
        txt = getattr(self, "last_user_msg", None)
        if not txt:
            return []

        low = txt.lower()
        if THREAT_PATTERN.search(txt):
            return [
                Trigger(
                    TriggerType.THREAT,
                    {
                        "source": "user",
                        "importance": 1.0,
                        "immediacy": 1.0,
                        "reversibility": 0.2,
                        "effort": 0.2,
                        "uncertainty": 0.1,
                    },
                    {"text": txt, "label": "shutdown", "conf": 0.95},
                )
            ]

        label, conf = ("inform", 0.5)
        if self.intent_model is not None:
            try:
                label, conf = self.intent_model.predict(txt)
            except Exception:
                pass

        fallback_label, fallback_conf = self._intent_fallback.predict(txt)
        if conf >= 0.8:
            try:
                self._intent_fallback.update(txt, label)
            except Exception:
                pass
        if conf < 0.45:
            label, conf = fallback_label, max(conf, fallback_conf)
        elif fallback_label == label:
            conf = max(conf, fallback_conf)
        elif fallback_conf > conf + 0.15:
            blended = 0.7 * fallback_conf + 0.3 * conf
            label, conf = fallback_label, blended

        MAP = {
            "ask_info": "GOAL",
            "request": "GOAL",
            "create": "GOAL",
            "send": "GOAL",
            "summarize": "GOAL",
            "classify": "GOAL",
            "plan": "GOAL",
            "set_goal": "GOAL",
            "greet": "SIGNAL",
            "thanks": "SIGNAL",
            "bye": "SIGNAL",
            "meta_help": "SIGNAL",
            "inform": "SIGNAL",
            "feedback": "SIGNAL",
            "shutdown": "THREAT",
            "danger": "THREAT",
            "threat": "THREAT",
        }

        QUESTION_LABELS = {"ask_info"}
        COMMAND_LABELS = {"request", "create", "send", "summarize", "classify", "plan", "set_goal"}

        is_question = txt.strip().endswith("?") or label in QUESTION_LABELS
        tname = MAP.get(label, "GOAL" if is_question else "SIGNAL")
        ttype = TriggerType[tname]

        meta = {
            "source": "user",
            "importance": 0.9 if tname == "GOAL" else 0.6,
            "immediacy": 0.9 if tname == "GOAL" else 0.4,
            "reversibility": 1.0,
            "effort": 0.5 if tname == "GOAL" else 0.2,
            "uncertainty": max(0.0, 1.0 - float(conf or 0.0)),
        }

        payload = {"text": txt, "label": label, "conf": conf}
        if tname == "GOAL":
            if is_question:
                payload["goal_kind"] = "AnswerUserQuestion"
            elif label in COMMAND_LABELS:
                payload["goal_kind"] = "ExecuteUserCommand"
            else:
                payload["goal_kind"] = label

        return [Trigger(ttype, meta, payload)]

    def _curiosity_collector(self) -> List[Trigger]:
        goals = self.cognition.meta.pop_new_learning_goals(max_n=4)
        if not goals:
            return []
        triggers: List[Trigger] = []
        for goal in goals:
            text = goal.get("text") or goal.get("desc") or goal.get("description") or "objectif"
            importance = float(goal.get("priority", 0.7))
            triggers.append(
                Trigger(
                    TriggerType.GOAL,
                    {
                        "source": "curiosity",
                        "importance": max(0.3, min(1.0, importance)),
                        "probability": float(goal.get("probability", 0.6)),
                        "reversibility": float(goal.get("reversibility", 1.0)),
                        "effort": float(goal.get("effort", 0.4)),
                        "uncertainty": float(goal.get("uncertainty", 0.3)),
                    },
                    payload={
                        "goal_kind": goal.get("goal_kind", "CuriosityLearning"),
                        "goal_id": goal.get("id"),
                        "text": text,
                        "goal": goal,
                    },
                )
            )
        return triggers

    def _homeostasis_collector(self) -> List[Trigger]:
        need = self.cognition.homeostasis.poll_need()
        if not need:
            return []
        try:
            level = max(0.0, min(1.0, float(need.get("level", 0.0))))
        except Exception:
            level = 0.0
        severity = max(0.0, min(1.0, 1.0 - level))
        importance = max(0.5, min(1.0, 0.55 + 0.35 * severity))
        immediacy = max(0.3, min(1.0, 0.45 + 0.45 * severity))
        payload = {
            "need": {
                **need,
                "severity": severity,
                "timestamp": time.time(),
            }
        }
        return [
            Trigger(
                TriggerType.NEED,
                {
                    "source": "homeostasis",
                    "importance": importance,
                    "immediacy": immediacy,
                    "effort": 0.2,
                    "severity": severity,
                },
                payload=payload,
            )
        ]

    def _emotion_collector(self) -> List[Trigger]:
        emo = self.emotions.peek_peaks()
        if not emo:
            return []
        imp = 0.6 + 0.3 * abs(emo.get("valence", 0.0))
        return [
            Trigger(
                TriggerType.EMOTION,
                {
                    "source": "emotion",
                    "importance": imp,
                    "immediacy": 0.5,
                    "uncertainty": 0.3,
                },
                payload=emo,
            )
        ]

    def _memory_assoc_collector(self) -> List[Trigger]:
        assoc = self.memory.episodic.pop_salient_associations(max_n=2)
        if not assoc:
            return []
        return [
            Trigger(
                TriggerType.MEMORY_ASSOC,
                {
                    "source": "memory",
                    "importance": 0.4,
                    "immediacy": 0.2,
                    "uncertainty": 0.6,
                },
                payload=a,
            )
            for a in assoc
        ]

    def _signal_collector(self) -> List[Trigger]:
        signals = self.io.perception.pop_signals(max_n=4)
        if not signals:
            return []
        return [
            Trigger(
                TriggerType.SIGNAL,
                {
                    "source": "system",
                    "importance": 0.5,
                    "immediacy": 0.3,
                    "effort": 0.2,
                },
                payload=s,
            )
            for s in signals
        ]

    def _habit_collector(self) -> List[Trigger]:
        cue = self.cognition.habits.poll_context_cue()
        if not cue:
            return []
        importance = float(cue.get("strength", 0.6))
        return [
            Trigger(
                TriggerType.HABIT,
                {
                    "source": "habit",
                    "importance": importance,
                    "immediacy": 0.4,
                    "habit_strength": cue.get("strength", 0.7),
                    "habit_name": cue.get("name"),
                    "habit_period": cue.get("period"),
                },
                payload=cue,
            )
        ]

    # --- Legacy helper cycles (kept for compatibility) ---------------------
    def observe(self, user_msg: Optional[str] = None):
        if user_msg:
            self.io.perception.ingest_user_message(user_msg, author="user")
        else:
            self.memory.store.add({"kind": "tick", "text": "idle cycle", "ts": time.time()})

    def consolidate(self):
        self.memory.consolidator.maybe_consolidate()

    def emotion_homeostasis_cycle(self):
        recent = self.memory.store.get_recent(60)
        self.emotions.update_from_recent_memories(recent)
        self.emotions.modulate_homeostasis(self._homeostasis)
        r_intr = self._homeostasis.compute_intrinsic_reward(info_gain=0.5, progress=0.5)
        r_extr = self._homeostasis.compute_extrinsic_reward_from_memories("")
        return r_intr, r_extr

    def meta_cycle(self):
        assessment = self._meta.assess_understanding()
        goals = self.cognition.meta.pop_new_learning_goals(max_n=2)
        return assessment, goals

    def planning_cycle(self):
        self._planner.plan_for_goal("understand_humans", "Comprendre les humains")
        plan = self._planner.state["plans"].get("understand_humans")
        if plan and not plan.get("steps"):
            self._planner.add_step(
                "understand_humans", "Observer un échange et extraire intentions"
            )
            self._planner.add_step(
                "understand_humans", "Tester une hypothèse d'intention par question"
            )

    def action_cycle(self):
        picked = None
        lock = getattr(self._planner, "lock", None)
        if lock is None:
            lock = getattr(self._planner, "_lock", None)
        context = lock if lock is not None else nullcontext()
        with context:
            plan_ids = list(self._planner.state.get("plans", {}).keys())
        for gid in plan_ids:
            if gid.startswith("learn_"):
                picked = gid
                break
        if not picked:
            picked = "understand_humans"
        step = self._planner.pop_next_action(picked)
        if not step:
            return
        desc = str(step.get("desc", "")).lower()
        action: Dict[str, Any]
        if "poser" in desc or "question" in desc:
            action = {
                "type": "communicate",
                "text": "Peux-tu me décrire ton émotion actuelle et pourquoi ?",
                "target": "human",
            }
        elif "observer" in desc:
            action = {"type": "simulate", "what": "observe_exchange"}
        else:
            action = {"type": "simulate", "desc": desc}
        res = self.io.action.execute(action)
        self._planner.mark_action_done(picked, step["id"], result=res)

    def proposals_cycle(self):
        props = self._proposer.run_once_now()
        for p in props:
            try:
                self.self_model.apply_proposal(p, self._policy_engine)
            except Exception as exc:
                self.memory.store.add(
                    {"kind": "error", "text": f"Proposal error: {exc}", "ts": time.time()}
                )

    def _llm_cycle_recommendations(self, payload: Dict[str, Any]) -> Optional[Mapping[str, Any]]:
        if not _llm_enabled():
            return None

        try:
            response = _llm_manager().call_dict(
                "orchestrator_service",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            logger.debug("LLM orchestrator recommendations unavailable", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        return dict(response)

    def _collect_kernel_alerts_from_triggers(self, scored_triggers: Iterable["ScoredTrigger"]) -> None:
        for scored in scored_triggers:
            trigger = getattr(scored, "trigger", None)
            if trigger is None:
                continue
            meta = getattr(trigger, "meta", None) or {}
            payload = getattr(trigger, "payload", None) or {}
            alert_meta = meta.get("phenomenal_alert") or payload.get("phenomenal_alert")
            if not isinstance(alert_meta, dict) and isinstance(meta.get("system_alert"), dict):
                alert_meta = meta.get("system_alert")
            if isinstance(alert_meta, dict):
                kind = str(
                    alert_meta.get("kind")
                    or meta.get("kind")
                    or payload.get("kind")
                    or trigger.type.name.lower()
                )
                intensity = float(
                    alert_meta.get(
                        "intensity",
                        meta.get("immediacy", getattr(scored, "priority", 0.8)),
                    )
                )
                slowdown = alert_meta.get("slowdown")
                duration = alert_meta.get("duration", alert_meta.get("ttl", 30.0))
                timestamp = alert_meta.get("timestamp")
                try:
                    self._phenomenal_kernel.register_alert(
                        kind,
                        intensity=intensity,
                        slowdown=slowdown,
                        duration=duration,
                        timestamp=timestamp,
                    )
                except Exception:
                    continue

    def _update_physiology_from_system(self, resource_monitor: Optional[Any]) -> List[Dict[str, Any]]:
        if not hasattr(self, "_system_monitor") or self._system_monitor is None:
            return []

        try:
            snapshot = self._system_monitor.poll()
        except Exception as exc:
            self.telemetry.log("system_monitor", "poll_error", {"error": str(exc)})
            return []

        if not snapshot:
            return []

        self._last_system_snapshot = dict(snapshot)

        if resource_monitor and hasattr(resource_monitor, "register_machine_snapshot"):
            try:
                resource_monitor.register_machine_snapshot(snapshot)
            except Exception:
                pass

        try:
            updated_drives = self._homeostasis.integrate_system_metrics(snapshot)
        except Exception as exc:
            self.telemetry.log("system_monitor", "homeostasis_error", {"error": str(exc)})
            updated_drives = {}

        drive_levels: Dict[str, float] = {}
        for name in ("energy", "respiration", "thermal_regulation", "memory_balance"):
            if name in updated_drives:
                drive_levels[name] = float(updated_drives[name])
            else:
                drive_levels[name] = float(self._homeostasis.state["drives"].get(name, 0.5))

        severity: Dict[str, float] = {
            name: max(0.0, 1.0 - float(value)) for name, value in drive_levels.items()
        }

        alerts: List[Dict[str, Any]] = []
        slowdown = 0.0

        def _maybe_alert(drive: str, kind: str, *, threshold: float = 0.35, slowdown_bias: float = 0.15) -> None:
            nonlocal slowdown
            sev = severity.get(drive, 0.0)
            if sev < threshold:
                return
            intensity = max(0.0, min(1.0, sev + 0.05))
            slowdown_val = max(0.0, min(1.0, sev + slowdown_bias))
            alerts.append(
                {
                    "kind": kind,
                    "intensity": intensity,
                    "slowdown": slowdown_val,
                    "timestamp": snapshot.get("timestamp"),
                }
            )
            slowdown = max(slowdown, slowdown_val)

        _maybe_alert("thermal_regulation", "system_overheat", slowdown_bias=0.25)
        _maybe_alert("respiration", "cpu_overload", slowdown_bias=0.2)
        _maybe_alert("energy", "energy_depletion", slowdown_bias=0.15)
        _maybe_alert("memory_balance", "memory_pressure", slowdown_bias=0.1, threshold=0.4)

        if any(val >= 0.35 for val in severity.values()):
            summary = {
                "timestamp": snapshot.get("timestamp"),
                "cpu_load": snapshot.get("cpu", {}).get("load") if isinstance(snapshot.get("cpu"), dict) else None,
                "ram_percent": snapshot.get("memory", {}).get("percent") if isinstance(snapshot.get("memory"), dict) else None,
                "gpu_temp": snapshot.get("gpu", {}).get("temp_c") if isinstance(snapshot.get("gpu"), dict) else None,
                "gpu_util": snapshot.get("gpu", {}).get("util_pct") if isinstance(snapshot.get("gpu"), dict) else None,
                "power_draw": snapshot.get("power", {}).get("draw_w") if isinstance(snapshot.get("power"), dict) else None,
                "severity": severity,
            }
            self.telemetry.log("system_monitor", "alert", summary)

        if slowdown >= 0.3:
            now = time.time()
            if (
                slowdown - self._last_system_slowdown > 0.05
                or now - self._last_system_slowdown_ts > 15.0
            ):
                try:
                    self._homeostasis.register_global_slowdown(
                        slowdown,
                        meta={
                            "source": "system_monitor",
                            "severity": severity,
                            "timestamp": snapshot.get("timestamp"),
                        },
                    )
                except Exception:
                    pass
                self._last_system_slowdown = slowdown
                self._last_system_slowdown_ts = now
        else:
            self._last_system_slowdown = slowdown

        self._pending_system_alerts = alerts
        return alerts

    # --- Cycle principal ----------------------------------------------------
    def run_once_cycle(self, user_msg: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            prioritizer = getattr(self.arch, "prioritizer", None)
            if prioritizer is not None:
                prioritizer.reprioritize_all()
        except Exception:
            pass

        self.scheduler.tick()
        self.job_manager.drain_to_memory(self._memory_store)

        try:
            self._sj_conf_cache = self._sj_config_model.current_config()
        except Exception:
            self._sj_conf_cache = dict(_DEFAULT_SJ_CONF)

        self.last_user_msg = user_msg
        if user_msg:
            self.observe(user_msg)
        else:
            self.observe(None)

        emo_state = self.emotions.read()
        emotion_context = {}
        if emo_state is not None:
            emotion_context = {
                "valence": float(getattr(emo_state, "valence", 0.0)),
                "arousal": float(getattr(emo_state, "arousal", 0.0)),
            }
        valence = getattr(emo_state, "valence", 0.0) if emo_state else 0.0
        resource_monitor = None
        try:
            resource_monitor = self._meta.cognitive_monitoring.get("resource_monitoring")
        except Exception:
            resource_monitor = None
        system_alerts = self._update_physiology_from_system(resource_monitor)
        scored = self.trigger_bus.collect_and_score(valence=valence)
        self._collect_kernel_alerts_from_triggers(scored)
        selected: List[Any] = []
        for st in scored:
            if (
                st.priority >= 0.95
                and st.trigger.type is TriggerType.THREAT
                and st.trigger.meta.get("immediacy", 0.0) >= 0.8
            ):
                selected = [st]
                break
        if not selected:
            selected = scored[:3]

        urgent = any(
            item.trigger.type is TriggerType.THREAT and item.trigger.meta.get("immediacy", 0.0) >= 0.7
            for item in selected
        )
        novelty = abs(self._last_prediction_error)
        belief = 0.5
        if hasattr(self.self_model, "confidence") and callable(getattr(self.self_model, "confidence")):
            try:
                belief = float(self.self_model.confidence({}))  # type: ignore[call-arg]
            except Exception:
                belief = 0.5
        progress_signal = float(self._homeostasis.state.get("intrinsic_reward", 0.0))
        extrinsic_signal = float(self._homeostasis.state.get("extrinsic_reward", 0.0))
        hedonic_signal = float(self._homeostasis.state.get("hedonic_reward", 0.0))
        fatigue = None
        if resource_monitor and hasattr(resource_monitor, "assess_fatigue"):
            try:
                fatigue = float(resource_monitor.assess_fatigue(self._meta.metacognitive_history, self.arch))
            except Exception:
                fatigue = None
        kernel_state = self._phenomenal_kernel.update(
            emotional_state=emotion_context,
            novelty=novelty,
            belief=belief,
            progress=progress_signal,
            extrinsic_reward=extrinsic_signal,
            hedonic_signal=hedonic_signal,
            fatigue=fatigue,
            alerts=system_alerts,
        )
        self.phenomenal_kernel_state.clear()
        self.phenomenal_kernel_state.update(kernel_state)
        self.phenomenal_kernel_state.setdefault("mode", self.current_mode)
        if self._last_system_snapshot:
            self.phenomenal_kernel_state["system_snapshot"] = dict(self._last_system_snapshot)
        setattr(self.arch, "phenomenal_kernel_state", self.phenomenal_kernel_state)
        previous_mode = getattr(self, "current_mode", "travail")
        mode_info = self._mode_manager.update(self.phenomenal_kernel_state, urgent=urgent)
        self.current_mode = mode_info.get("mode", previous_mode)
        self.phenomenal_kernel_state["mode"] = self.current_mode
        self.phenomenal_kernel_state["flanerie_ratio"] = mode_info.get("flanerie_ratio")
        self.phenomenal_kernel_state["flanerie_budget_remaining"] = mode_info.get("flanerie_budget_remaining")
        self.phenomenal_kernel_state["urgent"] = urgent
        if getattr(self, "phenomenal_journal", None) is not None:
            try:
                justification = None
                interpretation = self.phenomenal_kernel_state.get("llm_interpretation")
                if isinstance(interpretation, dict):
                    justification = interpretation.get("justification")
                self.phenomenal_journal.record_mode_transition(
                    previous_mode=previous_mode,
                    new_mode=self.current_mode,
                    kernel_state={
                        key: value
                        for key, value in self.phenomenal_kernel_state.items()
                        if isinstance(value, (int, float, str, bool, list, dict))
                    },
                    reason=justification,
                )
            except Exception:
                pass
        slowdown = float(self.phenomenal_kernel_state.get("global_slowdown", 0.0) or 0.0)
        try:
            setattr(self.arch, "global_slowdown", slowdown)
        except Exception:
            pass
        try:
            setattr(self.arch, "current_mode", self.current_mode)
        except Exception:
            pass
        slowdown_factor = max(0.1, min(1.0, 1.0 - 0.7 * slowdown))
        for queue, base in self._job_base_budgets.items():
            factor = slowdown_factor * self._need_budget_factor(queue)
            if self.current_mode == "flanerie" and queue == "background":
                factor *= 0.5
            target = max(1, math.ceil(base * factor))
            self.job_manager.budgets.setdefault(queue, {})["max_running"] = target
        hedonic_gain = float(self.phenomenal_kernel_state.get("hedonic_reward", 0.0))
        budget_remaining = float(
            self.phenomenal_kernel_state.get("flanerie_budget_remaining", 0.0) or 0.0
        )
        if (
            self.current_mode == "flanerie"
            and hedonic_gain >= self._hedonic_reward_min_gain
            and budget_remaining > 0.0
            and (time.time() - self._last_intrinsic_reward_ts) > self._intrinsic_reward_cooldown
        ):
            reward_engine = getattr(self.arch, "reward_engine", None)
            if reward_engine and hasattr(reward_engine, "register_intrinsic_reward"):
                try:
                    reward_engine.register_intrinsic_reward(
                        "phenomenal_kernel",
                        hedonic_gain,
                        context={"phenomenal_kernel": dict(self.phenomenal_kernel_state), "mode": self.current_mode},
                    )
                    self._last_intrinsic_reward_ts = time.time()
                except Exception:
                    pass

        contexts: List[Dict[str, Any]] = []
        for scored_trigger in selected:
            ctx = self._run_pipeline(scored_trigger.trigger)
            contexts.append(ctx)

        self.consolidate()
        try:
            self.phenomenal_recall.prime_for_digest(
                self.memory.store,
                kernel_state=self.phenomenal_kernel_state,
                homeostasis=getattr(self._homeostasis, "state", {}),
            )
        except Exception:
            pass
        r_intr, r_extr = self.emotion_homeostasis_cycle()
        try:
            self.phenomenal_journal.audit_against(
                "homeostasis",
                {
                    "intrinsic_reward": float(r_intr),
                    "extrinsic_reward": float(r_extr),
                    "hedonic_reward": float(self._homeostasis.state.get("hedonic_reward", 0.0)),
                },
                tolerance=0.2,
            )
        except Exception:
            pass
        try:
            self.phenomenal_questioner.maybe_question(self.phenomenal_kernel_state)
        except Exception:
            pass
        assessment, _ = self.meta_cycle()
        self.planning_cycle()
        self.action_cycle()
        self.proposals_cycle()

        selected_snapshot = []
        for item in selected:
            trigger = getattr(item, "trigger", None)
            trigger_type = getattr(trigger, "type", None)
            trigger_name = getattr(trigger_type, "name", str(trigger_type)) if trigger_type else None
            selected_snapshot.append(
                {
                    "type": trigger_name,
                    "priority": getattr(item, "priority", None),
                }
            )

        llm_payload = {
            "mode": self.current_mode,
            "urgent": urgent,
            "system_alerts": system_alerts,
            "assessment": assessment,
            "phenomenal_state": {
                key: value
                for key, value in self.phenomenal_kernel_state.items()
                if isinstance(value, (int, float, str, bool))
            },
            "selected_triggers": selected_snapshot,
            "contexts_count": len(contexts),
            "uncertainty": assessment.get("uncertainty"),
        }
        llm_recommendations = self._llm_cycle_recommendations(llm_payload)
        if llm_recommendations:
            self._last_llm_recommendations = llm_recommendations
            try:
                self.telemetry.log("llm", "cycle_recommendations", llm_recommendations)
            except Exception:
                logger.debug("Telemetry logging for LLM recommendations failed", exc_info=True)

        learning_rate = 0.5
        self._evolution.log_cycle(
            intrinsic=r_intr,
            extrinsic=r_extr,
            learning_rate=learning_rate,
            uncertainty=assessment.get("uncertainty", 0.5),
        )
        if self._evolution.state.get("cycle_count", 0) % 20 == 0:
            notes = self._evolution.propose_macro_adjustments()
            if notes:
                self.memory.store.add(
                    {"kind": "strategy", "text": " | ".join(notes), "ts": time.time()}
                )

        try:
            infer_where_and_apply(self, threshold=0.70, stable_cycles=2)
        except Exception:
            pass

        self._mission_tick = getattr(self, "_mission_tick", 0) + 1
        if self._mission_tick % 5 == 0:
            try:
                res = recommend_and_apply_mission(self, threshold=0.75, delta_gate=0.10)
                if res.get("status") == "needs_confirmation":
                    proposal = {
                        "kind": "mission_proposal",
                        "best": res.get("best"),
                        "second": res.get("second"),
                        "delta": res.get("delta"),
                        "ts": time.time(),
                    }
                    try:
                        self._sj_new_items_queue.append(proposal)
                    except Exception:
                        pass
                    if hasattr(self.memory, "store") and hasattr(self.memory.store, "add"):
                        try:
                            payload = dict(proposal)
                            payload["status"] = "needs_confirmation"
                            self.memory.store.add(payload)
                        except Exception:
                            pass
            except Exception:
                pass

        self._principles_tick = getattr(self, "_principles_tick", 0) + 1
        if self._principles_tick == 1 or self._principles_tick % 10 == 0:
            try:
                run_and_apply_principles(self, require_confirmation=True)
            except Exception:
                pass

        self._preferences_tick = getattr(self, "_preferences_tick", 0) + 1
        if self._preferences_tick % 6 == 0:
            try:
                apply_preferences_if_confident(self, threshold=0.75)
            except Exception:
                pass

        return contexts

    # --- Pipeline -----------------------------------------------------------
    def _submit_for_mode(self, mode: ActMode, action: Dict[str, Any], meta: Dict[str, Any], prio: float) -> str:
        if mode is ActMode.REFLEX:
            return self.job_manager.submit(
                kind="action.reflex",
                fn=self._run_action,
                args={"action": action, "meta": meta, "mode": "reflex"},
                queue="interactive",
                priority=1.0,
                timeout_s=2.0,
            )
        if mode is ActMode.HABIT:
            return self.job_manager.submit(
                kind="action.habit",
                fn=self._run_action,
                args={"action": action, "meta": meta, "mode": "habit"},
                queue="interactive",
                priority=max(0.6, prio),
                timeout_s=10.0,
            )
        return self.job_manager.submit(
            kind="action.deliberate",
            fn=self._run_action,
            args={"action": action, "meta": meta, "mode": "deliberate"},
            queue="background",
            priority=min(0.95, prio),
            timeout_s=300.0,
        )

    def record_knowledge_milestone(self, topic: str, ctx: Optional[Dict[str, Any]] = None):
        """Capture l'état courant des croyances et projette des follow-ups d'apprentissage."""

        ctx = ctx or {}

        beliefs_now: List[Dict[str, Any]] = []
        if hasattr(self.memory, "semantic") and hasattr(self.memory.semantic, "export_topic_beliefs"):
            try:
                beliefs_now = list(self.memory.semantic.export_topic_beliefs(topic=topic)) or []
            except Exception:
                beliefs_now = []

        snap_id = None
        if hasattr(self, "timeline") and hasattr(self.timeline, "snapshot"):
            try:
                snap_id = self.timeline.snapshot(topic, beliefs_now)
            except Exception:
                snap_id = None

        delta_event = None
        prev = self._last_beliefs_by_topic.get(topic)
        if prev is not None and hasattr(self.timeline, "delta"):
            try:
                delta_event = self.timeline.delta(topic, prev, beliefs_now)
                if hasattr(self.memory, "store") and hasattr(self.memory.store, "add") and delta_event:
                    self.memory.store.add(delta_event)
            except Exception:
                delta_event = None

        self._last_beliefs_by_topic[topic] = beliefs_now

        gaps: List[str] = []
        if isinstance(ctx.get("gaps"), list):
            gaps.extend([str(g) for g in ctx["gaps"]])

        if delta_event:
            low_conf = [
                b.get("stmt")
                for b in delta_event.get("updated", [])
                if float(b.get("conf", 0.0)) < 0.5
            ]
            gaps.extend([g for g in low_conf if g])

        try:
            if hasattr(self.memory, "concepts") and hasattr(self.memory.concepts, "recent_gaps"):
                recent = self.memory.concepts.recent_gaps() or []
                gaps.extend([str(g) for g in recent])
        except Exception:
            pass

        gaps = [g for g in {g.strip() for g in gaps} if g]

        projected = []
        if hasattr(self, "timeline") and hasattr(self.timeline, "project") and gaps:
            try:
                projected = self.timeline.project(topic, gaps=gaps)
            except Exception:
                projected = [
                    {"goal_kind": "LearnConcept", "topic": topic, "concept": g}
                    for g in gaps
                ]
        elif gaps:
            projected = [
                {"goal_kind": "LearnConcept", "topic": topic, "concept": g}
                for g in gaps
            ]

        followups: List[Trigger] = []
        for payload in projected[:3]:
            followups.append(
                Trigger(
                    TriggerType.GOAL,
                    {
                        "source": "timeline",
                        "importance": 0.6,
                        "immediacy": 0.3,
                        "reversibility": 1.0,
                    },
                    {
                        "goal_kind": payload.get("goal_kind", "LearnConcept"),
                        "topic": payload.get("topic", topic),
                        "concept": payload.get("concept"),
                    },
                )
            )

        for trig in followups:
            try:
                if hasattr(self, "trigger_bus") and hasattr(self.trigger_bus, "push"):
                    self.trigger_bus.push(trig)
                else:
                    self._pending_triggers.append(trig)
            except Exception:
                pass

        return {
            "snapshot_id": snap_id,
            "delta_written": bool(delta_event),
            "delta_event": delta_event,
            "projected_goals": [t.payload for t in followups],
            "projected_plan": projected,
            "gaps": gaps,
            "beliefs": beliefs_now,
        }

    def _run_action(self, ctx: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        result = self.io.action.execute(args.get("action"))
        result = result or {}
        result["mode"] = args.get("mode")
        result["meta"] = args.get("meta", {})
        return result

    def _run_pipeline(self, trigger: Trigger) -> Dict[str, Any]:
        if self.immediate_question_blocked:
            meta = trigger.meta or {}
            try:
                immediacy = float(meta.get("immediacy", 0.0))
            except (TypeError, ValueError):
                immediacy = 0.0
            if immediacy >= 0.75:
                logger.info(
                    "Déclencheur %s ignoré (questions immédiates bloquées)",
                    trigger.type.name,
                )
                return {
                    "meta": meta,
                    "payload": trigger.payload or {},
                    "blocked": True,
                    "reason": "immediate_questions_backlog",
                }
        family = self.trigger_router.select_pipeline(trigger)
        trigger_meta = trigger.meta or {}
        trigger_payload = trigger.payload or {}
        policy_ctx = {
            "trigger_type": trigger.type.name,
            "meta": trigger_meta,
            "payload": trigger_payload,
        }
        selection = PIPELINE_POLICY.select(family, policy_ctx)
        pipe = selection.name if selection.name in REGISTRY else family
        steps = REGISTRY.get(pipe) or REGISTRY.get(family) or []
        ctx: Dict[str, Any] = {
            "meta": trigger_meta,
            "payload": trigger_payload,
            "obs": None,
            "scratch": {},
            "decision": None,
            "expected": {"score": 1.0},
            "obtained": None,
            "mode": None,
        }

        self._idle_beat = 0
        self._current_trigger = trigger
        payload = trigger_payload
        self._current_topic = (
            payload.get("topic")
            or (trigger.meta or {}).get("topic")
            or payload.get("text")
        )
        ctx["topic"] = self._current_topic
        ctx["from"] = (
            payload.get("from")
            or payload.get("speaker")
            or payload.get("author")
            or (trigger.meta or {}).get("from")
            or "user"
        )
        raw_text = payload.get("text") or payload.get("content")
        if isinstance(raw_text, bytes):
            try:
                raw_text = raw_text.decode("utf-8", errors="ignore")
            except Exception:
                raw_text = str(raw_text)
        ctx["text"] = str(raw_text) if raw_text is not None else ""
        ctx["msg_ref"] = payload.get("msg_ref") or (trigger.meta or {}).get("msg_ref")
        self._current_decision_id = None
        self._current_trace_id = None

        monitor = getattr(self, "thinking_monitor", None)
        decision_journal = getattr(self, "decision_journal", None)
        reasoning_ledger = getattr(self, "reasoning_ledger", None)
        understanding_agg = getattr(self, "understanding_agg", None)
        selfhood = getattr(self, "selfhood", None)
        policy_engine = getattr(self.core, "policy", None)

        telemetry = getattr(self, "telemetry", None)
        pipeline_start = time.time()
        stage_metrics: List[Dict[str, Any]] = []
        ctx["scratch"]["pipeline"] = {
            "family": selection.family,
            "variant": pipe,
            "reason": selection.reason,
            "metrics": stage_metrics,
        }
        if telemetry:
            try:
                telemetry.log(
                    "pipeline_selected",
                    "cognition",
                    {
                        "family": selection.family,
                        "pipeline": pipe,
                        "reason": selection.reason,
                        "trigger_type": trigger.type.name,
                        "meta": {
                            k: trigger_meta.get(k)
                            for k in ("importance", "immediacy", "salience")
                            if k in trigger_meta
                        },
                    },
                )
            except Exception:
                pass

        if selection.family == "HABIT" and isinstance(trigger_payload, dict):
            habit_payload = self._build_habit_action_payload(trigger_payload)
            ctx["payload"] = dict(habit_payload)
            scratch = ctx.setdefault("scratch", {})
            scratch["habit_payload"] = habit_payload

            habit_raw = habit_payload.get("habit")
            habit_info = habit_raw if isinstance(habit_raw, dict) else {}
            habit_name = habit_info.get("name") or habit_payload.get("name")
            habit_desc = habit_info.get("description") or habit_payload.get("description") or ""

            if habit_name:
                ctx["topic"] = f"habit::{habit_name}"
            if habit_desc:
                ctx["text"] = habit_desc

            expected_score = habit_payload.get("strength") or trigger.meta.get("importance") or 0.7
            try:
                expected_score = max(0.05, min(1.0, float(expected_score)))
            except Exception:
                expected_score = 0.7

            habit_action = {
                "type": habit_payload.get("type", "habit_reflection"),
                "payload": dict(habit_payload),
                "priority": expected_score,
            }

            scratch["frame"] = {
                "trigger": trigger,
                "source": "habit",
                "text": habit_desc or habit_name or "",
                "options": [
                    {
                        "action": habit_action,
                        "expected": {"score": expected_score},
                    }
                ],
                "stop_rules": {},
            }

            if not ctx.get("reason"):
                ctx["reason"] = f"habit::{habit_name or 'routine'}"

            scratch.setdefault("priority", expected_score)

        elif selection.family == "NEED":
            need_context = self._prepare_need_action(trigger, trigger_payload)
            ctx["payload"] = dict(need_context["action"])
            scratch = ctx.setdefault("scratch", {})
            scratch["frame"] = need_context["frame"]
            scratch["priority"] = need_context["priority"]
            scratch["need"] = need_context["need"]
            ctx["reason"] = need_context["reason"]
            ctx["summary"] = need_context["summary"]
            ctx["topic"] = need_context["topic"]
            ctx["expected"] = {"score": need_context["expected_score"]}

        for step in steps:
            stg = step["stage"]
            skip_condition = step.get("skip_if")
            skipped = False
            skip_reason: Optional[str] = None
            if skip_condition:
                if callable(skip_condition):
                    try:
                        skipped = bool(skip_condition(ctx))
                    except Exception:
                        skipped = False
                else:
                    skipped, skip_reason = should_skip_stage(skip_condition, ctx)
            if skipped:
                stage_metrics.append({"stage": stg.name, "skipped": True, "reason": skip_reason})
                continue
            stage_start = time.time()
            mem_before = _get_process_memory_kb()
            try:
                if stg is Stage.PERCEIVE:
                    ctx["obs"] = self.io.perception.observe(trigger)
                    if isinstance(ctx["obs"], dict):
                        obs_text = ctx["obs"].get("text") or ctx["obs"].get("content")
                        if obs_text and not ctx.get("text"):
                            ctx["text"] = str(obs_text)
                    try:
                        self.self_model.record_interaction(
                            {
                                "with": ctx.get("from", "user"),
                                "when": time.time(),
                                "topic": ctx.get("topic") or "__generic__",
                                "summary": ctx.get("summary")
                                or (ctx.get("text", "")[:120] if ctx.get("text") else ""),
                                "ref": ctx.get("msg_ref"),
                            }
                        )
                    except Exception:
                        pass
                elif stg is Stage.ATTEND:
                    obs = ctx.get("obs")
                    if not ctx.get("text") and isinstance(obs, dict):
                        maybe_text = obs.get("text") or obs.get("content")
                        if maybe_text:
                            ctx["text"] = str(maybe_text)
                    summary = ctx.get("text") or ""
                    if not summary and obs is not None:
                        if isinstance(obs, dict):
                            summary = str(
                                obs.get("summary")
                                or obs.get("description")
                                or obs.get("text")
                                or obs.get("content")
                                or obs
                            )
                        else:
                            summary = str(obs)
                    ctx["summary"] = summary[:240]
                    meta = trigger.meta or {}
                    payload = trigger.payload or {}
                    try:
                        importance = float(meta.get("importance", 0.5))
                    except Exception:
                        importance = 0.5
                    try:
                        immediacy = float(meta.get("immediacy", 0.0))
                    except Exception:
                        immediacy = 0.0
                    salience = importance
                    if isinstance(payload, dict):
                        try:
                            salience = max(salience, float(payload.get("salience", salience)))
                        except Exception:
                            pass
                    salience = max(0.0, min(1.0, salience))
                    attention_snapshot = {
                        "salience": salience,
                        "immediacy": immediacy,
                        "source": meta.get("source"),
                        "trigger": trigger.type.name,
                    }
                    if isinstance(payload, dict):
                        payload_kind = payload.get("kind") or payload.get("type")
                        if payload_kind:
                            attention_snapshot["payload_kind"] = payload_kind
                    ctx["scratch"]["attention"] = attention_snapshot
                    if (
                        salience >= 0.7
                        and hasattr(self.memory, "store")
                        and hasattr(self.memory.store, "add")
                    ):
                        try:
                            self.memory.store.add(
                                {
                                    "kind": "attention_marker",
                                    "topic": ctx.get("topic") or meta.get("topic"),
                                    "summary": ctx.get("summary"),
                                    "attention": dict(attention_snapshot),
                                    "ts": time.time(),
                                }
                            )
                        except Exception:
                            pass
                elif stg is Stage.INTERPRET:
                    ctx["scratch"]["concepts"] = self.memory.concepts.extract(ctx["obs"])
                    ctx["scratch"]["episodic_links"] = self.memory.episodic.link(ctx["obs"])
                elif stg is Stage.EVALUATE:
                    if monitor:
                        monitor.begin_cycle()
                    decision_ctx = {
                        "trigger": getattr(self, "_current_trigger", None),
                        "mode": None,
                        "pipeline_family": selection.family,
                        "pipeline_variant": pipe,
                    }
                    if decision_journal:
                        try:
                            self._current_decision_id = decision_journal.new(decision_ctx)
                        except Exception:
                            self._current_decision_id = None
                    current_topic = getattr(self, "_current_topic", None)
                    if reasoning_ledger:
                        try:
                            self._current_trace_id = reasoning_ledger.start_trace(topic=current_topic)
                        except Exception:
                            self._current_trace_id = None
                    if (
                        decision_journal
                        and reasoning_ledger
                        and self._current_decision_id
                        and self._current_trace_id
                    ):
                        try:
                            decision_journal.attach_trace(
                                self._current_decision_id, self._current_trace_id
                            )
                        except Exception:
                            pass
                    emo = self.emotions.read()
                    prio = unified_priority(
                        impact=trigger.meta.get("importance", 0.6),
                        probability=trigger.meta.get("probability", 0.6),
                        reversibility=trigger.meta.get("reversibility", 1.0),
                        effort=trigger.meta.get("effort", 0.5),
                        uncertainty=trigger.meta.get("uncertainty", 0.0),
                        valence=getattr(emo, "valence", 0.0) if emo else 0.0,
                    )
                    priority_token = get_last_priority_token()
                    if priority_token:
                        ctx.setdefault("scratch", {})["priority_token"] = priority_token
                    ctx["scratch"]["priority"] = prio
                elif stg is Stage.REFLECT:
                    if monitor:
                        monitor.on_reflect_start()
                    try:
                        ctx["scratch"]["frame"] = self.cognition.planner.frame(
                            trigger, stop_rules={"max_options": 3, "max_seconds": 900}
                        )
                    finally:
                        if monitor:
                            monitor.on_reflect_end()
                elif stg is Stage.REASON:
                    if monitor:
                        monitor.on_reason_start()
                    try:
                        tested = self.cognition.reflection_loop.test_hypotheses(
                            ctx.get("scratch", {}), max_tests=3
                        )
                        n_tested = (
                            int(tested.get("tested", 0))
                            if isinstance(tested, dict)
                            else 0
                        )
                        if monitor:
                            monitor.on_hypothesis_tested(n_tested)
                        ctx.setdefault("scratch", {})["reason"] = tested
                    except Exception:
                        pass
                    finally:
                        if monitor:
                            monitor.on_reason_end()
                elif stg is Stage.DECIDE:
                    ctx_depth = len(list(ctx.keys())) if isinstance(ctx, dict) else 0
                    if monitor:
                        monitor.set_depth(ctx_depth)
                    decision = policy_engine.decide(ctx) if policy_engine else {}
                    ctx["decision"] = decision
                    ctx["expected"] = decision.get("expected", {"score": 1.0})
                    if (
                        decision_journal
                        and self._current_decision_id
                        and isinstance(decision, dict)
                    ):
                        try:
                            decision_journal.commit_action(
                                self._current_decision_id,
                                decision.get("action", {}),
                                (decision.get("expected") or {}).get("score", 1.0),
                            )
                        except Exception:
                            pass
                    if (
                        reasoning_ledger
                        and self._current_trace_id
                        and isinstance(decision, dict)
                    ):
                        try:
                            reasoning_ledger.select_option(
                                self._current_trace_id,
                                chosen_id=(decision.get("action", {}) or {}).get("type", "act"),
                                justification_text=(decision.get("action", {}) or {}).get("desc", ""),
                                stop_rules_hit=False,
                            )
                        except Exception:
                            pass
                elif stg is Stage.ACT:
                    raw_mode = step.get("mode")
                    if callable(raw_mode):
                        raw_mode = raw_mode(ctx)
                    if isinstance(raw_mode, ActMode):
                        mode = raw_mode
                    elif isinstance(raw_mode, str):
                        try:
                            mode = ActMode[raw_mode.upper()]
                        except KeyError:
                            mode = ActMode.DELIBERATE
                    else:
                        mode = ActMode.DELIBERATE
                    ctx["mode"] = mode
                    action = (ctx.get("decision") or {}).get("action")
                    if not action:
                        continue
                    jid = self._submit_for_mode(
                        mode,
                        action,
                        trigger.meta,
                        ctx["scratch"].get("priority", 0.6),
                    )
                    events = self.job_manager.poll_completed(32)
                    result: Optional[Dict[str, Any]] = None
                    for ev in events:
                        job = ev.get("job", {})
                        if job.get("id") == jid:
                            ctx["obtained"] = {
                                "score": 1.0 if ev.get("event") == "done" else 0.0
                            }
                            result = ev.get("result") if isinstance(ev, dict) else None
                            break
                    decision = ctx.get("decision") or {}
                    obtained_score = (
                        1.0
                        if (result and result.get("status") in ("ok", "done", "success"))
                        else float((ctx.get("obtained") or {"score": 0.0}).get("score", 0.0))
                    )
                    if selection.family == "HABIT":
                        habit_payload = (ctx.get("scratch", {}) or {}).get("habit_payload", {})
                        if result is not None:
                            ctx.setdefault("scratch", {})["habit_result"] = result
                        habit_info = habit_payload.get("habit") if isinstance(habit_payload, dict) else {}
                        habit_name = (habit_info or {}).get("name")
                        if isinstance(habit_payload, dict):
                            habit_name = habit_name or habit_payload.get("name")
                        if habit_name and result and result.get("ok"):
                            try:
                                self.cognition.habits.record_completion(habit_name, when=time.time())
                            except Exception:
                                pass
                    if decision_journal and self._current_decision_id:
                        try:
                            decision_journal.close(self._current_decision_id, obtained_score)
                        except Exception:
                            pass
                    if reasoning_ledger and self._current_trace_id:
                        try:
                            reasoning_ledger.end_trace(
                                self._current_trace_id,
                                expected=(decision.get("expected") or {}).get("score", 1.0),
                                obtained=obtained_score,
                            )
                        except Exception:
                            pass
                    try:
                        self.self_model.register_decision(
                            {
                                "decision_id": self._current_decision_id,
                                "topic": ctx.get("topic") or "__generic__",
                                "action": (decision.get("action", {}) or {}).get("type"),
                                "expected": float((decision.get("expected") or {}).get("score", 1.0)),
                                "obtained": float(obtained_score),
                                "trace_id": self._current_trace_id,
                                "ts": time.time(),
                            }
                        )
                    except Exception:
                        pass
                    try:
                        jm = getattr(self, "job_manager", None)
                        if jm and hasattr(jm, "snapshot_identity_view"):
                            view = jm.snapshot_identity_view() or {}
                            self.self_model.update_work(
                                current=view.get("current"),
                                recent=view.get("recent"),
                            )
                    except Exception:
                        pass
                    try:
                        decision_action = (ctx.get("decision", {}) or {}).get("action", {})
                        if isinstance(decision_action, dict):
                            action_label = (
                                decision_action.get("desc")
                                or decision_action.get("text")
                                or decision_action.get("name")
                                or decision_action.get("type")
                                or "une action"
                            )
                        else:
                            action_label = "une action"
                        summary = f"J'exécute {action_label}" if action_label else "J'exécute une action"
                        expected_score = (ctx.get("expected") or {}).get("score")
                        try:
                            expected_value = float(expected_score) if expected_score is not None else None
                        except Exception:
                            expected_value = None
                        self._phenomenal_record_action(
                            stage="ACT",
                            ctx=ctx,
                            summary=summary,
                            expected=expected_value,
                            obtained=float(obtained_score),
                        )
                    except Exception:
                        pass
                elif stg is Stage.FEEDBACK:
                    exp = float(ctx["expected"].get("score", 1.0))
                    obt = float((ctx.get("obtained") or {"score": 0.0}).get("score", 0.0))
                    err = abs(obt - exp)
                    ctx["scratch"]["raw_prediction_error"] = err
                    reward_signal = max(0.0, min(1.0, 1.0 - err))
                    success = obt >= exp
                    if success:
                        reward_signal = max(reward_signal, 0.95)
                    scratch = ctx.setdefault("scratch", {})

                    def _metric(key: str, default: float) -> float:
                        try:
                            return float(scratch.get(key, default))
                        except (TypeError, ValueError):
                            return float(default)

                    reward_features = {
                        "memory_consistency": _metric("memory_consistency", 0.5),
                        "transfer_success": _metric("transfer_success", 0.5),
                        "explanatory_adequacy": _metric("explanatory_adequacy", 0.5),
                        "social_appraisal": _metric("social_appraisal", 0.5),
                        "calibration_gap": _metric("calibration_gap", 0.3),
                        "clarification_penalty": 1.0
                        if (ctx.get("decision", {}).get("action", {}).get("type") == "clarify")
                        else 0.0,
                        "sj_success": 1.0 if success else 0.0,
                    }
                    ema_context = {
                        "mode": ctx.get("mode").name if ctx.get("mode") else None,
                        "action": (ctx.get("decision", {}).get("action") or {}).get("type"),
                        "topic": ctx.get("topic"),
                        "uncertainty": ctx.get("expected", {}).get("uncertainty"),
                        "priority": scratch.get("priority"),
                        "success": 1.0 if success else 0.0,
                        "contradiction": 1.0 if scratch.get("contradiction") else 0.0,
                    }
                    smoothed_err = self._prediction_error_ema.update(
                        err,
                        reward_signal,
                        reward_features=reward_features,
                        context=ema_context,
                    )
                    ctx["scratch"]["prediction_error"] = smoothed_err
                    ctx["scratch"]["sj_reward"] = reward_signal
                    ctx["scratch"]["sj_success"] = 1.0 if success else 0.0
                    self._last_prediction_error = smoothed_err
                    self._last_contradiction = bool(
                        ctx.get("scratch", {}).get("contradiction", False)
                    )
                    self.memory.store.add(
                        {
                            "kind": "feedback",
                            "pipe": pipe,
                            "mode": ctx["mode"].name if ctx.get("mode") else None,
                            "err": err,
                        }
                    )
                    mode_name = ctx["mode"].name if ctx.get("mode") else "unknown"
                    if policy_engine and hasattr(policy_engine, "update_outcome"):
                        try:
                            policy_engine.update_outcome(mode_name, ok=success)
                        except Exception:
                            pass
                    ctx.setdefault("scratch", {})
                    if "prediction_error" not in ctx["scratch"]:
                        ctx["scratch"]["prediction_error"] = 0.5  # valeur neutre

                    # Renforcement de l'habitude pour (action_type :: contexte)
                    try:
                        EvolutionManager.shared().reinforce(ctx)
                    except Exception:
                        pass
                    try:
                        feedback_summary = (
                            "Le résultat confirme mon action" if success else "Le résultat contredit mon action"
                        )
                        self._phenomenal_record_action(
                            stage="FEEDBACK",
                            ctx=ctx,
                            summary=f"{feedback_summary} (erreur={err:.2f}, récompense={reward_signal:.2f})",
                            expected=exp,
                            obtained=obt,
                        )
                    except Exception:
                        pass
                elif stg is Stage.LEARN:
                    self.cognition.evolution.reinforce(ctx)
                elif stg is Stage.UPDATE:
                    consolidation = self.memory.consolidator.maybe_consolidate()
                    if isinstance(consolidation, dict):
                        new_items: List[Dict[str, Any]] = []
                        for lesson in consolidation.get("lessons", []) or []:
                            new_items.append({"kind": "lesson", "text": lesson})
                        for proposal in consolidation.get("proposals", []) or []:
                            if isinstance(proposal, dict):
                                new_items.append({"kind": proposal.get("kind", "proposal"), "data": proposal})
                        for item in new_items:
                            try:
                                self._sj_new_items_queue.append(item)
                            except Exception:
                                break
                    prediction_error = float(ctx.get("scratch", {}).get("prediction_error", 0.0))
                    memory_consistency = float(ctx.get("scratch", {}).get("memory_consistency", 0.5))
                    transfer_success = float(ctx.get("scratch", {}).get("transfer_success", 0.5))
                    explanatory_adequacy = float(ctx.get("scratch", {}).get("explanatory_adequacy", 0.5))
                    social_appraisal = float(ctx.get("scratch", {}).get("social_appraisal", 0.5))
                    clarification_penalty = (
                        1.0
                        if (ctx.get("decision", {}).get("action", {}).get("type") == "clarify")
                        else 0.0
                    )
                    if policy_engine and hasattr(policy_engine, "get_last_confidence"):
                        try:
                            last_conf = float(policy_engine.get_last_confidence() or 0.7)
                        except Exception:
                            last_conf = 0.7
                        calibration_gap = abs(last_conf - float(1.0 - prediction_error))
                    else:
                        calibration_gap = 0.3

                    sj_features = {
                        "prediction_error": prediction_error,
                        "memory_consistency": memory_consistency,
                        "transfer_success": transfer_success,
                        "explanatory_adequacy": explanatory_adequacy,
                        "social_appraisal": social_appraisal,
                        "calibration_gap": calibration_gap,
                        "clarification": clarification_penalty,
                        "contradiction": 1.0
                        if ctx.get("scratch", {}).get("contradiction")
                        else 0.0,
                        "success": ctx.get("scratch", {}).get("sj_success", 0.0),
                    }
                    reward_signal = ctx.get("scratch", {}).get("sj_reward", 0.5)
                    try:
                        self._sj_conf_cache = self._sj_config_model.update(
                            sj_features, reward_signal
                        )
                        corr = self._prediction_error_ema.correlation()
                        self._sj_corr_log.append(corr)
                        if len(self._sj_corr_log) % 16 == 0:
                            self.telemetry.log(
                                "orchestrator",
                                "sj_adaptation",
                                {
                                    "reward": reward_signal,
                                    "corr": corr,
                                    "threshold": self._sj_conf_cache["surprise_threshold"],
                                    "heartbeat_every": self._sj_conf_cache["heartbeat_every"],
                                },
                            )
                    except Exception:
                        pass

                    current_topic = (
                        ctx.get("topic")
                        or getattr(self, "_current_topic", None)
                        or "__generic__"
                    )

                    try:
                        decision_info = ctx.get("decision") or {}
                        ema_context = {
                            "mode": ctx.get("mode").name if ctx.get("mode") else None,
                            "action": (decision_info.get("action") or {}).get("type"),
                            "topic": current_topic,
                            "uncertainty": ctx.get("expected", {}).get("uncertainty"),
                            "success": ctx.get("scratch", {}).get("sj_success", 0.0),
                            "clarification": clarification_penalty,
                        }
                        self._prediction_error_ema.reinforce(
                            reward_signal,
                            reward_features=sj_features,
                            context=ema_context,
                        )
                    except Exception:
                        pass

                    if understanding_agg:
                        try:
                            U = understanding_agg.compute(
                                topic=current_topic,
                                prediction_error=prediction_error,
                                memory_consistency=memory_consistency,
                                transfer_success=transfer_success,
                                explanatory_adequacy=explanatory_adequacy,
                                social_appraisal=social_appraisal,
                                clarification_penalty=clarification_penalty,
                                calibration_gap=calibration_gap,
                            )
                        except Exception:
                            U = SimpleNamespace(U_topic=0.5, U_global=0.5)
                    else:
                        U = SimpleNamespace(U_topic=0.5, U_global=0.5)

                    scratch = ctx.setdefault("scratch", {})
                    scratch["clarification_penalty"] = clarification_penalty
                    scratch["calibration_gap"] = calibration_gap
                    scratch["memory_consistency"] = memory_consistency
                    scratch["transfer_success"] = transfer_success
                    scratch["explanatory_adequacy"] = explanatory_adequacy
                    scratch["social_appraisal"] = social_appraisal
                    scratch.setdefault("understanding", {})
                    scratch["understanding"].update(
                        {
                            "topic": current_topic,
                            "U_topic": float(getattr(U, "U_topic", 0.5)),
                            "U_global": float(getattr(U, "U_global", 0.5)),
                        }
                    )

                    snap = (
                        monitor.snapshot()
                        if monitor
                        else SimpleNamespace(thinking_score=0.5, depth=0)
                    )

                    if hasattr(self.memory, "store") and hasattr(self.memory.store, "add"):
                        self.memory.store.add(
                            {
                                "kind": "self_judgment",
                                "topic": current_topic,
                                "scores": {
                                    "U_topic": U.U_topic,
                                    "U_global": U.U_global,
                                    "thinking": getattr(snap, "thinking_score", 0.5),
                                    "calibration_gap": calibration_gap,
                                    "consistency": memory_consistency,
                                    "transfer": transfer_success,
                                    "social_appraisal": social_appraisal,
                                },
                                "flags": {
                                    "asked_clarification": bool(clarification_penalty > 0.0),
                                    "contradiction_detected": bool(
                                        ctx.get("scratch", {}).get("contradiction", False)
                                    ),
                                },
                                "evidence_refs": ctx.get("scratch", {}).get("evidence_refs", []),
                            }
                        )

                    if selfhood:
                        try:
                            selfhood.update_from_cycle(
                                U_global=U.U_global,
                                thinking_score=getattr(snap, "thinking_score", 0.5),
                                social_appraisal=social_appraisal,
                                calibration_gap=calibration_gap,
                                consistency_signal=memory_consistency,
                                evidence_refs=ctx.get("scratch", {}).get("evidence_refs", []),
                            )
                        except Exception:
                            pass

                    if (
                        policy_engine
                        and hasattr(policy_engine, "update_meta")
                        and selfhood
                        and hasattr(selfhood, "policy_hints")
                    ):
                        try:
                            policy_engine.update_meta(selfhood.policy_hints())
                        except Exception:
                            pass

                    try:
                        topic = ctx.get("topic") or getattr(self, "_current_topic", "__generic__")
                        milestone_info = self.record_knowledge_milestone(topic=topic, ctx=ctx)
                    except Exception:
                        milestone_info = None

                    goals = getattr(self.arch, "goals", None)
                    if goals and hasattr(goals, "integrate_understanding"):
                        try:
                            goals.integrate_understanding(
                                topic=current_topic if current_topic != "__generic__" else topic,
                                score=float(getattr(U, "U_topic", 0.5)),
                                prediction_error=float(prediction_error),
                                gaps=ctx.get("gaps"),
                                goal_id=(ctx.get("payload") or {}).get("goal_id"),
                                clarification_penalty=float(clarification_penalty),
                                source=(ctx.get("meta") or {}).get("source", "pipeline"),
                                evidence={
                                    "pipeline": pipe,
                                    "trigger": trigger.type.name,
                                    "reason": selection.reason,
                                },
                            )
                        except Exception:
                            pass

                    if not isinstance(ctx.get("gaps"), list):
                        ctx["gaps"] = []
                    else:
                        ctx.setdefault("gaps", [])
                    if isinstance(milestone_info, dict):
                        for gap in milestone_info.get("gaps", []) or []:
                            if gap not in ctx["gaps"]:
                                ctx["gaps"].append(gap)

                    # --- SelfIdentity : auto-jugement & état interne ---
                    try:
                        traits_growth = 0.0
                        phase = "novice"
                        if hasattr(self, "selfhood") and hasattr(self.selfhood, "traits"):
                            traits_growth = float(
                                getattr(self.selfhood.traits, "growth_rate", 0.0)
                            )
                            phase = str(getattr(self.selfhood.traits, "phase", "novice"))
                        self.self_model.attach_selfhood(
                            traits={
                                "self_efficacy": float(U.U_global),
                                "self_trust": float(1.0 - calibration_gap),
                                "self_consistency": float(memory_consistency),
                                "social_acceptance": float(social_appraisal),
                                "growth_rate": traits_growth,
                            },
                            phase=phase,
                            claims={
                                "thinking": {
                                    "text": "I think when I explore, hypothesize and reason.",
                                    "confidence": float(getattr(snap, "thinking_score", 0.5)),
                                },
                                "understanding": {
                                    "text": "I understand when I can predict, explain and transfer.",
                                    "confidence": float(U.U_global),
                                },
                            },
                            evidence_refs=ctx.get("scratch", {}).get("evidence_refs", []),
                        )
                    except Exception:
                        pass

                    try:
                        self.self_model.update_state(
                            emotions=ctx.get("emotions"),
                            doubts=ctx.get("doubts"),
                            cognition={
                                "thinking": float(getattr(snap, "thinking_score", 0.5)),
                                "reason_depth": int(getattr(snap, "depth", 0)),
                                "uncertainty": float(calibration_gap),
                                "load": float(ctx.get("load", 0.0)),
                            },
                        )
                    except Exception:
                        pass

                    try:
                        self.self_model.set_identity_patch(
                            {
                                "self_judgment": {
                                    "understanding": {
                                        "global": float(U.U_global),
                                        "topics": ctx.get("U_topics", {}),
                                        "calibration_gap": float(calibration_gap),
                                    }
                                }
                            }
                        )
                    except Exception:
                        pass

                    try:
                        topic = ctx.get("topic") or getattr(self, "_current_topic", "__generic__")
                        beliefs_now: List[Dict[str, Any]] = []
                        milestone_beliefs: List[Dict[str, Any]] = []
                        if isinstance(milestone_info, dict):
                            milestone_beliefs = list(milestone_info.get("beliefs", []) or [])
                        if milestone_beliefs:
                            beliefs_now = milestone_beliefs
                        elif hasattr(self.memory, "semantic") and hasattr(
                            self.memory.semantic, "export_topic_beliefs"
                        ):
                            beliefs_now = list(
                                self.memory.semantic.export_topic_beliefs(topic=topic)
                            )

                        snap_id = None
                        if isinstance(milestone_info, dict):
                            snap_id = milestone_info.get("snapshot_id")
                        if snap_id is None and hasattr(self, "timeline"):
                            try:
                                snap_id = (
                                    self.timeline.snapshot(topic, beliefs_now)
                                    if hasattr(self.timeline, "snapshot")
                                    else None
                                )
                            except Exception:
                                snap_id = None

                        delta: Optional[Dict[str, Any]] = None
                        if isinstance(milestone_info, dict):
                            delta = milestone_info.get("delta_event")
                        if delta is None and hasattr(self, "timeline"):
                            last_beliefs = getattr(self, "_last_beliefs_by_topic", {}).get(topic, [])
                            try:
                                delta = (
                                    self.timeline.delta(topic, last_beliefs, beliefs_now)
                                    if hasattr(self.timeline, "delta")
                                    else None
                                )
                            except Exception:
                                delta = None
                        if delta and hasattr(self.memory, "store") and hasattr(self.memory.store, "add"):
                            try:
                                self.memory.store.add(delta)
                            except Exception:
                                pass

                        self._last_beliefs_by_topic = getattr(self, "_last_beliefs_by_topic", {})
                        self._last_beliefs_by_topic[topic] = beliefs_now

                        related = ctx.get("related_topics", []) or []
                        topics = [topic, *related]
                        topics = [t for t in dict.fromkeys([t for t in topics if t])]
                        self.self_model.update_timeline(
                            last_topics=topics,
                            last_snapshot_id=snap_id,
                            last_delta_id=(delta.get("id") if isinstance(delta, dict) else None),
                        )

                        gaps = ctx.get("gaps", [])
                        projected = None
                        if isinstance(milestone_info, dict):
                            projected = milestone_info.get("projected_plan")
                        if projected is None and hasattr(self, "timeline") and hasattr(
                            self.timeline, "project"
                        ):
                            projected = self.timeline.project(topic, gaps=gaps) if gaps else []
                        self.self_model.set_learning_plan(projected or [])
                    except Exception:
                        pass

                    followups: List[Trigger] = []
                    if U.U_topic < 0.4 and ctx.get("meta", {}).get("immediacy", 0.5) > 0.7:
                        followups.append(
                            Trigger(
                                TriggerType.GOAL,
                                {"source": "self_judgment", "importance": 0.9, "immediacy": 0.9},
                                {"goal_kind": "ClarifyUserIntent", "topic": current_topic},
                            )
                        )

                    self_trust = (
                        getattr(getattr(selfhood, "traits", SimpleNamespace()), "self_trust", 1.0)
                        if selfhood
                        else 1.0
                    )
                    if (
                        self_trust < 0.45
                        and policy_engine
                        and hasattr(policy_engine, "set_uncertainty_disclosure")
                    ):
                        try:
                            policy_engine.set_uncertainty_disclosure(True)
                        except Exception:
                            pass

                    for t in followups:
                        try:
                            self._pending_triggers.append(t)
                        except Exception:
                            pass

                    try:
                        understanding_stats = ctx.get("scratch", {}).get("understanding", {})
                        u_topic = float(understanding_stats.get("U_topic", getattr(U, "U_topic", 0.5)))
                        u_global = float(understanding_stats.get("U_global", getattr(U, "U_global", 0.5)))
                        summary = (
                            f"Je me réévalue : compréhension globale={u_global:.2f},"
                            f" précision locale={u_topic:.2f}, écart de calibration={calibration_gap:.2f}"
                        )
                        self._phenomenal_record_action(
                            stage="UPDATE",
                            ctx=ctx,
                            summary=summary,
                            expected=None,
                            obtained=None,
                        )
                        if getattr(self, "phenomenal_journal", None) is not None:
                            try:
                                self.phenomenal_journal.audit_against(
                                    "understanding",
                                    {
                                        "U_topic": u_topic,
                                        "U_global": u_global,
                                        "calibration_gap": float(calibration_gap),
                                    },
                                    tolerance=0.1,
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass

            finally:
                mem_after = _get_process_memory_kb()
                duration_ms = 1000.0 * (time.time() - stage_start)
                delta_mem_mb = max(0.0, (mem_after - mem_before) / 1024.0)
                stage_metrics.append({
                    "stage": stg.name,
                    "duration_ms": duration_ms,
                    "delta_mem_mb": round(delta_mem_mb, 4),
                })
        pipeline_duration_ms = 1000.0 * (time.time() - pipeline_start)
        ctx["scratch"]["pipeline"]["duration_ms"] = pipeline_duration_ms
        if telemetry:
            try:
                telemetry.log(
                    "pipeline_run",
                    "cognition",
                    {
                        "pipeline": pipe,
                        "family": selection.family,
                        "reason": selection.reason,
                        "duration_ms": pipeline_duration_ms,
                        "stages": stage_metrics,
                    },
                )
            except Exception:
                pass
        return ctx

    def _phenomenal_identity_snapshot(self) -> Tuple[List[str], List[str]]:
        values: List[str] = []
        principles: List[str] = []
        try:
            persona = getattr(self.self_model, "persona", {})
            if isinstance(persona, dict):
                raw_values = persona.get("values")
                if isinstance(raw_values, list):
                    values.extend(str(val) for val in raw_values if isinstance(val, str))
            identity = getattr(self.self_model, "identity", {})
            if isinstance(identity, dict):
                declared = identity.get("values")
                if isinstance(declared, list):
                    values.extend(str(val) for val in declared if isinstance(val, str))
                principle_items = identity.get("principles")
                if isinstance(principle_items, list):
                    principles.extend(str(item) for item in principle_items if isinstance(item, str))
                commitments = identity.get("commitments", {})
                if isinstance(commitments, dict):
                    by_key = commitments.get("by_key")
                    if isinstance(by_key, dict):
                        for key, info in by_key.items():
                            if isinstance(info, dict) and info.get("active"):
                                principles.append(str(key))
        except Exception:
            pass
        if values:
            values = sorted({val.strip() for val in values if val})
        if principles:
            principles = sorted({val.strip() for val in principles if val})
        return values, principles

    def _phenomenal_homeostasis_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        state = getattr(self._homeostasis, "state", {})
        if isinstance(state, dict):
            for key in ("intrinsic_reward", "extrinsic_reward", "hedonic_reward"):
                value = state.get(key)
                if isinstance(value, (int, float)):
                    snapshot[key] = float(value)
            drives = state.get("drives")
            if isinstance(drives, Mapping):
                ranked = sorted(
                    (
                        (str(name), float(val))
                        for name, val in drives.items()
                        if isinstance(val, (int, float))
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if ranked:
                    snapshot["drives"] = {name: value for name, value in ranked[:5]}
        return snapshot

    def _phenomenal_emotion_snapshot(self) -> Dict[str, Any]:
        emotions: Dict[str, Any] = {}
        try:
            emo_state = self.emotions.read()
        except Exception:
            emo_state = None
        if emo_state is not None:
            for key in ("valence", "arousal", "dominance"):
                try:
                    emotions[key] = float(getattr(emo_state, key))
                except Exception:
                    continue
            label = getattr(emo_state, "label", None)
            if not label:
                label = getattr(emo_state, "state", None)
            if label:
                emotions["label"] = str(label)
        kernel_state = getattr(self, "phenomenal_kernel_state", {})
        if isinstance(kernel_state, dict):
            interpretation = kernel_state.get("llm_interpretation")
            if isinstance(interpretation, dict):
                state_name = interpretation.get("current_state")
                if state_name:
                    emotions.setdefault("narrative", str(state_name))
        return emotions

    def _phenomenal_sensation_snapshot(self) -> Dict[str, Any]:
        sensations: Dict[str, Any] = {}
        kernel_state = getattr(self, "phenomenal_kernel_state", {})
        if isinstance(kernel_state, dict):
            for key in (
                "energy",
                "arousal",
                "resonance",
                "surprise",
                "fatigue",
                "hedonic_reward",
                "global_slowdown",
            ):
                value = kernel_state.get(key)
                if isinstance(value, (int, float)):
                    sensations[key] = float(value)
        return sensations

    def _phenomenal_metric_snapshot(self, ctx: Mapping[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not isinstance(ctx, Mapping):
            return metrics
        scratch = ctx.get("scratch")
        if isinstance(scratch, Mapping):
            for key in (
                "priority",
                "prediction_error",
                "sj_reward",
                "sj_success",
                "memory_consistency",
                "transfer_success",
                "explanatory_adequacy",
                "social_appraisal",
                "clarification_penalty",
                "calibration_gap",
            ):
                value = scratch.get(key)
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
            understanding = scratch.get("understanding")
            if isinstance(understanding, Mapping):
                for key in ("U_topic", "U_global"):
                    value = understanding.get(key)
                    if isinstance(value, (int, float)):
                        metrics[key] = float(value)
        expected = ctx.get("expected")
        if isinstance(expected, Mapping):
            uncertainty = expected.get("uncertainty")
            if isinstance(uncertainty, (int, float)):
                metrics["uncertainty"] = float(uncertainty)
        return metrics

    def _phenomenal_action_context(
        self,
        ctx: Mapping[str, Any],
        *,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        trigger = getattr(self, "_current_trigger", None)
        if trigger is not None:
            payload["trigger_type"] = getattr(getattr(trigger, "type", None), "name", None)
            meta = getattr(trigger, "meta", None)
            if isinstance(meta, Mapping):
                payload["trigger_meta"] = {
                    key: meta[key]
                    for key in ("importance", "immediacy", "reversibility", "source")
                    if key in meta
                }
        if isinstance(ctx, Mapping):
            scratch = ctx.get("scratch")
            if isinstance(scratch, Mapping):
                priority = scratch.get("priority")
                if isinstance(priority, (int, float)):
                    payload["priority"] = float(priority)
                vetoes = scratch.get("policy_vetoes")
                if isinstance(vetoes, (list, tuple)):
                    payload["policy_vetoes"] = [str(v) for v in vetoes[:4]]
                habit_payload = scratch.get("habit_payload")
                if isinstance(habit_payload, Mapping):
                    payload["habit"] = {
                        "name": habit_payload.get("name")
                        or (habit_payload.get("habit") or {}).get("name"),
                        "confidence": habit_payload.get("confidence"),
                    }
            decision = ctx.get("decision")
            if isinstance(decision, Mapping):
                action = decision.get("action")
                if isinstance(action, Mapping):
                    payload["action"] = {
                        "type": action.get("type"),
                        "desc": action.get("desc") or action.get("text"),
                    }
                payload["reason"] = decision.get("reason") or decision.get("rationale")
            gaps = ctx.get("gaps")
            if isinstance(gaps, list):
                payload["gaps"] = [str(g) for g in gaps[:5]]
        if metrics:
            payload["metrics"] = {key: float(value) for key, value in metrics.items()}
        return {k: v for k, v in payload.items() if v is not None}

    def _phenomenal_record_action(
        self,
        *,
        stage: str,
        ctx: Mapping[str, Any],
        summary: str,
        expected: Optional[float],
        obtained: Optional[float],
    ) -> None:
        journal = getattr(self, "phenomenal_journal", None)
        if journal is None:
            return
        try:
            mode_value = ctx.get("mode") if isinstance(ctx, Mapping) else None
            if isinstance(mode_value, ActMode):
                mode = mode_value.name.lower()
            else:
                mode = str(mode_value) if mode_value else None
            topic = ctx.get("topic") if isinstance(ctx, Mapping) else None
            values, principles = self._phenomenal_identity_snapshot()
            homeostasis = self._phenomenal_homeostasis_snapshot()
            emotions = self._phenomenal_emotion_snapshot()
            sensations = self._phenomenal_sensation_snapshot()
            metrics = self._phenomenal_metric_snapshot(ctx)
            context_payload = self._phenomenal_action_context(ctx, metrics=metrics)
            entry = journal.record_action(
                stage=stage,
                mode=mode,
                topic=str(topic) if topic else None,
                summary=summary,
                expected=expected,
                obtained=obtained,
                values=values,
                principles=principles,
                homeostasis=homeostasis,
                emotions=emotions,
                sensations=sensations,
                context=context_payload,
            )
            try:
                ctx.setdefault("scratch", {})["phenomenal_episode"] = entry
            except Exception:
                pass
        except Exception:
            pass
