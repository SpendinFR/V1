from __future__ import annotations
from typing import Dict, Any, Optional, Iterable, Mapping, Sequence
from dataclasses import dataclass, asdict, field
from collections import deque
from copy import deepcopy
import logging
import random
import statistics
import json
import os
import time
import uuid

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


def _now():
    return time.time()


def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


EXPERIMENTS_LOG = "logs/experiments.jsonl"


@dataclass
class Experiment:
    exp_id: str
    metric: str
    baseline: float
    target_change: float
    plan: Dict[str, Any]
    plan_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_cycles: int = 3
    created_at: float = 0.0
    status: str = "scheduled"
    notes: str = ""

    def to_jsonl(self) -> str:
        d = asdict(self)
        return json.dumps(json_sanitize(d), ensure_ascii=False)


class MetacognitionExperimenter:
    """
    Génère de petits tests concrets à partir des métriques,
    et consigne le résultat pour fermer la boucle.
    """

    def __init__(self, system_ref: Any = None):
        self.system = system_ref
        self.active: deque[Experiment] = deque(maxlen=50)
        self.plan_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.metric_stats: Dict[str, Dict[str, Any]] = {}
        self._load_history()

    def suggest_and_log_tests(self, performance_metrics: Dict[str, float]):
        """
        Pour chaque métrique clé, si sous un seuil → proposer un test.
        """
        created = 0
        last_metric: Optional[str] = None
        for metric, baseline in performance_metrics.items():
            if metric not in PLAN_LIBRARY:
                continue

            self._register_metric_baseline(metric, baseline)

            if self._has_active_experiment(metric):
                continue

            threshold = self._dynamic_threshold(metric)
            if baseline >= threshold:
                continue

            llm_proposal = self._propose_experiment_with_llm(metric, baseline, threshold)
            if llm_proposal and llm_proposal.get("skip"):
                continue

            if llm_proposal and "plan" in llm_proposal:
                plan_payload = deepcopy(llm_proposal["plan"])
                parameters = deepcopy(llm_proposal.get("parameters", {}))
                if parameters:
                    plan_payload.setdefault("parameters", parameters)
                adaptive_change = float(llm_proposal["target_change"])
                plan_id = llm_proposal.get("plan_id", plan_payload.get("strategy", ""))
                duration_cycles = int(llm_proposal.get("duration_cycles", 3))
                notes = llm_proposal.get("notes", "")
            else:
                plan_cfg = self._select_plan(metric)
                if not plan_cfg:
                    continue

                adaptive_change = self._adaptive_target_change(metric, baseline, plan_cfg)
                plan_payload = deepcopy(plan_cfg.get("plan", {}))
                parameters = deepcopy(plan_cfg.get("parameters", {}))
                if parameters:
                    plan_payload.setdefault("parameters", parameters)
                plan_id = plan_cfg.get("id", plan_payload.get("strategy", ""))
                duration_cycles = int(plan_cfg.get("duration", 3))
                notes = ""

            exp = Experiment(
                exp_id=str(uuid.uuid4())[:8],
                metric=metric,
                baseline=float(baseline),
                target_change=float(adaptive_change),
                plan=plan_payload,
                plan_id=plan_id,
                parameters=parameters,
                duration_cycles=duration_cycles,
                created_at=_now(),
                status="scheduled",
                notes=notes,
            )

            self._append_jsonl(EXPERIMENTS_LOG, exp.to_jsonl())
            self.active.append(exp)
            created += 1
            last_metric = metric

        if created and self.system:
            try:
                self.system._record_metacognitive_event(
                    event_type="experiment_planned",
                    domain=self._metric_domain(last_metric) if last_metric else None,
                    description=f"{created} test(s) planifié(s) pour optimisation métriques.",
                    significance=min(0.3 + 0.1 * created, 0.7),
                    confidence=0.6,
                )
            except Exception:
                pass

    def record_outcome(self, metric: str, new_value: float):
        """
        Enregistre un résultat pour un test ciblant 'metric' (si actif).
        """
        exp = None
        for e in reversed(self.active):
            if e.metric == metric and e.status in ("scheduled", "running"):
                exp = e
                break
        if not exp:
            return

        goal = exp.baseline * (1.0 + exp.target_change)
        success = bool(new_value >= goal)

        outcome = {
            "exp_id": exp.exp_id,
            "metric": metric,
            "baseline": exp.baseline,
            "observed": float(new_value),
            "goal": goal,
            "success": success,
            "measured_at": _now(),
            "plan_id": exp.plan_id,
            "parameters": exp.parameters,
        }
        self._append_jsonl(
            EXPERIMENTS_LOG,
            json.dumps(json_sanitize({"outcome": outcome}), ensure_ascii=False),
        )
        exp.status = "done" if success else "failed"
        self._update_plan_stats(metric, exp.plan_id, success)
        self._register_metric_outcome(metric, success, new_value)

        if self.system:
            try:
                self.system._record_metacognitive_event(
                    event_type="experiment_result",
                    domain=self._metric_domain(metric),
                    description=(
                        f"Résultat test {metric}: {'OK' if success else 'KO'} "
                        f"(observé={new_value:.2f}, cible={goal:.2f})"
                    ),
                    significance=0.5 if success else 0.3,
                    confidence=0.7,
                )
            except Exception:
                pass

    @staticmethod
    def _append_jsonl(path: str, line: str):
        _ensure_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load_history(self):
        if not os.path.exists(EXPERIMENTS_LOG):
            return

        exp_lookup: Dict[str, Dict[str, Any]] = {}
        try:
            with open(EXPERIMENTS_LOG, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(data, dict) and "outcome" in data:
                        outcome = data["outcome"]
                        metric = outcome.get("metric")
                        if not metric:
                            lookup = exp_lookup.get(outcome.get("exp_id"), {})
                            metric = lookup.get("metric")
                        if not metric:
                            continue

                        plan_id = outcome.get("plan_id")
                        if not plan_id:
                            lookup = exp_lookup.get(outcome.get("exp_id"), {})
                            plan_id = lookup.get("plan_id")

                        self._update_plan_stats(metric, plan_id or "", bool(outcome.get("success")))
                        observed = outcome.get("observed")
                        if observed is not None:
                            self._register_metric_outcome(metric, bool(outcome.get("success")), float(observed))
                        continue

                    if isinstance(data, dict):
                        metric = data.get("metric")
                        if not metric:
                            continue
                        baseline = data.get("baseline")
                        if baseline is not None:
                            self._register_metric_baseline(metric, float(baseline))

                        plan_id = data.get("plan_id") or data.get("plan", {}).get("strategy", "")
                        if data.get("exp_id"):
                            exp_lookup[data["exp_id"]] = {"metric": metric, "plan_id": plan_id}
        except OSError:
            return

    def _update_plan_stats(self, metric: str, plan_id: str, success: bool):
        if not plan_id:
            plan_id = "_legacy"
        metric_stats = self.plan_stats.setdefault(metric, {})
        entry = metric_stats.setdefault(plan_id, {"success": 0, "failure": 0})
        if success:
            entry["success"] += 1
        else:
            entry["failure"] += 1

    def _register_metric_baseline(self, metric: str, baseline: float):
        stats = self.metric_stats.setdefault(
            metric,
            {
                "baselines": deque(maxlen=120),
                "success_count": 0,
                "failure_count": 0,
                "outcomes": deque(maxlen=120),
            },
        )
        stats["baselines"].append(float(baseline))

    def _register_metric_outcome(self, metric: str, success: bool, observed: float):
        stats = self.metric_stats.setdefault(
            metric,
            {
                "baselines": deque(maxlen=120),
                "success_count": 0,
                "failure_count": 0,
                "outcomes": deque(maxlen=120),
            },
        )
        stats["outcomes"].append(float(observed))
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1

    def _metric_domain(self, metric: str) -> Any:
        if not self.system or not hasattr(self.system, "CognitiveDomain"):
            return None
        domain_enum = self.system.CognitiveDomain
        try:
            if metric == "relationship_depth":
                return getattr(domain_enum, "SOCIAL")
            if metric in {"recall_accuracy", "memory_capacity"}:
                return getattr(domain_enum, "MEMORY")
            if metric in {"reasoning_speed"}:
                return getattr(domain_enum, "REASONING")
        except AttributeError:
            return getattr(domain_enum, "LEARNING", None)
        return getattr(domain_enum, "LEARNING", None)

    def _dynamic_threshold(self, metric: str) -> float:
        stats = self.metric_stats.get(metric, {})
        baselines: Iterable[float] = list(stats.get("baselines", []))
        if baselines:
            recent = list(baselines)[-20:]
            threshold = self._percentile(recent, 0.35)
            trend_boost = self._recent_trend_penalty(recent)
            threshold = max(0.5, min(0.9, threshold + trend_boost))
            return threshold
        return 0.7

    def _recent_trend_penalty(self, values: Iterable[float]) -> float:
        seq = list(values)
        if len(seq) < 3:
            return 0.0
        delta = seq[-1] - seq[0]
        if delta >= 0:
            return 0.0
        return max(-0.08, min(0.08, -delta * 0.5))

    def _plan_candidates(self, metric: str) -> Sequence[Mapping[str, Any]]:
        return PLAN_LIBRARY.get(metric, [])

    @staticmethod
    def _sanitize_plan_for_prompt(plan: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "id": plan.get("id", ""),
            "plan": plan.get("plan", {}),
            "parameters": plan.get("parameters", {}),
            "duration": plan.get("duration", 3),
            "base_improve": plan.get("base_improve", 0.08),
            "aggressiveness": plan.get("aggressiveness", 1.0),
        }

    @staticmethod
    def _sanitize_plan_payload(plan: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(plan, Mapping):
            return None
        strategy = str(plan.get("strategy", "")).strip()
        details = str(plan.get("details", "")).strip()
        if not strategy and not details:
            return None
        payload: Dict[str, Any] = {}
        payload["strategy"] = strategy or "custom_strategy"
        payload["details"] = details or ""
        if "domain" in plan:
            payload["domain"] = str(plan.get("domain", "")).strip() or None
        return {k: v for k, v in payload.items() if v is not None}

    def _propose_experiment_with_llm(
        self,
        metric: str,
        baseline: float,
        threshold: float,
    ) -> Optional[Dict[str, Any]]:
        plan_candidates = list(self._plan_candidates(metric))
        payload = {
            "metric": metric,
            "baseline": baseline,
            "threshold": threshold,
            "recent_stats": self.metric_stats.get(metric, {}),
            "plan_history": self.plan_stats.get(metric, {}),
            "plan_candidates": [
                self._sanitize_plan_for_prompt(plan) for plan in plan_candidates
            ],
        }

        response = try_call_llm_dict(
            "metacognition_experiment_planner",
            input_payload=json_sanitize(payload),
            logger=LOGGER,
        )

        if not response:
            return None

        should_plan = response.get("should_plan", True)
        if isinstance(should_plan, str):
            should_plan = should_plan.strip().lower() not in {"", "false", "no", "0"}
        if not bool(should_plan):
            return {"skip": True}

        plan_payload = self._sanitize_plan_payload(response.get("plan"))
        if not plan_payload:
            return None

        plan_id_raw = response.get("plan_id") or plan_payload.get("strategy", "")
        plan_id = str(plan_id_raw).strip() or plan_payload.get("strategy", "custom")

        parameters_raw = response.get("parameters", {})
        parameters: Dict[str, Any]
        if isinstance(parameters_raw, Mapping):
            parameters = {str(k): v for k, v in parameters_raw.items()}
        else:
            parameters = {}

        duration_raw = response.get("duration_cycles", 3)
        try:
            duration_cycles = max(1, int(duration_raw))
        except (TypeError, ValueError):
            duration_cycles = 3

        try:
            target_change = float(response.get("target_change", 0.0))
        except (TypeError, ValueError):
            target_change = 0.0

        reference_plan = next(
            (plan for plan in plan_candidates if plan.get("id") == plan_id),
            plan_candidates[0] if plan_candidates else None,
        )

        if target_change <= 0.0:
            reference_cfg = reference_plan or {
                "base_improve": 0.08,
                "aggressiveness": 1.0,
            }
            target_change = self._adaptive_target_change(
                metric,
                baseline,
                reference_cfg,
            )

        target_change = max(0.02, min(0.4, float(target_change)))

        notes = str(response.get("notes", "")).strip()
        try:
            confidence = float(response.get("confidence", None))
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None:
            notes = (notes + " " if notes else "") + f"(confiance LLM={confidence:.2f})"

        return {
            "plan": plan_payload,
            "plan_id": plan_id,
            "parameters": parameters,
            "target_change": target_change,
            "duration_cycles": duration_cycles,
            "notes": notes,
        }

    def _select_plan(self, metric: str) -> Optional[Dict[str, Any]]:
        plans = PLAN_LIBRARY.get(metric)
        if not plans:
            return None
        stats = self.plan_stats.get(metric, {})
        best_score = -1.0
        chosen: Optional[Dict[str, Any]] = None
        for plan in plans:
            record = stats.get(plan["id"], {"success": 0, "failure": 0})
            score = random.betavariate(record["success"] + 1, record["failure"] + 1)
            if score > best_score:
                best_score = score
                chosen = plan
        return deepcopy(chosen) if chosen else None

    def _adaptive_target_change(self, metric: str, baseline: float, plan_cfg: Dict[str, Any]) -> float:
        base = float(plan_cfg.get("base_improve", 0.08))
        aggressiveness = float(plan_cfg.get("aggressiveness", 1.0))
        stats = self.metric_stats.get(metric, {})
        baselines = list(stats.get("baselines", []))
        if len(baselines) >= 4:
            recent = baselines[-12:]
            mean_val = statistics.mean(recent)
            stdev = statistics.pstdev(recent) if len(recent) > 1 else 0.0
            deficit = max(0.0, mean_val - baseline)
            dynamic = base * aggressiveness + 0.4 * deficit
            if stdev > 0:
                volatility_factor = max(0.55, 1.0 - min(stdev, 0.4))
                dynamic *= volatility_factor
        else:
            dynamic = base * aggressiveness
        return max(0.02, min(0.4, dynamic))

    def _has_active_experiment(self, metric: str) -> bool:
        return any(
            exp.metric == metric and exp.status in {"scheduled", "running"}
            for exp in self.active
        )

    @staticmethod
    def _percentile(values: Iterable[float], q: float) -> float:
        seq = sorted(values)
        if not seq:
            return 0.0
        q = max(0.0, min(1.0, q))
        k = (len(seq) - 1) * q
        f = int(k)
        c = min(f + 1, len(seq) - 1)
        if f == c:
            return seq[f]
        d0 = seq[f] * (c - k)
        d1 = seq[c] * (k - f)
        return d0 + d1


ABILITY_TO_METRIC = {
    "reasoning_speed": "reasoning_speed",
    "learning_efficiency": "learning_rate",
    "memory_capacity": "memory_capacity",
    "relationship_management": "relationship_depth",
}


def calibrate_self_model(
    self_model: Any, performance_tracking: Dict[str, list], learning_rate: float = 0.1
) -> Dict[str, float]:
    """
    Compare auto-évaluations vs performances (dernière valeur),
    puis ajuste doucement les capacités du self_model.
    Retourne les deltas appliqués.
    """
    applied = {}
    for ability, metric in ABILITY_TO_METRIC.items():
        hist = performance_tracking.get(metric, [])
        if not hist:
            continue
        values = [float(item.get("value", 0.5)) for item in hist[-15:]]
        observed = values[-1]
        moving_avg = statistics.fmean(values) if values else observed
        adaptive_lr = _adaptive_learning_rate(values, learning_rate)
        blended_target = 0.6 * observed + 0.4 * moving_avg
        current = float(self_model.cognitive_abilities.get(ability, 0.5))
        new_val = (1 - adaptive_lr) * current + adaptive_lr * blended_target
        self_model.cognitive_abilities[ability] = max(0.0, min(1.0, new_val))
        applied[ability] = self_model.cognitive_abilities[ability] - current

    hist_load = performance_tracking.get("cognitive_load", [])
    if hist_load:
        load_values = [float(item.get("value", 0.5)) for item in hist_load[-15:]]
        observed_load = load_values[-1]
        observed_attention = 1.0 - observed_load
        adaptive_lr = _adaptive_learning_rate(load_values, learning_rate)
        current = float(self_model.cognitive_abilities.get("attention_control", 0.5))
        new_val = (1 - adaptive_lr) * current + adaptive_lr * observed_attention
        self_model.cognitive_abilities["attention_control"] = max(
            0.0, min(1.0, new_val)
        )
        applied["attention_control"] = (
            self_model.cognitive_abilities["attention_control"] - current
        )

    return applied


PLAN_LIBRARY: Dict[str, list] = {
    "reasoning_speed": [
        {
            "id": "focused_cycles_soft",
            "plan": {
                "strategy": "focused_cycles",
                "details": "2 cycles attention_focused puis 1 cycle normal",
            },
            "parameters": {"focus_cycles": 2, "normal_cycles": 1},
            "duration": 3,
            "base_improve": 0.08,
            "aggressiveness": 0.9,
        },
        {
            "id": "tempo_ramp",
            "plan": {
                "strategy": "tempo_ramp",
                "details": "augmenter la cadence de 5% par cycle pendant 3 cycles",
            },
            "parameters": {"increment": 0.05, "cycles": 3},
            "duration": 3,
            "base_improve": 0.12,
            "aggressiveness": 1.1,
        },
    ],
    "learning_rate": [
        {
            "id": "spaced_review_light",
            "plan": {
                "strategy": "spaced_review",
                "details": "répéter 2 prompts similaires à 1 min d'intervalle",
            },
            "parameters": {"repetitions": 2, "interval_sec": 60},
            "duration": 4,
            "base_improve": 0.07,
            "aggressiveness": 0.95,
        },
        {
            "id": "meta_reflection",
            "plan": {
                "strategy": "meta_reflection",
                "details": "1 question méta + 1 reformulation ciblée",
            },
            "parameters": {"reflection_depth": 2},
            "duration": 3,
            "base_improve": 0.1,
            "aggressiveness": 1.05,
        },
    ],
    "recall_accuracy": [
        {
            "id": "elaborative_q_default",
            "plan": {
                "strategy": "elaborative_q",
                "details": "ajouter 2 'pourquoi' à chaque explication",
            },
            "parameters": {"why_count": 2},
            "duration": 3,
            "base_improve": 0.08,
            "aggressiveness": 1.0,
        },
        {
            "id": "memory_anchor",
            "plan": {
                "strategy": "memory_palace",
                "details": "associer chaque point clé à une image vive",
            },
            "parameters": {"anchors": 3},
            "duration": 4,
            "base_improve": 0.09,
            "aggressiveness": 1.1,
        },
    ],
    "relationship_depth": [
        {
            "id": "personal_checkins",
            "plan": {
                "strategy": "relationship_checkin",
                "details": "1 question personnelle ciblée + 1 partage sincère",
            },
            "parameters": {"checkins": 1, "share_detail": True},
            "duration": 3,
            "base_improve": 0.07,
            "aggressiveness": 1.0,
        },
        {
            "id": "memory_refresh",
            "plan": {
                "strategy": "relationship_memory_refresh",
                "details": "relire 2 souvenirs relationnels et préparer un suivi",
            },
            "parameters": {"refresh_count": 2},
            "duration": 4,
            "base_improve": 0.06,
            "aggressiveness": 0.9,
        },
    ],
}


def _adaptive_learning_rate(values: Iterable[float], base_lr: float) -> float:
    seq = [float(v) for v in values if v is not None]
    if not seq:
        return base_lr
    recent = seq[-8:]
    if len(recent) < 2:
        return max(0.03, min(0.35, base_lr))
    trend = recent[-1] - recent[0]
    volatility = statistics.pstdev(recent) if len(recent) > 1 else 0.0
    lr = base_lr * (1 + max(-0.3, min(0.3, trend)))
    lr *= 1.0 / (1.0 + 2.5 * volatility)
    return max(0.03, min(0.35, lr))
