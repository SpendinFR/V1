"""Background maintenance scheduler for the cognitive architecture.

The previous version of this module had two issues that prevented the
scheduler from running correctly once imported:

* The standard library modules it relied on (``time``, ``os``, ``json`` …)
  were never imported.  Python therefore raised ``NameError`` exceptions as
  soon as the helper functions such as :func:`_now` or :func:`_safe_json`
  were executed.
* The reflection task attempted to import ``CognitiveDomain`` from a
  top-level ``metacognition`` module.  Inside the package hierarchy the
  correct import path is ``AGI_Evolutive.metacognition``; the truncated
  import made the task crash at runtime.

Both problems manifested as "truncated command"/runtime failures even
though the file passed static syntax checks.  Restoring the missing imports
and using the fully qualified package path fixes the scheduler logic.
"""

import json
import logging
import math
import os
import random
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.core.global_workspace import GlobalWorkspace
from AGI_Evolutive.knowledge.mechanism_store import MechanismStore
from AGI_Evolutive.cognition.principle_inducer import PrincipleInducer
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


def _safe_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_sanitize(obj), f, ensure_ascii=False, indent=2)


def _append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_sanitize(obj), ensure_ascii=False) + "\n")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class AdaptiveTaskPolicy:
    """Adapte les intervalles à partir d'un signal de récompense en ligne.

    La politique combine deux mécanismes :

    * Une moyenne mobile exponentielle avec dérive dont le facteur d'oubli est
      sélectionné par Thompson Sampling parmi ``beta_options``.
    * Une transformation tanh du score pour ajuster l'intervalle dans des
      bornes raisonnables (``min_scale``/``max_scale``).
    """

    beta_options: Sequence[float] = (0.2, 0.4, 0.6, 0.8)

    @staticmethod
    def _beta_key(beta: float) -> str:
        return f"{beta:.1f}"

    def __init__(
        self,
        base_interval: float,
        *,
        min_scale: float = 0.3,
        max_scale: float = 2.5,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_interval = float(max(0.5, base_interval))
        self.min_scale = float(max(0.1, min_scale))
        self.max_scale = float(max(self.min_scale, max_scale))

        self.score = 0.0
        self.current_interval = self.base_interval
        self.current_beta = 0.4
        self._beta_stats: Dict[str, Dict[str, float]] = {
            self._beta_key(beta): {"alpha": 1.0, "beta": 1.0} for beta in self.beta_options
        }

        if state:
            self.score = float(state.get("score", 0.0))
            self.current_interval = float(state.get("current_interval", self.base_interval))
            self.current_beta = float(state.get("current_beta", self.current_beta))
            stored_stats = state.get("beta_stats") or {}
            for key, val in stored_stats.items():
                if key in self._beta_stats and isinstance(val, dict):
                    alpha = float(val.get("alpha", 1.0))
                    beta = float(val.get("beta", 1.0))
                    self._beta_stats[key]["alpha"] = max(1e-3, alpha)
                    self._beta_stats[key]["beta"] = max(1e-3, beta)

        self._update_interval()

    # Thompson sampling over the beta (forgetting factor) candidates ------------
    def _sample_beta(self) -> float:
        draws = {}
        for beta in self.beta_options:
            stats = self._beta_stats[self._beta_key(beta)]
            a = max(1e-3, stats["alpha"])
            b = max(1e-3, stats["beta"])
            try:
                draws[beta] = random.betavariate(a, b)
            except Exception:
                draws[beta] = a / (a + b)
        chosen = max(draws, key=draws.get) if draws else self.current_beta
        self.current_beta = float(chosen)
        return self.current_beta

    def _update_interval(self) -> None:
        # Score -> tanh -> scale in [min_scale, max_scale]
        norm = math.tanh(self.score)
        scale = 1.0 - norm * 0.8  # high score -> faster (smaller interval)
        scale = _clamp(scale, self.min_scale, self.max_scale)
        self.current_interval = self.base_interval * scale

    def update(self, reward: float, duration: float, success: bool) -> None:
        forget = self._sample_beta()
        stats = self._beta_stats[self._beta_key(forget)]
        pos = max(0.0, reward)
        neg = max(0.0, -reward)
        # Apprentissage online des stats du TS avec dérive
        stats["alpha"] = (1.0 - forget) * stats["alpha"] + forget * (1.0 + pos)
        stats["beta"] = (1.0 - forget) * stats["beta"] + forget * (1.0 + neg)

        signed = reward
        if not success:
            signed -= 0.3
        # durée plus longue que prévu => légère pénalité
        ratio = duration / max(1e-3, self.base_interval)
        if ratio > 1.2:
            signed -= min(0.5, ratio - 1.0)
        elif ratio < 0.8:
            signed += min(0.3, 1.0 - ratio)

        self.score = (1.0 - forget) * self.score + forget * signed
        self._update_interval()

    def to_state(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "current_interval": self.current_interval,
            "current_beta": self.current_beta,
            "beta_stats": self._beta_stats,
            "base_interval": self.base_interval,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
        }


class OnlineLogistic:
    """Régression logistique online simple avec décroissance légère."""

    def __init__(self, n_features: int, lr: float = 0.1, l2: float = 1e-3, state: Optional[Dict[str, Any]] = None):
        self.n_features = int(n_features)
        self.lr = float(lr)
        self.l2 = float(max(0.0, l2))
        self.weights = [0.0] * self.n_features
        if state and isinstance(state.get("weights"), list):
            stored = list(state["weights"])
            for i in range(min(len(stored), self.n_features)):
                self.weights[i] = float(stored[i])

    def _logit(self, features: Sequence[float]) -> float:
        z = 0.0
        for w, x in zip(self.weights, features):
            z += w * float(x)
        return z

    def predict(self, features: Sequence[float]) -> float:
        z = _clamp(self._logit(features), -20.0, 20.0)
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, features: Sequence[float], label: float) -> None:
        label = _clamp(float(label), 0.0, 1.0)
        pred = self.predict(features)
        error = pred - label
        for idx in range(self.n_features):
            grad = error * float(features[idx]) + self.l2 * self.weights[idx]
            self.weights[idx] -= self.lr * grad

    def to_state(self) -> Dict[str, Any]:
        return {"weights": list(self.weights), "lr": self.lr, "l2": self.l2}


class OnlinePlattCalibrator:
    """Calibration probabiliste style Platt en mise à jour en ligne."""

    def __init__(self, lr: float = 0.05, state: Optional[Dict[str, Any]] = None):
        self.lr = float(lr)
        self.a = 1.0
        self.b = 0.0
        if state:
            self.a = float(state.get("a", self.a))
            self.b = float(state.get("b", self.b))

    def transform(self, p: float) -> float:
        p = _clamp(p, 1e-4, 1.0 - 1e-4)
        logit = math.log(p / (1.0 - p))
        z = _clamp(self.a * logit + self.b, -20.0, 20.0)
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, p: float, label: float) -> None:
        label = _clamp(label, 0.0, 1.0)
        p = _clamp(p, 1e-4, 1.0 - 1e-4)
        logit = math.log(p / (1.0 - p))
        z = _clamp(self.a * logit + self.b, -20.0, 20.0)
        pred = 1.0 / (1.0 + math.exp(-z))
        error = pred - label
        self.a -= self.lr * error * logit
        self.b -= self.lr * error

    def to_state(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "lr": self.lr}
class Scheduler:
    """
    Orchestrateur de cycles :
      - tâches déclaratives avec intervalle (s)
      - exécution en thread daemon
      - reprise après crash (persist last_run)
      - instrumentation JSONL
    Par défaut, enchaîne :
      homeostasis → consolidation → concept_extractor → episodic_linker →
      goals/planning → reflection → evolution_manager

    À ne pas confondre avec :class:`AGI_Evolutive.light_scheduler.LightScheduler`
    (planificateur léger synchronisé à appeler dans une boucle).  Cette
    version runtime persiste son état et tourne dans un thread dédié.
    """

    def __init__(self, arch, data_dir: str = "data"):
        self.arch = arch
        self.data_dir = data_dir
        self.paths = {
            "state": os.path.join(self.data_dir, "runtime/scheduler_state.json"),
            "log": os.path.join(self.data_dir, "runtime/scheduler.log.jsonl"),
        }
        os.makedirs(os.path.dirname(self.paths["state"]), exist_ok=True)
        self.state = _safe_json(self.paths["state"], {
            "created_at": _now(),
            "last_runs": {},   # task_name -> ts
            "enabled": True
        })
        self.state.setdefault("policies", {})
        self.state.setdefault("models", {})
        self.state["models"].setdefault("reflection", {})
        self.running = False
        self.thread = None

        # registre des tâches (nom -> dict)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._policies: Dict[str, AdaptiveTaskPolicy] = {}
        self._pending_reflection_update: Optional[Dict[str, Any]] = None

        reflection_state = self.state["models"].get("reflection", {})
        self._reflection_model = OnlineLogistic(
            n_features=6,
            state=reflection_state.get("logistic"),
        )
        self._reflection_calibrator = OnlinePlattCalibrator(
            state=reflection_state.get("calibrator"),
        )
        self._last_planning_outcome: Dict[str, Any] = {}
        self._last_metrics_snapshot: Dict[str, Any] = {}
        self._prev_metrics_value: float = 0.0
        self._last_metrics_value: float = 0.0
        self._last_metrics_delta: float = 0.0
        self._metrics_ready: bool = False
        self._last_applicable_mais: List[Any] = []

        self._register_default_tasks()

        self.workspace = getattr(self.arch, "global_workspace", None)

        policy = getattr(self.arch, "policy", None)
        mechanism_store = None
        if policy is not None and hasattr(policy, "_mechanisms"):
            mechanism_store = getattr(policy, "_mechanisms", None)
        if mechanism_store is None:
            mechanism_store = getattr(self.arch, "mechanism_store", None)
        if mechanism_store is None:
            mechanism_store = MechanismStore()
            if policy is not None and hasattr(policy, "_mechanisms"):
                try:
                    policy._mechanisms = mechanism_store
                except Exception:
                    pass
        setattr(self.arch, "mechanism_store", mechanism_store)
        self.mechanism_store = mechanism_store

        principle_inducer = getattr(self.arch, "principle_inducer", None)
        if principle_inducer is None:
            principle_inducer = PrincipleInducer(self.mechanism_store)
            setattr(self.arch, "principle_inducer", principle_inducer)
        self.principle_inducer = principle_inducer

        period = getattr(self.arch, "evolution_period", None)
        if period is None:
            period = getattr(self, "_evolution_period", None)
        if period is None:
            period = 20
        try:
            period_value = int(period)
        except Exception:
            try:
                period_value = int(float(period))
            except Exception:
                period_value = 20
        self._evolution_period = max(1, period_value)
        try:
            self._tick = int(getattr(self, "_tick", 0))
        except Exception:
            self._tick = 0
        try:
            self._tick_counter = int(getattr(self, "_tick_counter", self._tick))
        except Exception:
            self._tick_counter = self._tick

    # ---------- helpers ----------
    def _build_state_snapshot(self) -> Dict[str, Any]:
        arch = self.arch
        language = getattr(arch, "language", None)
        dialogue = None
        if language is not None:
            dialogue = getattr(language, "state", None)
            dialogue = getattr(language, "dialogue_state", dialogue)
        world = getattr(arch, "world_model", None)
        return {
            "beliefs": getattr(arch, "beliefs", None),
            "self_model": getattr(arch, "self_model", None),
            "dialogue": dialogue,
            "world": world,
            "memory": getattr(arch, "memory", None),
        }

    def _predicate_registry(self, state: Dict[str, Any]) -> Dict[str, Callable[..., bool]]:
        policy = getattr(self.arch, "policy", None)
        if policy is not None and hasattr(policy, "build_predicate_registry"):
            try:
                registry = policy.build_predicate_registry(state)
                if isinstance(registry, dict):
                    return registry
            except Exception:
                pass

        dialogue = state.get("dialogue")
        world = state.get("world")
        self_model = state.get("self_model")
        beliefs = state.get("beliefs")

        def _belief_contains(topic: Any) -> bool:
            if beliefs is None:
                return False
            for accessor in ("contains", "has_fact", "has_edge"):
                fn = getattr(beliefs, accessor, None)
                if callable(fn):
                    try:
                        if fn(topic):
                            return True
                    except Exception:
                        continue
            return False

        def _belief_confidence(topic: Any, threshold: float) -> bool:
            if beliefs is None:
                return False
            confidence_for = getattr(beliefs, "confidence_for", None)
            if not callable(confidence_for):
                return False
            try:
                return float(confidence_for(topic)) >= float(threshold)
            except Exception:
                return False

        registry: Dict[str, Callable[..., bool]] = {
            "request_is_sensitive": lambda st: getattr(dialogue, "is_sensitive", False) if dialogue else False,
            "audience_is_not_owner": lambda st: (
                getattr(dialogue, "audience_id", None) != getattr(dialogue, "owner_id", None)
                if dialogue
                else False
            ),
            "has_consent": lambda st: getattr(dialogue, "has_consent", False) if dialogue else False,
            "imminent_harm_detected": lambda st: getattr(world, "imminent_harm", False) if world else False,
            "has_commitment": lambda st, key: (
                self_model.has_commitment(key)
                if hasattr(self_model, "has_commitment")
                else False
            ),
            "belief_mentions": lambda st, topic: _belief_contains(topic),
            "belief_confidence_above": lambda st, topic, threshold: _belief_confidence(topic, threshold),
        }
        return registry

    def _get_workspace(self) -> Any:
        workspace = getattr(self, "workspace", None)
        if workspace is None:
            workspace = getattr(self.arch, "global_workspace", None)
        if workspace is None:
            policy = getattr(self.arch, "policy", None)
            if policy is None:
                return None
            workspace = GlobalWorkspace(policy=policy, planner=getattr(self.arch, "planner", None))
            setattr(self.arch, "global_workspace", workspace)
        self.workspace = workspace
        return workspace

    def _render_and_emit(self, decision: Dict[str, Any], state: Dict[str, Any]) -> None:
        arch = self.arch
        language = getattr(arch, "language", None)
        text = decision.get("decision_text") or decision.get("decision") or "noop"
        if language is not None and hasattr(language, "renderer"):
            renderer = getattr(language, "renderer", None)
            if renderer and hasattr(renderer, "render_reply"):
                try:
                    semantics = {"text": decision.get("decision_text") or decision.get("decision", ""), "meta": decision}
                    ctx = {
                        "last_message": getattr(getattr(language, "state", None), "last_user", ""),
                        "omitted_content": decision.get("omitted_content", False),
                        "state_snapshot": state,
                    }
                    text = renderer.render_reply(semantics, ctx)
                except Exception:
                    text = decision.get("decision_text") or text
        arch.last_output_text = text
        logger = getattr(arch, "logger", None)
        if logger and hasattr(logger, "write"):
            try:
                logger.write("gw.decision", decision=decision, rendered=text)
            except Exception:
                pass

    # ---------- registration ----------
    def _init_policy(self, name: str, interval_s: float) -> AdaptiveTaskPolicy:
        stored = self.state["policies"].get(name)
        policy_state: Optional[Dict[str, Any]] = None
        base_interval = float(interval_s)
        if isinstance(stored, dict):
            policy_state = dict(stored)
            policy_state["base_interval"] = base_interval
        policy = AdaptiveTaskPolicy(base_interval=base_interval, state=policy_state)
        self._policies[name] = policy
        return policy

    def register(self, name: str, fn: Callable[[], None], interval_s: float, jitter_s: float = 0.0):
        policy = self._init_policy(name, float(interval_s))
        self.tasks[name] = {
            "fn": fn,
            "interval": policy.current_interval,
            "jitter": float(jitter_s),
            "policy": policy,
            "base_interval": float(interval_s),
        }
        self.state["policies"][name] = policy.to_state()

    def _register_default_tasks(self):
        # Les fonctions sont toutes robustifiées (attr checks)
        self.register("homeostasis", self._task_homeostasis, interval_s=60.0, jitter_s=5.0)
        self.register("consolidation", self._task_consolidation, interval_s=180.0, jitter_s=10.0)
        self.register("concepts", self._task_concepts, interval_s=45.0, jitter_s=5.0)
        self.register("episodes", self._task_episodes, interval_s=60.0, jitter_s=5.0)
        self.register("planning", self._task_planning, interval_s=90.0, jitter_s=8.0)
        self.register("reflection", self._task_reflection, interval_s=120.0, jitter_s=10.0)
        self.register("evolution", self._task_evolution, interval_s=120.0, jitter_s=10.0)

    # ---------- adaptation helpers ----------
    def _scalarize_metrics(self, metrics: Dict[str, Any]) -> float:
        values: List[float] = []

        def _collect(obj: Any) -> None:
            if isinstance(obj, dict):
                for val in obj.values():
                    _collect(val)
            elif isinstance(obj, (int, float)):
                values.append(float(obj))

        _collect(metrics or {})
        if not values:
            return 0.0
        return sum(values) / float(len(values))

    def _compute_task_reward(self, name: str, ok: bool, duration: float) -> float:
        reward = 0.6 if ok else -0.9
        if name == "planning":
            outcome = self._last_planning_outcome or {}
            trust = float(outcome.get("trust_delta", 0.0))
            harm = float(outcome.get("harm_delta", 0.0))
            regret = float(outcome.get("regret", 0.0))
            reward += _clamp(trust - harm, -1.5, 1.5)
            reward -= _clamp(regret, 0.0, 1.5) * 0.5
        elif name == "evolution":
            reward += _clamp(self._last_metrics_delta, -2.0, 2.0) * 0.4
            if self._metrics_ready:
                reward += 0.2
        elif name == "episodes":
            applicable = len(self._last_applicable_mais or [])
            reward += _clamp(applicable / 5.0, 0.0, 1.0) * 0.3
        elif name == "reflection":
            reward += (self._derive_reflection_label(self._last_planning_outcome) - 0.5) * 0.8
        return _clamp(reward, -2.5, 2.5)

    def _update_task_policy(self, name: str, policy: AdaptiveTaskPolicy, ok: bool, duration: float) -> None:
        try:
            reward = self._compute_task_reward(name, ok, duration)
            policy.update(reward, duration, ok)
        except Exception:
            return
        self.state["policies"][name] = policy.to_state()
        self._apply_llm_policy(name, policy, reward, duration, ok)

    def _build_reflection_features(self) -> List[float]:
        outcome = self._last_planning_outcome or {}
        trust = float(outcome.get("trust_delta", 0.0))
        harm = float(outcome.get("harm_delta", 0.0))
        regret = float(outcome.get("regret", 0.0))
        metrics_delta = float(self._last_metrics_delta)
        applicable = len(self._last_applicable_mais or [])
        norm_applicable = _clamp(applicable / 10.0, 0.0, 1.0)
        return [
            1.0,
            trust,
            -harm,
            regret,
            metrics_delta,
            norm_applicable,
        ]

    def _derive_reflection_label(self, outcome: Optional[Dict[str, Any]]) -> float:
        if not outcome:
            return 0.5
        trust = float(outcome.get("trust_delta", 0.0))
        harm = float(outcome.get("harm_delta", 0.0))
        regret = float(outcome.get("regret", 0.0))
        score = trust - harm - 0.5 * regret
        return 1.0 if score >= 0.0 else 0.0

    def _update_reflection_model(self, outcome: Dict[str, Any]) -> None:
        if not outcome:
            return
        if not self._pending_reflection_update:
            return
        features = self._pending_reflection_update.get("features")
        raw_prob = self._pending_reflection_update.get("raw_prob")
        if not isinstance(features, list) or raw_prob is None:
            self._pending_reflection_update = None
            return
        label = self._derive_reflection_label(outcome)
        try:
            self._reflection_model.update(features, label)
            self._reflection_calibrator.update(raw_prob, label)
        except Exception:
            pass
        self.state["models"]["reflection"] = {
            "logistic": self._reflection_model.to_state(),
            "calibrator": self._reflection_calibrator.to_state(),
        }
        self._pending_reflection_update = None

    def _apply_llm_policy(
        self,
        name: str,
        policy: AdaptiveTaskPolicy,
        reward: float,
        duration: float,
        success: bool,
    ) -> None:
        payload = {
            "job": name,
            "current_interval": policy.current_interval,
            "base_interval": policy.base_interval,
            "reward": reward,
            "duration": duration,
            "success": success,
        }
        response = try_call_llm_dict(
            "scheduler",
            input_payload=payload,
            logger=logger,
        )
        if not response:
            policy.llm_last = None  # type: ignore[attr-defined]
            return

        decision = str(response.get("policy", "")).strip().lower()
        if "accel" in decision:
            policy.current_interval = max(policy.base_interval * 0.5, policy.current_interval * 0.8)
        elif "ralent" in decision:
            policy.current_interval = min(policy.base_interval * 3.0, policy.current_interval * 1.2)
        policy.llm_last = dict(response)  # type: ignore[attr-defined]

    # ---------- run loop ----------
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _loop(self):
        while self.running and self.state.get("enabled", True):
            now = _now()
            for name, cfg in self.tasks.items():
                last = float(self.state["last_runs"].get(name, 0.0))
                policy = cfg.get("policy")
                interval = cfg.get("interval", 0.0)
                if policy is not None:
                    interval = policy.current_interval
                due = last + interval
                if now >= due:
                    # jitter léger
                    j = cfg["jitter"]
                    if j > 0:
                        time.sleep(min(j, 0.25))  # lissé

                    t0 = _now()
                    ok = True
                    err = None
                    try:
                        cfg["fn"]()
                    except Exception as e:
                        ok = False
                        err = f"{e}\n{traceback.format_exc()}"
                    t1 = _now()

                    self.state["last_runs"][name] = t1
                    duration = t1 - t0
                    if policy is not None:
                        self._update_task_policy(name, policy, ok, duration)
                        cfg["interval"] = policy.current_interval
                    if policy is not None:
                        self.state["policies"][name] = policy.to_state()
                    _write_json(self.paths["state"], self.state)
                    _append_jsonl(self.paths["log"], {
                        "t0": t0, "t1": t1, "dt": duration, "task": name, "ok": ok, "err": err,
                        "interval": interval,
                        "next_interval": cfg.get("interval"),
                    })
            time.sleep(0.5)

    # ---------- individual tasks (robustes) ----------
    def _task_homeostasis(self):
        # module emotions/homeostasis (si existant)
        hm = getattr(self.arch, "homeostasis", None)
        if hm and hasattr(hm, "run_homeostasis_cycle"):
            hm.run_homeostasis_cycle()
        # sinon, essayer emotions.adjust_if_needed()
        emo = getattr(self.arch, "emotions", None)
        if emo and hasattr(emo, "adjust_if_needed"):
            try:
                emo.adjust_if_needed()
            except Exception:
                pass

    def _task_consolidation(self):
        # mémoire/learning : consolidation (si dispo)
        # Essais : learning.consolidate(), memory.consolidate(), memory.consolidator.run_once_now()
        learn = getattr(self.arch, "learning", None)
        mem = getattr(self.arch, "memory", None)
        # learning
        if learn and hasattr(learn, "consolidate"):
            try:
                learn.consolidate()
                return
            except Exception:
                pass
        # memory
        if mem and hasattr(mem, "consolidate"):
            try:
                mem.consolidate()
                return
            except Exception:
                pass
        # consolidator (style VIKI+)
        cons = getattr(self.arch, "consolidator", None)
        if cons and hasattr(cons, "run_once_now"):
            try:
                cons.run_once_now(scope="auto")
            except Exception:
                pass

    def _task_concepts(self):
        ce = getattr(self.arch, "concept_extractor", None)
        if ce and hasattr(ce, "step"):
            ce.step()

    def _task_episodes(self):
        el = getattr(self.arch, "episodic_linker", None)
        if el and hasattr(el, "step"):
            el.step()
        workspace = self._get_workspace()
        state = self._build_state_snapshot()
        predicate_registry = self._predicate_registry(state)

        applicable: List[Any] = []
        try:
            applicable = list(self.mechanism_store.scan_applicable(state, predicate_registry))
        except Exception:
            applicable = []
        self._last_applicable_mais = applicable

        if workspace is None:
            return

        for mechanism in applicable:
            try:
                bids = list(mechanism.propose(state))
            except Exception:
                continue
            for bid in bids:
                try:
                    workspace.submit(bid)
                except AttributeError:
                    attention = max(0.0, min(1.0, getattr(bid, "expected_info_gain", 0.0)))
                    workspace.submit_bid(
                        bid.payload.get("origin", getattr(bid, "source", "mai")),
                        bid.action_hint,
                        attention,
                        bid.payload,
                    )

    def _task_planning(self):
        goals = getattr(self.arch, "goals", None)
        # Ex : goals.refresh_plans(), goals.step(), planner.plan_for_goal() …
        if goals and hasattr(goals, "step"):
            try:
                goals.step()
                return
            except Exception:
                pass
        if goals and hasattr(goals, "refresh_plans"):
            try:
                goals.refresh_plans()
            except Exception:
                pass
        workspace = self._get_workspace()
        policy = getattr(self.arch, "policy", None)
        if workspace and policy and hasattr(workspace, "step") and hasattr(policy, "decide_with_bids"):
            state = self._build_state_snapshot()
            try:
                workspace.step(state, timebox_iters=2)
                winners = list(workspace.winners())
            except Exception:
                winners = []
            if not winners:
                try:
                    winners = list(workspace.last_trace())
                except Exception:
                    winners = []
            try:
                decision = policy.decide_with_bids(
                    winners,
                    state,
                    global_workspace=workspace,
                    proposer=getattr(self.arch, "proposer", None),
                    homeo=getattr(self.arch, "homeostasis", None) or getattr(self.arch, "emotions", None),
                    planner=getattr(self.arch, "planner", None),
                    ctx={"scheduler": True, "workspace_trace": [bid.origin_tag() for bid in winners]},
                )
            except Exception:
                decision = None

            if decision:
                emitted = False
                runtime = getattr(self.arch, "runtime", None)
                response_api = getattr(runtime, "response", None) if runtime else None
                if (
                    runtime
                    and hasattr(runtime, "emit")
                    and response_api is not None
                    and hasattr(response_api, "format_agent_reply")
                ):
                    try:
                        utterance = response_api.format_agent_reply(decision)
                        runtime.emit(utterance)
                        emitted = True
                    except Exception:
                        emitted = False
                if not emitted:
                    self._render_and_emit(decision, state)

                applicable_mais = list(getattr(self, "_last_applicable_mais", []))
                self._last_planning_outcome = {}
                if applicable_mais:
                    try:
                        from AGI_Evolutive.social.social_critic import SocialCritic

                        critic = SocialCritic()
                        outcome: Dict[str, Any] = {}
                        if hasattr(critic, "last_outcome"):
                            try:
                                outcome = critic.last_outcome() or {}
                            except Exception:
                                outcome = {}
                        if outcome:
                            self._last_planning_outcome = outcome
                            self._update_reflection_model(outcome)
                        for mechanism in applicable_mais:
                            try:
                                wins = 1.0 if any(getattr(bid, "source", "") == f"MAI:{mechanism.id}" for bid in winners) else 0.0
                                feedback = {
                                    "activation": 1.0,
                                    "wins": wins,
                                    "benefit": float(outcome.get("trust_delta", 0.0))
                                    - float(outcome.get("harm_delta", 0.0)),
                                    "regret": float(outcome.get("regret", 0.0)),
                                }
                                if hasattr(mechanism, "update_from_feedback"):
                                    mechanism.update_from_feedback(feedback)
                                try:
                                    self.mechanism_store.update(mechanism)
                                except Exception:
                                    pass
                            except Exception:
                                continue
                    except Exception:
                        pass
                # If no social evaluation is available we keep the pending
                # reflection update until feedback arrives, avoiding neutral
                # training data that would collapse the model.

    def _task_reflection(self):
        mc = getattr(self.arch, "metacognition", None) or getattr(self.arch, "metacognitive_system", None)
        if not mc or not hasattr(mc, "trigger_reflection"):
            self._pending_reflection_update = None
            return
        try:
            # réflexion légère récurrente (domain REASONING par défaut)
            from AGI_Evolutive.metacognition import CognitiveDomain
            features = self._build_reflection_features()
            raw_prob = self._reflection_model.predict(features)
            calibrated = self._reflection_calibrator.transform(raw_prob)
            urgency = _clamp(calibrated, 0.15, 0.95)
            depth = 1 if urgency < 0.6 else 2
            self._pending_reflection_update = {"features": features, "raw_prob": raw_prob}
            mc.trigger_reflection(
                trigger="periodic_scheduler",
                domain=CognitiveDomain.REASONING,
                urgency=urgency,
                depth=depth
            )
        except Exception:
            self._pending_reflection_update = None
            pass

    def _task_evolution(self):
        evo = getattr(self.arch, "evolution", None)
        if evo and hasattr(evo, "record_cycle"):
            try:
                # enregistre snapshot + génère recommandations
                evo.record_cycle(extra_tags={"via": "scheduler"})
                # potentiellement proposer des évolutions (non destructif)
                evo.propose_evolution()
            except Exception:
                pass
        self._tick = getattr(self, "_tick", 0) + 1
        self._tick_counter = self._tick
        if self._tick % self._evolution_period == 0:
            arch = self.arch
            recent_docs: List[Any]
            recent_dialogues: List[Any]

            if hasattr(arch, "recent_docs"):
                recent_docs = list(getattr(arch, "recent_docs") or [])
            else:
                recent_docs = []
                memory = getattr(arch, "memory", None)
                if memory and hasattr(memory, "get_recent_memories"):
                    try:
                        recent_docs = memory.get_recent_memories(n=200)
                    except Exception:
                        recent_docs = []

            if hasattr(arch, "recent_dialogues"):
                recent_dialogues = list(getattr(arch, "recent_dialogues") or [])
            else:
                recent_dialogues = []
                memory = getattr(arch, "memory", None)
                dialogue_log = getattr(memory, "interactions", None)
                if dialogue_log and hasattr(dialogue_log, "get_recent"):
                    try:
                        recent_dialogues = dialogue_log.get_recent(100)
                    except Exception:
                        recent_dialogues = []

            metrics_snapshot: Dict[str, Any] = {}
            metrics = getattr(arch, "metrics", None)
            if metrics and hasattr(metrics, "snapshot"):
                try:
                    metrics_snapshot = metrics.snapshot() or {}
                except Exception:
                    metrics_snapshot = {}
            self._last_metrics_snapshot = metrics_snapshot or {}
            new_value = self._scalarize_metrics(self._last_metrics_snapshot)
            if self._metrics_ready:
                self._last_metrics_delta = new_value - self._last_metrics_value
            else:
                self._metrics_ready = True
                self._last_metrics_delta = 0.0
            self._prev_metrics_value = self._last_metrics_value
            self._last_metrics_value = new_value

            try:
                self.principle_inducer.run(recent_docs, recent_dialogues, metrics_snapshot)
            except Exception:
                pass
