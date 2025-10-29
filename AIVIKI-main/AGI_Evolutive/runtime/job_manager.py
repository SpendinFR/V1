# Gestionnaire de jobs complet (priorités, deux files, budgets, annulation,
# idempotence, progrès, persistance JSONL, drain des complétions côté thread principal).
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List, Deque, Tuple, Set, Iterable
import time, threading, heapq, os, json, uuid, traceback, collections, math, random, logging

from AGI_Evolutive.utils.llm_service import try_call_llm_dict, update_urgent_state


LOGGER = logging.getLogger(__name__)


_THREAD_CONTEXT = threading.local()


def _push_thread_tokens(ctx_id: str, tokens: Iterable[str]) -> None:
    token_set = {t for t in tokens if t}
    if not token_set:
        return
    stack: List[Tuple[str, Set[str]]] = getattr(_THREAD_CONTEXT, "stack", [])
    stack.append((ctx_id, token_set))
    _THREAD_CONTEXT.stack = stack


def _pop_thread_tokens(ctx_id: str) -> None:
    stack: List[Tuple[str, Set[str]]] = getattr(_THREAD_CONTEXT, "stack", [])
    if not stack:
        return
    for idx in range(len(stack) - 1, -1, -1):
        if stack[idx][0] == ctx_id:
            stack.pop(idx)
            break
    if stack:
        _THREAD_CONTEXT.stack = stack
    else:
        try:
            delattr(_THREAD_CONTEXT, "stack")
        except AttributeError:
            pass


def _current_thread_tokens() -> Set[str]:
    stack: List[Tuple[str, Set[str]]] = getattr(_THREAD_CONTEXT, "stack", [])
    tokens: Set[str] = set()
    for _, subset in stack:
        tokens.update(subset)
    return tokens


def _now() -> float:
    return time.time()


@dataclass(order=True)
class _PQItem:
    # heapq ordonne par (neg_prio, created, seq, job_id)
    sort_key: Tuple[float, float, int] = field(init=False, repr=False)
    neg_prio: float
    created_ts: float
    seq: int
    job_id: str

    def __post_init__(self):
        self.sort_key = (self.neg_prio, self.created_ts, self.seq)


@dataclass
class Job:
    id: str
    kind: str  # ex: "io", "compute", "nlp"
    queue: str  # "interactive" | "background"
    priority: float  # 0..1
    fn: Optional[Callable] = field(repr=False, default=None)
    args: Dict[str, Any] = field(default_factory=dict)
    key: Optional[str] = None  # idempotence key (même job)
    timeout_s: Optional[float] = None
    status: str = "queued"  # queued|running|done|error|cancelled
    created_ts: float = field(default_factory=_now)
    started_ts: float = 0.0
    finished_ts: float = 0.0
    progress: float = 0.0  # 0..1
    result: Any = None
    error: Optional[str] = None
    trace: Optional[str] = None
    base_priority: float = 0.0
    predicted_priority: float = 0.0
    context_features: Dict[str, float] = field(default_factory=dict, repr=False)
    metrics: Dict[str, Any] = field(default_factory=dict, repr=False)
    priority_tokens: Set[str] = field(default_factory=set, repr=False)


class JobContext:
    """Contexte passé à la fonction du job (pour progrès, cancel, timeouts)."""

    def __init__(self, jm: "JobManager", job_id: str):
        self._jm = jm
        self._job_id = job_id

    def update_progress(self, p: float):
        self._jm._update_progress(self._job_id, max(0.0, min(1.0, float(p))))

    def cancelled(self) -> bool:
        return self._jm._is_cancelled(self._job_id)

    def pause_if_urgent_active(self, timeout: float = 0.2) -> bool:
        """Bloque si une chaîne urgente est active (pour jobs coopératifs)."""

        return self._jm.pause_if_urgent_active(timeout=timeout)


class _OnlinePriorityModel:
    """Petit modèle logistique online pour ajuster les priorités."""

    def __init__(
        self,
        lr: float = 0.2,
        l2: float = 1e-4,
        min_weight: float = 0.1,
        max_weight: float = 0.6,
        exploration: float = 0.05,
    ):
        self.lr = lr
        self.l2 = l2
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.exploration = exploration
        self.weights: Dict[str, float] = collections.defaultdict(float)
        self._updates = 0

    def _iter_features(self, feats: Dict[str, float]):
        for key, value in feats.items():
            try:
                v = float(value)
            except Exception:
                continue
            if math.isnan(v) or math.isinf(v):
                continue
            yield key, v
        yield "__bias__", 1.0

    def _linear_sum(self, feats: Dict[str, float]) -> float:
        return sum(self.weights[key] * value for key, value in self._iter_features(feats))

    def predict(self, feats: Dict[str, float]) -> float:
        z = self._linear_sum(feats)
        if z >= 30:
            return 1.0
        if z <= -30:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, feats: Dict[str, float], reward: float) -> float:
        reward = max(0.0, min(1.0, float(reward)))
        pred = self.predict(feats)
        error = pred - reward
        for key, value in self._iter_features(feats):
            grad = error * value + self.l2 * self.weights[key]
            self.weights[key] -= self.lr * grad
        self._updates += 1
        return pred

    @property
    def confidence(self) -> float:
        return min(self.max_weight, self.min_weight + 0.02 * self._updates)

    @property
    def noise_scale(self) -> float:
        return max(0.0, self.exploration / (1.0 + 0.05 * self._updates))

    @property
    def updates(self) -> int:
        return self._updates


class JobManager:
    """
    Deux files : interactive (latence faible) et background (batch).
    Priorités [0..1], budgets par tick, idempotence, annulation, timeouts,
    persistance JSONL, complétions drainables par le thread principal.
    """

    def __init__(self, arch, data_dir: str = "data", workers_interactive: int = 1, workers_background: int = 2):
        self.arch = arch
        self.data_dir = data_dir
        self.paths = {
            "log": os.path.join(self.data_dir, "runtime/jobs.log.jsonl"),
            "state": os.path.join(self.data_dir, "runtime/jobs.state.json"),
        }
        os.makedirs(os.path.dirname(self.paths["log"]), exist_ok=True)

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._model_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        self._seq = 0
        self._jobs: Dict[str, Job] = {}
        self._key_map: Dict[str, str] = {}  # key -> job_id (idempotence)
        self._cancelled: set[str] = set()
        self._pq_inter: List[_PQItem] = []
        self._pq_back: List[_PQItem] = []
        self._completions: Deque[Dict[str, Any]] = collections.deque(maxlen=512)
        self._priority_model = _OnlinePriorityModel()
        self._urgent_jobs: Set[str] = set()
        self._urgent_token_counts: Dict[str, int] = collections.defaultdict(int)
        self._manual_urgent_counts: Dict[str, int] = collections.defaultdict(int)
        self._urgent_contexts: Dict[str, Set[str]] = {}
        self._urgent_state: bool = False
        self._urgent_condition = threading.Condition(self._lock)

        # budgets (ajuste si besoin)
        self.budgets = {
            "interactive": {"max_running": workers_interactive},
            "background": {"max_running": workers_background},
        }
        self._lane_capacity = {
            "interactive": max(1, int(workers_interactive)),
            "background": max(1, int(workers_background)),
        }
        self._lane_stats: Dict[str, Dict[str, Any]] = {
            "interactive": {"count": 0},
            "background": {"count": 0},
        }
        self._running_inter = 0
        self._running_back = 0

        # Démarre les workers
        self._alive = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        try:
            from AGI_Evolutive.utils import llm_service

            llm_service.register_urgent_gate(
                check_active=self.has_active_urgent_chain,
                allow_current=self.current_thread_has_urgent_permission,
            )
        except Exception:
            pass
        try:
            update_urgent_state(False)
        except Exception:
            pass

    def _current_focus_topic(self) -> Optional[str]:
        arch = getattr(self, "arch", None)
        if arch is None:
            return None
        return getattr(arch, "_current_topic", None) or getattr(arch, "current_topic", None)

    def _build_context_features_locked(self, job: Job, base_priority: float) -> Dict[str, float]:
        """Construit des features (suppose self._lock détenu)."""
        features: Dict[str, float] = {
            "base_priority": float(base_priority),
            "queue_interactive": 1.0 if job.queue == "interactive" else 0.0,
            "queue_background": 1.0 if job.queue == "background" else 0.0,
            "priority_offset": float(base_priority) - 0.5,
        }
        features[f"kind={job.kind}"] = 1.0
        running_inter = self._running_inter
        running_back = self._running_back
        pq_inter = len(self._pq_inter)
        pq_back = len(self._pq_back)
        cap_inter = max(1, self._lane_capacity["interactive"])
        cap_back = max(1, self._lane_capacity["background"])
        features["running_inter_norm"] = running_inter / cap_inter
        features["running_back_norm"] = running_back / cap_back
        features["queue_inter_norm"] = pq_inter / float(cap_inter * 4)
        features["queue_back_norm"] = pq_back / float(cap_back * 4)
        features["budget_inter_norm"] = self.budgets["interactive"]["max_running"] / cap_inter
        features["budget_back_norm"] = self.budgets["background"]["max_running"] / cap_back
        load_total = (running_inter + pq_inter + running_back + pq_back)
        features["global_pressure"] = load_total / float(cap_inter + cap_back)
        if job.timeout_s is not None:
            features["timeout_norm"] = min(1.0, float(job.timeout_s) / 60.0)
            features["has_timeout"] = 1.0
        else:
            features["has_timeout"] = 0.0
        args_len = len(job.args or {})
        features["args_size_norm"] = min(1.0, args_len / 8.0)
        focus = self._current_focus_topic()
        if focus:
            features[f"focus={str(focus)[:48]}"] = 1.0
        return features

    @staticmethod
    def _blend_priority(base_priority: float, model_pred: float, confidence: float, noise: float) -> float:
        alpha = max(0.0, min(1.0, confidence))
        blended = (1.0 - alpha) * float(base_priority) + alpha * float(model_pred) + float(noise)
        return max(0.0, min(1.0, blended))

    # ------------------- API publique -------------------
    def submit(
        self,
        *,
        kind: str,
        fn: Callable[[JobContext, Dict[str, Any]], Any],
        args: Dict[str, Any] | None = None,
        queue: str = "background",
        priority: float = 0.5,
        key: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> str:
        """Dépose un job. Idempotent si `key` est fournie."""
        args = args or {}
        priority = max(0.0, min(1.0, float(priority)))
        queue = "interactive" if queue == "interactive" else "background"

        with self._lock:
            if key and key in self._key_map:
                jid = self._key_map[key]
                j = self._jobs.get(jid)
                if j and j.status in {"queued", "running"}:
                    return jid

            jid = str(uuid.uuid4())
            trigger_token, chain_tokens, source_token = self._extract_priority_tokens(args)
            lane_override_hint = self._should_force_interactive(args)
            effective_queue = "interactive" if lane_override_hint else queue
            job = Job(
                id=jid,
                kind=kind,
                queue=effective_queue,
                priority=priority,
                fn=fn,
                args=args,
                key=key,
                timeout_s=timeout_s,
            )
            job.base_priority = priority
            job.metrics["base_priority"] = priority
            context_features = self._build_context_features_locked(job, priority)
            job.context_features = context_features
            llm_overlay = self._llm_rescore(job, context_features)
            with self._model_lock:
                model_pred = self._priority_model.predict(context_features)
                confidence = self._priority_model.confidence
                noise_scale = self._priority_model.noise_scale
            noise = random.uniform(-noise_scale, noise_scale) if noise_scale > 0 else 0.0
            adjusted_priority = self._blend_priority(priority, model_pred, confidence, noise)
            if llm_overlay is not None:
                adjusted_priority = max(0.0, min(1.0, float(llm_overlay)))
            override = self._priority_override(job, args)
            if override is not None:
                adjusted_priority = override[0]
                job.metrics.setdefault("overrides", {})["trigger_priority"] = override[1]
            job.predicted_priority = model_pred
            job.priority = adjusted_priority
            if lane_override_hint:
                job.metrics.setdefault("overrides", {})["lane"] = lane_override_hint
            self._register_priority_tokens(job, trigger_token, chain_tokens, source_token)
            job.metrics.update(
                {
                    "model_prediction": model_pred,
                    "confidence": confidence,
                    "exploration_noise": noise,
                    "adjusted_priority": adjusted_priority,
                    "context_size": len(context_features),
                }
            )
            if llm_overlay is not None:
                job.metrics.setdefault("llm_guidance", {})["priority"] = adjusted_priority
            if context_features:
                job.metrics["context_preview"] = dict(list(context_features.items())[:8])
            self._jobs[jid] = job
            if key:
                self._key_map[key] = jid
            self._push_pq(job)
            self._log({"event": "submit", "job": self._job_view(job)})
            with self._condition:
                self._condition.notify_all()
            return jid

    def _llm_rescore(self, job: Job, context_features: Dict[str, float]) -> Optional[float]:
        payload = {
            "job": {
                "id": job.id,
                "kind": job.kind,
                "queue": job.queue,
                "priority": job.priority,
                "base_priority": job.base_priority,
                "timeout_s": job.timeout_s,
                "args_keys": sorted(list(job.args.keys())),
            },
            "context_features": context_features,
        }
        response = try_call_llm_dict(
            "runtime_job_manager",
            input_payload=payload,
            logger=LOGGER,
        )
        if not response:
            return None
        priority = response.get("priority")
        if isinstance(priority, (int, float)):
            job.metrics.setdefault("llm_guidance", {})["justification"] = response.get("justification", "")
            return float(max(0.0, min(1.0, priority)))
        return None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs and self._jobs[job_id].status in {"queued", "running"}:
                self._cancelled.add(job_id)
                self._log({"event": "cancel", "job_id": job_id})
                return True
            return False

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            return self._job_view(j) if j else None

    def poll_completed(self, max_n: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with self._lock:
            for _ in range(min(max_n, len(self._completions))):
                out.append(self._completions.popleft())
        return out

    def drain_to_memory(self, memory) -> int:
        """A appeler côté thread principal (ex: dans _tick_background_systems)."""
        done = self.poll_completed(128)
        n = 0
        for ev in done:
            try:
                memory.add_memory({"kind": "job_event", "content": ev.get("event", ""), "metadata": ev})
                n += 1
            except Exception:
                pass
        return n

    # ------------------- internes -------------------
    def _push_pq(self, job: Job):
        self._seq += 1
        effective_priority = float(job.priority)
        urgent_types = {"SIGNAL", "THREAT", "NEED"}
        if job.priority_tokens and job.priority_tokens.intersection(urgent_types):
            effective_priority += 1e-3
        item = _PQItem(
            neg_prio=-effective_priority,
            created_ts=job.created_ts,
            seq=self._seq,
            job_id=job.id,
        )
        if job.queue == "interactive":
            heapq.heappush(self._pq_inter, item)
        else:
            heapq.heappush(self._pq_back, item)

    def _select_next_job_locked(self) -> Tuple[Optional[str], Optional[str]]:
        lane_item: Optional[Tuple[str, _PQItem]] = None
        urgent_active = any(count > 0 for count in self._urgent_token_counts.values())
        urgent_interactive_pending = self._has_pending_urgent_interactive_locked()
        if self._pq_inter:
            lane_item = ("interactive", self._pq_inter[0])
        if self._pq_back:
            back_item = self._pq_back[0]
            if lane_item is None:
                lane_item = ("background", back_item)
            elif not (urgent_active or urgent_interactive_pending):
                current_lane, current_item = lane_item
                current_prio = -float(current_item.neg_prio)
                back_prio = -float(back_item.neg_prio)
                if back_prio > current_prio:
                    lane_item = ("background", back_item)
                elif math.isclose(back_prio, current_prio):
                    if current_lane != "interactive":
                        lane_item = ("interactive", self._pq_inter[0])
                    else:
                        if back_item.created_ts < current_item.created_ts:
                            lane_item = ("background", back_item)
        if lane_item is None:
            return None, None
        lane, item = lane_item
        pq = self._pq_inter if lane == "interactive" else self._pq_back
        heapq.heappop(pq)
        return lane, item.job_id

    def _pop_next_any(self) -> Tuple[Optional[str], Optional[str]]:
        with self._condition:
            while self._alive:
                if self._running_inter or self._running_back:
                    self._condition.wait(timeout=0.1)
                    continue
                lane, jid = self._select_next_job_locked()
                if jid is not None:
                    return lane, jid
                self._condition.wait(timeout=0.1)
            return None, None

    def _has_pending_urgent_interactive_locked(self) -> bool:
        if not self._pq_inter:
            return False
        urgent_types = {"SIGNAL", "THREAT", "NEED"}
        for item in self._pq_inter:
            job = self._jobs.get(item.job_id)
            if job and job.priority_tokens and job.priority_tokens.intersection(urgent_types):
                return True
        return False

    def _should_defer_background_locked(self) -> bool:
        if not self._pq_inter:
            return False
        if self._has_pending_urgent_interactive_locked():
            return True
        if self._urgent_state:
            return True
        return False

    def _compute_reward(self, j: Job) -> Tuple[float, float]:
        finished = j.finished_ts or _now()
        started = j.started_ts or j.created_ts or finished
        latency = max(0.0, finished - started)
        success = 1.0 if j.status == "done" else 0.0
        progress_bonus = max(0.0, min(1.0, j.progress)) * 0.2
        ref_latency = 2.0 if j.queue == "interactive" else 10.0
        latency_penalty = min(0.7, latency / max(0.1, ref_latency) * 0.7)
        reward = success + progress_bonus - latency_penalty
        if j.status == "error":
            reward *= 0.5
        if j.status == "cancelled":
            reward *= 0.2
        return max(0.0, min(1.0, reward)), latency

    def _update_priority_model(self, job: Job, reward: float):
        if not job.context_features:
            return
        with self._model_lock:
            self._priority_model.update(job.context_features, reward)

    def _lane_snapshot(self, lane: str) -> Tuple[int, int, int]:
        with self._lock:
            if lane == "interactive":
                return (
                    len(self._pq_inter),
                    self._running_inter,
                    self.budgets["interactive"]["max_running"],
                )
            return (
                len(self._pq_back),
                self._running_back,
                self.budgets["background"]["max_running"],
            )

    @staticmethod
    def _ema(previous: Optional[float], value: float, alpha: float) -> float:
        if previous is None:
            return value
        return (1.0 - alpha) * previous + alpha * value

    def _record_completion_metrics(self, job: Job, reward: float, latency: float):
        lane = job.queue
        stats = self._lane_stats.get(lane)
        if stats is None:
            return
        ema_alpha = 0.2
        queue_len, running, lane_budget = self._lane_snapshot(lane)
        load = queue_len + running
        with self._stats_lock:
            stats["count"] = stats.get("count", 0) + 1
            stats["ema_reward"] = self._ema(stats.get("ema_reward"), reward, ema_alpha)
            stats["ema_latency"] = self._ema(stats.get("ema_latency"), latency, ema_alpha)
            stats["ema_load"] = self._ema(stats.get("ema_load"), float(load), ema_alpha)
            stats["ema_success"] = self._ema(stats.get("ema_success"), 1.0 if job.status == "done" else 0.0, ema_alpha)
            stats["last_latency"] = latency
            stats["last_reward"] = reward
            stats["last_load"] = load
            stats["last_status"] = job.status
            job.metrics["lane_budget_before"] = lane_budget
            self._maybe_adjust_budgets(lane, stats)
            with self._lock:
                job.metrics["lane_budget_after"] = self.budgets[lane]["max_running"]
        job.metrics["lane_load_after"] = load

    def _maybe_adjust_budgets(self, lane: str, stats: Dict[str, Any]):
        now = _now()
        last_adjust = stats.get("last_adjust_ts", 0.0)
        if stats.get("count", 0) < 8:
            return
        if now - last_adjust < 5.0:
            return
        ema_reward = stats.get("ema_reward")
        ema_load = stats.get("ema_load")
        if ema_reward is None or ema_load is None:
            stats["last_adjust_ts"] = now
            return
        with self._lock:
            current = self.budgets[lane]["max_running"]
        capacity = self._lane_capacity[lane]
        target = current
        if ema_load > (current * 1.2) and ema_reward >= 0.4 and current < capacity:
            target = min(capacity, current + 1)
        elif ema_reward < 0.2 and current > 1:
            target = max(1, current - 1)
        stats["last_adjust_ts"] = now
        if target != current:
            with self._lock:
                self.budgets[lane]["max_running"] = target
            self._log(
                {
                    "event": "budget_update",
                    "lane": lane,
                    "max_running": target,
                    "ema_reward": ema_reward,
                    "ema_load": ema_load,
                }
            )

    def _start_job(self, j: Job):
        j.status = "running"
        j.started_ts = _now()
        if j.queue == "interactive":
            self._running_inter += 1
        else:
            self._running_back += 1
        self._log({"event": "start", "job": self._job_view(j)})

    def _finish_job(self, j: Job, ok: bool, result: Any = None, error: Optional[str] = None, trace: Optional[str] = None):
        self._release_priority_tokens(j)
        j.finished_ts = _now()
        j.status = "done" if ok else ("cancelled" if (j.id in self._cancelled) else "error")
        j.result = result
        j.error = error
        j.trace = trace
        if j.queue == "interactive":
            self._running_inter = max(0, self._running_inter - 1)
        else:
            self._running_back = max(0, self._running_back - 1)
        reward, latency = self._compute_reward(j)
        success_flag = 1.0 if j.status == "done" else 0.0
        j.metrics.update(
            {
                "reward": reward,
                "latency_s": latency,
                "success": success_flag,
                "finished_ts": j.finished_ts,
            }
        )
        event_payload = {
            "event": j.status,
            "job": self._job_view(j),
            "reward": reward,
            "latency": latency,
            "priority": {
                "base": j.base_priority,
                "predicted": j.predicted_priority,
                "adjusted": j.priority,
            },
            "metrics": dict(j.metrics),
        }
        if error:
            event_payload["error"] = error
        self._update_priority_model(j, reward)
        self._record_completion_metrics(j, reward, latency)
        with self._lock:
            self._completions.append(event_payload)
        self._log(event_payload)
        self._persist_state()

    def _update_progress(self, job_id: str, p: float):
        with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return
            j.progress = p
            self._log({"event": "progress", "job_id": job_id, "progress": float(p)})

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def _worker_loop(self):
        while self._alive:
            lane, jid = self._pop_next_any()
            if jid is None:
                time.sleep(0.01)
                continue
            with self._lock:
                j = self._jobs.get(jid)
                if j and lane == "background" and self._should_defer_background_locked():
                    self._push_pq(j)
                    self._condition.notify_all()
                    continue
            if not j:
                with self._condition:
                    self._condition.notify_all()
                continue
            # cancellation avant start ?
            if self._is_cancelled(j.id):
                self._finish_job(j, ok=False, error="cancelled_before_start")
                continue
            self._start_job(j)
            ctx = JobContext(self, j.id)
            ok, result, err, tr = True, None, None, None
            thread_ctx_id = f"job:{j.id}"
            _push_thread_tokens(thread_ctx_id, j.priority_tokens or set())
            try:
                # timeout soft: on laisse la fonction vérifier ctx.cancelled() périodiquement
                result = j.fn(ctx, j.args or {})
            except Exception as e:
                ok, err = False, str(e)
                tr = traceback.format_exc()
            finally:
                _pop_thread_tokens(thread_ctx_id)
            # cancellation après exécution ?
            if self._is_cancelled(j.id) and ok:
                ok, err = False, "cancelled"
            self._finish_job(j, ok=ok, result=result, error=err, trace=tr)
            with self._condition:
                self._condition.notify_all()

    def _job_view(self, j: Optional[Job]) -> Optional[Dict[str, Any]]:
        if not j:
            return None
        return {
            "id": j.id,
            "kind": j.kind,
            "queue": j.queue,
            "priority": j.priority,
            "status": j.status,
            "progress": j.progress,
            "created_ts": j.created_ts,
            "started_ts": j.started_ts,
            "finished_ts": j.finished_ts,
            "timeout_s": j.timeout_s,
            "key": j.key,
        }

    def _log(self, obj: Dict[str, Any]):
        try:
            os.makedirs(os.path.dirname(self.paths["log"]), exist_ok=True)
            with open(self.paths["log"], "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _persist_state(self):
        try:
            state = {jid: self._job_view(j) for jid, j in self._jobs.items()}
            with open(self.paths["state"], "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def wait_for(self, job_id: str, *, poll_interval: float = 0.05, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        deadline = None
        if timeout is not None:
            deadline = time.time() + max(0.0, float(timeout))
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status in {"done", "error", "cancelled"}:
                    event = {
                        "event": "done" if job.status == "done" else job.status,
                        "job": self._job_view(job),
                        "result": job.result,
                        "error": job.error,
                        "trace": job.trace,
                    }
                    if job.metrics:
                        event["metrics"] = dict(job.metrics)
                    return event
            if deadline is not None and time.time() >= deadline:
                return None
            time.sleep(max(0.001, poll_interval))

    def is_low_load(self) -> bool:
        with self._lock:
            total = (
                len(self._pq_inter)
                + len(self._pq_back)
                + self._running_inter
                + self._running_back
            )
        return total <= 1

    def has_active_urgent_chain(self) -> bool:
        """Indique si une chaîne prioritaire (SIGNAL/THREAT/NEED) est active."""

        with self._lock:
            active = self._compute_urgent_state_locked()
            self._set_urgent_state_locked(active)
            return active

    def pause_if_urgent_active(self, timeout: float = 0.2) -> bool:
        """Bloque brièvement l'appelant si une chaîne urgente est active."""

        with self._urgent_condition:
            if not self._urgent_state:
                return False
            self._urgent_condition.wait(timeout)
            return True

    def _compute_urgent_state_locked(self) -> bool:
        urgent_types = {"SIGNAL", "THREAT", "NEED"}
        if any(count > 0 for count in self._urgent_token_counts.values()):
            return True
        if any(count > 0 for count in self._manual_urgent_counts.values()):
            return True
        for job_id in list(self._urgent_jobs):
            job = self._jobs.get(job_id)
            if not job:
                continue
            if job.status in {"queued", "running"} and job.priority_tokens.intersection(urgent_types):
                return True
        return False

    def _set_urgent_state_locked(self, active: bool) -> None:
        changed = active != self._urgent_state
        self._urgent_state = active
        if changed or active:
            # Toujours notifier les threads en attente pour réduire la latence.
            self._urgent_condition.notify_all()
        try:
            update_urgent_state(self._urgent_state)
        except Exception:
            pass

    def current_thread_has_urgent_permission(self) -> bool:
        urgent_types = {"SIGNAL", "THREAT", "NEED"}
        return any(token in urgent_types for token in _current_thread_tokens())

    def activate_urgent_context(self, tokens: Iterable[str]) -> Optional[str]:
        """Force une fenêtre d'urgence tant que les jobs dérivés ne sont pas soumis."""

        urgent_types = {"SIGNAL", "THREAT", "NEED"}
        normalized: Set[str] = set()
        for tok in tokens:
            if not isinstance(tok, str):
                continue
            token = tok.strip().upper()
            if token and token in urgent_types:
                normalized.add(token)
        if not normalized:
            return None
        ctx_id = str(uuid.uuid4())
        with self._lock:
            self._urgent_contexts[ctx_id] = set(normalized)
            for token in normalized:
                self._manual_urgent_counts[token] += 1
            self._set_urgent_state_locked(True)
        _push_thread_tokens(ctx_id, normalized)
        return ctx_id

    def deactivate_urgent_context(self, ctx_id: Optional[str]) -> None:
        if not ctx_id:
            return
        with self._lock:
            tokens = self._urgent_contexts.pop(ctx_id, None)
            if not tokens:
                _pop_thread_tokens(ctx_id)
                return
            for token in tokens:
                count = self._manual_urgent_counts.get(token, 0)
                if count <= 1:
                    self._manual_urgent_counts.pop(token, None)
                else:
                    self._manual_urgent_counts[token] = count - 1
            self._set_urgent_state_locked(self._compute_urgent_state_locked())
        _pop_thread_tokens(ctx_id)

    def _extract_priority_tokens(
        self, args: Dict[str, Any]
    ) -> Tuple[Optional[str], Set[str], Optional[str]]:
        if not isinstance(args, dict):
            return None, set(), None
        meta = args.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        trigger_raw = meta.get("trigger_type")
        trigger_token = trigger_raw.strip().upper() if isinstance(trigger_raw, str) else None
        chain_tokens: Set[str] = set()
        chain = meta.get("priority_chain")
        if isinstance(chain, (list, tuple, set)):
            for item in chain:
                if isinstance(item, str):
                    token = item.strip().upper()
                    if token:
                        chain_tokens.add(token)
        if trigger_token:
            chain_tokens.add(trigger_token)
        source_raw = meta.get("source") or meta.get("origin")
        source_token = source_raw.strip().lower() if isinstance(source_raw, str) else None
        return trigger_token, chain_tokens, source_token

    def _priority_override(self, job: Job, args: Dict[str, Any]) -> Optional[Tuple[float, str]]:
        trigger_token, chain_tokens, source_token = self._extract_priority_tokens(args)
        urgent_types = {"SIGNAL", "THREAT", "NEED"}
        if trigger_token in urgent_types:
            return 1.0, f"trigger:{trigger_token}"
        urgent_chain = chain_tokens.intersection(urgent_types)
        if urgent_chain:
            joined = ",".join(sorted(urgent_chain))
            return 1.0, f"chain:{joined}"
        if source_token in {"user", "interaction"}:
            return max(float(job.priority), 0.9), f"source:{source_token}"
        return None

    def _register_priority_tokens(
        self,
        job: Job,
        trigger_token: Optional[str],
        chain_tokens: Set[str],
        source_token: Optional[str],
    ) -> None:
        tokens: Set[str] = set()
        urgent_types = {"SIGNAL", "THREAT", "NEED"}
        if trigger_token:
            tokens.add(trigger_token)
        tokens.update(chain_tokens)
        if tokens:
            job.priority_tokens = set(tokens)
            job.metrics.setdefault("priority_tokens", sorted(tokens))
        if source_token:
            job.metrics.setdefault("priority_source", source_token)
        urgent_tokens = {tok for tok in tokens if tok in urgent_types}
        if urgent_tokens:
            overrides = job.metrics.setdefault("overrides", {})
            overrides["urgent_tokens"] = sorted(urgent_tokens)
            with self._lock:
                self._urgent_jobs.add(job.id)
                for tok in urgent_tokens:
                    self._urgent_token_counts[tok] += 1
                self._set_urgent_state_locked(True)

    def _release_priority_tokens(self, job: Job) -> None:
        urgent_types = {"SIGNAL", "THREAT", "NEED"}
        urgent_tokens = {tok for tok in job.priority_tokens if tok in urgent_types}
        if not urgent_tokens:
            return
        with self._lock:
            self._urgent_jobs.discard(job.id)
            for tok in urgent_tokens:
                count = self._urgent_token_counts.get(tok, 0)
                if count <= 1:
                    self._urgent_token_counts.pop(tok, None)
                else:
                    self._urgent_token_counts[tok] = count - 1
            self._set_urgent_state_locked(self._compute_urgent_state_locked())
        job.priority_tokens.clear()

    def _should_force_interactive(self, args: Dict[str, Any]) -> Optional[str]:
        trigger_token, chain_tokens, _ = self._extract_priority_tokens(args)
        urgent_types = {"SIGNAL", "THREAT", "NEED"}
        if trigger_token in urgent_types:
            return f"trigger:{trigger_token}"
        urgent_chain = chain_tokens.intersection(urgent_types)
        if urgent_chain:
            return f"chain:{','.join(sorted(urgent_chain))}"
        return None

    def snapshot_identity_view(self) -> Dict[str, Any]:
        """Small summary of running and recent jobs for the SelfModel."""
        now = time.time()
        view: Dict[str, Any] = {
            "current": {
                "focus_topic": None,
                "jobs_running": [],
                "loads": {"interactive": 0, "background": 0},
            },
            "recent": [],
        }
        try:
            with self._lock:
                jobs = list(self._jobs.values())
                pq_inter = len(self._pq_inter)
                pq_back = len(self._pq_back)
                running_inter = self._running_inter
                running_back = self._running_back
                budget_inter = self.budgets["interactive"]["max_running"]
                budget_back = self.budgets["background"]["max_running"]

            arch = getattr(self, "arch", None)
            focus_topic = None
            if arch is not None:
                focus_topic = (
                    getattr(arch, "_current_topic", None)
                    or getattr(arch, "current_topic", None)
                )
            view["current"]["focus_topic"] = focus_topic

            running_jobs = [j for j in jobs if j.status == "running"]
            view["current"]["jobs_running"] = [
                {
                    "id": j.id,
                    "kind": j.kind,
                    "since": j.started_ts or j.created_ts,
                }
                for j in running_jobs
            ]

            view["current"]["loads"]["interactive"] = running_inter + pq_inter
            view["current"]["loads"]["background"] = running_back + pq_back
            view["current"]["budgets"] = {
                "interactive": budget_inter,
                "background": budget_back,
            }

            with self._stats_lock:
                lane_metrics = {
                    lane: {
                        "ema_reward": stats.get("ema_reward"),
                        "ema_latency": stats.get("ema_latency"),
                        "ema_load": stats.get("ema_load"),
                        "count": stats.get("count", 0),
                    }
                    for lane, stats in self._lane_stats.items()
                }
            with self._model_lock:
                lane_metrics["model"] = {
                    "confidence": self._priority_model.confidence,
                    "noise_scale": self._priority_model.noise_scale,
                    "updates": self._priority_model.updates,
                }
            view["current"]["scheduler_metrics"] = lane_metrics

            completed = [
                j
                for j in jobs
                if j.status in {"done", "error", "cancelled"}
            ]
            completed.sort(
                key=lambda j: j.finished_ts or j.created_ts,
                reverse=True,
            )
            for job in completed[:50]:
                finished_ts = job.finished_ts or job.created_ts or now
                created_ts = job.created_ts or finished_ts
                latency_ms = int(max(0.0, (finished_ts - created_ts)) * 1000.0)
                status = job.status
                if status == "cancelled":
                    status = "error"
                view["recent"].append(
                    {
                        "job_id": job.id,
                        "kind": job.kind,
                        "status": status,
                        "ts": finished_ts,
                        "latency_ms": latency_ms,
                    }
                )

            log_path = self.paths.get("log")
            if log_path and os.path.exists(log_path):
                added = {
                    (entry.get("job_id"), entry.get("ts"))
                    for entry in view["recent"]
                    if entry.get("job_id") is not None
                }
                with open(log_path, "r", encoding="utf-8") as handle:
                    tail = handle.readlines()[-500:]
                for line in tail:
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    if event.get("event") not in {"job_done", "job_error"}:
                        continue
                    job_id = event.get("job_id")
                    ts = float(event.get("ts", now))
                    key = (job_id, ts)
                    if key in added:
                        continue
                    status = "done" if event.get("event") == "job_done" else "error"
                    latency_ms = int(float(event.get("latency", 0.0)) * 1000.0)
                    view["recent"].append(
                        {
                            "job_id": job_id,
                            "kind": event.get("kind"),
                            "status": status,
                            "ts": ts,
                            "latency_ms": latency_ms,
                        }
                    )
                    added.add(key)

            view["recent"].sort(key=lambda entry: entry.get("ts", 0.0), reverse=True)
            view["recent"] = view["recent"][:200]
        except Exception:
            pass
        return view
