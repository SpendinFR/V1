from __future__ import annotations

import math
import random
import threading
import time

import logging

from typing import Any, Dict, List, Mapping, Optional

from AGI_Evolutive.goals.dag_store import GoalDAG
from AGI_Evolutive.reasoning.structures import (
    Evidence,
    Hypothesis,
    Test,
    episode_record,
)
from AGI_Evolutive.runtime.logger import JSONLLogger
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


class AutonomyCore:
    """
    Scheduler d'autonomie : en idle, choisit un sous-but à forte EVI,
    exécute UNE petite étape, logge, et propose un prochain test.
    """

    def __init__(self, arch, logger: JSONLLogger, dag: GoalDAG):
        self.arch = arch
        self.logger = logger
        self.dag = dag
        self.running = True
        self.thread = None
        self.idle_interval = 20  # secondes sans input pour tick
        self._last_user_time = time.time()
        self._tick = 0
        # Online GLM (logistic regression) weights for hypothesis prior/EVI.
        self._policy_weights: Dict[str, float] = {
            "bias": -0.2,
            "progress": -0.8,
            "progress_sq": -0.3,
            "belief": 0.4,
            "belief_sq": 0.2,
            "novelty": 0.35,
            "novelty_sq": 0.15,
            "evi": 0.6,
            "evi_progress": 0.4,
        }
        self._last_weight_snapshot = dict(self._policy_weights)
        self._learning_rate = 0.15
        self._max_step = 0.05
        self._weight_drift_threshold = 0.12
        self._ema_betas = (0.2, 0.4, 0.6, 0.8)
        self._ema_trackers: Dict[str, Dict[str, Any]] = {
            name: self._init_ema_tracker()
            for name in ("belief", "novelty", "progress")
        }

    def notify_user_activity(self):
        self._last_user_time = time.time()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                idle_for = time.time() - self._last_user_time
                if idle_for >= self.idle_interval:
                    self.tick()
                    self._last_user_time = time.time()  # évite boucle frénétique
                time.sleep(1.0)
            except Exception as e:
                self.logger.write("autonomy.error", error=str(e))
                time.sleep(5)

    def tick(self):
        self._tick += 1
        # 1) Choix d'objectif
        pick = self.dag.choose_next_goal()
        goal_id, evi, progress = pick["id"], pick["evi"], pick["progress"]

        # 2) Hypothèse & test minimal
        # Prépare signaux pour le modèle online (avec EMA adaptatives).
        belief_raw = 0.6
        try:
            self_model = getattr(self.arch, "self_model", None)
            if self_model and hasattr(self_model, "belief_confidence"):
                belief_raw = float(self_model.belief_confidence({}))
        except Exception as exc:
            self.logger.write("autonomy.warn", stage="belief_confidence", error=str(exc))

        novelty_raw = 0.7
        try:
            memory = getattr(self.arch, "memory", None)
            if memory and hasattr(memory, "get_recent_memories"):
                raw_recent = memory.get_recent_memories(n=100)
                recent = [m for m in raw_recent if isinstance(m, dict)] if isinstance(raw_recent, list) else []
            else:
                recent = []
            typs = [m.get("type", m.get("kind")) for m in recent]
            same = sum(1 for tpe in typs if tpe == "update")
            novelty_raw = max(0.2, min(0.95, 1.0 - (0.02 * max(0, 5 - same))))
        except Exception as exc:
            self.logger.write("autonomy.warn", stage="novelty_eval", error=str(exc))

        belief = self._adaptive_ema_step("belief", belief_raw)
        novelty_fam = self._adaptive_ema_step("novelty", novelty_raw)
        progress_smoothed = self._adaptive_ema_step("progress", progress)

        features = self._compute_features(
            progress_smoothed,
            belief,
            novelty_fam,
            evi,
        )
        score = self._linear_score(features)
        prior = self._sigmoid(score)
        expected_info_gain = max(0.05, min(0.95, self._sigmoid(score + 0.35)))

        llm_guidance = self._query_llm_guidance(
            goal_id=goal_id,
            features=features,
            belief=belief,
            novelty=novelty_fam,
            progress=progress_smoothed,
            evi=evi,
            proposals=proposals,
        )

        h = [
            Hypothesis(
                content=f"Une micro-étape sur {goal_id} accélère la compréhension",
                prior=prior,
            )
        ]
        t = [
            Test(
                description=f"Lire/agréger 3 traces récentes et distiller 1 règle pour {goal_id}",
                cost_est=0.15,
                expected_information_gain=expected_info_gain,
            )
        ]

        # 3) "Exécution" symbolique (sans I/O lourde ici)
        rule = self._distill_micro_rule(goal_id)
        ev_confidence = self._sigmoid(score + 0.2)
        ev = Evidence(notes=f"Règle distillée: {rule}", confidence=ev_confidence)

        # 3bis) Décider d'une action via la policy (si dispo)
        proposer = getattr(self.arch, "proposer", None)
        policy = getattr(self.arch, "policy", None)
        homeo = getattr(self.arch, "homeostasis", None) or getattr(self.arch, "homeo", None)
        planner = getattr(self.arch, "planner", None)
        memory = getattr(self.arch, "memory", None)
        proposals: List[Dict[str, Any]] = []
        if proposer and hasattr(proposer, "run_once_now"):
            try:
                raw_props = proposer.run_once_now() or []
                if isinstance(raw_props, list):
                    proposals = [p for p in raw_props if isinstance(p, dict)]
            except Exception as exc:
                self.logger.write("autonomy.warn", stage="proposer", error=str(exc))

        decision: Dict[str, Any] = {"decision": "noop", "reason": "no policy", "confidence": 0.5}
        if policy and hasattr(policy, "decide"):
            try:
                decision = policy.decide(
                    proposals,
                    self_state={"tick": self._tick},
                    proposer=proposer,
                    homeo=homeo,
                    planner=planner,
                    ctx={"belief_confidence": belief, "novelty_familiarity": novelty_fam},
                )
            except Exception as exc:
                decision = {"decision": "error", "reason": str(exc), "confidence": 0.3}
                self.logger.write("autonomy.warn", stage="policy_decide", error=str(exc))

        if llm_guidance:
            decision = dict(decision)
            decision.setdefault("llm_guidance", llm_guidance)

        # 4) Mise à jour DAG + logs
        progress_step = max(0.002, min(0.02, prior * 0.02))
        progress_after = self.dag.bump_progress(progress_step)
        ep = episode_record(
            user_msg="[idle]",
            hypotheses=h,
            chosen_index=0,
            tests=t,
            evidence=ev,
            result_text=f"Micro-étape sur {goal_id}: {rule}",
            final_confidence=self._sigmoid(score + 0.1),
        )
        self.logger.write(
            "autonomy.tick",
            goal=goal_id,
            evi=evi,
            progress_before=progress,
            progress_after=progress_after,
            episode=ep,
            policy_decision=decision,
            hedonic_signal=getattr(self.arch, "phenomenal_kernel_state", {}).get("hedonic_reward") if getattr(self.arch, "phenomenal_kernel_state", None) else None,
        )

        # 5) Ping métacognition (si existante)
        try:
            if hasattr(self.arch, "metacognition") and self.arch.metacognition:
                self.arch.metacognition._record_metacognitive_event(
                    event_type="autonomy_step",
                    domain=
                    self.arch.metacognition.CognitiveDomain.LEARNING
                    if hasattr(self.arch.metacognition, "CognitiveDomain")
                    else None,
                    description=f"Idle→micro-étape sur {goal_id}",
                    significance=min(0.3 + 0.4 * evi, 0.9),
                    confidence=0.6,
                )
        except Exception:
            pass

        # 6) Feedback à la policy
        try:
            executed = decision.get("proposal") if isinstance(decision, dict) else None
            success = bool(decision.get("decision") == "apply") if isinstance(decision, dict) else False
            if policy and hasattr(policy, "register_outcome") and executed:
                policy.register_outcome(executed, success)
        except Exception as exc:
            self.logger.write("autonomy.warn", stage="policy_feedback", error=str(exc))

        progress_delta = max(0.0, progress_after - progress)
        reward_signal = min(1.0, 0.4 * evi + 5.0 * progress_delta)
        kernel_state = getattr(self.arch, "phenomenal_kernel_state", None)
        hedonic_reward = 0.0
        mode = "travail"
        if isinstance(kernel_state, dict):
            try:
                hedonic_reward = float(kernel_state.get("hedonic_reward", 0.0))
            except Exception:
                hedonic_reward = 0.0
            mode = kernel_state.get("mode") or kernel_state.get("mode_suggestion") or "travail"
        if hedonic_reward:
            blend = 0.5 if mode == "flanerie" else 0.2
            hedonic_scaled = max(0.0, min(1.0, 0.5 + 0.5 * hedonic_reward))
            reward_signal = max(0.0, min(1.0, (1.0 - blend) * reward_signal + blend * hedonic_scaled))
        if isinstance(decision, dict) and decision.get("decision") == "apply":
            reward_signal = min(
                1.0,
                reward_signal + 0.3 * float(decision.get("confidence", 0.5)),
            )
        self._update_policy_weights(features, reward_signal)
        self._adaptive_ema_feedback("belief", reward_signal)
        self._adaptive_ema_feedback("novelty", reward_signal)
        self._adaptive_ema_feedback("progress", reward_signal)

    def _distill_micro_rule(self, goal_id: str) -> str:
        """
        Distille une mini-règle depuis l'historique récent pour garder l'agent 'vivant'.
        (Heuristique très simple et sûre.)
        """
        try:
            # lit les 30 derniers événements dialogue/autonomy (si dispo via logger → non trivial)
            # ici : renvoie une règle statique contextualisée pour démarrer
            if goal_id == "understand_humans":
                return "Toujours expliciter l'hypothèse et demander 1 validation binaire."
            if goal_id == "self_modeling":
                return "Journaliser 'ce que j'ai appris' au moins une fois par tour."
            if goal_id == "tooling_mastery":
                return "Proposer un patch minimal plutôt qu'un grand refactor."
            return "Faire un pas plus petit mais mesurable."
        except Exception:
            return "Faire un pas plus petit mais mesurable."

    def _compute_features(
        self,
        progress: float,
        belief: float,
        novelty: float,
        evi: float,
    ) -> Dict[str, float]:
        return {
            "bias": 1.0,
            "progress": progress,
            "progress_sq": progress * progress,
            "belief": belief,
            "belief_sq": belief * belief,
            "novelty": novelty,
            "novelty_sq": novelty * novelty,
            "evi": evi,
            "evi_progress": evi * progress,
        }

    def _sigmoid(self, value: float) -> float:
        if value < -60:
            return 0.0
        if value > 60:
            return 1.0
        return 1.0 / (1.0 + math.exp(-value))

    def _linear_score(self, features: Dict[str, float]) -> float:
        score = 0.0
        for key, val in features.items():
            weight = self._policy_weights.get(key, 0.0)
            score += weight * val
        return score

    def _update_policy_weights(self, features: Dict[str, float], reward: float) -> None:
        reward = max(0.0, min(1.0, reward))
        prediction = self._sigmoid(self._linear_score(features))
        error = reward - prediction
        gradient_scale = prediction * (1 - prediction)
        drift_detected = False
        for key, value in features.items():
            grad = error * gradient_scale * value
            grad = max(-self._max_step, min(self._max_step, grad))
            update = self._learning_rate * grad
            if abs(update) < 1e-5:
                continue
            new_weight = self._policy_weights.get(key, 0.0) + update
            if abs(new_weight - self._last_weight_snapshot.get(key, new_weight)) >= self._weight_drift_threshold:
                drift_detected = True
            self._policy_weights[key] = new_weight
        if drift_detected:
            self.logger.write(
                "autonomy.drift",
                weights=self._policy_weights,
                reward=reward,
                prediction=prediction,
            )
            self._last_weight_snapshot = dict(self._policy_weights)

    def _init_ema_tracker(self) -> Dict[str, Any]:
        return {
            "states": {
                beta: {"value": None, "alpha": 1.0, "beta": 1.0}
                for beta in self._ema_betas
            },
            "last_choice": None,
        }

    def _adaptive_ema_step(self, key: str, new_value: float) -> float:
        tracker = self._ema_trackers[key]
        best_beta = None
        best_sample = None
        for beta, params in tracker["states"].items():
            alpha = max(1e-3, params["alpha"])
            beta_param = max(1e-3, params["beta"])
            # Thompson sampling: draw a Beta sample; fallback to posterior mean if random fails.
            try:
                sample = random.betavariate(alpha, beta_param)
            except Exception:
                sample = alpha / (alpha + beta_param)
            if best_sample is None or sample > best_sample:
                best_sample = sample
                best_beta = beta
        if best_beta is None:
            best_beta = self._ema_betas[0]
        tracker["last_choice"] = best_beta
        state = tracker["states"][best_beta]
        prev = state["value"]
        if prev is None:
            state["value"] = new_value
        else:
            state["value"] = (1 - best_beta) * prev + best_beta * new_value
        return float(state["value"])

    def _adaptive_ema_feedback(self, key: str, reward: float) -> None:
        tracker = self._ema_trackers[key]
        chosen_beta = tracker.get("last_choice")
        if chosen_beta is None:
            return
        params = tracker["states"][chosen_beta]
        if reward >= 0.5:
            params["alpha"] += 1.0
        else:
            params["beta"] += 1.0

    def _query_llm_guidance(
        self,
        *,
        goal_id: str,
        features: Dict[str, float],
        belief: float,
        novelty: float,
        progress: float,
        evi: float,
        proposals: List[Dict[str, Any]],
    ) -> Optional[Mapping[str, Any]]:
        payload = {
            "goal_id": goal_id,
            "features": features,
            "belief": belief,
            "novelty": novelty,
            "progress": progress,
            "evi": evi,
            "proposals": proposals,
        }

        response = try_call_llm_dict(
            "autonomy_core",
            input_payload=payload,
            logger=LOGGER,
        )

        if isinstance(response, Mapping):
            return dict(response)
        return None
