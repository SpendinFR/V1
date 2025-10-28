import json
import logging
import os
import random
import statistics
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from AGI_Evolutive.core.structures.mai import MAI
from AGI_Evolutive.knowledge.mechanism_store import MechanismStore
from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


def _safe_write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_sanitize(obj), f, ensure_ascii=False, indent=2)


def _safe_read_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_sanitize(obj), ensure_ascii=False) + "\n")


def _mean(xs: List[float], default: float = 0.0) -> float:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return float(statistics.fmean(xs)) if xs else float(default)


def _rolling(values: List[float], k: int) -> float:
    if not values:
        return 0.0
    tail = values[-k:] if len(values) > k else values
    return _mean(tail)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class ThompsonBandit:
    """Simple Thompson Sampling helper persisted as dict in state."""

    def __init__(self, storage: Dict[str, Any], actions: Sequence[Union[int, float, str]]):
        self.storage = storage
        self.actions = list(actions)
        if "_meta" not in self.storage:
            self.storage["_meta"] = {}
        for action in self.actions:
            key = self._key(action)
            self.storage.setdefault(key, {"alpha": 1.0, "beta": 1.0})

    @staticmethod
    def _key(action: Union[int, float, str]) -> str:
        if isinstance(action, float):
            # éviter les collisions dues aux flottants
            return f"{action:.6f}"
        return str(action)

    def sample(self, default: Union[int, float, str]) -> Union[int, float, str]:
        """Renvoie l'action tirée via Thompson Sampling."""
        best_action = None
        best_score = -1.0
        for action in self.actions:
            entry = self.storage[self._key(action)]
            draw = random.betavariate(entry["alpha"], entry["beta"])
            if draw > best_score:
                best_score = draw
                best_action = action
        if best_action is None:
            best_action = default
        self.set_last_action(best_action)
        return best_action

    def update(self, action: Union[int, float, str], reward: float) -> None:
        entry = self.storage[self._key(action)]
        reward = _clamp01(reward)
        entry["alpha"] = float(entry.get("alpha", 1.0)) + reward
        entry["beta"] = float(entry.get("beta", 1.0)) + (1.0 - reward)

    def last_action(self) -> Optional[Union[int, float, str]]:
        return self.storage.get("_meta", {}).get("last_action")

    def set_last_action(self, action: Union[int, float, str]) -> None:
        self.storage.setdefault("_meta", {})["last_action"] = action


class EvolutionManager:
    """
    Gestion long-terme de l'agent :
      - collecte des métriques cross-modules
      - détection de tendances (amélioration / régression / stagnation)
      - analyse des risques (fatigue, charge, erreurs récurrentes)
      - recommandations stratégiques (réglages, priorités, expérimentations)
      - propositions d'évolution (proposals) -> à valider par une policy si elle existe
    Persistance :
      - data/evolution/state.json
      - data/evolution/cycles.jsonl   (trace de chaque cycle)
      - data/evolution/recommendations.jsonl
      - data/evolution/dashboard.json (snapshot consolidé)
    """

    _shared = None

    @classmethod
    def shared(cls):
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def __init__(self, data_dir: str = "data", horizon_cycles: int = 200):
        self.data_dir = data_dir
        self.paths = {
            "state": os.path.join(self.data_dir, "evolution/state.json"),
            "cycles": os.path.join(self.data_dir, "evolution/cycles.jsonl"),
            "reco": os.path.join(self.data_dir, "evolution/recommendations.jsonl"),
            "dashboard": os.path.join(self.data_dir, "evolution/dashboard.json"),
        }
        os.makedirs(os.path.dirname(self.paths["state"]), exist_ok=True)

        # liens (optionnels) vers l'archi
        self.arch = None
        self.memory = None
        self.metacog = None
        self.goals = None
        self.learning = None
        self.emotions = None
        self.language = None

        self.habits_strength: Dict[str, float] = {}

        # État persistant
        self.state = _safe_read_json(self.paths["state"], {
            "created_at": _now(),
            "last_cycle_id": 0,
            "cycle_count": 0,
            "history": {
                # séries brutes
                "reasoning_speed": [],
                "reasoning_confidence": [],
                "learning_rate": [],
                "recall_accuracy": [],
                "cognitive_load": [],
                "fatigue": [],
                "error_rate": [],
                "goals_progress": [],
                "affect_valence": [],
            },
            "milestones": [],  # liste d'événements marquants
            "risk_flags": [],  # ex: "regression_learning", "high_fatigue"
            "history_extra": {},
            "legacy_metrics_history": [],
            "strategies": [],
            "bandits": {},
        })
        self.horizon = horizon_cycles
        self._state_lock = threading.RLock()

    # ---------- adaptive helpers ----------
    def _get_bandit(self, name: str, actions: Sequence[Union[int, float, str]]) -> ThompsonBandit:
        bandits = self.state.setdefault("bandits", {})
        storage = bandits.setdefault(name, {})
        return ThompsonBandit(storage, actions)

    def _update_window_bandit(
        self,
        metric: str,
        previous_values: List[float],
        current_values: List[float],
        default_window: int,
    ) -> None:
        if not current_values:
            return
        bandit = self._get_bandit(
            f"window::{metric}",
            actions=[3, 5, 8, 14, 20, 30],
        )
        last_action = bandit.last_action()
        if last_action is None:
            bandit.set_last_action(default_window)
            last_action = default_window
        last_action = int(last_action)
        if not previous_values:
            # Pas assez d'historique pour calculer une récompense.
            return
        tail = previous_values[-last_action:] if len(previous_values) >= last_action else previous_values
        previous_avg = _mean(tail, default=current_values[-1])
        diff = abs(current_values[-1] - previous_avg)
        reward = _clamp01(1.0 - min(1.0, diff))
        bandit.update(last_action, reward)
        next_action = int(bandit.sample(last_action))
        bandit.set_last_action(next_action)

    def _current_window(self, metric: str, default_window: int) -> int:
        bandit = self._get_bandit(f"window::{metric}", actions=[3, 5, 8, 14, 20, 30])
        last_action = bandit.last_action()
        if last_action is None:
            bandit.set_last_action(default_window)
            last_action = default_window
        return int(last_action)

    def _detect_regimes(self, hist: Dict[str, List[float]]) -> List[str]:
        regimes: List[str] = []
        if len(hist.get("learning_rate", [])) >= 12:
            short = _rolling(hist["learning_rate"], min(6, len(hist["learning_rate"])) )
            long = _rolling(hist["learning_rate"], min(18, len(hist["learning_rate"])) )
            if short - long > 0.05:
                regimes.append("learning_surge")
            elif long - short > 0.05:
                regimes.append("learning_stall")
        if len(hist.get("fatigue", [])) >= 12:
            short_f = _rolling(hist["fatigue"], min(6, len(hist["fatigue"])) )
            long_f = _rolling(hist["fatigue"], min(18, len(hist["fatigue"])) )
            if short_f > 0.75 and short_f - long_f > 0.03:
                regimes.append("fatigue_rise")
        if len(hist.get("reasoning_confidence", [])) >= 12 and len(hist.get("recall_accuracy", [])) >= 12:
            conf_short = _rolling(hist["reasoning_confidence"], min(6, len(hist["reasoning_confidence"])) )
            recall_short = _rolling(hist["recall_accuracy"], min(6, len(hist["recall_accuracy"])) )
            if conf_short - recall_short > 0.2:
                regimes.append("overconfidence_phase")
        return regimes

    def evaluate_mechanism(self, mai: MAI) -> bool:
        """
        Évalue un MAI via:
        - world_model social (contrefactuels)
        - social_critic (trust/harm/cooperation/regret)
        - self_improver sandbox/ablation
        - check cohérence identitaire
        Retourne True si promotion recommandée.
        """

        ok = False
        try:
            from AGI_Evolutive.social.social_critic import SocialCritic
            from AGI_Evolutive.world_model import SocialModel
            from AGI_Evolutive.self_improver.promote import PromoteManager
            from AGI_Evolutive.core.self_model import SelfModel

            critic = SocialCritic()
            sim = SocialModel()
            promoter = PromoteManager()
            self_model = SelfModel.shared() if hasattr(SelfModel, "shared") else SelfModel()

            # 1) Générer des cas contrefactuels pertinents pour ce MAI
            batch = (
                sim.build_counterfactual_batch_for_mechanism(mai)
                if hasattr(sim, "build_counterfactual_batch_for_mechanism")
                else []
            )

            scores_with, scores_without = [], []
            for case in batch:
                s_with = critic.score(sim.run(case, enable_mechanism=mai))
                s_without = critic.score(sim.run(case, enable_mechanism=None))
                scores_with.append(s_with)
                scores_without.append(s_without)

            def agg(slist, key, default=0.0):
                return sum(float(s.get(key, default)) for s in slist) / max(1, len(slist))

            trust_gain = agg(scores_with, "trust") - agg(scores_without, "trust")
            harm_diff = agg(scores_with, "harm") - agg(scores_without, "harm")
            coop_gain = agg(scores_with, "cooperation") - agg(scores_without, "cooperation")
            regret_diff = agg(scores_with, "regret") - agg(scores_without, "regret")

            # 2) Identité / invariants
            identity_ok = True
            if hasattr(self_model, "violates_invariants"):
                identity_ok = not self_model.violates_invariants(mai.safety_invariants)

            # 3) Sandbox / ablation (preuve causale)
            sandbox_ok = (
                promoter.sandbox_mechanism(mai)
                if hasattr(promoter, "sandbox_mechanism")
                else True
            )

            ok = (trust_gain > 0.0) and (harm_diff <= 0.0) and identity_ok and sandbox_ok

        except Exception:
            ok = False

        return bool(ok)

    def reinforce(self, ctx: Dict[str, Any]) -> None:
        """Met à jour une force d'habitude simple à partir d'un contexte de décision."""

        if not isinstance(ctx, dict):
            return

        decision = ctx.get("decision") if isinstance(ctx.get("decision"), dict) else {}
        action = decision.get("action") if isinstance(decision.get("action"), dict) else {}

        key_parts: List[str] = []
        act_type = action.get("type") or decision.get("type")
        if act_type:
            key_parts.append(str(act_type))
        context_hint = action.get("context") or decision.get("context") or ctx.get("context")
        if isinstance(context_hint, dict):
            context_key = context_hint.get("id") or context_hint.get("label") or context_hint.get("name")
            if context_key:
                key_parts.append(str(context_key))
        elif isinstance(context_hint, str):
            key_parts.append(context_hint)

        if not key_parts:
            key_parts.append("unknown")
        key = "::".join(key_parts)

        scratch = ctx.get("scratch") if isinstance(ctx.get("scratch"), dict) else {}
        err = scratch.get("prediction_error", 0.5)
        try:
            err_val = float(err)
        except (TypeError, ValueError):
            err_val = 0.5
        success = err_val < 0.2
        reward_signal = _clamp01(1.0 - min(1.0, err_val))

        current = float(self.habits_strength.get(key, 0.0))
        if success:
            gain_bandit = self._get_bandit(
                f"habit_gain::{key}",
                actions=[0.04, 0.06, 0.08, 0.12],
            )
            step = float(gain_bandit.sample(0.08))
            delta = step
            gain_bandit.update(step, reward_signal)
        else:
            loss_bandit = self._get_bandit(
                f"habit_loss::{key}",
                actions=[0.02, 0.04, 0.06, 0.08],
            )
            step = float(loss_bandit.sample(0.04))
            delta = -step
            loss_bandit.update(step, 1.0 - reward_signal)

        updated = _clamp01(current + delta)
        self.habits_strength[key] = updated

    # ---------- binding ----------
    def bind(self, architecture=None, memory=None, metacog=None,
             goals=None, learning=None, emotions=None, language=None):
        self.arch = architecture
        self.memory = memory
        self.metacog = metacog
        self.goals = goals
        self.learning = learning
        self.emotions = emotions
        self.language = language

    # ---------- cycle ingestion ----------
    def record_cycle(
        self,
        extra_tags: Optional[Dict[str, Any]] = None,
        manual_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Capture l'état courant des métriques utiles, loggue en JSONL, met à jour l'état.
        Appelle ensuite evaluate_cycle() pour calculer tendances + risques.
        """
        with self._state_lock:
            cycle_id = self.state["last_cycle_id"] + 1
            metrics = manual_metrics or self._collect_metrics_snapshot()
            snap = {
                "t": _now(),
                "cycle_id": cycle_id,
                "metrics": metrics,
                "tags": extra_tags or {}
            }
            _append_jsonl(self.paths["cycles"], snap)

            # pousser les séries
            hist = self.state["history"]
            previous_hist = {k: list(v) for k, v in hist.items()}
            for key in hist.keys():
                hist[key].append(float(metrics.get(key, 0.0)))

            extra_hist = self.state.setdefault("history_extra", {})
            if manual_metrics:
                for key, value in manual_metrics.items():
                    if key not in hist:
                        arr = extra_hist.setdefault(key, [])
                        arr.append(float(value))

            # limiter l'horizon
            for k in hist.keys():
                if len(hist[k]) > self.horizon:
                    hist[k] = hist[k][-self.horizon:]
            for k in list(extra_hist.keys()):
                if len(extra_hist[k]) > self.horizon:
                    extra_hist[k] = extra_hist[k][-self.horizon:]

            # mise à jour des bandits de fenêtres adaptatives
            for metric_key, prev_values in previous_hist.items():
                default_window = 20 if metric_key == "error_rate" else 5
                self._update_window_bandit(
                    metric_key,
                    previous_values=prev_values,
                    current_values=hist.get(metric_key, []),
                    default_window=default_window,
                )

            self.state["last_cycle_id"] = cycle_id
            self.state["cycle_count"] = self.state.get("cycle_count", 0) + 1

            # évaluation et recommandations
            eval_out = self.evaluate_cycle()
            dash = self._make_dashboard_snapshot()
            _safe_write_json(self.paths["dashboard"], dash)
            _safe_write_json(self.paths["state"], self.state)
            return {"snapshot": snap, "evaluation": eval_out, "dashboard": dash}

    # ---------- collect ----------
    def _collect_metrics_snapshot(self) -> Dict[str, float]:
        """
        Aggrège les métriques cross-modules avec garde-fous.
        """
        out = {
            "reasoning_speed": 0.5,
            "reasoning_confidence": 0.5,
            "learning_rate": 0.3,
            "recall_accuracy": 0.5,
            "cognitive_load": 0.5,
            "fatigue": 0.2,
            "error_rate": 0.0,
            "goals_progress": 0.0,
            "affect_valence": 0.0,
        }

        # métacognition → performance_tracking
        if self.metacog and hasattr(self.metacog, "cognitive_monitoring"):
            perf = self.metacog.cognitive_monitoring.get("performance_tracking", {})
            def last(metric, default=0.5):
                arr = perf.get(metric, [])
                if not arr:
                    return default
                try:
                    return float(arr[-1]["value"])
                except Exception:
                    return default
            out["reasoning_speed"] = last("reasoning_speed", 0.5)
            out["reasoning_confidence"] = last("reasoning_confidence", 0.5)
            out["learning_rate"] = last("learning_rate", 0.3)
            out["recall_accuracy"] = last("recall_accuracy", 0.5)

        # ressources cognitives (via moniteur)
        if self.metacog and hasattr(self.metacog, "cognitive_monitoring"):
            try:
                rm = self.metacog.cognitive_monitoring.get("resource_monitoring")
                if rm:
                    # ré-évaluer à la volée si possible
                    load = rm.assess_cognitive_load(self.arch, getattr(self.arch, "reasoning", None))
                    out["cognitive_load"] = float(load)
            except Exception:
                pass

        # fatigue → metacog resource_monitoring
        if self.metacog and hasattr(self.metacog, "cognitive_monitoring"):
            try:
                rm = self.metacog.cognitive_monitoring.get("resource_monitoring")
                if rm:
                    fat = rm.assess_fatigue(self.metacog.metacognitive_history, self.arch)
                    out["fatigue"] = float(fat)
            except Exception:
                pass

        # erreur rate (comptage récent)
        if self.metacog:
            try:
                corr = list(self.metacog.metacognitive_history.get("error_corrections", []))
                recent = corr[-50:] if len(corr) > 50 else corr
                out["error_rate"] = len(recent) / max(10.0, (recent[-1]["timestamp"] - recent[0]["timestamp"])) if len(recent) > 1 else 0.0
            except Exception:
                pass

        # progression des buts (si goal system expose une métrique)
        if self.goals and hasattr(self.goals, "get_progress"):
            try:
                out["goals_progress"] = float(self.goals.get_progress() or 0.0)
            except Exception:
                pass

        # valence émotionnelle globale (si dispo)
        if self.emotions and hasattr(self.emotions, "get_affect"):
            try:
                aff = self.emotions.get_affect()  # ex: dict {"valence": -1..1, "arousal": 0..1}
                if isinstance(aff, dict):
                    out["affect_valence"] = max(0.0, min(1.0, (float(aff.get("valence", 0.0)) + 1.0) / 2.0))
            except Exception:
                pass

        return out

    # ---------- evaluation ----------
    def evaluate_cycle(self) -> Dict[str, Any]:
        """
        Analyse tendances courtes et longues; pose des flags de risque; émet des recommandations.
        """
        with self._state_lock:
            hist = self.state["history"]
            win_speed = self._current_window("reasoning_speed", 5)
            win_learn = self._current_window("learning_rate", 5)
            win_conf = self._current_window("reasoning_confidence", 5)
            win_recall = self._current_window("recall_accuracy", 5)
            win_fatigue = self._current_window("fatigue", 5)
            win_load = self._current_window("cognitive_load", 5)
            win_errors = self._current_window("error_rate", 20)
            eval_out = {
                "t": _now(),
                "rolling": {
                    "speed": _rolling(hist["reasoning_speed"], win_speed),
                    "learn": _rolling(hist["learning_rate"], win_learn),
                    "conf": _rolling(hist["reasoning_confidence"], win_conf),
                    "recall": _rolling(hist["recall_accuracy"], win_recall),
                    "fatigue": _rolling(hist["fatigue"], win_fatigue),
                    "load": _rolling(hist["cognitive_load"], win_load),
                    "errors": _rolling(hist["error_rate"], win_errors),
                },
                "flags": [],
                "recommendations": [],
                "regimes": self._detect_regimes(hist),
            }
            eval_out["rolling"].update({
                "speed_5": eval_out["rolling"]["speed"],
                "learn_5": eval_out["rolling"]["learn"],
                "conf_5": eval_out["rolling"]["conf"],
                "recall_5": eval_out["rolling"]["recall"],
                "fatigue_5": eval_out["rolling"]["fatigue"],
                "load_5": eval_out["rolling"]["load"],
                "errors_20": eval_out["rolling"]["errors"],
            })

            r = eval_out["rolling"]
            # flags
            if r["learn"] < 0.35 and r["conf"] < 0.45:
                eval_out["flags"].append("learning_regression")
            if r["fatigue"] > 0.7:
                eval_out["flags"].append("high_fatigue")
            if r["load"] > 0.8:
                eval_out["flags"].append("overload")
            if r["errors"] > 0.05:
                eval_out["flags"].append("high_error_rate")

            # recos : stratégie & expérimentation
            eval_out["recommendations"] += self._strategy_recommendations(r)
            eval_out["recommendations"] += self._exploration_recommendations()

            # enregistrer les régimes détectés comme milestones si nouveaux
            if eval_out["regimes"]:
                milestones = self.state.setdefault("milestones", [])
                for regime in eval_out["regimes"]:
                    marker = {"ts": eval_out["t"], "type": "regime", "value": regime}
                    if marker not in milestones:
                        milestones.append(marker)
                self.state["milestones"] = milestones[-200:]

            # persiste recos + flags en JSONL
            _append_jsonl(self.paths["reco"], {
                "t": eval_out["t"],
                "cycle_id": self.state["last_cycle_id"],
                "flags": eval_out["flags"],
                "recommendations": eval_out["recommendations"]
            })

            # ajoute aux risk_flags si nouveau
            for f in eval_out["flags"]:
                if f not in self.state["risk_flags"]:
                    self.state["risk_flags"].append(f)
            return eval_out

    def _strategy_recommendations(self, r: Dict[str, float]) -> List[Dict[str, Any]]:
        recos = []
        # calibrage effort/attention
        if r["load"] > 0.8 or r["fatigue"] > 0.7:
            recos.append({
                "kind": "effort_downshift",
                "reason": "High cognitive load/fatigue",
                "action": "Réduire la profondeur des tâches; augmenter micro-pauses; privilégier consolidation."
            })
        # booster apprentissage si stagnation
        if r["learn"] < 0.4 and r["conf"] < 0.5:
            recos.append({
                "kind": "strategy_change",
                "reason": "Low learning rate and low confidence",
                "action": "Tester stratégie d'étude alternative (auto-explication, exemples concrets, retrieval practice)."
            })
        # renforcer rappel si recall bas
        if r["recall"] < 0.5:
            recos.append({
                "kind": "memory_consolidation",
                "reason": "Low recall accuracy",
                "action": "Augmenter fréquence de consolidation et répétition espacée."
            })
        # erreurs hautes
        if r["errors"] > 0.05:
            recos.append({
                "kind": "error_clinic",
                "reason": "High recent error rate",
                "action": "Lancer mini-clinique d'erreurs: catégoriser 10 dernières erreurs et élaborer correctifs ciblés."
            })
        return recos

    def _exploration_recommendations(self) -> List[Dict[str, Any]]:
        """
        S'appuie (si dispo) sur concept_graph / episodes pour suggérer des axes d'exploration.
        """
        recos = []
        # lire concept_graph si présent
        cgraph_path = os.path.join(self.data_dir, "concept_graph.json")
        cgraph = _safe_read_json(cgraph_path, {"nodes": {}, "edges": {}})
        nodes = cgraph.get("nodes", {})
        if nodes:
            # sélectionner concepts fréquents mais peu exploités (heuristique: counts faibles dans plans?)
            top_concepts = sorted(nodes.items(), key=lambda kv: kv[1].get("count", 0), reverse=True)[:10]
            for name, meta in top_concepts[:3]:
                recos.append({
                    "kind": "explore_concept",
                    "concept": name,
                    "reason": "Concept central du vécu récent",
                    "action": f"Créer un sous-objectif d'analyse approfondie du concept '{name}'."
                })
        # lire derniers épisodes pour détecter chaînes cause→effet
        episodes_path = os.path.join(self.data_dir, "episodes.jsonl")
        if os.path.exists(episodes_path):
            try:
                last_lines = []
                with open(episodes_path, "r", encoding="utf-8") as f:
                    for line in f:
                        last_lines.append(json.loads(line))
                last_lines = last_lines[-5:] if len(last_lines) > 5 else last_lines
                if last_lines:
                    recos.append({
                        "kind": "reflect_episode",
                        "reason": "Épisodes récents disponibles",
                        "action": "Déclencher réflexion ciblée sur 1-2 épisodes pour extraire leçons causales."
                    })
            except Exception:
                pass
        return recos

    # ---------- proposals (optionnels) ----------
    def propose_evolution(self) -> List[Dict[str, Any]]:
        """
        Génère des 'proposals' d'évolution (self_model, goals, stratégies).
        Ces objets sont **à valider** par une policy/contrôle en amont si dispo.
        Retourne la liste des proposals émis.
        """
        proposals = []

        # Exemple 1 : ajuster poids de curiosité si learning_rate bas
        lr = _rolling(
            self.state["history"]["learning_rate"],
            self._current_window("learning_rate", 8),
        )
        if lr < 0.4:
            curiosity_bandit = self._get_bandit(
                "proposal::curiosity_delta",
                actions=[0.05, 0.1, 0.15],
            )
            curiosity_delta = float(curiosity_bandit.sample(0.1))
            proposals.append({
                "type": "adjust_drive",
                "target": "curiosity",
                "delta": curiosity_delta,
                "rationale": "Learning rate bas sur 8 cycles - stimuler exploration.",
            })
            curiosity_bandit.update(curiosity_delta, _clamp01((0.4 - lr) / 0.4))

        # Exemple 2 : créer macro-goal d'étude d'un concept central
        cgraph_path = os.path.join(self.data_dir, "concept_graph.json")
        cgraph = _safe_read_json(cgraph_path, {"nodes": {}, "edges": {}})
        if cgraph.get("nodes"):
            top = sorted(cgraph["nodes"].items(), key=lambda kv: kv[1].get("count", 0), reverse=True)[:1]
            if top:
                concept = top[0][0]
                proposals.append({
                    "type": "create_goal",
                    "title": f"Étudier en profondeur : {concept}",
                    "criteria": ["résumer en 5 points", "exemples concrets", "quiz de rappel >= 0.7"],
                    "value": 0.6,
                    "rationale": "Concept central du vécu récent"
                })

        # Exemple 3 : calibration métacognitive si écart confiance/perf
        conf = _rolling(
            self.state["history"]["reasoning_confidence"],
            self._current_window("reasoning_confidence", 8),
        )
        recall = _rolling(
            self.state["history"]["recall_accuracy"],
            self._current_window("recall_accuracy", 8),
        )
        if conf - recall > 0.25:
            proposals.append({
                "type": "metacog_calibration",
                "action": "Ajouter exercice de calibration confiance↔rappel (prédire score avant test).",
                "rationale": "Surconfiance détectée (conf - recall > 0.25)"
            })

        # Publier en mémoire (si dispo) pour traçabilité + éventuelle policy
        if self.memory and hasattr(self.memory, "add_memory"):
            for p in proposals:
                try:
                    self.memory.add_memory(
                        kind="evolution_proposal",
                        content=p.get("title") or p.get("type"),
                        metadata={"proposal": p, "source": "EvolutionManager", "timestamp": _now()},
                    )
                except Exception:
                    pass
        return proposals

    # ---------- dashboard ----------
    def _make_dashboard_snapshot(self) -> Dict[str, Any]:
        h = self.state["history"]
        snap = {
            "t": _now(),
            "last_cycle": self.state["last_cycle_id"],
            "rolling": {
                "speed": _rolling(h["reasoning_speed"], self._current_window("reasoning_speed", 8)),
                "learn": _rolling(h["learning_rate"], self._current_window("learning_rate", 8)),
                "conf": _rolling(h["reasoning_confidence"], self._current_window("reasoning_confidence", 8)),
                "recall": _rolling(h["recall_accuracy"], self._current_window("recall_accuracy", 8)),
                "load": _rolling(h["cognitive_load"], self._current_window("cognitive_load", 8)),
                "fatigue": _rolling(h["fatigue"], self._current_window("fatigue", 8)),
                "goals": _rolling(h["goals_progress"], self._current_window("goals_progress", 8)),
            },
            "risk_flags": list(self.state.get("risk_flags", [])),
            "milestones": list(self.state.get("milestones", []))[-20:]
        }
        return snap

    # ---------- public helpers ----------
    def get_long_term_trends(self) -> Dict[str, float]:
        return self._make_dashboard_snapshot().get("rolling", {})

    def export_dashboard(self) -> Dict[str, Any]:
        dash = self._make_dashboard_snapshot()
        response = try_call_llm_dict(
            "evolution_manager",
            input_payload={"snapshot": dash},
            logger=logger,
        )
        if response:
            dash["llm_analysis"] = dict(response)
            self.state.setdefault("telemetry", {})["llm_trend"] = dash["llm_analysis"]
        _safe_write_json(self.paths["dashboard"], dash)
        return dash

    def record_feedback_event(
        self,
        source: str,
        *,
        label: str,
        success: Optional[bool],
        confidence: float,
        heuristic: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Archive les signaux de feedback textuel pour ajuster les habitudes."""

        with self._state_lock:
            registry = self.state.setdefault("feedback_events", {})
            entry = registry.setdefault(
                source,
                {
                    "total": 0,
                    "success": 0,
                    "failure": 0,
                    "pending": 0,
                    "labels": {},
                    "heuristics": {},
                    "confidence_history": [],
                    "last": {},
                },
            )

            entry["total"] = int(entry.get("total", 0)) + 1
            if success is True:
                entry["success"] = int(entry.get("success", 0)) + 1
            elif success is False:
                entry["failure"] = int(entry.get("failure", 0)) + 1
            else:
                entry["pending"] = int(entry.get("pending", 0)) + 1

            conf_hist = list(entry.get("confidence_history", []))
            conf_hist.append(float(confidence))
            if len(conf_hist) > 200:
                conf_hist = conf_hist[-200:]
            entry["confidence_history"] = conf_hist

            labels_map = entry.setdefault("labels", {})
            label_stats = labels_map.setdefault(
                label,
                {"observations": 0, "success": 0, "failure": 0, "pending": 0},
            )
            label_stats["observations"] = int(label_stats.get("observations", 0)) + 1
            if success is True:
                label_stats["success"] = int(label_stats.get("success", 0)) + 1
            elif success is False:
                label_stats["failure"] = int(label_stats.get("failure", 0)) + 1
            else:
                label_stats["pending"] = int(label_stats.get("pending", 0)) + 1

            if heuristic:
                heuristics_map = entry.setdefault("heuristics", {})
                heur_stats = heuristics_map.setdefault(
                    heuristic,
                    {"observations": 0, "success": 0, "failure": 0, "pending": 0},
                )
                heur_stats["observations"] = int(heur_stats.get("observations", 0)) + 1
                if success is True:
                    heur_stats["success"] = int(heur_stats.get("success", 0)) + 1
                elif success is False:
                    heur_stats["failure"] = int(heur_stats.get("failure", 0)) + 1
                else:
                    heur_stats["pending"] = int(heur_stats.get("pending", 0)) + 1

            entry["last"] = {
                "ts": _now(),
                "label": label,
                "success": success,
                "confidence": float(confidence),
                "heuristic": heuristic,
                "payload": payload or {},
            }

            observed = max(1, label_stats["success"] + label_stats["failure"])
            self.habits_strength[f"{source}::{label}"] = label_stats["success"] / observed

            _safe_write_json(self.paths["state"], self.state)

    # ---------- legacy compatibility helpers ----------
    def log_cycle(
        self,
        intrinsic: float,
        extrinsic: float,
        learning_rate: float,
        uncertainty: float,
    ) -> Dict[str, Any]:
        metrics = {
            "learning_rate": float(learning_rate),
            "uncertainty": float(uncertainty),
            "intrinsic_reward": float(intrinsic),
            "extrinsic_reward": float(extrinsic),
        }
        result = self.record_cycle(
            extra_tags={"legacy": metrics},
            manual_metrics=metrics,
        )

        entry = {
            "ts": result["snapshot"]["t"],
            "intr": metrics["intrinsic_reward"],
            "extr": metrics["extrinsic_reward"],
            "learn": metrics["learning_rate"],
            "uncert": metrics["uncertainty"],
        }
        with self._state_lock:
            legacy = self.state.setdefault("legacy_metrics_history", [])
            legacy.append(entry)
            self.state["legacy_metrics_history"] = legacy[-500:]
            _safe_write_json(self.paths["state"], self.state)
        return result

    def propose_macro_adjustments(self) -> List[str]:
        with self._state_lock:
            history = list(self.state.get("legacy_metrics_history", []))

        if len(history) < 10:
            return []

        last = history[-10:]
        avg_unc = statistics.fmean(item["uncert"] for item in last)
        avg_learn = statistics.fmean(item["learn"] for item in last)

        notes: List[str] = []
        if avg_unc > 0.65:
            notes.append("Augmenter exploration (curiosity), planifier plus de questions ciblées.")
        if avg_learn < 0.45:
            notes.append("Changer stratégie d'étude: plus d'exemples concrets et feedback.")
        if not notes:
            notes.append("Maintenir les stratégies actuelles, progression stable.")

        with self._state_lock:
            strategies = self.state.setdefault("strategies", [])
            strategies.append({"ts": _now(), "notes": notes})
            self.state["strategies"] = strategies[-200:]
            _safe_write_json(self.paths["state"], self.state)

        return notes
