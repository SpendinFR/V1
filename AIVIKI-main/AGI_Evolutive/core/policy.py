from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from math import sqrt
from typing import Callable, Dict, Any, List, Optional, Tuple

import logging
import random, copy, json, os, time

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.knowledge.mechanism_store import MechanismStore
from AGI_Evolutive.core.structures.mai import MAI, Bid
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


def rag_quality_signal(signals: dict) -> float:
    if not signals:
        return 0.0
    top1 = float(signals.get("rag_top1", 0.0))
    mean = float(signals.get("rag_mean", 0.0))
    div  = float(signals.get("rag_diversity", 0.0))
    n    = float(signals.get("rag_docs", 0.0))
    return max(0.0, min(1.0, 0.45*top1 + 0.35*mean + 0.15*div + 0.05*min(n/5.0,1.0)))


class _ThompsonBidSelector:
    """Beta–Bernoulli Thompson sampling over action success, folded into the same multi-criteria score."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.alpha = defaultdict(lambda: prior_alpha)
        self.beta = defaultdict(lambda: prior_beta)

    def _key(self, b: Bid) -> str:
        # clé stable par type d’action ; ajuste si tu veux plus fin
        return f"{b.action_hint}"

    def update(self, action_hint: str, success: bool) -> None:
        a = self.alpha[action_hint]
        b = self.beta[action_hint]
        if success:
            self.alpha[action_hint] = a + 1.0
        else:
            self.beta[action_hint] = b + 1.0

    def _sample_p(self, k: str) -> float:
        # échantillon Beta via deux Gamma
        a = self.alpha[k]
        b = self.beta[k]
        x = random.gammavariate(a, 1.0)
        y = random.gammavariate(b, 1.0)
        return (x / (x + y)) if (x + y) else 0.5

    def choose(self, bids: List[Bid]) -> Bid:
        if not bids:
            raise ValueError("bids list cannot be empty")

        def score(b: Bid):
            k = self._key(b)
            p = self._sample_p(k)  # “propension au succès” estimée
            base = (b.expected_info_gain - 0.3 * b.cost)
            return (p * base, b.urgency, b.affect_value)

        return max(bids, key=score)


class PolicyEngine:
    """Stores lightweight policy directives and strategy hints."""

    def __init__(self, path: str = "data/policy.json"):
        self.path = path
        self.state: Dict[str, Any] = {
            "drive_targets": {},
            "hints": []
        }
        self._load()
        # --- Policy telemetry / stats (fichier séparé, on ne touche pas à path)
        self._stats_path = "runtime/policy_stats.json"
        self._stats = defaultdict(lambda: {"s": 0, "n": 0})  # succès/essais par type de proposition
        self._last_confidence = 0.55
        self.last_decision: Optional[Dict[str, Any]] = None
        self._mechanisms = MechanismStore()
        self._bandit_selector = _ThompsonBidSelector()
        self.self_model: Optional[Any] = None

        # Stats légères pour la policy (modes d'action)
        self.stats = {
            "success": 0,
            "fail": 0,
            "by_mode": {"reflex": 0, "habit": 0, "deliberate": 0},
        }

        # charge si existe (best-effort)
        try:
            if os.path.exists(self._stats_path):
                with open(self._stats_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    for k, v in raw.items():
                        self._stats[k] = {"s": int(v.get("s", 0)), "n": int(v.get("n", 0))}
        except Exception:
            pass

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.state = json.load(fh)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(json_sanitize(self.state), fh, ensure_ascii=False, indent=2)

    def _stats_save(self) -> None:
        try:
            directory = os.path.dirname(self._stats_path) or "."
            os.makedirs(directory, exist_ok=True)
            with open(self._stats_path, "w", encoding="utf-8") as f:
                json.dump({k: dict(v) for k, v in self._stats.items()}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _key(self, proposal: Dict[str, Any]) -> str:
        return f"{proposal.get('type')}|{'/'.join(map(str, proposal.get('path', [])))}"

    def register_outcome(self, proposal: Dict[str, Any], success: bool) -> None:
        """À appeler APRES exécution d’une proposition (true/false)."""
        try:
            k = self._key(proposal)
            self._stats[k]["n"] += 1
            self._stats[k]["s"] += 1 if success else 0
            self._stats_save()
        except Exception:
            pass
        # Propager le feedback au bandit des bids si l'action est connue
        try:
            if self.last_decision and isinstance(self._bandit_selector, _ThompsonBidSelector):
                action = self.last_decision.get("decision")
                if isinstance(action, str) and action:
                    self._bandit_selector.update(action, bool(success))
        except Exception:
            pass
        mode = None
        if isinstance(proposal, dict):
            mode = proposal.get("mode")
            if mode is None:
                meta = proposal.get("meta")
                if isinstance(meta, dict):
                    mode = meta.get("mode")
        self.update_outcome(mode or "", success)

    def _wilson_lower_bound(self, s: int, n: int, z: float = 1.96) -> float:
        """Borne inférieure de Wilson (intervalle de confiance) = prudente."""
        if n <= 0:
            return 0.55  # prior doux
        p = s / n
        denom = 1 + (z * z) / n
        centre = p + (z * z) / (2 * n)
        margin = z * sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
        return max(0.0, min(1.0, (centre - margin) / denom))

    def _freq_conf(self, proposal: Dict[str, Any]) -> float:
        st = self._stats[self._key(proposal)]
        return self._wilson_lower_bound(st["s"], st["n"])

    def _stability_probe(self, proposer, homeo, runs: int = 5, jitter: float = 0.03) -> float:
        """Accord du top-1 sous petites perturbations des drives : 1 = très stable."""
        try:
            if proposer is None or homeo is None or not hasattr(proposer, "run_once_now"):
                return 0.5
            state = getattr(homeo, "state", {})
            if not isinstance(state, dict):
                return 0.5
            drives = state.get("drives", {})
            if not isinstance(drives, dict):
                return 0.5
            base = dict(drives)
            tops = []
            for _ in range(runs):
                perturbed = {k: max(0.0, min(1.0, v + random.uniform(-jitter, jitter))) for k, v in base.items()}
                drives.update(perturbed)
                props = proposer.run_once_now() or []
                tops.append(str(props[:1]))
            drives.update(base)
            most = max(tops.count(t) for t in set(tops))
            return most / runs
        except Exception:
            return 0.5

    def _value_and_risk(self, planner, proposal: Dict[str, Any], trials: int = 5) -> Tuple[float, float]:
        """Utilise Planner.simulate_action si dispo : renvoie (valeur≈[0..1], risque≈[0..1])."""
        try:
            if planner is None or not hasattr(planner, "simulate_action"):
                return 0.6, 0.25
            vals: List[float] = []
            for _ in range(trials):
                try:
                    v = float(planner.simulate_action(proposal))
                    vals.append(v)
                except Exception:
                    pass
            if not vals:
                return 0.6, 0.25
            m = sum(vals) / len(vals)
            # normalisation sigmoïde grossière pour valeur
            val = 1.0 / (1.0 + (2.718281828) ** (-m))
            # écart-type (borné)
            mu = m
            std = (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5
            risk = min(1.0, std / 5.0)
            return float(val), float(risk)
        except Exception:
            return 0.6, 0.25

    def confidence_for(self,
                       proposal: Dict[str, Any],
                       *,
                       proposer=None,
                       homeo=None,
                       planner=None,
                       ctx: Optional[Dict[str, Any]] = None) -> float:
        """Score final 0..1 : mélange fréquence (Wilson), stabilité, valeur, croyance, avec pénalité risque/novelty."""
        ctx = ctx or {}
        freq = self._freq_conf(proposal)                    # 0..1
        stab = self._stability_probe(proposer, homeo)       # 0..1
        val, risk = self._value_and_risk(planner, proposal) # 0..1, 0..1
        try:
            model = getattr(self, "self_model", None)
            if model and hasattr(model, "belief_confidence"):
                belief = float(model.belief_confidence(ctx))
            else:
                belief = float(ctx.get("belief_confidence", 0.6))
        except Exception:
            belief = float(ctx.get("belief_confidence", 0.6))
        novelty_fam = float(ctx.get("novelty_familiarity", 0.7))  # 0..1 (1 = familier, 0 = inédit)

        w = {"freq": 0.35, "stab": 0.25, "val": 0.25, "belief": 0.15}
        raw = w["freq"] * freq + w["stab"] * stab + w["val"] * val + w["belief"] * belief
        penalty = 0.20 * risk + 0.10 * (1.0 - novelty_fam)
        conf = max(0.0, min(1.0, raw - penalty))
        self._last_confidence = conf
        return conf

    def confidence(self) -> float:
        """Confiance globale (agrégée) — fallback si le test veut une API simple."""
        if not self._stats:
            return self._last_confidence
        vals = [self._wilson_lower_bound(v["s"], v["n"]) for v in self._stats.values()]
        vals.sort()
        median = vals[len(vals) // 2]
        return float(0.5 * median + 0.5 * self._last_confidence)

    # ------------------------------------------------------------------
    # MAI helpers
    def build_predicate_registry(self, state: Dict[str, Any]) -> Dict[str, Callable[..., bool]]:
        def _get_dialogue(st: Dict[str, Any]) -> Any:
            return st.get("dialogue")

        def _get_self_model(st: Dict[str, Any]) -> Any:
            return st.get("self_model")

        def _get_beliefs(st: Dict[str, Any]) -> Any:
            return st.get("beliefs")

        def _belief_confidence_above(st: Dict[str, Any], topic: Any, threshold: Any) -> bool:
            beliefs = _get_beliefs(st)
            if beliefs is None:
                return False
            confidence_for = getattr(beliefs, "confidence_for", None)
            if not callable(confidence_for):
                return False
            try:
                confidence = float(confidence_for(topic))
                return confidence >= float(threshold)
            except (TypeError, ValueError):
                return False

        def _has_commitment(st: Dict[str, Any], key: Any) -> bool:
            model = getattr(self, "self_model", None)
            if model is None:
                model = _get_self_model(st)
            if not hasattr(model, "has_commitment"):
                return False
            try:
                return bool(model.has_commitment(str(key)))
            except Exception:
                return False

        predicate_registry: Dict[str, Callable[..., bool]] = {
            "request_is_sensitive": lambda st: getattr(_get_dialogue(st), "is_sensitive", False),
            "audience_is_not_owner": lambda st: getattr(_get_dialogue(st), "audience_id", None)
            != getattr(_get_dialogue(st), "owner_id", None),
            "has_consent": lambda st: getattr(_get_dialogue(st), "has_consent", False),
            "imminent_harm_detected": lambda st: getattr(st.get("world"), "imminent_harm", False),
            "has_commitment": _has_commitment,
            "belief_mentions": lambda st, topic: bool(
                getattr(_get_beliefs(st), "contains", lambda _: False)(topic)
                if hasattr(_get_beliefs(st), "contains")
                else getattr(_get_beliefs(st), "has_fact", lambda _: False)(topic)
                if hasattr(_get_beliefs(st), "has_fact")
                else False
            ),
            "belief_confidence_above": _belief_confidence_above,
        }
        return predicate_registry

    def ingest_mechanism_bids(
        self,
        global_workspace: Any,
        state: Dict[str, Any],
        predicate_registry: Optional[Dict[str, Callable[..., bool]]] = None,
    ) -> List[MAI]:
        predicate_registry = predicate_registry or self.build_predicate_registry(state)
        emitted: List[MAI] = []
        if global_workspace is None or not hasattr(global_workspace, "submit"):
            return emitted
        try:
            mechanisms = list(self._mechanisms.scan_applicable(state, predicate_registry))
        except Exception:
            mechanisms = []
        for mechanism in mechanisms:
            try:
                for bid in mechanism.propose(state):
                    try:
                        global_workspace.submit(bid)
                    except AttributeError:
                        global_workspace.submit_bid(
                            bid.payload.get("origin", "mai"),
                            bid.action_hint,
                            max(0.0, min(1.0, bid.expected_info_gain)),
                            bid.payload,
                        )
                emitted.append(mechanism)
            except Exception:
                continue
        return emitted

    def explain(self, proposal: Optional[Dict[str, Any]] = None, **kw) -> Dict[str, Any]:
        """Renvoie les composantes (freq/stab/val/risk/belief/novelty) + score final."""
        if proposal is None:
            return {
                "confidence": self.confidence(),
                "note": "global"
            }
        proposer = kw.get("proposer")
        homeo = kw.get("homeo")
        planner = kw.get("planner")
        ctx = kw.get("ctx", {})
        freq = self._freq_conf(proposal)
        stab = self._stability_probe(proposer, homeo)
        val, risk = self._value_and_risk(planner, proposal)
        belief = float(ctx.get("belief_confidence", 0.6))
        novelty_fam = float(ctx.get("novelty_familiarity", 0.7))
        w = {"freq": 0.35, "stab": 0.25, "val": 0.25, "belief": 0.15}
        raw = w["freq"] * freq + w["stab"] * stab + w["val"] * val + w["belief"] * belief
        penalty = 0.20 * risk + 0.10 * (1.0 - novelty_fam)
        conf = max(0.0, min(1.0, raw - penalty))
        return {
            "components": {"freq": freq, "stab": stab, "val": val, "risk": risk, "belief": belief, "novelty_familiarity": novelty_fam},
            "weights": w,
            "penalty": penalty,
            "confidence": conf,
        }
    def adjust_drive_target(self, drive: str, target: float):
        self.state.setdefault("drive_targets", {})[drive] = float(max(0.0, min(1.0, target)))
        self.state.setdefault("history", []).append({
            "ts": time.time(),
            "event": "drive_target",
            "drive": drive,
            "target": target
        })
        self.state["history"] = self.state["history"][-200:]
        self._save()

    def register_hint(self, hint: Dict[str, Any]):
        entry = dict(hint)
        entry["ts"] = time.time()
        self.state.setdefault("hints", []).append(entry)
        self.state["hints"] = self.state["hints"][-100:]
        self._save()

    def validate_tactic(self, rule: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Stub pour validation de tactique : autorise par défaut."""
        return {"decision": "allow", "reason": "default_allow"}

    def recent_frictions(self, window_sec: int = 600) -> int:
        """Stub pour comptage des frictions récentes."""
        return 0

    def validate_proposal(self, proposal: Dict[str, Any], self_state: Dict[str, Any]) -> Dict[str, Any]:
        path = proposal.get("path", [])
        if not path:
            return {"decision": "deny", "reason": "path manquant"}

        if path[0] == "core_immutable":
            return {"decision": "deny", "reason": "noyau protégé"}

        if path == ["identity", "name"] and isinstance(proposal.get("value"), str) and len(proposal["value"]) > 20:
            return {"decision": "needs_human", "reason": "changement identité important"}

        return {"decision": "allow", "reason": "OK"}

    def compute_frame_utility(
        self,
        frame: Any,
        *,
        weights: Optional[Dict[str, float]] = None,
        components: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute aggregate utility for a conversational frame."""

        def _clamp01(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        def _collect_from(obj: Any, keys: List[str]) -> Dict[str, float]:
            collected: Dict[str, float] = {}
            for key in keys:
                source = None
                if isinstance(obj, dict):
                    source = obj.get(key)
                else:
                    source = getattr(obj, key, None)
                if isinstance(source, dict):
                    for name, val in source.items():
                        try:
                            collected[name] = float(val)
                        except (TypeError, ValueError):
                            continue
            return collected

        comp_sources = _collect_from(frame, ["utilities", "utility_components", "scores", "drives"])
        if components:
            for key, val in components.items():
                try:
                    comp_sources[key] = float(val)
                except (TypeError, ValueError):
                    continue

        w_sources = _collect_from(frame, ["weights", "drive_weights", "priorities"])
        if weights:
            for key, val in weights.items():
                try:
                    w_sources[key] = float(val)
                except (TypeError, ValueError):
                    continue

        U_survive = _clamp01(
            comp_sources.get("survive", comp_sources.get("Survive", 0.5))
        )
        U_evolve = _clamp01(
            comp_sources.get("evolve", comp_sources.get("Evolve", 0.5))
        )
        U_interact = _clamp01(
            comp_sources.get("interact", comp_sources.get("Interact", 0.5))
        )

        w_survive = w_sources.get("survive", w_sources.get("Survive", 1.0))
        w_evolve = w_sources.get("evolve", w_sources.get("Evolve", 1.0))
        w_interact = w_sources.get("interact", w_sources.get("Interact", 1.0))

        try:
            w_survive = float(w_survive)
        except (TypeError, ValueError):
            w_survive = 1.0
        try:
            w_evolve = float(w_evolve)
        except (TypeError, ValueError):
            w_evolve = 1.0
        try:
            w_interact = float(w_interact)
        except (TypeError, ValueError):
            w_interact = 1.0

        total_w = w_survive + w_evolve + w_interact
        if total_w <= 0:
            w_survive = w_evolve = w_interact = 1.0 / 3.0
        else:
            w_survive /= total_w
            w_evolve /= total_w
            w_interact /= total_w

        rq = rag_quality_signal(getattr(frame, "signals", {}))
        if isinstance(frame, dict):
            rq = rag_quality_signal(frame.get("signals", {}))

        # “Survivre” → éviter erreurs si support faible
        U_survive *= (0.7 + 0.6*rq)

        # “Évoluer” → exploiter lorsque support solide
        U_evolve  *= (0.8 + 0.5*rq)

        # “Interagir” → si support moyen/faible, favoriser clarification
        if rq < 0.35:
            U_interact *= 0.85
        elif rq < 0.6:
            U_interact *= 1.0
        else:
            U_interact *= 1.1

        U = w_survive*U_survive + w_evolve*U_evolve + w_interact*U_interact
        return max(0.0, min(1.0, U))

    def decide(
        self,
        proposals_or_ctx,
        self_state: Optional[Dict[str, Any]] = None,
        *,
        proposer=None,
        homeo=None,
        planner=None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sélectionne la meilleure proposition + confiance, avec abstention contrôlée.

        Compatible avec l'API historique (liste de propositions) et une API contextuelle
        simplifiée ``decide({"frame": ..., "scratch": ...})`` utilisée par le
        nouveau pipeline Policy/Planner.
        """

        if isinstance(proposals_or_ctx, dict) and self_state is None and proposer is None and homeo is None and planner is None and ctx is None:
            return self._decide_from_ctx(proposals_or_ctx)

        proposals: List[Dict[str, Any]] = list(proposals_or_ctx or [])
        self_state = self_state or {}
        ctx = ctx or {}
        if not proposals:
            self.last_decision = {"decision": "noop", "reason": "no proposals", "confidence": 0.5}
            return self.last_decision

        priority_hint = self._extract_priority(ctx)

        scored: List[Tuple[str, float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
        for idx, p in enumerate(proposals):
            gate = self.validate_proposal(p, self_state)  # conserve ta logique actuelle (allow/deny/needs_human)
            if gate.get("decision") == "deny":
                continue

            conf = self.confidence_for(p, proposer=proposer, homeo=homeo, planner=planner, ctx=ctx)
            # bonus léger sur actions "act" / malus sur "needs_human"
            bonus = 0.05 if p.get("type") in {"act", "do_step"} else 0.0
            if gate.get("decision") == "needs_human":
                conf -= 0.15
            final = max(0.0, min(1.0, conf + bonus))
            if priority_hint is not None:
                final = max(0.0, min(1.0, 0.65 * final + 0.35 * priority_hint))

            proposal_id = self._proposal_id(p, idx)
            scored.append((proposal_id, final, p, gate, {"conf": conf, "bonus": bonus, "priority_hint": priority_hint}))

        if not scored:
            self.last_decision = {"decision": "noop", "reason": "all denied", "confidence": 0.45}
            return self.last_decision

        scored.sort(key=lambda t: t[1], reverse=True)
        llm_selection = self._llm_select_proposal(scored, ctx=ctx, self_state=self_state)
        if llm_selection is not None:
            best_id, best_score, llm_meta = llm_selection
            found = next((item for item in scored if item[0] == best_id), None)
            if found is not None:
                _, _, best_p, best_gate, meta = found
                meta = dict(meta)
                meta["llm"] = llm_meta
            else:
                _, best_score, best_p, best_gate, meta = scored[0]
                meta = dict(meta)
                meta.setdefault("llm", {}).update(llm_meta or {})
        else:
            _, best_score, best_p, best_gate, meta = scored[0]

        if best_gate.get("decision") == "needs_human":
            self.last_decision = {
                "decision": "needs_human",
                "reason": best_gate.get("reason", "human validation required"),
                "confidence": best_score,
                "proposal": best_p,
                "policy_reason": best_gate.get("reason", ""),
                "meta": meta,
            }
            return self.last_decision

        # seuil d’abstention (risk-coverage simple)
        ABSTAIN = 0.58
        if best_score < ABSTAIN:
            self.last_decision = {
                "decision": "needs_human",
                "reason": "low confidence",
                "confidence": best_score,
                "proposal": best_p,
                "policy_reason": best_gate.get("reason", ""),
                "meta": meta,
            }
            return self.last_decision

        self.last_decision = {
            "decision": "apply",
            "proposal": best_p,
            "policy_reason": best_gate.get("reason", ""),
            "confidence": best_score,
            "meta": meta,
        }
        return self.last_decision

    def _proposal_id(self, proposal: Dict[str, Any], idx: int) -> str:
        for key in ("id", "proposal_id", "action", "name"):
            value = proposal.get(key)
            if isinstance(value, str) and value:
                return value
        return f"proposal_{idx}"

    def _llm_select_proposal(
        self,
        scored: List[Tuple[str, float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
        *,
        ctx: Optional[Dict[str, Any]],
        self_state: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        payload = {
            "proposals": [
                {
                    "id": proposal_id,
                    "heuristic_score": score,
                    "gate": gate,
                    "meta": meta,
                    "proposal": proposal,
                }
                for proposal_id, score, proposal, gate, meta in scored
            ],
            "context": ctx or {},
            "self_state": self_state or {},
        }
        response = try_call_llm_dict(
            "policy_engine",
            input_payload=payload,
            logger=LOGGER,
        )
        if not response:
            return None
        proposal_id = response.get("proposal_id")
        if not isinstance(proposal_id, str):
            return None
        value_estimate = response.get("value_estimate")
        if not isinstance(value_estimate, (int, float)):
            return None
        stability = response.get("stability")
        meta = {
            "stability": float(stability) if isinstance(stability, (int, float)) else None,
            "rationale": response.get("rationale", ""),
            "notes": response.get("notes", ""),
        }
        return proposal_id, max(0.0, min(1.0, float(value_estimate))), meta

    def _extract_priority(self, ctx: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(ctx, dict):
            return None
        scratch = ctx.get("scratch") if isinstance(ctx.get("scratch"), dict) else None
        if scratch is None:
            return None
        priority = scratch.get("priority")
        if isinstance(priority, dict):
            for key in ("score", "value", "priority", "p"):
                if key in priority and isinstance(priority[key], (int, float)):
                    priority = priority[key]
                    break
        if isinstance(priority, (int, float)):
            return max(0.0, min(1.0, float(priority)))
        return None

    def _decide_from_ctx(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx = ctx or {}
        scratch = ctx.get("scratch") if isinstance(ctx.get("scratch"), dict) else {}
        priority_hint = self._extract_priority({"scratch": scratch})
        action = self._compose_action_from_frame_or_goal(ctx)
        expected_score = priority_hint if priority_hint is not None else 1.0
        decision: Dict[str, Any] = {
            "action": action,
            "expected": {"score": float(max(0.0, min(1.0, expected_score)))},
        }
        reason = ctx.get("reason")
        if reason:
            decision["reason"] = reason
        mode = ctx.get("mode") or scratch.get("mode")
        if mode:
            decision["mode"] = mode
        payload = ctx.get("payload")
        if isinstance(payload, dict) and payload:
            decision.setdefault("context", {})
            decision["context"]["payload"] = payload
        self.last_decision = decision
        return decision

    def _compose_action_from_frame_or_goal(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        payload = ctx.get("payload")
        if isinstance(payload, dict) and payload.get("type"):
            return dict(payload)

        action: Dict[str, Any] = {"type": "reflect", "payload": {}}
        frame = ctx.get("frame")
        goal = ctx.get("goal")

        if isinstance(goal, dict) and goal:
            action_type = goal.get("action_type") or goal.get("type") or "plan_goal"
            action = {
                "type": action_type,
                "payload": {
                    "goal_id": goal.get("id") or goal.get("goal_id"),
                    "description": goal.get("description") or goal.get("desc"),
                    "context": goal.get("context"),
                },
            }

        elif frame:
            intent = None
            text = None
            if isinstance(frame, dict):
                intent = frame.get("intent") or frame.get("act")
                text = frame.get("text") or frame.get("surface_form")
            else:
                intent = getattr(frame, "intent", None)
                text = getattr(frame, "text", None) or getattr(frame, "surface_form", None)

            intent = (intent or "").lower()
            payload_text = text or ctx.get("reason") or ""
            mapping = {
                "ask": "message_user",
                "request": "message_user",
                "summarize": "plan",
                "plan": "plan",
                "inform": "write_memory",
                "reflect": "reflect",
            }
            action_type = mapping.get(intent, "reflect")
            if action_type == "message_user":
                action = {
                    "type": "message_user",
                    "payload": {"text": payload_text or "Peux-tu préciser ?"},
                }
            elif action_type == "write_memory":
                action = {
                    "type": "write_memory",
                    "payload": {"kind": "note", "text": payload_text},
                }
            elif action_type == "plan":
                action = {
                    "type": "plan",
                    "payload": {"frame": frame if isinstance(frame, dict) else getattr(frame, "__dict__", {})},
                }
            else:
                action = {"type": "reflect", "payload": {"focus": payload_text}}

        elif isinstance(ctx.get("reason"), str):
            action = {"type": "reflect", "payload": {"reason": ctx["reason"]}}

        return action

    def decide_with_bids(
        self,
        winners: Optional[List[Bid]],
        state: Dict[str, Any],
        *,
        global_workspace: Any = None,
        proposals: Optional[List[Dict[str, Any]]] = None,
        proposer=None,
        homeo=None,
        planner=None,
        ctx: Optional[Dict[str, Any]] = None,
        bandit_selector=None,
    ) -> Dict[str, Any]:
        ctx = ctx or {}
        predicate_registry = self.build_predicate_registry(state)
        emitted = []
        if global_workspace is not None:
            emitted = self.ingest_mechanism_bids(global_workspace, state, predicate_registry)
            if winners is None:
                try:
                    winners = list(global_workspace.winners())
                except Exception:
                    winners = []
        winners = list(winners or [])

        def violates_hard_invariant(bid: Bid) -> bool:
            return False

        feasible = [bid for bid in winners if not violates_hard_invariant(bid)]

        def dominates(a: Bid, b: Bid) -> bool:
            score_a = (a.expected_info_gain - 0.3 * a.cost, a.urgency, a.affect_value)
            score_b = (b.expected_info_gain - 0.3 * b.cost, b.urgency, b.affect_value)
            return score_a > score_b

        chosen: Optional[Bid] = None
        bandit_applied = False
        for candidate in feasible:
            if all(dominates(candidate, other) for other in feasible if other is not candidate):
                chosen = candidate
                break

        selector = bandit_selector or self._bandit_selector
        if chosen is None and feasible:
            try:
                chosen = selector.choose(feasible)
                bandit_applied = True
            except Exception:
                chosen = random.choice(feasible)

        decision_bundle: Dict[str, Any] = {
            "decision": "noop",
            "originating_bids": [bid.origin_tag() for bid in winners],
            "alternatives_rejetees": [asdict(bid) for bid in winners if bid is not chosen],
            "mai_emitted": [mai.id for mai in emitted],
            "evidence_refs": [
                asdict(ref)
                for mai in emitted
                for ref in getattr(mai, "provenance_docs", [])
            ],
        }

        if chosen is not None:
            decision_bundle.update(
                {
                    "decision": chosen.action_hint,
                    "chosen_bid": chosen.serialise(),
                    "confidence": max(0.0, min(1.0, chosen.expected_info_gain)),
                }
            )
        else:
            decision_bundle["chosen_bid"] = None

        decision_bundle["justification"] = {
            "dominance_checked": bool(feasible),
            "bandit_used": bandit_applied,
            "ctx": dict(ctx),
            "feasible_count": len(feasible),
        }

        if chosen is None and proposals is not None:
            try:
                fallback = self.decide(
                    proposals,
                    self_state=state.get("self_state"),
                    proposer=proposer,
                    homeo=homeo,
                    planner=planner,
                    ctx=ctx,
                )
                decision_bundle.setdefault("fallback_decision", fallback)
            except Exception:
                pass

        self.last_decision = decision_bundle
        return decision_bundle

    def update_outcome(self, mode: str, ok: bool) -> None:
        mode_key = (mode or "").lower()
        if ok:
            self.stats["success"] += 1
        else:
            self.stats["fail"] += 1
        if mode_key in self.stats["by_mode"]:
            self.stats["by_mode"][mode_key] += 1
