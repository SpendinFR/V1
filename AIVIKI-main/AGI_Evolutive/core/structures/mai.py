"""Core dataclasses and helpers for policy/knowledge/runtime integration.

This module defines:
- EvidenceRef: references to supporting material.
- ImpactHypothesis: qualitative hypothesis about applying an MAI.
- Bid: lightweight container used by MAIs to propose actions to the workspace.
- MAI: Mechanistic Actionable Insight with preconditions and bid generation.

Conventions:
- 'source' fields are string tags (e.g., "MAI:<id>", "planner", "critic").
- Bid expiration uses either an absolute 'expires_at' (epoch seconds) or a
  relative duration via 'expires_in' / 'ttl_s' in configs.
"""

from __future__ import annotations

import logging
import time
import uuid
import random
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from AGI_Evolutive.cognition.meta_cognition import OnlineLinear
from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)

# ---------- Types ----------
Expr = Dict[str, Any]  # {"op":"and","args":[...]} | {"op":"atom","name":"has_commitment","args":[...]}


# ---------- Dataclasses ----------
@dataclass
class EvidenceRef:
    """Reference to supporting material for an MAI."""
    source: Optional[str] = None            # e.g. "doc:xyz.pdf#p3", "episode:<id>"
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    kind: Optional[str] = None              # e.g. "doc", "episode", "web"
    weight: float = 1.0


@dataclass
class ImpactHypothesis:
    """Qualitative hypothesis about the impact of applying an MAI."""
    trust_delta: float = 0.0
    harm_delta: float = 0.0
    identity_coherence_delta: float = 0.0
    competence_delta: float = 0.0
    regret_delta: float = 0.0
    uncertainty: float = 0.5
    confidence: float = 0.0
    rationale: Optional[str] = None
    caveats: Optional[str] = None


@dataclass
class Bid:
    """Bid emitted by an MAI or another attentional mechanism."""
    source: str                              # e.g., "MAI:<id>" | "planner" | "critic"
    action_hint: str                         # e.g., "AskConsent", "ClarifyIntent", ...
    target: Optional[Any] = None
    rationale: Optional[str] = None
    expected_info_gain: float = 0.0
    urgency: float = 0.0
    affect_value: float = 0.0
    cost: float = 0.0
    expires_at: Optional[float] = None       # epoch seconds
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)

    def serialise(self) -> Dict[str, Any]:
        data = asdict(self)
        # ensure a self-describing origin is always present
        data.setdefault("payload", {})
        data["payload"].setdefault("origin", self.source)
        return data


# ---------- Expression evaluation ----------
def eval_expr(expr: Expr, state: Mapping[str, Any],
              predicate_registry: Mapping[str, Callable[..., bool]]) -> bool:
    """Evaluate a minimal boolean expression tree against the provided state.

    Supported ops:
      - {"op":"atom","name":<predicate>,"args":[...]}
      - {"op":"and","args":[...]}
      - {"op":"or","args":[...]}
      - {"op":"not","args":[<expr>]}
    Unknown ops or missing predicates evaluate to False (fail-safe).
    """
    if not isinstance(expr, Mapping):
        return False

    op = expr.get("op")
    if op == "atom":
        name = expr.get("name")
        args = expr.get("args", [])
        func = predicate_registry.get(name) if isinstance(name, str) else None
        if func is None:
            return False
        try:
            return bool(func(state, *args))
        except TypeError:
            # allow predicates with (state) signature if args provided accidentally
            try:
                return bool(func(state))
            except Exception:
                return False
        except Exception:
            return False

    if op == "and":
        return all(eval_expr(e, state, predicate_registry) for e in expr.get("args", []))
    if op == "or":
        return any(eval_expr(e, state, predicate_registry) for e in expr.get("args", []))
    if op == "not":
        args = expr.get("args", [])
        return not eval_expr(args[0], state, predicate_registry) if args else True
    return False


# ---------- MAI ----------
@dataclass
class MAI:
    """Mechanistic Actionable Insight."""
    id: str
    version: int = 1
    docstring: str = ""
    title: str = ""
    summary: str = ""
    status: str = "draft"

    expected_impact: ImpactHypothesis = field(default_factory=ImpactHypothesis)
    provenance_docs: List[EvidenceRef] = field(default_factory=list)
    provenance_episodes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner: Optional[str] = None

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Preconditions (legacy list form) and/or structured expression
    preconditions: List[Any] = field(default_factory=list)
    precondition_expr: Expr = field(default_factory=dict)

    # Bid sources
    bids: List[Mapping[str, Any]] = field(default_factory=list)  # explicit inline bid configs
    propose_spec: Dict[str, Any] = field(default_factory=dict)   # {"bids":[ {...}, ... ]}

    # Safety and runtime
    safety_invariants: List[str] = field(default_factory=list)
    runtime_counters: Dict[str, float] = field(default_factory=lambda: {
        "activation": 0.0,
        "wins": 0.0,
        "benefit": 0.0,
        "regret": 0.0,
        "rollbacks": 0.0,
    })
    adaptive_state: Dict[str, Any] = field(default_factory=dict)

    _ADAPTIVE_MODELS: ClassVar[Tuple[str, ...]] = ("expected_info_gain", "urgency", "affect_value", "cost")
    _DEFAULT_BANDIT_OFFSETS: ClassVar[Tuple[Dict[str, float], ...]] = (
        {"expected_info_gain": 0.0, "urgency": 0.0, "affect_value": 0.0, "cost": 0.0},
        {"expected_info_gain": 0.15, "urgency": 0.1, "affect_value": 0.05, "cost": 0.05},
        {"expected_info_gain": -0.1, "urgency": -0.05, "affect_value": 0.1, "cost": -0.05},
        {"expected_info_gain": 0.05, "urgency": 0.2, "affect_value": -0.05, "cost": 0.1},
    )

    # ---- Helpers ----
    def _iter_preconditions(self) -> Iterable[Any]:
        yield from self.preconditions
        meta_pre = self.metadata.get("preconditions") if isinstance(self.metadata, Mapping) else None
        if isinstance(meta_pre, (list, tuple)):
            for cond in meta_pre:
                yield cond

    def _iter_bid_configs(self) -> Iterable[Mapping[str, Any]]:
        # explicit configs
        for conf in self.bids:
            if isinstance(conf, Mapping):
                yield conf

        # from metadata
        meta_bids = self.metadata.get("bids") if isinstance(self.metadata, Mapping) else None
        if isinstance(meta_bids, list):
            for conf in meta_bids:
                if isinstance(conf, Mapping):
                    yield conf

        # from propose_spec
        for conf in self.propose_spec.get("bids", []):
            if isinstance(conf, Mapping):
                yield conf

        # fallback default if nothing was provided
        if not self.bids and not meta_bids and not self.propose_spec.get("bids"):
            yield {
                "action_hint": self.metadata.get("action_hint", "ClarifyIntent"),
                "expected_info_gain": float(self.expected_impact.confidence),
                "affect_value": float(self.expected_impact.trust_delta),
                "urgency": float(self.metadata.get("urgency", 0.0)),
                "cost": float(self.metadata.get("cost", 0.0)),
            }

    # ---- Logic ----
    def is_applicable(self, state: Mapping[str, Any],
                      predicate_registry: Mapping[str, Callable[..., bool]]) -> bool:
        if self.precondition_expr:
            try:
                return eval_expr(self.precondition_expr, dict(state), dict(predicate_registry))
            except Exception:
                return False

        # legacy list of predicates
        for cond in self._iter_preconditions():
            name: Optional[str]
            args: Sequence[Any]
            negate = False

            if isinstance(cond, str):
                name, args = cond, ()
            elif isinstance(cond, Mapping):
                name = cond.get("name") or cond.get("predicate")
                negate = bool(cond.get("negate"))
                raw_args = cond.get("args", ())
                if isinstance(raw_args, Sequence) and not isinstance(raw_args, (str, bytes)):
                    args = tuple(raw_args)
                elif raw_args is None:
                    args = ()
                else:
                    args = (raw_args,)
            else:
                # unsupported entry, treat as failing precondition
                return False

            pred = predicate_registry.get(name)
            if pred is None:
                return False
            try:
                result = pred(state, *args)
            except TypeError:
                result = pred(state)
            except Exception:
                result = False

            result = not bool(result) if negate else bool(result)
            if not result:
                return False

        return True

    def propose(self, state: Mapping[str, Any]) -> List[Bid]:
        now = time.time()
        proposals: List[Bid] = []
        features = self._collect_features(state)
        models = self._ensure_models()
        predictions = self._predict_metrics(models, features)
        offsets, chosen_idx = self._select_bandit_offsets()
        last_base: Dict[str, float] = {}
        last_adapted: Dict[str, float] = {}
        for conf in self._iter_bid_configs():
            action_hint = str(conf.get("action_hint", "ClarifyIntent"))
            expected_info_gain = float(conf.get("expected_info_gain", self.expected_impact.confidence))
            urgency = float(conf.get("urgency", 0.0))
            affect_value = float(conf.get("affect_value", self.expected_impact.trust_delta))
            cost = float(conf.get("cost", 0.0))
            rationale = conf.get("rationale")
            target = conf.get("target")

            base_metrics = {
                "expected_info_gain": expected_info_gain,
                "urgency": urgency,
                "affect_value": affect_value,
                "cost": cost,
            }
            adapted = self._blend_metrics(base_metrics, predictions, offsets)
            last_base = dict(base_metrics)
            last_adapted = dict(adapted)
            expected_info_gain = adapted["expected_info_gain"]
            urgency = adapted["urgency"]
            affect_value = adapted["affect_value"]
            cost = adapted["cost"]

            # normalize expiration
            expires_at = conf.get("expires_at")
            if expires_at is None:
                rel = conf.get("expires_in", conf.get("ttl_s"))
                if rel is not None:
                    try:
                        expires_at = now + float(rel)
                    except Exception:
                        expires_at = None

            # payload with origin
            payload_raw = conf.get("payload", {})
            payload = dict(payload_raw) if isinstance(payload_raw, Mapping) else {}
            payload.setdefault("origin", f"MAI:{self.id}")

            proposals.append(
                Bid(
                    source=f"MAI:{self.id}",
                    action_hint=action_hint,
                    target=target,
                    rationale=rationale,
                    expected_info_gain=expected_info_gain,
                    urgency=urgency,
                    affect_value=affect_value,
                    cost=cost,
                    expires_at=expires_at,
                    payload=payload,
                    evidence_refs=list(self.provenance_docs),
                )
                )
        if proposals:
            self._record_last_context(features, last_base, predictions, offsets, chosen_idx, last_adapted)
        enriched = self._enrich_bids_with_llm(proposals, state, features, predictions)
        return enriched if enriched is not None else proposals

    def touch(self) -> None:
        self.updated_at = time.time()

    def _enrich_bids_with_llm(
        self,
        proposals: List[Bid],
        state: Mapping[str, Any],
        features: Mapping[str, float],
        predictions: Mapping[str, float],
    ) -> Optional[List[Bid]]:
        if not proposals:
            return None

        payload = {
            "mai_id": self.id,
            "state_features": dict(features),
            "predictions": dict(predictions),
            "bids": [
                {
                    "index": idx,
                    "action_hint": bid.action_hint,
                    "expected_info_gain": bid.expected_info_gain,
                    "urgency": bid.urgency,
                    "affect_value": bid.affect_value,
                    "cost": bid.cost,
                    "rationale": bid.rationale,
                }
                for idx, bid in enumerate(proposals)
            ],
            "recent_state_keys": sorted(list(state.keys()))[:20],
        }
        response = try_call_llm_dict(
            "mai_bid_coach",
            input_payload=json_sanitize(payload),
            logger=LOGGER,
            max_retries=2,
        )
        if not isinstance(response, Mapping):
            return self._heuristic_bid_guidance(proposals)

        prioritized = response.get("prioritized_bids")
        if not isinstance(prioritized, list):
            return self._heuristic_bid_guidance(proposals)

        clones = [
            replace(bid, payload=dict(bid.payload), evidence_refs=list(bid.evidence_refs))
            for bid in proposals
        ]
        ordered: List[Bid] = []
        seen: set[int] = set()
        for entry in prioritized:
            if not isinstance(entry, Mapping):
                continue
            idx = entry.get("index")
            if not isinstance(idx, int) or not (0 <= idx < len(clones)):
                continue
            clone = clones[idx]
            adjustments = entry.get("adjustments")
            if isinstance(adjustments, Mapping):
                if "expected_info_gain" in adjustments:
                    clone.expected_info_gain = float(
                        max(0.0, min(1.0, float(adjustments["expected_info_gain"])))
                    )
                if "urgency" in adjustments:
                    clone.urgency = float(max(0.0, min(1.0, float(adjustments["urgency"]))))
                if "affect_value" in adjustments:
                    clone.affect_value = float(max(-1.0, min(1.0, float(adjustments["affect_value"]))))
                if "cost" in adjustments:
                    clone.cost = float(max(0.0, float(adjustments["cost"])))
            reason = self._clean_text(entry.get("reason"))
            if reason:
                clone.payload.setdefault("annotations", {})
                clone.payload["annotations"]["llm_reason"] = reason
            notes = self._clean_text(entry.get("notes"))
            if notes:
                clone.payload.setdefault("annotations", {})
                clone.payload["annotations"]["llm_notes"] = notes
            ordered.append(clone)
            seen.add(idx)

        if not ordered:
            return self._heuristic_bid_guidance(proposals)

        for idx, clone in enumerate(clones):
            if idx not in seen:
                ordered.append(clone)

        return ordered

    def _heuristic_bid_guidance(self, proposals: List[Bid]) -> List[Bid]:
        scored = []
        for bid in proposals:
            score = (
                0.6 * bid.expected_info_gain
                + 0.25 * bid.urgency
                + 0.15 * max(0.0, bid.affect_value)
                - 0.1 * max(0.0, bid.cost)
            )
            scored.append((score, bid))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]

    @staticmethod
    def _clean_text(value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        return ""

    def update_from_feedback(self, delta: Mapping[str, float]) -> None:
        for key, value in delta.items():
            self.runtime_counters[key] = self.runtime_counters.get(key, 0.0) + float(value)
        self._update_bandit(delta)
        self._train_models(delta)
        self.touch()

    # ---- Adaptive helpers ----
    def _ensure_models(self) -> Dict[str, OnlineLinear]:
        cache = getattr(self, "_adaptive_models_cache", None)
        if cache is not None:
            return cache
        models_state = {}
        if isinstance(self.adaptive_state, Mapping):
            models_state = self.adaptive_state.get("models", {}) if isinstance(self.adaptive_state.get("models"), Mapping) else {}
        cache = {}
        for name in self._ADAPTIVE_MODELS:
            state = models_state.get(name) if isinstance(models_state, Mapping) else None
            cache[name] = OnlineLinear.from_state(
                state,
                bounds=(0.0, 1.0),
                lr=0.04,
                l2=0.0015,
                max_grad=0.25,
                warmup=10,
                init_weight=0.1,
            )
        setattr(self, "_adaptive_models_cache", cache)
        return cache

    def _collect_features(self, state: Mapping[str, Any]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        runtime = self.runtime_counters
        activation = float(runtime.get("activation", 0.0))
        wins = float(runtime.get("wins", 0.0))
        regret = float(runtime.get("regret", 0.0))
        benefit = float(runtime.get("benefit", 0.0))
        rollbacks = float(runtime.get("rollbacks", 0.0))
        total = max(1.0, activation)
        features["activation_norm"] = min(activation / 100.0, 1.0)
        features["win_rate"] = max(0.0, min(1.0, wins / total))
        features["benefit_avg"] = max(-1.0, min(1.0, benefit / total))
        features["regret_avg"] = max(0.0, min(1.0, regret / total))
        features["rollback_rate"] = max(0.0, min(1.0, rollbacks / total))

        if isinstance(state, Mapping):
            meta = state.get("meta") if isinstance(state.get("meta"), Mapping) else {}
            drives = state.get("drives") if isinstance(state.get("drives"), Mapping) else {}
            context = state.get("context") if isinstance(state.get("context"), Mapping) else {}
            features["state_uncertainty"] = self._safe_float(meta.get("uncertainty", meta.get("confidence", 0.5)), 0.5) if meta else 0.5
            features["state_urgency"] = self._safe_float(meta.get("urgency", meta.get("pressure", 0.3)), 0.3) if meta else 0.3
            features["drive_survive"] = self._safe_float(drives.get("survive", 0.5), 0.5) if drives else 0.5
            features["drive_evolve"] = self._safe_float(drives.get("evolve", 0.5), 0.5) if drives else 0.5
            features["drive_interact"] = self._safe_float(drives.get("interact", 0.5), 0.5) if drives else 0.5
            if context:
                features["context_complexity"] = self._safe_float(context.get("complexity", 0.3), 0.3)
                features["context_priority"] = self._safe_float(context.get("priority", 0.3), 0.3)
            workspace = state.get("workspace") if isinstance(state.get("workspace"), Mapping) else {}
            if workspace:
                features["workspace_focus"] = self._safe_float(workspace.get("focus", 0.4), 0.4)
                features["workspace_load"] = self._safe_float(workspace.get("load", 0.5), 0.5)
        return {k: float(v) for k, v in features.items() if isinstance(v, (int, float))}

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _predict_metrics(
        self,
        models: Mapping[str, OnlineLinear],
        features: Mapping[str, float],
    ) -> Dict[str, float]:
        predictions: Dict[str, float] = {}
        for name, model in models.items():
            try:
                predictions[name] = float(model.predict(dict(features)))
            except Exception:
                predictions[name] = 0.5
        return predictions

    def _select_bandit_offsets(self) -> Tuple[Dict[str, float], int]:
        if not isinstance(self.adaptive_state, dict):
            self.adaptive_state = {}
        bandit = self.adaptive_state.setdefault("bandit", {})
        offsets = bandit.setdefault("offsets", [dict(candidate) for candidate in self._DEFAULT_BANDIT_OFFSETS])
        stats = bandit.setdefault("stats", {str(idx): {"alpha": 1.0, "beta": 1.0} for idx in range(len(offsets))})
        best_idx = 0
        best_score = -1.0
        for idx, _ in enumerate(offsets):
            key = str(idx)
            entry = stats.setdefault(key, {"alpha": 1.0, "beta": 1.0})
            alpha = max(1e-3, float(entry.get("alpha", 1.0)))
            beta = max(1e-3, float(entry.get("beta", 1.0)))
            sample = random.betavariate(alpha, beta)
            if sample > best_score:
                best_score = sample
                best_idx = idx
        bandit["last_choice"] = best_idx
        return offsets[best_idx], best_idx

    def _blend_metrics(
        self,
        base: Mapping[str, float],
        predictions: Mapping[str, float],
        offsets: Mapping[str, float],
    ) -> Dict[str, float]:
        adapted: Dict[str, float] = {}
        models = self._ensure_models()
        for name in self._ADAPTIVE_MODELS:
            base_val = float(base.get(name, 0.0))
            pred = float(predictions.get(name, base_val))
            offset = float(offsets.get(name, 0.0))
            model = models.get(name)
            confidence = float(model.confidence()) if model else 0.0
            exploration_weight = max(0.0, 0.4 * (1.0 - confidence))
            offset_value = max(0.0, min(1.0, base_val + offset))
            blended = (
                (1.0 - confidence - exploration_weight) * base_val
                + confidence * pred
                + exploration_weight * offset_value
            )
            adapted[name] = max(0.0, min(1.0, blended))
        return adapted

    def _record_last_context(
        self,
        features: Mapping[str, float],
        base: Mapping[str, float],
        predictions: Mapping[str, float],
        offsets: Mapping[str, float],
        candidate_idx: int,
        final_metrics: Mapping[str, float],
    ) -> None:
        if not isinstance(self.adaptive_state, dict):
            self.adaptive_state = {}
        self.adaptive_state["last_context"] = {
            "ts": time.time(),
            "features": dict(features),
            "base": dict(base),
            "predictions": dict(predictions),
            "offsets": dict(offsets),
            "candidate": int(candidate_idx),
            "final": dict(final_metrics),
        }

    def _update_bandit(self, delta: Mapping[str, float]) -> None:
        if not isinstance(self.adaptive_state, dict):
            return
        bandit = self.adaptive_state.get("bandit")
        if not isinstance(bandit, dict):
            return
        stats = bandit.get("stats")
        if not isinstance(stats, dict):
            return
        choice = bandit.get("last_choice")
        if choice is None:
            return
        entry = stats.setdefault(str(int(choice)), {"alpha": 1.0, "beta": 1.0})
        activation = max(0.0, float(delta.get("activation", 0.0)))
        wins = max(0.0, float(delta.get("wins", 0.0)))
        benefit = float(delta.get("benefit", 0.0))
        regret = max(0.0, float(delta.get("regret", 0.0)))
        success = max(0.0, wins + max(0.0, benefit))
        failure = max(0.0, activation - wins + regret)
        entry["alpha"] = max(1e-3, float(entry.get("alpha", 1.0)) + success)
        entry["beta"] = max(1e-3, float(entry.get("beta", 1.0)) + failure)

    def _train_models(self, delta: Mapping[str, float]) -> None:
        if not isinstance(self.adaptive_state, dict):
            return
        last = self.adaptive_state.get("last_context")
        if not isinstance(last, Mapping):
            return
        features = last.get("features")
        if not isinstance(features, Mapping):
            return
        activation = max(0.0, float(delta.get("activation", 0.0)))
        wins = max(0.0, float(delta.get("wins", 0.0)))
        benefit = float(delta.get("benefit", 0.0))
        regret = max(0.0, float(delta.get("regret", 0.0)))
        total = max(1.0, activation)
        success_ratio = max(0.0, min(1.0, wins / total))
        net_benefit = max(-1.0, min(1.0, benefit - regret))
        regret_norm = max(0.0, min(1.0, regret))
        targets = {
            "expected_info_gain": max(0.0, min(1.0, 0.4 * success_ratio + 0.6 * max(0.0, benefit))),
            "affect_value": max(0.0, min(1.0, 0.5 + 0.4 * net_benefit)),
            "urgency": max(0.0, min(1.0, 0.2 + 0.5 * success_ratio + 0.3 * regret_norm)),
            "cost": max(0.0, min(1.0, 0.4 - 0.3 * net_benefit + 0.2 * regret_norm)),
        }
        models = self._ensure_models()
        for name, target in targets.items():
            model = models.get(name)
            if model is None:
                continue
            try:
                model.update(dict(features), float(target))
            except Exception:
                continue
        self.adaptive_state.setdefault("models", {})
        model_state = self.adaptive_state["models"]
        if isinstance(model_state, dict):
            for name, model in models.items():
                try:
                    model_state[name] = model.to_state()
                except Exception:
                    continue


# ---------- Factory ----------
def new_mai(docstring: str, precondition_expr: Expr, propose_spec: Dict[str, Any],
            evidence: List[EvidenceRef], safety: List[str]) -> MAI:
    return MAI(
        id=str(uuid.uuid4()),
        version=1,
        docstring=docstring,
        precondition_expr=precondition_expr,
        propose_spec=propose_spec,
        provenance_docs=evidence,
        safety_invariants=safety,
    )


__all__ = ["EvidenceRef", "ImpactHypothesis", "Bid", "MAI", "eval_expr", "new_mai"]
