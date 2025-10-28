# Représentation robuste d'un "pattern d’interaction" comme règle sociale testable
# Forme : ⟨ Contexte → Tactique → Effets_attendus ⟩ + incertitude (postérieurs Beta par effet)
# Intègre un appel LLM optionnel (fallback heuristique conservé). 100% JSON-sérialisable.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import time, math, hashlib, json, logging

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


class OnlineLinear:
    """Mini-estimateur linéaire en ligne avec régularisation L2."""

    def __init__(self, init: Optional[Dict[str, float]] = None,
                 lr: float = 0.05,
                 l2: float = 0.01,
                 bounds: Tuple[float, float] = (0.0, 1.0)):
        self.lr = lr
        self.l2 = l2
        self.bounds = bounds
        self.weights: Dict[str, float] = dict(init or {})
        self.prior: Dict[str, float] = dict(init or {})

    def ensure_defaults(self, defaults: Dict[str, float]):
        for k, v in defaults.items():
            if k not in self.weights:
                self.weights[k] = v
                self.prior.setdefault(k, v)

    def current_weights(self, defaults: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if defaults:
            self.ensure_defaults(defaults)
        return dict(self.weights)

    def predict(self, features: Dict[str, float]) -> float:
        return sum(self.weights.get(k, 0.0) * features.get(k, 0.0) for k in features.keys())

    def update(self, features: Dict[str, float], target: float,
               defaults: Optional[Dict[str, float]] = None):
        if defaults:
            self.ensure_defaults(defaults)
        if not features:
            return
        pred = clamp(self.predict(features))
        error = pred - clamp(target)
        for name, value in features.items():
            if name not in self.weights:
                base = defaults.get(name, 0.0) if defaults else 0.0
                self.weights[name] = base
                self.prior.setdefault(name, base)
            grad = error * value + self.l2 * (self.weights[name] - self.prior.get(name, 0.0))
            self.weights[name] = clamp(self.weights[name] - self.lr * grad,
                                       self.bounds[0], self.bounds[1])

    def state(self) -> Dict[str, float]:
        return dict(self.weights)


class AdaptiveEMA:
    """Sélection adaptative du coefficient beta d'EMA via suivi d'erreur."""

    def __init__(self,
                 betas: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.6, 0.8),
                 state: Optional[Dict[str, Any]] = None,
                 baseline: float = 0.5):
        self.betas = betas
        state = state or {}
        self.values: Dict[float, float] = {
            float(b): float(state.get("values", {}).get(str(b), baseline)) for b in betas
        }
        self.errors: Dict[float, float] = {
            float(b): float(state.get("errors", {}).get(str(b), 0.5 + abs(b - 0.3) * 0.01))
            for b in betas
        }
        self.active_beta: float = float(state.get("active_beta", 0.3))
        self._last_baseline = baseline

    @classmethod
    def from_state(cls, state: Optional[Dict[str, Any]], baseline: float) -> "AdaptiveEMA":
        return cls(state=state, baseline=baseline)

    def _ensure(self, baseline: float):
        for b in self.betas:
            if b not in self.values:
                self.values[b] = baseline
            if b not in self.errors:
                self.errors[b] = 0.5 + abs(b - 0.3) * 0.01
        self._last_baseline = baseline

    def update(self, reward: float, baseline: float) -> float:
        self._ensure(baseline)
        reward = clamp(reward)
        best_beta = self.active_beta
        best_error = None
        best_value = baseline
        for b in self.betas:
            pred = self.values[b]
            err = abs(reward - pred)
            self.errors[b] = 0.9 * self.errors[b] + 0.1 * err
            new_value = (1.0 - b) * pred + b * reward
            self.values[b] = clamp(new_value)
            if best_error is None or self.errors[b] < best_error:
                best_error = self.errors[b]
                best_beta = b
                best_value = self.values[b]
        self.active_beta = best_beta
        return clamp(best_value)

    def state(self) -> Dict[str, Any]:
        return {
            "values": {str(k): v for k, v in self.values.items()},
            "errors": {str(k): v for k, v in self.errors.items()},
            "active_beta": self.active_beta,
        }

# ---------- Petites bases ----------
TS = float

def _now() -> TS:
    return time.time()

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))

# ---------- Prédicats symboliques de contexte ----------
@dataclass
class Predicate:
    """
    Prédicat symbolique: "dialogue_act == insinuation", "risk_level <= medium", etc.
    ops supportés: 'eq','neq','in','nin','ge','gt','le','lt','exists','missing'
    """
    key: str
    op: str
    value: Any = None
    weight: float = 1.0  # importance dans le match global

    def test(self, ctx: Dict[str, Any]) -> Tuple[bool, float]:
        v = ctx
        for k in self.key.split("."):
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                v = None
                break

        ok = False
        if self.op == "exists":
            ok = v is not None
        elif self.op == "missing":
            ok = v is None
        elif self.op == "eq":
            ok = (v == self.value)
        elif self.op == "neq":
            ok = (v != self.value)
        elif self.op == "in":
            ok = (v in (self.value or []))
        elif self.op == "nin":
            ok = (v not in (self.value or []))
        elif self.op == "ge":
            try: ok = float(v) >= float(self.value)
            except: ok = False
        elif self.op == "gt":
            try: ok = float(v) > float(self.value)
            except: ok = False
        elif self.op == "le":
            try: ok = float(v) <= float(self.value)
            except: ok = False
        elif self.op == "lt":
            try: ok = float(v) < float(self.value)
            except: ok = False

        # score partiel = weight si ok, sinon 0
        return ok, (self.weight if ok else 0.0)

# ---------- Tactique (transformation pragmatique) ----------
@dataclass
class TacticSpec:
    """
    Tactique abstraite (pas un texte), paramétrable.
    Exemples: 'reformulation_empathique', 'banter_leger', 'question_socratique', 'ack_gracieux'
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.name}:{json.dumps(self.params, sort_keys=True, ensure_ascii=False)}"

# ---------- Effets observables & postérieurs ----------
@dataclass
class EffectPosterior:
    """
    Modélise un effet attendu avec incertitude via une Beta(α,β).
    Exemple d’effets: 'reduce_uncertainty', 'continue_dialogue', 'positive_valence', 'acceptance_marker'
    """
    alpha: float = 1.0
    beta: float = 1.0
    # Fenêtre glissante approximative via "decay" (0..1); ex: 0.98 → oublie en douceur
    decay: float = 0.995
    decay_candidates: Tuple[float, ...] = (0.95, 0.98, 0.995)
    decay_posteriors: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    decay_choice: float = field(default=0.995, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.decay_posteriors:
            self.decay_posteriors = {float(c): (1.0, 1.0) for c in self.decay_candidates}
        else:
            cleaned: Dict[float, Tuple[float, float]] = {}
            for k, v in self.decay_posteriors.items():
                if isinstance(v, dict):
                    cleaned[float(k)] = (float(v.get("alpha", 1.0)), float(v.get("beta", 1.0)))
                else:
                    cleaned[float(k)] = (float(v[0]), float(v[1]))
            self.decay_posteriors = cleaned
        if self.decay not in self.decay_posteriors:
            self.decay_posteriors[self.decay] = (1.0, 1.0)
        self.decay_choice = self.decay

    def _select_decay(self):
        # Thompson discret simplifié → choisir la valeur à espérance max.
        best_decay = self.decay
        current = self.decay_posteriors.get(self.decay)
        if current:
            a_cur, b_cur = current
            denom = a_cur + b_cur
            best_score = a_cur / denom if denom > 0 else 0.5
        else:
            best_score = -1.0
        for decay_value, (a, b) in self.decay_posteriors.items():
            denom = a + b
            score = a / denom if denom > 0 else 0.5
            if score > best_score:
                best_score = score
                best_decay = decay_value
        self.decay = best_decay
        self.decay_choice = best_decay

    def _update_decay_bandit(self, success: bool):
        a, b = self.decay_posteriors.get(self.decay_choice, (1.0, 1.0))
        if success:
            a += 1.0
        else:
            b += 1.0
        self.decay_posteriors[self.decay_choice] = (a, b)

    def expected(self) -> float:
        return self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.5

    def conf_int95(self) -> Tuple[float, float]:
        # Approximation rapide : ±1.96 * sqrt(p(1-p)/(n+3)) (n ≈ alpha+beta-2)
        n = max(0.0, self.alpha + self.beta - 2.0)
        p = self.expected()
        if n <= 0: return (max(0.0, p-0.5), min(1.0, p+0.5))
        import math
        se = math.sqrt(p * (1 - p) / (n + 3.0))
        return (clamp(p - 1.96 * se), clamp(p + 1.96 * se))

    def observe(self, success: bool):
        self._select_decay()
        # Décroissance douce pour oublier le très ancien
        self.alpha = 1.0 + (self.alpha - 1.0) * self.decay
        self.beta  = 1.0 + (self.beta  - 1.0) * self.decay
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self._update_decay_bandit(success)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "decay": self.decay,
            "decay_candidates": list(self.decay_candidates),
            "decay_posteriors": {str(k): {"alpha": v[0], "beta": v[1]} for k, v in self.decay_posteriors.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectPosterior":
        decay_posteriors = data.get("decay_posteriors") or {}
        return cls(
            alpha=float(data.get("alpha", 1.0)),
            beta=float(data.get("beta", 1.0)),
            decay=float(data.get("decay", 0.995)),
            decay_candidates=tuple(float(x) for x in data.get("decay_candidates", (0.95, 0.98, 0.995))),
            decay_posteriors={float(k): (float(v.get("alpha", 1.0)), float(v.get("beta", 1.0)))
                              for k, v in decay_posteriors.items()},
        )

# ---------- Règle sociale testable ----------
@dataclass
class InteractionRule:
    """
    Représente ⟨ Contexte → Tactique → Effets_attendus ⟩ + incertitude
    - context_predicates: liste de Predicate à matcher sur le contexte courant
    - tactic: TacticSpec
    - effects: dict nom_effet -> EffectPosterior
    - stats: usage_count, last_used_ts, confidence, ema_reward (aggrégat interne)
    - provenance: comment/depuis où (inbox:..., curated:..., mined:...)
    - tags: ex. ["mined","social","human-like"]
    """
    id: str
    version: str = "1.0"
    context_predicates: List[Predicate] = field(default_factory=list)
    tactic: TacticSpec = field(default_factory=lambda: TacticSpec("noop", {}))
    effects: Dict[str, EffectPosterior] = field(default_factory=dict)
    created_ts: TS = field(default_factory=_now)
    last_used_ts: TS = 0.0
    usage_count: int = 0
    confidence: float = 0.5    # synthèse (ex: moyenne des expected() d’effets clés)
    ema_reward: float = 0.5    # score agrégé multi-sources (Social Critic), 0..1
    cooldown: float = 0.0      # anti-abus probabiliste (utilisé par le sélecteur, pas ici)
    provenance: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=lambda: ["social"])
    learned_effect_weights: Dict[str, float] = field(default_factory=dict)
    learned_predicate_weights: Dict[str, float] = field(default_factory=dict)
    ema_adapter_state: Dict[str, Any] = field(default_factory=dict)
    effect_weight_learner: Optional[OnlineLinear] = field(default=None, init=False, repr=False, compare=False)
    predicate_weight_learner: Optional[OnlineLinear] = field(default=None, init=False, repr=False, compare=False)
    _ema_adapter: Optional[AdaptiveEMA] = field(default=None, init=False, repr=False, compare=False)
    _last_predicate_features: Dict[str, float] = field(default_factory=dict, init=False, repr=False, compare=False)

    # ----------- Usine de construction -----------
    @staticmethod
    def build(predicates: List[Predicate],
              tactic: TacticSpec,
              effects_names: Optional[List[str]] = None,
              provenance: Optional[Dict[str, Any]] = None) -> "InteractionRule":
        effects = {}
        for name in (
            effects_names
            or [
                "reduce_uncertainty",
                "continue_dialogue",
                "positive_valence",
                "acceptance_marker",
                "relationship_growth",
            ]
        ):
            effects[name] = EffectPosterior()
        rid = _hash(tactic.key() + "|" + "|".join(f"{p.key}:{p.op}:{p.value}" for p in predicates))
        return InteractionRule(
            id=rid,
            context_predicates=predicates,
            tactic=tactic,
            effects=effects,
            provenance=provenance or {}
        )

    # ----------- Helpers internes -----------
    def _ensure_learners(self):
        if self.effect_weight_learner is None:
            defaults = self._default_effect_weights()
            merged = dict(defaults)
            merged.update(self.learned_effect_weights)
            self.effect_weight_learner = OnlineLinear(init=merged, bounds=(0.05, 0.95))
            self.learned_effect_weights = self.effect_weight_learner.state()
        if self.predicate_weight_learner is None:
            defaults = {p.key: p.weight for p in self.context_predicates}
            defaults.update(self.learned_predicate_weights)
            self.predicate_weight_learner = OnlineLinear(init=defaults, bounds=(0.2, 2.0))
            self.learned_predicate_weights = self.predicate_weight_learner.state()
        if self._ema_adapter is None:
            self._ema_adapter = AdaptiveEMA.from_state(self.ema_adapter_state, baseline=self.ema_reward)

    def _default_effect_weights(self) -> Dict[str, float]:
        return {
            "reduce_uncertainty": 0.30,
            "continue_dialogue":  0.22,
            "positive_valence":   0.22,
            "acceptance_marker":  0.12,
            "relationship_growth": 0.14,
        }

    def _get_ema_adapter(self) -> AdaptiveEMA:
        self._ensure_learners()
        return self._ema_adapter

    # ----------- Matching contexte -----------
    def match_score(self, ctx: Dict[str, Any]) -> float:
        """
        Score de pertinence [0..1] basé sur les prédicats.
        On pondère par weight et normalise.
        """
        if not self.context_predicates:
            return 0.5  # neutre si aucune contrainte
        self._ensure_learners()
        defaults = {p.key: p.weight for p in self.context_predicates}
        weights_map = self.predicate_weight_learner.current_weights(defaults)
        self.learned_predicate_weights = self.predicate_weight_learner.state()
        total_w = 0.0
        got = 0.0
        features: Dict[str, float] = {}
        for p in self.context_predicates:
            ok, _ = p.test(ctx)
            weight = max(0.0, weights_map.get(p.key, p.weight))
            total_w += weight
            if ok:
                got += weight
            features[p.key] = 1.0 if ok else 0.0
        self._last_predicate_features = features
        if total_w <= 0.0:
            total_w = 1.0
        return clamp(got / total_w)

    # ----------- Utilité attendue -----------
    def expected_utility(self, weights: Optional[Dict[str, float]] = None,
                         exploration_bonus: float = 0.0) -> float:
        """
        Utilité attendue d’appliquer la tactique, combinant effets (postérieurs),
        confiance synthétique, et un petit bonus d’exploration si désiré.
        """
        base = self._default_effect_weights()
        if weights:
            base.update(weights)
        self._ensure_learners()
        learned = self.effect_weight_learner.current_weights(base)
        self.learned_effect_weights = dict(learned)
        combined = dict(base)
        combined.update(learned)

        u = 0.0
        for name, w in combined.items():
            post = self.effects.get(name)
            if post:
                u += w * post.expected()
            else:
                u += w * 0.5  # neutre si inconnu

        # synthèse de "confidence" : moyenne des expected()
        self.confidence = sum((self.effects.get(k, EffectPosterior()).expected() for k in combined.keys())) / len(combined)

        # exploration: bonus léger quand on n'a pas encore beaucoup essayé
        n = max(1, self.usage_count)
        bonus = exploration_bonus * (1.0 / math.sqrt(n))
        return clamp(u + bonus)

    # ----------- Cycle de vie -----------
    def register_use(self):
        self.usage_count += 1
        self.last_used_ts = _now()

    def observe_outcome(self, outcome: Dict[str, Any]):
        """
        Mise à jour des postérieurs d’effets à partir d’un 'outcome' symbolique.
        outcome: {
            "reduced_uncertainty": bool,
            "continued": bool,
            "valence": float (-1..+1),
            "accepted": bool,
            "relationship_growth": float (0..1),
            "reward": float (0..1),   # Social Critic agrégé
        }
        """
        self._ensure_learners()
        # reduce_uncertainty
        if "reduced_uncertainty" in outcome:
            self.effects.setdefault("reduce_uncertainty", EffectPosterior()).observe(bool(outcome["reduced_uncertainty"]))
        # continue_dialogue
        if "continued" in outcome:
            self.effects.setdefault("continue_dialogue", EffectPosterior()).observe(bool(outcome["continued"]))
        # positive_valence (seuil 0)
        if "valence" in outcome:
            self.effects.setdefault("positive_valence", EffectPosterior()).observe(float(outcome["valence"]) > 0.0)
        # acceptance_marker (ex: “ok, merci”, “d’accord”)
        if "accepted" in outcome:
            self.effects.setdefault("acceptance_marker", EffectPosterior()).observe(bool(outcome["accepted"]))

        if "relationship_growth" in outcome:
            try:
                rel_value = float(outcome["relationship_growth"])
            except (TypeError, ValueError):
                rel_value = 0.5
            threshold = 0.55
            self.effects.setdefault("relationship_growth", EffectPosterior()).observe(rel_value >= threshold)

        # reward agrégé (si fourni)
        if "reward" in outcome:
            r = clamp(float(outcome["reward"]), 0.0, 1.0)
            adapter = self._get_ema_adapter()
            self.ema_reward = round(adapter.update(r, self.ema_reward), 4)
            self.ema_adapter_state = adapter.state()

            effect_defaults = self._default_effect_weights()
            features = {}
            for name in set(list(effect_defaults.keys()) + list(self.effects.keys())):
                post = self.effects.get(name)
                features[name] = post.expected() if post else 0.5
            self.effect_weight_learner.update(features, r, defaults=effect_defaults)
            self.learned_effect_weights = self.effect_weight_learner.state()

            if self._last_predicate_features:
                predicate_defaults = {p.key: p.weight for p in self.context_predicates}
                self.predicate_weight_learner.update(self._last_predicate_features, r, defaults=predicate_defaults)
                self.learned_predicate_weights = self.predicate_weight_learner.state()

    # ----------- Sérialisation JSON pour Memory -----------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "interaction_rule",
            "id": self.id,
            "version": self.version,
            "context_predicates": [asdict(p) for p in self.context_predicates],
            "tactic": {"name": self.tactic.name, "params": self.tactic.params},
            "effects": {k: v.to_dict() for k, v in self.effects.items()},
            "created_ts": self.created_ts,
            "last_used_ts": self.last_used_ts,
            "usage_count": self.usage_count,
            "confidence": self.confidence,
            "ema_reward": self.ema_reward,
            "cooldown": self.cooldown,
            "provenance": self.provenance,
            "tags": list(self.tags),
            "learned_effect_weights": dict(self.learned_effect_weights),
            "learned_predicate_weights": dict(self.learned_predicate_weights),
            "ema_adapter_state": self._ema_adapter.state() if self._ema_adapter else dict(self.ema_adapter_state),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InteractionRule":
        preds = [Predicate(**p) for p in d.get("context_predicates", [])]
        effs = {k: EffectPosterior.from_dict(v) for k, v in (d.get("effects") or {}).items()}
        t = d.get("tactic") or {}
        return InteractionRule(
            id=d["id"],
            version=d.get("version", "1.0"),
            context_predicates=preds,
            tactic=TacticSpec(name=t.get("name","noop"), params=t.get("params") or {}),
            effects=effs,
            created_ts=float(d.get("created_ts", _now())),
            last_used_ts=float(d.get("last_used_ts", 0.0)),
            usage_count=int(d.get("usage_count", 0)),
            confidence=float(d.get("confidence", 0.5)),
            ema_reward=float(d.get("ema_reward", 0.5)),
            cooldown=float(d.get("cooldown", 0.0)),
            provenance=d.get("provenance") or {},
            tags=list(d.get("tags") or ["social"]),
            learned_effect_weights=dict(d.get("learned_effect_weights") or {}),
            learned_predicate_weights=dict(d.get("learned_predicate_weights") or {}),
            ema_adapter_state=d.get("ema_adapter_state") or {},
        )

# ---------- Contexte de décision : builder depuis l’architecture ----------
class ContextBuilder:
    """
    Construit un contexte symbolique exploitable par les ``Predicate.test``.

    Ici on prépare un dictionnaire compact dédié au moteur de règles
    sociales (actes de dialogue, polarité, risques, persona…).  Il ne faut
    pas le confondre avec :class:`AGI_Evolutive.conversation.context.ContextBuilder`
    qui, lui, assemble un résumé narratif pour l'interface de conversation.
    """

    @staticmethod
    def _sanitize_mapping(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(data, dict):
            return None
        sanitized: Dict[str, Any] = {}
        for key, value in data.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, (str, float, int, bool)) or value is None:
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                collected: List[Any] = []
                for item in list(value)[:6]:
                    if isinstance(item, (str, float, int, bool)):
                        collected.append(item)
                if collected:
                    sanitized[key] = collected
            elif isinstance(value, dict):
                nested = ContextBuilder._sanitize_mapping(value)
                if nested:
                    sanitized[key] = nested
        return sanitized or None

    @staticmethod
    def _recent_history(arch) -> List[Dict[str, str]]:
        state = getattr(arch, "conversation_state", None)
        turns = getattr(state, "recent_turns", None)
        history: List[Dict[str, str]] = []
        if isinstance(turns, list):
            for turn in turns[-6:]:
                if not isinstance(turn, dict):
                    continue
                speaker = turn.get("speaker")
                text = turn.get("text")
                if isinstance(speaker, str) and isinstance(text, str):
                    sp = speaker.strip()
                    tx = text.strip()
                    if sp and tx:
                        history.append({"speaker": sp, "text": tx})
        return history

    @staticmethod
    def _last_text(source: Any, candidates: Tuple[str, ...]) -> Optional[str]:
        if source is None:
            return None
        for name in candidates:
            value = getattr(source, name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _llm_payload(arch, baseline: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        state = getattr(arch, "conversation_state", None)
        last_user = ContextBuilder._last_text(
            state,
            ("last_user_message", "last_user_text", "last_user", "last_user_msg"),
        )
        if not last_user:
            candidate = getattr(arch, "last_user_msg", None)
            if isinstance(candidate, str) and candidate.strip():
                last_user = candidate.strip()
        last_agent = ContextBuilder._last_text(
            state,
            ("last_agent_reply", "last_response", "last_agent_message", "last_rendered_reply"),
        )
        history = ContextBuilder._recent_history(arch)
        payload: Dict[str, Any] = {
            "baseline": ContextBuilder._sanitize_mapping(baseline),
            "extra": ContextBuilder._sanitize_mapping(extra) if isinstance(extra, dict) else None,
            "recent_messages": {
                "user": last_user,
                "assistant": last_agent,
            },
            "history": history or None,
            "user_id": getattr(arch, "last_user_id", None) or getattr(arch, "user_id", None),
        }
        recent = {
            key: value for key, value in payload["recent_messages"].items() if isinstance(value, str) and value
        }
        if recent:
            payload["recent_messages"] = recent
        else:
            payload.pop("recent_messages")
        if payload.get("extra") is None:
            payload.pop("extra")
        if payload.get("history") is None:
            payload.pop("history")
        return {k: v for k, v in payload.items() if v}

    @staticmethod
    def _llm_enrich(arch, baseline: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        payload = ContextBuilder._llm_payload(arch, baseline, extra)
        response = try_call_llm_dict(
            "social_interaction_context",
            input_payload=payload,
            logger=LOGGER,
        )
        return dict(response) if isinstance(response, dict) else None

    @staticmethod
    def build(arch, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {
            # actes de dialogue + topics mis par tes analyseurs
            "dialogue_act": getattr(getattr(arch, "conversation_state", None), "last_dialogue_act", None),
            "topic_cluster": getattr(getattr(arch, "conversation_state", None), "topic_cluster", None),
            # polarité / valence (heuristique locale)
            "polarity": getattr(getattr(arch, "affect_state", None), "polarity", 0.0),
            "valence": getattr(getattr(arch, "affect_state", None), "valence", 0.0),
            # sous-entendus/implicature (si ton analyzer en met une piste)
            "implicature_hint": getattr(getattr(arch, "conversation_state", None), "implicature_hint", None),
            # style utilisateur estimé / persona de l’agent
            "user_style": getattr(getattr(arch, "conversation_state", None), "user_style", {}),
            "persona_alignment": ContextBuilder._persona_alignment(arch),
            # risques / garde-fous policy
            "risk_level": ContextBuilder._risk_level(arch),
            # recence d’usage (pour anti-abus en sélection)
            "recence_usage": getattr(getattr(arch, "conversation_state", None), "recent_rule_usage", 0.0),
            # goal parent actif le plus saillant
            "parent_goal": ContextBuilder._active_parent_goal(arch),
        }
        baseline_copy = dict(ctx)
        if extra:
            ctx.update(extra)
        llm_response = ContextBuilder._llm_enrich(arch, baseline_copy, extra)
        if llm_response:
            context_updates = llm_response.get("context")
            if isinstance(context_updates, dict):
                for key, value in context_updates.items():
                    if key == "topics" and isinstance(value, list):
                        topics = [str(v) for v in value if isinstance(v, str)]
                        if topics:
                            ctx["topics"] = topics
                        continue
                    if isinstance(value, (str, float, int, bool)):
                        ctx[key] = value
                    elif isinstance(value, dict):
                        ctx[key] = value
            meta: Dict[str, Any] = {}
            if "confidence" in llm_response:
                try:
                    meta["confidence"] = float(llm_response["confidence"])
                except (TypeError, ValueError):
                    pass
            notes = llm_response.get("notes")
            if isinstance(notes, str) and notes.strip():
                meta["notes"] = notes.strip()
            if meta:
                ctx.setdefault("_llm_context", meta)
        return ctx

    @staticmethod
    def _persona_alignment(arch) -> float:
        """
        Raccourci: retourne [0..1] sur "ce contexte correspond à mes valeurs/voix".
        Implémentation simple: si persona.values contient des concepts voisins du topic/act.
        (Tu peux enrichir via Ontology/Beliefs si dispo.)
        """
        try:
            persona = getattr(arch.self_model, "state", {}).get("persona", {})
            vals = set(v.lower() for v in persona.get("values", []))
            topics = set(getattr(getattr(arch, "conversation_state", None), "topic_cluster", []) or [])
            # intersection simple
            return clamp(len(vals & topics) / max(1, len(topics)))
        except Exception:
            return 0.5

    @staticmethod
    def _risk_level(arch) -> str:
        """
        Estimation simple: si Policy a récemment "needs_human/deny" dans ce fil → medium/high.
        """
        try:
            pol = getattr(arch, "policy", None)
            if pol and hasattr(pol, "recent_frictions"):
                fr = int(pol.recent_frictions(window_sec=600))
                if fr >= 2: return "high"
                if fr == 1: return "medium"
            return "low"
        except Exception:
            return "low"

    @staticmethod
    def _active_parent_goal(arch) -> Optional[str]:
        try:
            parents = getattr(arch.planner, "state", {}).get("parents", {})
            stack = getattr(arch.planner, "state", {}).get("active_stack", [])
            if not stack: return None
            g = stack[-1]
            return parents.get(g) or g
        except Exception:
            return None

# ---------- Aide: fabriquer rapidement une règle typique ----------
def make_rule_insinuation_banter(provenance: Optional[Dict[str, Any]] = None) -> InteractionRule:
    preds = [
        Predicate(key="dialogue_act", op="eq", value="insinuation", weight=1.2),
        Predicate(key="risk_level",  op="in", value=["low","medium"], weight=0.8),
        Predicate(key="persona_alignment", op="ge", value=0.3, weight=0.6),
    ]
    tactic = TacticSpec("banter_leger", {"soft": True, "max_len_delta": 40})
    return InteractionRule.build(preds, tactic, provenance=provenance or {"source":"curated"})
