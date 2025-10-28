# AGI_Evolutive/social/tactic_selector.py
# Sélection contextuelle d'une tactique sociale (bandit contextuel LinUCB plein)
# Combine: match des prédicats, utilité attendue (effets postérieurs), reward EMA,
# anti-répétition (récence), garde-fous Policy, et incertitude (terme UCB).
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import logging
import time, math, random

import numpy as np

from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)

from AGI_Evolutive.social.interaction_rule import (
    InteractionRule, ContextBuilder, Predicate, TacticSpec, clamp
)

LOGGER = logging.getLogger(__name__)


def _now() -> float: return time.time()

# ---------------- Config par défaut ----------------
_DEFAULT_CFG = {
    "thresholds": {
        "match_min": 0.35,        # score de match (prédicats) minimal pour considérer une règle
        "score_min": 0.30,        # score final minimal pour appliquer une tactique
        "recent_sec": 300,        # pénalité si la règle a été utilisée < 300s
    },
    "weights": {
        "match":     0.40,        # poids du match symbolique
        "utility":   0.30,        # poids de l'utilité attendue (effets postérieurs)
        "ema_reward":0.20,        # poids du reward EMA appris (Social Critic)
        "bandit":    0.20,        # poids du terme UCB (incertitude / exploration)
        "recency":  -0.15,        # pénalité si réutilisation récente
    },
    "utility_weights": {          # pondération des effets dans expected_utility()
        "reduce_uncertainty": 0.30,
        "continue_dialogue":  0.22,
        "positive_valence":   0.22,
        "acceptance_marker":  0.12,
        "relationship_growth": 0.14,
    },
    "bandit": {
        "alpha": 0.6,             # intensité de l'exploration UCB
        "dim": 16,                # dimension du vecteur de contexte φ(s)
        "type": "linucb_full",   # modèle par défaut (compatible diag)
        "l2": 1.0,                # régularisation de la matrice A
        "forget": 0.98            # facteur d'oubli exponentiel (<1.0 pour drift)
    },
    "epsilon": 0.08               # epsilon-greedy: chance de prendre la 2e meilleure
}

# ---------------- LinUCB plein avec oubli ----------------
class FullLinUCB:
    """LinUCB complet avec mise à jour Sherman–Morrison et facteur d'oubli."""

    TYPE = "linucb_full"

    def __init__(self, dim: int, alpha: float = 0.6, l2: float = 1.0, forget: float = 1.0):
        self.d = dim
        self.alpha = alpha
        self.l2 = l2
        self.forget = max(0.8, min(1.0, float(forget)))  # bornes de sécurité
        self.A = np.eye(dim, dtype=float) * float(l2)
        self.A_inv = np.linalg.inv(self.A)
        self.b = np.zeros(dim, dtype=float)
        self.n = 0

    def _theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def score(self, x: List[float]) -> float:
        vec = np.asarray(x, dtype=float)
        theta = self._theta()
        mean = float(theta @ vec)
        conf_sq = float(vec @ self.A_inv @ vec)
        conf_sq = max(conf_sq, 0.0)
        conf = math.sqrt(conf_sq)
        return float(mean + self.alpha * conf)

    def update(self, x: List[float], r01: float) -> None:
        vec = np.asarray(x, dtype=float)
        reward = float(r01)
        self.n += 1

        if self.forget < 1.0:
            self.A *= self.forget
            self.b *= self.forget
            self.A_inv /= self.forget

        Av = self.A_inv @ vec
        denom = 1.0 + float(vec @ Av)
        if denom <= 1e-8:
            self.A += np.outer(vec, vec)
            self.A_inv = np.linalg.inv(self.A)
        else:
            outer = np.outer(Av, Av) / denom
            self.A_inv -= outer
            self.A += np.outer(vec, vec)

        self.b += reward * vec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.TYPE,
            "dim": self.d,
            "alpha": self.alpha,
            "l2": self.l2,
            "forget": self.forget,
            "A": self.A.tolist(),
            "A_inv": self.A_inv.tolist(),
            "b": self.b.tolist(),
            "n": self.n,
        }

    @classmethod
    def load(cls, payload: Dict[str, Any], default_cfg: Dict[str, Any]) -> "FullLinUCB":
        dim = int(payload.get("dim", default_cfg["bandit"]["dim"]))
        alpha = float(payload.get("alpha", default_cfg["bandit"]["alpha"]))
        l2 = float(payload.get("l2", default_cfg["bandit"].get("l2", 1.0)))
        forget = float(payload.get("forget", default_cfg["bandit"].get("forget", 1.0)))
        inst = cls(dim=dim, alpha=alpha, l2=l2, forget=forget)
        if "A" in payload and isinstance(payload["A"], list):
            mat = np.asarray(payload["A"], dtype=float)
            if mat.shape == (dim, dim):
                inst.A = mat
        if "A_inv" in payload and isinstance(payload["A_inv"], list):
            mat_inv = np.asarray(payload["A_inv"], dtype=float)
            if mat_inv.shape == (dim, dim):
                inst.A_inv = mat_inv
        else:
            inst.A_inv = np.linalg.inv(inst.A)
        if "b" in payload and isinstance(payload["b"], list):
            vec = np.asarray(payload["b"], dtype=float)
            if vec.shape == (dim,):
                inst.b = vec
        inst.n = int(payload.get("n", 0))
        return inst

    @classmethod
    def from_diag(cls, diag: "DiagLinUCB", default_cfg: Dict[str, Any]) -> "FullLinUCB":
        inst = cls(
            dim=diag.d,
            alpha=diag.alpha,
            l2=default_cfg["bandit"].get("l2", 1.0),
            forget=default_cfg["bandit"].get("forget", 1.0),
        )
        inst.A = np.diag(np.asarray(diag.D, dtype=float))
        inst.A_inv = np.linalg.inv(inst.A)
        inst.b = np.asarray(diag.b, dtype=float)
        inst.n = diag.n
        return inst

# ---------------- Modèle logistique online pour les poids ----------------
class OnlineLogisticWeights:
    """Régression logistique online pour pondérer match/utilité/etc."""

    def __init__(self,
                 feature_keys: List[str],
                 lr: float = 0.05,
                 l2: float = 0.001,
                 warmup: int = 25,
                 init_weights: Optional[Dict[str, float]] = None,
                 state: Optional[Dict[str, Any]] = None) -> None:
        self.feature_keys = list(feature_keys)
        self.lr = lr
        self.l2 = l2
        self.warmup = warmup
        self._weights = np.zeros(len(self.feature_keys) + 1, dtype=float)
        self._updates = 0
        if init_weights:
            for idx, key in enumerate(self.feature_keys, start=1):
                if key in init_weights:
                    self._weights[idx] = float(init_weights[key])
        if state:
            self._load_state(state)

    def _sigmoid(self, z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        vec = [1.0]
        for key in self.feature_keys:
            vec.append(float(features.get(key, 0.0)))
        return np.asarray(vec, dtype=float)

    def predict(self, features: Dict[str, float], fallback: Optional[float] = None) -> float:
        x = self._vectorize(features)
        z = float(self._weights @ x)
        p = self._sigmoid(z)
        if self._updates < self.warmup and fallback is not None:
            return float(fallback)
        return p

    def update(self, features: Dict[str, float], label: float) -> None:
        x = self._vectorize(features)
        y = max(0.0, min(1.0, float(label)))
        z = float(self._weights @ x)
        p = self._sigmoid(z)
        error = p - y
        grad = error * x
        reg = np.concatenate(([0.0], self._weights[1:]))
        self._weights -= self.lr * (grad + self.l2 * reg)
        self._updates += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_keys": list(self.feature_keys),
            "lr": self.lr,
            "l2": self.l2,
            "warmup": self.warmup,
            "weights": self._weights.tolist(),
            "updates": self._updates,
        }

    def _load_state(self, state: Dict[str, Any]) -> None:
        if isinstance(state, dict):
            weights = state.get("weights")
            if isinstance(weights, list) and len(weights) == len(self.feature_keys) + 1:
                self._weights = np.asarray(weights, dtype=float)
            self._updates = int(state.get("updates", self._updates))

    @classmethod
    def load(cls, feature_keys: List[str], state: Optional[Dict[str, Any]], init_weights: Optional[Dict[str, float]]) -> "OnlineLogisticWeights":
        lr = 0.05
        l2 = 0.001
        warmup = 25
        if isinstance(state, dict):
            lr = float(state.get("lr", lr))
            l2 = float(state.get("l2", l2))
            warmup = int(state.get("warmup", warmup))
        return cls(feature_keys=feature_keys, lr=lr, l2=l2, warmup=warmup, init_weights=init_weights, state=state)


# ---------------- Petit bandit diagonal (sans numpy) ----------------
class DiagLinUCB:
    """
    LinUCB diagonalisé (pas d'inversion de matrice générale).
    - D : diag des A (variances par feature) -> liste de taille d
    - b : vecteur cumul des rewards pondérés par x -> liste de taille d
    Score UCB: x·θ + α * sqrt( sum_i (x_i^2 / D_i) )
    Update: D_i += x_i^2 ; b_i += r * x_i ; θ_i = b_i / D_i
    """
    def __init__(self, dim: int, alpha: float = 0.6):
        self.d = dim
        self.alpha = alpha
        self.D = [1.0] * dim   # variances init (identité)
        self.b = [0.0] * dim   # cumul
        self.n = 0             # nombre d'updates

    def score(self, x: List[float]) -> float:
        theta = [ (self.b[i] / self.D[i]) for i in range(self.d) ]
        # x·θ
        mean = sum( (x[i] * theta[i]) for i in range(self.d) )
        # incertitude
        conf = math.sqrt( sum( (x[i]*x[i]) / self.D[i] for i in range(self.d) ) )
        return float(mean + self.alpha * conf)

    def update(self, x: List[float], r01: float):
        self.n += 1
        for i in range(self.d):
            xi = x[i]
            self.D[i] += xi * xi
            self.b[i] += r01 * xi

    @staticmethod
    def load(dic: Dict[str, Any]) -> "DiagLinUCB":
        d = DiagLinUCB(dim=int(dic.get("dim", len(dic.get("D",[])) or _DEFAULT_CFG["bandit"]["dim"])),
                       alpha=float(dic.get("alpha", _DEFAULT_CFG["bandit"]["alpha"])))
        D = dic.get("D"); b = dic.get("b")
        if isinstance(D, list) and len(D) == d.d: d.D = [float(v) for v in D]
        if isinstance(b, list) and len(b) == d.d: d.b = [float(v) for v in b]
        d.n = int(dic.get("n", 0))
        return d

    def to_dict(self) -> Dict[str, Any]:
        return {"dim": self.d, "alpha": self.alpha, "D": self.D, "b": self.b, "n": self.n}

# ---------------- Sélecteur ----------------
class TacticSelector:
    """
    Choisit une InteractionRule pertinente pour le contexte courant.
    Combine: match (prédicats), utilité attendue, reward EMA, bandit (incertitude), recency.
    Vérifie la Policy si dispo. Garde-fous: risque/contexte/persona.
    """
    SCORE_FEATURES = ["match", "utility", "ema_reward", "ucb", "recency"]

    def __init__(self, arch, cfg: Optional[Dict[str, Any]] = None):
        self.arch = arch
        self.cfg = cfg or getattr(arch, "tactic_selector_cfg", None) or _DEFAULT_CFG
        self._score_model = self._init_score_model()

    def _init_score_model(self) -> OnlineLogisticWeights:
        state_container = getattr(self.arch, "tactic_selector_state", None)
        if state_container is None or not isinstance(state_container, dict):
            state_container = {}
            setattr(self.arch, "tactic_selector_state", state_container)
        init_weights = self.cfg.get("weights", {})
        stored = state_container.get("score_model")
        model = OnlineLogisticWeights.load(self.SCORE_FEATURES, stored, init_weights)
        state_container["score_model"] = model.to_dict()
        return model

    def _persist_score_model(self) -> None:
        state_container = getattr(self.arch, "tactic_selector_state", None)
        if isinstance(state_container, dict):
            state_container["score_model"] = self._score_model.to_dict()

    def _create_bandit(self) -> FullLinUCB:
        cfg_src = self.cfg if "bandit" in self.cfg else _DEFAULT_CFG
        bcfg = cfg_src.get("bandit", {})
        return FullLinUCB(
            dim=int(bcfg.get("dim", _DEFAULT_CFG["bandit"]["dim"])),
            alpha=float(bcfg.get("alpha", _DEFAULT_CFG["bandit"]["alpha"])),
            l2=float(bcfg.get("l2", _DEFAULT_CFG["bandit"].get("l2", 1.0))),
            forget=float(bcfg.get("forget", _DEFAULT_CFG["bandit"].get("forget", 1.0))),
        )

    def _load_bandit(self, state: Optional[Dict[str, Any]]) -> FullLinUCB:
        cfg_src = self.cfg if "bandit" in self.cfg else _DEFAULT_CFG
        if isinstance(state, dict):
            btype = state.get("type")
            if btype == FullLinUCB.TYPE:
                return FullLinUCB.load(state, cfg_src)
            if "A" in state or "A_inv" in state:
                return FullLinUCB.load(state, cfg_src)
            if "D" in state and "b" in state:
                diag = DiagLinUCB.load(state)
                return FullLinUCB.from_diag(diag, cfg_src)
        return self._create_bandit()

    # ---------- Construction du contexte φ(s) ----------
    def _risk_num(self, level: str) -> float:
        if level in ("high","élevé"): return 1.0
        if level in ("medium","moyen"): return 0.5
        return 0.0

    def _one_hot(self, val: Optional[str], vocab: List[str]) -> List[float]:
        v = [0.0]*len(vocab)
        if val is None: return v
        try:
            i = vocab.index(val)
            v[i] = 1.0
        except ValueError:
            pass
        return v

    def _phi(self, ctx: Dict[str, Any], rule: Dict[str, Any]) -> List[float]:
        """
        Vecteur de contexte fixe (dim = cfg["bandit"]["dim"]).
        Compact, interprétable, sans dépendances.
        """
        dim = int(self.cfg["bandit"]["dim"])
        # Slots (≤ dim). Si moins, on padde par des zéros.
        # 0: intercept
        feats: List[float] = [1.0]

        # 1: match score (prédicats)
        # le match dépend de la règle -> recalcul ici
        try:
            r = InteractionRule.from_dict(rule)
            ms = r.match_score(ctx)
        except Exception:
            ms = 0.5
        feats.append(ms)

        # 2: risk level num
        feats.append(self._risk_num(str(ctx.get("risk_level","low")).lower()))

        # 3: persona alignment [0..1]
        feats.append(float(ctx.get("persona_alignment", 0.5)))

        # 4: valence ([-1..1] -> [0..1])
        val = float(ctx.get("valence", 0.0))
        feats.append((val + 1.0)/2.0)

        # 5: recence_usage (0..1; plus c'est haut, plus c'est récent/abus)
        feats.append(float(ctx.get("recence_usage", 0.0)))

        # 6..k: one-hot dialogue_act
        acts_vocab = ["question","compliment","insinuation","disagreement","confusion","ack","thanks","clarify","explain","statement"]
        feats += self._one_hot(ctx.get("dialogue_act"), acts_vocab)

        # 6+k..: one-hot implicature_hint (quelques hints)
        hints_vocab = ["sous-entendu","ironie","taquinerie","double-entendre"]
        feats += self._one_hot(ctx.get("implicature_hint"), hints_vocab)

        # tronquer/padder à dim
        if len(feats) < dim:
            feats += [0.0] * (dim - len(feats))
        elif len(feats) > dim:
            feats = feats[:dim]
        return feats

    # ---------- Politique / garde-fous ----------
    def _allowed_by_policy(self, rule: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[bool, str]:
        # 1) Risk gates simples si pas d'API policy
        risk = str(ctx.get("risk_level","low")).lower()
        tactic = (rule.get("tactic") or {}).get("name","")
        if risk == "high" and tactic in {"banter_leger"}:
            return False, "risk_high_no_banter"

        # 2) API policy si dispo
        pol = getattr(self.arch, "policy", None)
        if pol and hasattr(pol, "validate_tactic"):
            try:
                res = pol.validate_tactic(rule, ctx)
                if isinstance(res, dict):
                    dec = res.get("decision","allow")
                    return (dec == "allow"), f"policy:{dec}"
            except Exception:
                pass
        # 3) défaut
        return True, "ok"

    # ---------- Scoring combiné ----------
    def _compute_score_components(self,
                                  rule: Dict[str, Any],
                                  ctx: Dict[str, Any],
                                  now: Optional[float] = None) -> Tuple[Dict[str, float], Dict[str, Any], FullLinUCB, List[float]]:
        th = self.cfg["thresholds"]
        now = now or _now()

        try:
            rr = InteractionRule.from_dict(rule)
            match = rr.match_score(ctx)
        except Exception:
            match = 0.0

        if match < float(th["match_min"]):
            return {}, {"skip": "low_match", "match": round(match, 3)}, self._create_bandit(), []

        try:
            util = rr.expected_utility(weights=self.cfg.get("utility_weights"), exploration_bonus=0.0)
        except Exception:
            util = 0.5

        ema = float(rule.get("ema_reward", 0.5))

        bandit_state = rule.get("bandit") or {}
        bandit_model = self._load_bandit(bandit_state)
        x = self._phi(ctx, rule)
        ucb = bandit_model.score(x)

        last_used = float(rule.get("last_used_ts", 0.0))
        recency_factor = max(0.0, 1.0 - ((now - last_used) / 60.0))
        diversity_penalty = 0.08 * recency_factor
        ucb -= diversity_penalty

        cost = float(rule.get("cost", 0.0))
        ucb -= 0.10 * cost

        recent_pen = 0.0
        if last_used > 0 and (now - last_used) < float(th["recent_sec"]):
            recent_pen = 1.0

        components = {
            "match": float(match),
            "utility": float(util),
            "ema_reward": float(ema),
            "ucb": float(ucb),
            "recency": float(recent_pen),
        }

        meta = {
            "match": round(match, 3),
            "utility": round(util, 3),
            "ema_reward": round(ema, 3),
            "ucb": round(ucb, 3),
            "recent_pen": recent_pen,
            "tactic": (rule.get("tactic") or {}).get("name", ""),
        }

        return components, meta, bandit_model, x

    def _score_rule(self, rule: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        components, meta, _, _ = self._compute_score_components(rule, ctx)
        if not components:
            return 0.0, meta

        W = self.cfg["weights"]
        linear_score = (
            float(W["match"]) * components["match"] +
            float(W["utility"]) * components["utility"] +
            float(W["ema_reward"]) * components["ema_reward"] +
            float(W["bandit"]) * components["ucb"] +
            float(W["recency"]) * components["recency"]
        )
        linear_score = clamp(linear_score, 0.0, 1.0)

        score = self._score_model.predict(components, fallback=linear_score)
        meta["score_linear"] = round(linear_score, 3)
        return float(score), meta

    # ---------- Pick principal ----------
    def pick(self, ctx: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Retourne (rule_dict, why) ou (None, None) si rien de suffisamment pertinent.
        Ne modifie pas la mémoire ici (le renderer tracera et marquera l'usage).
        """
        arch = self.arch
        ctx = ctx or ContextBuilder.build(arch)

        # 1) récupérer les règles depuis la mémoire
        try:
            rules = arch.memory.get_recent_memories(kind="interaction_rule", limit=400) or []
        except Exception:
            rules = []
        if not rules:
            return (None, None)

        # 2) filtrage simple Policy/garde-fous + scoring
        candidates: List[Dict[str, Any]] = []
        for r in rules:
            allow, reason = self._allowed_by_policy(r, ctx)
            if not allow:
                continue
            components, meta, bandit_model, x = self._compute_score_components(r, ctx)
            if not components:
                continue
            W = self.cfg["weights"]
            linear_score = clamp(
                float(W["match"]) * components["match"]
                + float(W["utility"]) * components["utility"]
                + float(W["ema_reward"]) * components["ema_reward"]
                + float(W["bandit"]) * components["ucb"]
                + float(W["recency"]) * components["recency"],
                0.0,
                1.0,
            )
            score = self._score_model.predict(components, fallback=linear_score)
            meta["policy"] = reason
            meta["score_linear"] = round(linear_score, 3)
            candidate = {
                "rule": r,
                "score": float(score),
                "components": components,
                "meta": meta,
                "bandit_model": bandit_model,
                "context_vector": x,
            }
            candidates.append(candidate)

        if not candidates:
            return (None, None)

        llm_priorities = self._llm_rank_tactics(ctx, candidates)

        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for entry in candidates:
            rule = entry["rule"]
            meta = dict(entry["meta"])
            base_score = float(entry["score"])
            if llm_priorities:
                llm_data = llm_priorities[0].get(rule.get("id"))
                if not llm_data:
                    tactic_name = (rule.get("tactic") or {}).get("name")
                    if tactic_name:
                        llm_data = llm_priorities[1].get(tactic_name)
                if llm_data:
                    priority = max(0.0, min(1.0, llm_data.get("priority", 0.0)))
                    base_score = 0.6 * priority + 0.4 * base_score
                    meta["llm"] = llm_data
            scored.append((base_score, rule, meta))

        if not scored:
            return (None, None)

        scored.sort(key=lambda t: t[0], reverse=True)
        best_s, best_r, best_why = scored[0]

        eps = float(self.cfg.get("epsilon", 0.08))
        if len(scored) > 1 and random.random() < eps:
            s2, r2, w2 = scored[1]
            if s2 > float(self.cfg["thresholds"]["score_min"]):
                best_s, best_r, best_why = s2, r2, {**w2, "explore": "epsilon"}

        # epsilon: si la 2e est proche, on peut explorer
        eps = float(self.cfg.get("epsilon", 0.08))
        if len(scored) > 1 and random.random() < eps:
            s2, r2, w2 = scored[1]
            if s2 > float(self.cfg["thresholds"]["score_min"]):
                best_s, best_r, best_why = s2, r2, {**w2, "explore":"epsilon"}

        if best_s < float(self.cfg["thresholds"]["score_min"]):
            return (None, None)

        # 4) self-prompting court (explication interne) — pour traçabilité
        best_why["score"] = round(best_s,3)
        best_why["reason"] = (
            f"match={best_why['match']}, util={best_why['utility']}, ema={best_why['ema_reward']}, "
            f"ucb={best_why['ucb']}, policy={best_why.get('policy','ok')}"
        )

        return (best_r, best_why)

    def _llm_rank_tactics(
        self,
        ctx: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]]:
        if not candidates or not is_llm_enabled():
            return None

        payload = {
            "context": {k: v for k, v in ctx.items() if isinstance(k, str)},
            "candidates": [],
        }
        for entry in candidates[:8]:
            rule = entry["rule"]
            tactic = (rule.get("tactic") or {}).get("name")
            payload["candidates"].append(
                {
                    "rule_id": rule.get("id"),
                    "tactic": tactic,
                    "components": entry["components"],
                    "meta": entry["meta"],
                }
            )

        if not payload["candidates"]:
            return None

        try:
            response = get_llm_manager().call_dict(
                "social_tactic_selector",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM social tactic selector unavailable", exc_info=True)
            return None

        if not isinstance(response, dict):
            return None

        by_rule: Dict[str, Dict[str, Any]] = {}
        by_name: Dict[str, Dict[str, Any]] = {}
        for entry in response.get("tactics", []):
            if not isinstance(entry, dict):
                continue
            info = {
                "utility": float(entry.get("utility", 0.0) or 0.0),
                "risk": float(entry.get("risk", 0.0) or 0.0),
                "explanation": entry.get("explanation"),
            }
            priority = info["utility"] - info["risk"]
            info["priority"] = max(0.0, min(1.0, (priority + 1.0) / 2.0))
            rid = entry.get("rule_id")
            if isinstance(rid, str):
                by_rule[rid] = info
            name = entry.get("name") or entry.get("tactic")
            if isinstance(name, str):
                by_name[name] = info

        notes = response.get("notes")
        if notes:
            LOGGER.debug("LLM tactic selector notes: %s", notes)

        return by_rule, by_name

    # ---------- Hook d'update bandit (appelé par Social Critic APRÈS feedback) ----------
    def bandit_update(self, rule_id: str, ctx: Dict[str, Any], reward01: float) -> Optional[Dict[str, Any]]:
        """
        Met à jour l'état du bandit de la règle (LinUCB plein) à partir du contexte utilisé et du reward 0..1.
        Persiste en mémoire (update/add). Retourne la règle persistée (dict) ou None.
        """
        try:
            rules = self.arch.memory.get_recent_memories(kind="interaction_rule", limit=500) or []
        except Exception:
            rules = []
        rule = None
        for r in rules:
            if r.get("id") == rule_id:
                rule = r; break
        if not rule:
            return None

        components, _, bandit_model, x = self._compute_score_components(rule, ctx)
        if not x:
            x = self._phi(ctx, rule)
        bandit_model.update(x, float(reward01))
        rule["bandit"] = bandit_model.to_dict()

        # légère adaptation de cooldown en fonction du reward (punitif si très bas)
        cd = float(rule.get("cooldown", 0.0))
        if reward01 < 0.25:
            cd = min(1.0, cd + 0.15)
        elif reward01 > 0.75:
            cd = max(0.0, cd * 0.9)
        rule["cooldown"] = cd

        if components:
            self._score_model.update(components, float(reward01))
            self._persist_score_model()

        if hasattr(self.arch.memory, "update_memory"):
            self.arch.memory.update_memory(rule)
        else:
            self.arch.memory.add_memory(rule)
        return rule


# ---------------------------------------------------------------------------
# Lightweight style bandit helper (for macro selection, optional usage)


class _ThompsonBandit:
    def __init__(self) -> None:
        self.alpha = defaultdict(lambda: 1.0)
        self.beta = defaultdict(lambda: 1.0)

    def sample(self, arm: str) -> float:
        a = self.alpha[arm]
        b = self.beta[arm]
        x = random.gammavariate(a, 1.0)
        y = random.gammavariate(b, 1.0)
        return x / (x + y) if (x + y) else 0.5

    def update(self, arm: str, reward: float) -> None:
        if reward > 0:
            self.alpha[arm] += reward
        elif reward < 0:
            self.beta[arm] += -reward


class StyleMacroBandit:
    """Petit bandit Thompson-sampling pour choisir un macro style."""

    DEFAULT_ARMS = ["sobre", "taquin", "coach", "deadpan"]

    def __init__(self, arms: Optional[List[str]] = None) -> None:
        self.arms = arms or list(self.DEFAULT_ARMS)
        self._bandit = _ThompsonBandit()
        self._last_arm: Optional[str] = None

    def pick(self, context: Optional[Dict[str, Any]] = None) -> str:
        scored = [(self._bandit.sample(arm), arm) for arm in self.arms]
        scored.sort(key=lambda item: item[0], reverse=True)
        self._last_arm = scored[0][1]
        return self._last_arm

    def feedback(self, reward: float, arm: Optional[str] = None) -> None:
        choice = arm or self._last_arm
        if not choice:
            return
        self._bandit.update(choice, float(reward))
