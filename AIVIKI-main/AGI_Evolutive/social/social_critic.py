# Social Critic : calcule des résultats observables + un reward multi-source
# et met à jour les InteractionRule (postérieurs + ema_reward).
from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import time, json, random, re, logging
from collections import deque

from AGI_Evolutive.social.adaptive_lexicon import AdaptiveLexicon
from AGI_Evolutive.social.interaction_rule import (
    InteractionRule, clamp
)
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


def _now() -> float: return time.time()

LOGGER = logging.getLogger(__name__)

# ----------------- lexiques FR simples (tu peux enrichir) -----------------
POS_MARKERS = [
    "j'adore", "génial", "top", "parfait", "bravo", "merci", "super", "nickel",
    "impeccable", "parfaitement", "très bien", "bien vu", "ça me va", "ok c'est bon"
]
NEG_MARKERS = [
    "trop", "arrête", "stop", "pas ça", "relou", "ça me saoule", "mauvais", "bof",
    "je n'aime pas", "non pas comme ça", "pas terrible", "déçu", "insupportable"
]
ACCEPTANCE = ["ok", "d'accord", "merci", "compris", "noté", "ça marche", "bien reçu"]

REL_PRONOUNS = [
    "je", "j'", "moi", "mon", "ma", "mes", "nous", "notre", "nos",
    "toi", "t'", "te", "ton", "ta", "tes",
]
REL_KEYWORDS = [
    "ensemble", "relation", "lien", "connex", "ami", "amie", "partage", "partager",
    "dis-moi", "parle-moi", "histoire", "journee", "journée", "ressenti", "confiance",
    "prends soin", "souvenir", "souviens",
]
REL_QUESTIONS = [
    "comment", "qu'est-ce", "que fais", "tu as fait", "raconte", "parle de",
]
REL_DISCLOSURE = [
    "aujourd'hui", "hier", "demain", "perso", "pers", "famille", "travail", "projet",
]


def _contains_any(s: str, words: List[str]) -> bool:
    s = s.lower()
    return any(w in s for w in words)


def _sentiment_heuristic(s: str) -> float:
    """Renvoie une valence approx ∈ [-1,1]. Zéro si neutre."""
    s = (s or "").lower()
    pos = sum(1 for w in POS_MARKERS if w in s)
    neg = sum(1 for w in NEG_MARKERS if w in s)
    if pos == 0 and neg == 0:
        return 0.0
    val = (pos - neg) / max(1.0, (pos + neg))
    return max(-1.0, min(1.0, val))


def _acceptance_marker(s: str) -> bool:
    return _contains_any(s, ACCEPTANCE)


def _relationship_signal(
    message: str,
    decision_trace: Optional[Dict[str, Any]] = None,
    pre_ctx: Optional[Dict[str, Any]] = None,
    post_ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    text = (message or "").strip()
    if not text:
        return 0.2, {"topics": [], "sentiment": 0.0}
    low = text.lower()
    tokens = re.findall(r"\w+", low)
    length_bonus = min(len(tokens) / 45.0, 1.0)

    pronoun_hits = sum(1 for pron in REL_PRONOUNS if pron in low)
    pronoun_score = min(1.0, pronoun_hits / 4.0)

    relational_hits = sum(1 for w in REL_KEYWORDS if w in low)
    relational_score = min(1.0, relational_hits / 3.0)

    disclosure_hits = sum(1 for w in REL_DISCLOSURE if w in low)
    disclosure_score = min(1.0, disclosure_hits / 4.0)

    question_score = 0.0
    if "?" in text:
        question_score = 0.55
    if any(w in low for w in REL_QUESTIONS):
        question_score = max(question_score, 0.85)

    sentiment = _sentiment_heuristic(message)
    sentiment_gate = 1.0 if sentiment >= 0.0 else 0.6

    topics: List[str] = []
    topic_signal = 0.0
    ctx_topics = []
    if isinstance(post_ctx, dict):
        ctx_topics = post_ctx.get("conversation_topics") or []
    elif isinstance(pre_ctx, dict):
        ctx_topics = pre_ctx.get("conversation_topics") or []
    for item in ctx_topics:
        if isinstance(item, str):
            topic = item.strip()
            if topic and topic not in topics:
                topics.append(topic)
    if topics:
        topic_signal = min(1.0, len(topics) / 3.0)

    shared_recall = 0.0
    try:
        shared_recall = 1.0 if (post_ctx or {}).get("shared_reference") else 0.0
    except Exception:
        shared_recall = 0.0

    raw_score = (
        0.22 * pronoun_score
        + 0.22 * relational_score
        + 0.14 * disclosure_score
        + 0.18 * question_score
        + 0.12 * length_bonus
        + 0.12 * topic_signal
    )
    score = clamp((raw_score * sentiment_gate) + 0.08 * shared_recall, 0.0, 1.0)

    meta = {
        "topics": topics,
        "sentiment": sentiment,
        "length": len(tokens),
        "question_score": question_score,
    }
    return score, meta


# ----------------- Critic -----------------
class ContextualWeightLearner:
    """Apprend des poids par contexte via descente de gradient en ligne.

    On garde un dictionnaire de contextes -> poids (+ vitesse d'oubli) afin de
    ne pas casser les heuristiques existantes. Les poids sont initialisés avec
    les valeurs configurées et ré-encadrés dans [0, 1] sauf indication
    contraire. Une légère injection de bruit permet de ré-explorer en cas de
    dérive.
    """

    def __init__(self,
                 base_weights: Dict[str, float],
                 component_names: List[str],
                 lr: float = 0.15,
                 l2: float = 1e-3,
                 forget: float = 0.02,
                 noise: float = 0.03):
        self._base = base_weights
        self._components = component_names
        self._lr = lr
        self._l2 = l2
        self._forget = forget
        self._noise = noise
        self._ctx_weights: Dict[Tuple[Any, ...], Dict[str, float]] = {}

    # bornes douces pour garder des poids raisonnables
    def _clamp(self, name: str, value: float) -> float:
        if name == "identity_consist":
            return clamp(value, 0.0, 0.4)
        if name == "continue_dialogue":
            return clamp(value, 0.0, 0.35)
        if name == "valence":
            return clamp(value, 0.0, 0.35)
        if name == "uncertainty_delta":
            return clamp(value, 0.0, 0.45)
        if name == "explicit_feedback":
            return clamp(value, 0.1, 0.65)
        if name == "acceptance":
            return clamp(value, 0.0, 0.35)
        if name == "relationship_growth":
            return clamp(value, 0.0, 0.35)
        return clamp(value, -0.35, 0.35)

    def _init_ctx(self, ctx_key: Tuple[Any, ...]) -> Dict[str, float]:
        weights = {
            name: float(self._base.get(name, 0.0))
            for name in self._components
        }
        self._ctx_weights[ctx_key] = weights
        return weights

    def get(self, ctx_key: Optional[Tuple[Any, ...]]) -> Dict[str, float]:
        if not ctx_key:
            # renvoie une copie pour éviter les mutations externes
            return {
                name: float(self._base.get(name, 0.0))
                for name in self._components
            }
        if ctx_key not in self._ctx_weights:
            return dict(self._init_ctx(ctx_key))
        return dict(self._ctx_weights[ctx_key])

    def update(self,
               ctx_key: Optional[Tuple[Any, ...]],
               feature_vec: List[float],
               reward: float,
               confidence: float = 1.0) -> None:
        if not feature_vec or len(feature_vec) != len(self._components):
            return
        if confidence <= 0.0:
            return
        ctx = ctx_key or tuple()
        weights = self._ctx_weights.get(ctx)
        if weights is None:
            weights = self._init_ctx(ctx)
        # produit scalaire estimé
        est = sum(weights[name] * val for name, val in zip(self._components, feature_vec))
        err = (reward - est)
        scaled_lr = self._lr * clamp(confidence, 0.05, 1.0)
        for name, val in zip(self._components, feature_vec):
            grad = (-2.0 * err * val) + (self._l2 * weights[name])
            update = -scaled_lr * grad
            weights[name] = self._clamp(name, weights[name] + update)
        # oubli doux vers la base
        if self._forget > 0.0:
            for name in self._components:
                base = float(self._base.get(name, weights[name]))
                weights[name] = (1.0 - self._forget) * weights[name] + self._forget * base

    def inject_noise(self,
                     ctx_key: Optional[Tuple[Any, ...]],
                     component: Optional[str] = None,
                     strength: Optional[float] = None) -> None:
        ctx = ctx_key or tuple()
        weights = self._ctx_weights.get(ctx)
        if weights is None:
            weights = self._init_ctx(ctx)
        comps = [component] if component else self._components
        amp = strength if strength is not None else self._noise
        for name in comps:
            if name not in weights:
                continue
            jitter = random.uniform(-amp, amp)
            weights[name] = self._clamp(name, weights[name] + jitter)


class SocialCritic:
    """
    Calcule un outcome post-réponse + reward multi-source avec incertitude.
    Sources (exemples, toutes optionnelles) :
      - Explicite (markers FR positifs/négatifs)
      - Implicite : baisse des pending_questions, poursuite du fil, valence heuristique
      - Policy friction (pénalité si friction récente)
      - Consistance identitaire (bonus si persona_alignment acceptable)
    Poids configurables via data/social_critic_config.json (optionnel).
    """

    def __init__(self, arch):
        self.arch = arch
        self.cfg = self._load_cfg()
        self.lex = getattr(self.arch, "lexicon", None)
        needs_adaptive = self.lex is None or not all(
            callable(getattr(self.lex, attr, None)) for attr in ("match", "observe_message")
        )
        if needs_adaptive:
            # seed avec tes POS/NEG statiques pour ne pas “perdre” ton actuel
            self.arch.lexicon_seeds = {
                "pos": [
                    "j'adore",
                    "génial",
                    "top",
                    "parfait",
                    "bravo",
                    "merci",
                    "super",
                    "nickel",
                    "impeccable",
                    "parfaitement",
                    "très bien",
                    "bien vu",
                    "ça me va",
                    "ok c'est bon",
                ],
                "neg": [
                    "trop",
                    "arrête",
                    "stop",
                    "pas ça",
                    "relou",
                    "ça me saoule",
                    "mauvais",
                    "bof",
                    "je n'aime pas",
                    "non pas comme ça",
                    "pas terrible",
                    "déçu",
                    "insupportable",
                ],
            }
            self.lex = AdaptiveLexicon(self.arch)
            setattr(self.arch, "lexicon", self.lex)

        self._reward_history: List[float] = []
        self._reward_perf_baseline: float = 0.6
        self._last_calibration_ts: float = 0.0
        components = [
            "explicit_feedback",
            "uncertainty_delta",
            "continue_dialogue",
            "valence",
            "acceptance",
            "relationship_growth",
            "identity_consist",
        ]
        self._component_names = components
        learner_cfg = self.cfg.get("weight_learner", {})
        self._weight_learner = ContextualWeightLearner(
            self.cfg.get("weights", {}),
            component_names=components,
            lr=float(learner_cfg.get("lr", 0.15)),
            l2=float(learner_cfg.get("l2", 1e-3)),
            forget=float(learner_cfg.get("forget", 0.02)),
            noise=float(learner_cfg.get("exploration_noise", 0.03)),
        )
        self._component_stats: Dict[str, Dict[str, float]] = {
            name: {"fast": 0.5, "slow": 0.5, "last_drift": 0.0}
            for name in components
        }
        self._relationship_baseline: float = 0.5
        self._relationship_history: deque[Dict[str, Any]] = deque(maxlen=60)
        self._last_relationship_event: float = 0.0

    def _load_cfg(self) -> Dict[str, Any]:
        path = getattr(self.arch, "social_critic_cfg_path", "data/social_critic_config.json")
        try:
            import os
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        # défaut raisonnable
        return {
            "weights": {
                "explicit_feedback": 0.45,   # j'adore / trop / etc.
                "uncertainty_delta": 0.20,   # baisse des pending_questions
                "continue_dialogue": 0.10,   # fil continue
                "valence":           0.12,   # valence heuristique
                "acceptance":        0.10,   # ok / d'accord / merci
                "relationship_growth": 0.12, # profondeur d'échange
                "identity_consist":  0.05,   # alignment persona
                "policy_friction":  -0.15    # pénalité si friction
            },
            "min_confidence": 0.15
        }

    # ----------- contexte auxiliaire avant/après ----------
    def _pending_questions_count(self) -> int:
        try:
            qm = getattr(self.arch, "question_manager", None)
            return len(getattr(qm, "pending_questions", []) or [])
        except Exception:
            return 0

    def _policy_friction_recent(self, window_sec: int = 300) -> int:
        try:
            pol = getattr(self.arch, "policy", None)
            if pol and hasattr(pol, "recent_frictions"):
                return int(pol.recent_frictions(window_sec=window_sec))
        except Exception:
            pass
        return 0

    @staticmethod
    def _sanitize_for_llm(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(data, dict):
            return None
        sanitized: Dict[str, Any] = {}
        for key, value in data.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, (str, float, int, bool)) or value is None:
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                simple_items: List[Any] = []
                for item in list(value)[:6]:
                    if isinstance(item, (str, float, int, bool)):
                        simple_items.append(item)
                if simple_items:
                    sanitized[key] = simple_items
            elif isinstance(value, dict):
                nested = SocialCritic._sanitize_for_llm(value)
                if nested:
                    sanitized[key] = nested
        return sanitized or None

    def _llm_payload(
        self,
        user_msg: str,
        decision_trace: Dict[str, Any],
        pre_ctx: Optional[Dict[str, Any]],
        post_ctx: Optional[Dict[str, Any]],
        heuristics: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "message": user_msg or "",
            "heuristics": heuristics,
            "pre_context": self._sanitize_for_llm(pre_ctx),
            "post_context": self._sanitize_for_llm(post_ctx),
            "decision_trace": self._sanitize_for_llm(decision_trace),
            "user_id": getattr(self.arch, "last_user_id", None) or getattr(self.arch, "user_id", None),
        }
        return payload

    def _llm_assess(
        self,
        user_msg: str,
        decision_trace: Dict[str, Any],
        pre_ctx: Optional[Dict[str, Any]],
        post_ctx: Optional[Dict[str, Any]],
        heuristics: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        payload = self._llm_payload(user_msg, decision_trace, pre_ctx, post_ctx, heuristics)
        response = try_call_llm_dict(
            "social_critic_assessment",
            input_payload=payload,
            logger=LOGGER,
        )
        return dict(response) if isinstance(response, dict) else None

    # ----------- calcul de l'outcome ----------
    def compute_outcome(self,
                        user_msg: str,
                        decision_trace: Dict[str, Any],
                        pre_ctx: Optional[Dict[str, Any]] = None,
                        post_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retourne: {
          "reduced_uncertainty": bool,
          "continued": bool,
          "valence": float [-1..1],
          "accepted": bool,
          "reward": float [0..1],
          "confidence": float [0..1],
          "components": {
              <nom composant>: {"w": float, "v": float, "contrib": float}
          }
        }
        """
        ctx_key = self._context_key(decision_trace or {}, post_ctx or {})
        W = self._weight_learner.get(ctx_key)
        # compléter avec les poids hors apprentissage
        for name, value in self.cfg.get("weights", {}).items():
            W.setdefault(name, value)

        # pré / post comptage des incertitudes
        pre_q  = int((pre_ctx or {}).get("pending_questions_count", self._pending_questions_count()))
        post_q = int((post_ctx or {}).get("pending_questions_count", self._pending_questions_count()))
        reduced_unc = (post_q < pre_q)

        # fil continu : trivial si user a répondu et n'a pas “coupé” grossièrement
        cont = True
        low = (user_msg or "").strip().lower()
        if not low or low in {"stop", "bye", "au revoir"}:
            cont = False

        # valence + acceptance
        val = _sentiment_heuristic(user_msg or "")
        acc = _acceptance_marker(user_msg or "")

        relationship_depth, rel_meta = _relationship_signal(
            user_msg or "",
            decision_trace,
            pre_ctx,
            post_ctx,
        )
        baseline_before = float(getattr(self, "_relationship_baseline", 0.5))
        # feedback explicite (hybride: statique + lexique appris)
        user_id = getattr(self.arch, "user_id", None)
        match_fn = getattr(self.lex, "match", None)
        exp_pos = _contains_any(low, POS_MARKERS) or (
            callable(match_fn) and match_fn(user_msg, polarity="pos", user_id=user_id)
        )
        exp_neg = _contains_any(low, NEG_MARKERS) or (
            callable(match_fn) and match_fn(user_msg, polarity="neg", user_id=user_id)
        )
        explicit = 0.5
        if exp_pos and not exp_neg:
            explicit = 1.0
        elif exp_neg and not exp_pos:
            explicit = 0.0
        elif exp_pos and exp_neg:
            explicit = 0.5

        # identité (alignement persona du post-contexte si dispo)
        identity_consist = float((post_ctx or {}).get("persona_alignment", 0.5))

        # policy friction
        frictions = self._policy_friction_recent(window_sec=300)
        pol_pen = 1.0 if frictions == 0 else (0.7 if frictions == 1 else 0.4)

        heuristics_snapshot = {
            "reduced_uncertainty": bool(reduced_unc),
            "continue_dialogue": bool(cont),
            "valence": float(val),
            "acceptance": bool(acc),
            "explicit_feedback": float(explicit),
            "identity_consist": float(identity_consist),
            "relationship_depth": float(relationship_depth),
            "pending_questions_before": pre_q,
            "pending_questions_after": post_q,
            "policy_penalty": float(pol_pen),
        }

        llm_data = self._llm_assess(user_msg or "", decision_trace or {}, pre_ctx, post_ctx, heuristics_snapshot)
        llm_reward_hint: Optional[float] = None
        llm_confidence: Optional[float] = None
        llm_markers: List[Dict[str, Any]] = []
        if llm_data:
            try:
                raw_hint = llm_data.get("reward_hint")
                if raw_hint is not None:
                    llm_reward_hint = clamp(float(raw_hint), 0.0, 1.0)
            except (TypeError, ValueError):
                llm_reward_hint = None
            try:
                if "confidence" in llm_data:
                    llm_confidence = clamp(float(llm_data.get("confidence", 0.0)), 0.0, 1.0)
            except (TypeError, ValueError):
                llm_confidence = None
            signals = llm_data.get("signals") if isinstance(llm_data.get("signals"), dict) else {}
            if signals:
                if "reduced_uncertainty" in signals:
                    reduced_unc = bool(signals.get("reduced_uncertainty"))
                if "continue_dialogue" in signals:
                    cont = bool(signals.get("continue_dialogue"))
                if "valence" in signals:
                    try:
                        val = clamp(float(signals.get("valence", val)), -1.0, 1.0)
                    except (TypeError, ValueError):
                        pass
                if "acceptance" in signals:
                    acc = bool(signals.get("acceptance"))
                if "relationship_depth" in signals:
                    try:
                        relationship_depth = clamp(float(signals.get("relationship_depth", relationship_depth)), 0.0, 1.0)
                    except (TypeError, ValueError):
                        pass
                if "identity_consistency" in signals:
                    try:
                        identity_consist = clamp(float(signals.get("identity_consistency", identity_consist)), 0.0, 1.0)
                    except (TypeError, ValueError):
                        pass
                feedback = signals.get("explicit_feedback")
                if isinstance(feedback, dict):
                    polarity = str(feedback.get("polarity") or "").strip().lower()
                    try:
                        fb_conf = clamp(float(feedback.get("confidence", 0.7)), 0.0, 1.0)
                    except (TypeError, ValueError):
                        fb_conf = 0.7
                    if polarity == "positive":
                        exp_pos, exp_neg = True, False
                        explicit = max(explicit, fb_conf)
                    elif polarity == "negative":
                        exp_pos, exp_neg = False, True
                        explicit = min(explicit, 1.0 - fb_conf)
                    elif polarity in {"mixed", "both"}:
                        exp_pos = exp_neg = True
                        explicit = 0.5
                    elif polarity == "neutral":
                        exp_pos = exp_neg = False
                        explicit = 0.5
            markers_payload = llm_data.get("markers")
            if isinstance(markers_payload, list):
                for item in markers_payload:
                    if isinstance(item, dict):
                        llm_markers.append(item)

        if llm_markers:
            for marker in llm_markers:
                polarity = str(marker.get("polarity") or "").strip().lower()
                if polarity == "positive":
                    exp_pos = True
                elif polarity == "negative":
                    exp_neg = True

        growth_signal = clamp(0.5 + (relationship_depth - baseline_before) * 0.8, 0.0, 1.0)
        self._relationship_baseline = 0.92 * baseline_before + 0.08 * relationship_depth

        if isinstance(llm_data, dict):
            llm_topics = llm_data.get("topics")
            if isinstance(llm_topics, list):
                topics = [t for t in llm_topics if isinstance(t, str)]
                if topics:
                    rel_meta = dict(rel_meta or {})
                    rel_meta.setdefault("topics", topics)

        snapshot = self._build_relationship_snapshot(
            user_msg or "",
            rel_meta,
            decision_trace,
            pre_ctx,
            post_ctx,
            relationship_depth,
            growth_signal,
        )

        # agrégation reward (0..1), pondérée
        # mapping valence [-1..1] -> [0..1]
        val01 = (val + 1.0) / 2.0
        reward = 0.0
        comp = {}

        parts = [
            ("explicit_feedback", explicit),
            ("uncertainty_delta", 1.0 if reduced_unc else 0.0),
            ("continue_dialogue", 1.0 if cont else 0.0),
            ("valence",           val01),
            ("acceptance",        1.0 if acc else 0.0),
            ("relationship_growth", growth_signal),
            ("identity_consist",  identity_consist),
        ]
        for name, v in parts:
            w = float(W.get(name, 0.0))
            reward += w * v
            comp[name] = {"w": w, "v": round(v, 3), "contrib": round(w * v, 3)}

        # policy friction (pénalité multiplicative)
        reward = reward * max(0.0, min(1.0, 1.0 + float(W.get("policy_friction", -0.15)) * (1.0 - pol_pen)))
        reward = clamp(reward, 0.0, 1.0)

        if llm_reward_hint is not None:
            reward = clamp(0.7 * reward + 0.3 * llm_reward_hint, 0.0, 1.0)

        if llm_data:
            try:
                setattr(self.arch, "_social_critic_last_llm", llm_data)
            except Exception:
                pass

        # confiance de l'estimation (plus on a de signaux forts, plus c'est fiable)
        support = 0.0
        support += 0.35 if (exp_pos or exp_neg) else 0.0
        support += 0.20 if reduced_unc else 0.0
        support += 0.10 if cont else 0.0
        support += 0.10 if acc else 0.0
        support += 0.18 if growth_signal >= 0.55 else 0.05
        support += 0.10  # base
        if llm_confidence is not None:
            support += 0.08 * llm_confidence
        confidence = clamp(max(self.cfg.get("min_confidence", 0.15), support), 0.0, 1.0)
        if llm_confidence is not None:
            confidence = clamp(0.6 * confidence + 0.4 * llm_confidence, 0.0, 1.0)

        # apprentissage du lexique (pas binaire — reward et confidence pondèrent)
        observe_fn = getattr(self.lex, "observe_message", None)
        if callable(observe_fn):
            try:
                observe_fn(
                    user_msg,
                    reward01=reward,
                    confidence=confidence,
                    user_id=user_id,
                )
            except Exception:
                pass

        outcome = {
            "reduced_uncertainty": reduced_unc,
            "continued": cont,
            "valence": round(val, 3),
            "accepted": acc,
            "reward": round(reward, 4),
            "confidence": round(confidence, 3),
            "components": comp,
            "context_key": ctx_key,
            "relationship_depth": round(relationship_depth, 4),
            "relationship_growth": round(growth_signal, 4),
            "relationship_snapshot": snapshot,
            "features": {name: float(val) for name, val in parts},
        }
        if llm_data:
            outcome["llm"] = {
                "confidence": llm_confidence,
                "notes": llm_data.get("notes"),
                "reward_hint": llm_reward_hint,
            }
        return outcome

    # ----------- application à la règle ----------
    def update_rule_with_outcome(self, rule_id: str, outcome: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Charge la règle depuis la mémoire, observe l'outcome et persiste.
        Retourne le dict final persisté (ou None si pas trouvé).
        """
        # récupérer la règle la plus récente (kind="interaction_rule", id=rule_id)
        try:
            rules = self.arch.memory.get_recent_memories(kind="interaction_rule", limit=500) or []
        except Exception:
            rules = []
        rule_dict = None
        for r in rules:
            if r.get("id") == rule_id:
                rule_dict = r
                break
        if not rule_dict:
            return None

        try:
            rule = InteractionRule.from_dict(rule_dict)
            # mise à jour des postérieurs + ema
            rule.observe_outcome(outcome)
            # persister (update si possible, sinon re-add)
            newd = rule.to_dict()
            if hasattr(self.arch.memory, "update_memory"):
                self.arch.memory.update_memory(newd)
            else:
                self.arch.memory.add_memory(newd)
            # trace
            try:
                self.arch.memory.add_memory({
                    "kind":"reward_event",
                    "rule_id": rule_id,
                    "outcome": outcome,
                    "ts": _now()
                })
            except Exception:
                pass

            try:
                reward_val = float((outcome or {}).get("reward", 0.5))
            except Exception:
                reward_val = 0.5
            self._reward_history.append(reward_val)
            window = int(getattr(self.arch, "social_calibration_window", 30))
            if window > 0 and len(self._reward_history) > window:
                self._reward_history = self._reward_history[-window:]
            alpha = float(getattr(self.arch, "social_calibration_alpha", 0.2))
            alpha = max(0.05, min(0.5, alpha))
            ema = None
            for r in self._reward_history:
                ema = r if ema is None else (alpha * r + (1.0 - alpha) * ema)
            if ema is not None:
                baseline = getattr(self, "_reward_perf_baseline", ema)
                self._reward_perf_baseline = (0.98 * baseline) + (0.02 * ema)
                drift = baseline - ema
                cooldown = max(10.0, float(getattr(self.arch, "social_calibration_cooldown", 30.0)))
                if drift > 0.04 and (_now() - getattr(self, "_last_calibration_ts", 0.0)) > cooldown:
                    weights = self.cfg.setdefault("weights", {})
                    adjust = min(0.02, drift * 0.1)
                    if adjust > 0.0:
                        weights["explicit_feedback"] = max(
                            0.25, float(weights.get("explicit_feedback", 0.45)) - adjust
                        )
                        weights["continue_dialogue"] = min(
                            0.22, float(weights.get("continue_dialogue", 0.10)) + adjust * 0.5
                        )
                        weights["valence"] = min(
                            0.22, float(weights.get("valence", 0.15)) + adjust * 0.5
                        )
                        self._last_calibration_ts = _now()
            self._learn_from_outcome(outcome)
            try:
                rel_depth = outcome.get("relationship_depth")
            except Exception:
                rel_depth = None
            snapshot_payload = outcome.get("relationship_snapshot")
            growth_value = outcome.get("relationship_growth")
            self._record_relationship_metric(rel_depth, snapshot=snapshot_payload, growth=growth_value)
            return newd
        except Exception:
            return None

    def _record_relationship_metric(
        self,
        value: Optional[float],
        *,
        snapshot: Optional[Dict[str, Any]] = None,
        growth: Optional[float] = None,
    ) -> None:
        if value is None:
            return
        try:
            depth = float(value)
        except (TypeError, ValueError):
            return
        metacog = getattr(self.arch, "metacognition", None)
        if not metacog or not hasattr(metacog, "cognitive_monitoring"):
            return
        try:
            tracking = metacog.cognitive_monitoring.setdefault("performance_tracking", {})
        except Exception:
            return
        try:
            hist = tracking.setdefault("relationship_depth", [])
            hist.append({"timestamp": _now(), "value": clamp(depth, 0.0, 1.0), "context": "social_reward"})
        except Exception:
            pass
        if snapshot:
            snapshot = dict(snapshot)
            snapshot.setdefault("timestamp", _now())
            self._store_relationship_snapshot(snapshot)
        self._last_relationship_event = _now()
        experimenter = getattr(metacog, "experimenter", None)
        if experimenter and hasattr(experimenter, "record_outcome"):
            try:
                experimenter.record_outcome("relationship_depth", new_value=clamp(depth, 0.0, 1.0))
            except Exception:
                pass
        self._emit_relationship_event(depth, snapshot=snapshot, growth=growth)

    def _build_relationship_snapshot(
        self,
        message: str,
        rel_meta: Dict[str, Any],
        decision_trace: Optional[Dict[str, Any]],
        pre_ctx: Optional[Dict[str, Any]],
        post_ctx: Optional[Dict[str, Any]],
        depth: float,
        growth: float,
    ) -> Optional[Dict[str, Any]]:
        text = (message or "").strip()
        if not text:
            return None
        if depth < 0.35 and growth < 0.55:
            return None
        meta_topics = [t for t in rel_meta.get("topics", []) if isinstance(t, str)] if isinstance(rel_meta, dict) else []
        ctx_topics = []
        for ctx in (pre_ctx, post_ctx):
            if isinstance(ctx, dict):
                for item in ctx.get("conversation_topics", []) or []:
                    if isinstance(item, str):
                        val = item.strip()
                        if val and val not in meta_topics and val not in ctx_topics:
                            ctx_topics.append(val)
        topics = meta_topics or ctx_topics
        if not topics and len(text) < 20:
            return None
        agent_reply = None
        trace = decision_trace or {}
        for key in ("rendered_reply", "response_text", "assistant_reply", "message"):
            val = trace.get(key)
            if isinstance(val, str) and val.strip():
                agent_reply = val.strip()
                break
        shared_reference = False
        if isinstance(post_ctx, dict):
            shared_reference = bool(post_ctx.get("shared_reference"))
        payload = {
            "message": text,
            "topics": topics,
            "sentiment": rel_meta.get("sentiment") if isinstance(rel_meta, dict) else 0.0,
            "depth": clamp(depth, 0.0, 1.0),
            "growth": clamp(growth, 0.0, 1.0),
            "agent_reply": agent_reply,
            "shared_reference": shared_reference,
        }
        if isinstance(post_ctx, dict):
            payload["user_traits"] = {k: post_ctx.get(k) for k in ("user_segment", "mood", "persona_alignment") if k in post_ctx}
        return payload

    def _store_relationship_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if not snapshot:
            return
        memory = getattr(self.arch, "memory", None)
        if not memory or not hasattr(memory, "add_memory"):
            return
        latest = self._relationship_history[-1] if self._relationship_history else None
        if latest and latest.get("message") == snapshot.get("message"):
            return
        entry = {
            "kind": "relationship_snapshot",
            "timestamp": snapshot.get("timestamp", _now()),
            "message": snapshot.get("message"),
            "topics": snapshot.get("topics", []),
            "depth": snapshot.get("depth"),
            "growth": snapshot.get("growth"),
            "sentiment": snapshot.get("sentiment"),
            "agent_reply": snapshot.get("agent_reply"),
            "shared_reference": snapshot.get("shared_reference", False),
            "metadata": snapshot.get("user_traits") or {},
        }
        try:
            memory.add_memory(entry)
        except Exception:
            pass
        self._relationship_history.append(entry)

    def _emit_relationship_event(
        self,
        depth: float,
        *,
        snapshot: Optional[Dict[str, Any]] = None,
        growth: Optional[float] = None,
    ) -> None:
        metacog = getattr(self.arch, "metacognition", None)
        if not metacog:
            return
        event_fn = getattr(metacog, "_record_metacognitive_event", None)
        domain_enum = getattr(metacog, "CognitiveDomain", None)
        if not callable(event_fn) or domain_enum is None:
            return
        try:
            domain = getattr(domain_enum, "SOCIAL")
        except AttributeError:
            domain = getattr(domain_enum, "LANGUAGE", None)
        if domain is None:
            return
        growth_val = clamp(float(growth) if growth is not None else depth, 0.0, 1.0)
        centred = depth - 0.5
        magnitude = abs(centred)
        sentiment = 0.0
        if snapshot:
            try:
                sentiment = float(snapshot.get("sentiment", 0.0))
            except (TypeError, ValueError):
                sentiment = 0.0
        event_type = "relationship_gain" if growth_val >= 0.55 and depth >= 0.45 else "relationship_regress" if depth < 0.35 else "relationship_update"
        description = "Actualisation du lien relationnel"
        if snapshot and snapshot.get("topics"):
            topic_preview = ", ".join(snapshot["topics"][:2])
            description = f"Lien relationnel autour de {topic_preview}"
        significance = clamp(0.35 + magnitude * 0.6, 0.0, 1.0)
        confidence = clamp(0.55 + (growth_val - 0.5) * 0.5, 0.1, 0.95)
        try:
            event_fn(
                event_type,
                domain,
                description,
                significance,
                confidence,
                emotional_valence=sentiment,
                cognitive_load=max(0.0, 0.4 - confidence * 0.3),
            )
        except Exception:
            return

    # ----------- apprentissage pondéré & dérive ----------
    def _normalize_ctx_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple, set)):
            return tuple(self._normalize_ctx_value(v) for v in value)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (self._normalize_ctx_value(k), self._normalize_ctx_value(v))
                    for k, v in value.items()
                )
            )
        return str(value)

    def _context_key(self, decision_trace: Dict[str, Any], post_ctx: Dict[str, Any]) -> Tuple[Any, ...]:
        persona = (decision_trace or {}).get("persona_id") or getattr(self.arch, "persona_id", None)
        channel = (decision_trace or {}).get("channel") or getattr(self.arch, "channel", None)
        tactic = (decision_trace or {}).get("selected_tactic")
        user_segment = post_ctx.get("user_segment") if isinstance(post_ctx, dict) else None
        return tuple(
            self._normalize_ctx_value(val)
            for val in (persona, channel, tactic, user_segment)
        )

    def _learn_from_outcome(self, outcome: Optional[Dict[str, Any]]) -> None:
        if not outcome:
            return
        ctx_key = outcome.get("context_key")
        try:
            ctx_tuple = tuple(ctx_key) if ctx_key else tuple()
        except TypeError:
            ctx_tuple = tuple()
        features_map = outcome.get("features") or {}
        feature_vec = [float(features_map.get(name, 0.0)) for name in self._component_names]
        reward = float(outcome.get("reward", 0.5))
        confidence = float(outcome.get("confidence", 0.5))
        self._weight_learner.update(ctx_tuple, feature_vec, reward, confidence)
        self._update_component_stats(ctx_tuple, features_map, reward)

    def _update_component_stats(self,
                               ctx_key: Tuple[Any, ...],
                               features: Dict[str, float],
                               reward: float) -> None:
        now = _now()
        if not isinstance(features, dict):
            return
        for name, stats in self._component_stats.items():
            val = float(features.get(name, 0.0)) * reward
            fast = stats["fast"]
            slow = stats["slow"]
            stats["fast"] = fast + 0.35 * (val - fast)
            stats["slow"] = slow + 0.08 * (val - slow)
            drift = abs(stats["fast"] - stats["slow"])
            if drift > 0.12 and (now - stats.get("last_drift", 0.0)) > 45.0:
                # légère ré-exploration ciblée sur le composant qui dérive
                self._weight_learner.inject_noise(ctx_key, component=name, strength=0.05)
                stats["last_drift"] = now
