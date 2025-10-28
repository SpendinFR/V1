from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Any, List, Tuple, Optional, Iterable

import logging
import random
import re
import time
import math

from AGI_Evolutive.social.tactic_selector import TacticSelector
from AGI_Evolutive.social.interaction_rule import ContextBuilder
from AGI_Evolutive.core.structures.mai import MAI

from .nlg import NLGContext, apply_mai_bids_to_nlg, paraphrase_light, join_tokens
from .style_critic import StyleCritic
from AGI_Evolutive.runtime.response import humanize_reasoning_block
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]{1,}")
DIRECT_QUESTION_PATTERN = re.compile(r"\?\s*(?:$|[)\]»\"']?$)")


LOGGER = logging.getLogger(__name__)


def _tokens(s: str) -> set:
    """Return a normalized token set robust to accents/case."""

    text = (s or "").casefold()
    return {token for token in TOKEN_PATTERN.findall(text) if len(token) >= 3}


def _build_language_state_snapshot(arch, ctx: Dict[str, Any]) -> Dict[str, Any]:
    dialogue_ctx = ctx.get("dialogue") or ctx.get("dialogue_state")
    if dialogue_ctx is None:
        dialogue_ctx = getattr(getattr(arch, "language", None), "state", None)
    return {
        "beliefs": getattr(arch, "beliefs", None),
        "self_model": getattr(arch, "self_model", None),
        "dialogue": dialogue_ctx,
        "world": getattr(arch, "world_model", None),
    }


def _apply_action_hint(text: str, hint: str) -> str:
    hint = (hint or "").strip()
    lower = text.lower()
    if hint == "AskConsent":
        prefix = "Avant de poursuivre, pourrais-tu confirmer que je peux partager ces informations ? "
        if not lower.startswith(prefix.lower()):
            return prefix + text
    elif hint == "RefusePolitely":
        apology = "Je suis désolé, je ne peux pas partager cette information."
        if apology.lower() not in lower:
            return f"{apology} {text}".strip()
    elif hint == "PartialReveal":
        note = "(Je partage uniquement ce qui est approprié pour protéger la confidentialité.)"
        if note.lower() not in lower:
            return f"{text}\n\n{note}"
    elif hint == "RephraseRespectfully":
        marker = "Je vais reformuler avec plus de délicatesse :"
        if marker.lower() not in lower:
            return f"{marker} {text}"
    return text


class OnlineLogisticModel:
    """Simple online logistic regression with bounded weights."""

    def __init__(
        self,
        features: Iterable[str],
        *,
        learning_rate: float = 0.05,
        weight_bounds: Tuple[float, float] = (-3.0, 3.0),
        bias: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_bounds = weight_bounds
        self.weights: Dict[str, float] = {name: 0.0 for name in features}
        self.bias = bias

    def _clip(self, value: float) -> float:
        lo, hi = self.weight_bounds
        return max(lo, min(hi, value))

    def _linear(self, feats: Dict[str, float]) -> float:
        score = self.bias
        for name, value in feats.items():
            score += self.weights.get(name, 0.0) * value
        return score

    def predict(self, feats: Dict[str, float]) -> float:
        score = self._linear(feats)
        try:
            return 1.0 / (1.0 + math.exp(-score))
        except OverflowError:
            return 0.0 if score < 0 else 1.0

    def update(self, feats: Dict[str, float], label: float, *, sample_weight: float = 1.0) -> None:
        label = float(label)
        pred = self.predict(feats)
        grad = (pred - label) * sample_weight
        lr = self.learning_rate
        for name, value in feats.items():
            w = self.weights.get(name, 0.0)
            w -= lr * grad * value
            self.weights[name] = self._clip(w)
        self.bias = self._clip(self.bias - lr * grad)


@dataclass
class OrnamentDecision:
    kind: str
    decided: bool
    probability: float
    features: Dict[str, float]
    meta: Dict[str, Any]


class OnlineTextClassifier:
    """Lightweight text openness classifier (online logistic)."""

    FEATURE_NAMES = (
        "bias",
        "len_norm",
        "has_question",
        "exclam_density",
        "emoji_density",
        "has_bonjour",
        "has_negation",
        "contains_slash",
    )

    def __init__(self) -> None:
        features = [name for name in self.FEATURE_NAMES if name != "bias"]
        self.model = OnlineLogisticModel(features, learning_rate=0.03, weight_bounds=(-2.5, 2.5), bias=-0.3)

    @staticmethod
    def extract_features(text: str) -> Dict[str, float]:
        text = (text or "").strip()
        length = len(text)
        words = len(text.split()) or 1
        emojis = len(re.findall(r"[\U0001F300-\U0001FAFF]", text))
        exclam = text.count("!")
        features: Dict[str, float] = {
            "len_norm": min(1.0, length / 220.0),
            "has_question": 1.0 if re.search(r"\?\s*$", text) else 0.0,
            "exclam_density": min(1.0, exclam / max(1, words)),
            "emoji_density": min(1.0, emojis / max(1, words)),
            "has_bonjour": 1.0 if re.search(r"\bbon(jour|soir)\b", text, re.IGNORECASE) else 0.0,
            "has_negation": 1.0 if re.search(r"\b(pas|plus|jamais|rien)\b", text, re.IGNORECASE) else 0.0,
            "contains_slash": 1.0 if "/" in text else 0.0,
        }
        return features

    def predict(self, text: str) -> float:
        feats = self.extract_features(text)
        return self.model.predict(feats)

    def update(self, text: str, liked: bool, weight: float = 1.0) -> None:
        feats = self.extract_features(text)
        self.model.update(feats, 1.0 if liked else 0.0, sample_weight=weight)


class LanguageRenderer:
    def __init__(self, voice_profile, lexicon, ranker=None):
        self.voice = voice_profile
        self.lex = lexicon
        self.ranker = ranker
        # anti-spam / fréquence
        self._cooldown = {"past": 0.0, "colloc": 0.0}
        self._last_used = {"past": "", "colloc": ""}

        # seuils réglables (valeurs par défaut, peuvent être apprises)
        self.THRESH = {
            "past_relevance": 0.25,   # pertinence mini lien passé
            "colloc_relevance": 0.20, # pertinence mini collocation
            "conf_min": 0.55,         # confiance mini pour ornement
            "chance_colloc": 0.35,    # proba de tenter une collocation (modulée)
            "quote_prob": 0.35,
        }

        self.critic = StyleCritic(max_chars=1200)
        self._rand = random.Random()

        # modèles adaptatifs
        self._past_policy = OnlineLogisticModel(
            [
                "relevance",
                "confidence",
                "budget_ratio",
                "cooldown",
                "direct_question",
                "duplication",
                "openness",
            ],
            learning_rate=0.04,
            weight_bounds=(-2.0, 2.5),
            bias=0.1,
        )
        # initialise pour coller au comportement existant
        self._past_policy.weights.update(
            {
                "relevance": 3.4,
                "confidence": 0.9,
                "budget_ratio": 0.4,
                "cooldown": -0.8,
                "direct_question": -1.2,
                "duplication": -1.5,
                "openness": 0.6,
            }
        )

        self._colloc_policy = OnlineLogisticModel(
            [
                "relevance",
                "confidence",
                "budget_ratio",
                "cooldown",
                "randomized",
                "openness",
            ],
            learning_rate=0.035,
            weight_bounds=(-2.5, 2.5),
            bias=-0.2,
        )
        self._colloc_policy.weights.update(
            {
                "relevance": 2.8,
                "confidence": 0.6,
                "budget_ratio": 0.5,
                "cooldown": -1.1,
                "randomized": 0.3,
                "openness": 0.45,
            }
        )

        self._text_classifier = OnlineTextClassifier()
        self._pending_adaptive_events: List[OrnamentDecision] = []


    def rand(self) -> float:
        return self._rand.random()

    # ---------- utilitaires ----------
    def apply_action_hint(self, text: str, hint: str) -> str:
        """Expose l'utilitaire d'application d'hints MAI pour d'autres pipelines."""

        return _apply_action_hint(text, hint)

    def _confidence(self) -> float:
        # essaie d’extraire une confiance globale depuis la policy
        try:
            pol = self.voice.self_model.arch.policy
            if hasattr(pol, "confidence"):
                return float(pol.confidence())
        except Exception:
            pass
        return 0.6

    def _budget_chars(self, ctx: Dict[str, Any]) -> int:
        # budget d’ornement selon style utilisateur / voix
        st = self.voice.style()
        user = (ctx.get("user_style") or {})
        base = 160  # budget max d’ornement
        if user.get("prefers_long"):
            base += 80
        if st.get("conciseness", 0.6) > 0.7:
            base -= 60
        return max(0, base)

    def _decrease_cooldowns(self, store: Optional[Dict[str, float]] = None) -> None:
        # refroidit un peu à chaque appel
        target = store if store is not None else self._cooldown
        for k in target:
            target[k] = max(0.0, target[k] - 0.34)

    # ---------- sélection “lien au passé” ----------
    def _pick_relevant_moment(self, user_msg: str, ctx: Dict[str, Any]) -> Tuple[str, float]:
        moments: List[str] = ctx.get("key_moments") or []
        if not moments:
            return ("", 0.0)
        utoks = _tokens(user_msg)
        best, score = "", 0.0
        for m in moments[-8:]:
            mtoks = _tokens(m)
            if not mtoks:
                continue
            jacc = len(utoks & mtoks) / max(1, len(utoks | mtoks))
            if jacc > score:
                best, score = m, jacc
        return (best, score)

    # ---------- sélection “collocation aimée” ----------
    def _pick_collocation(self, ctx: Dict[str, Any]) -> Tuple[str, float]:
        topics = set(ctx.get("topics") or [])
        cand = self.lex.sample_collocation(novelty=0.3)
        if not cand:
            return ("", 0.0)
        rel = len(_tokens(cand) & topics) / max(1, len(_tokens(cand)))
        return (cand, rel)

    # ---------- décor et rendu ----------
    def _decorate_with_voice(self, text: str) -> str:
        st = self.voice.style()
        if st.get("formality", 0.4) > 0.75 and not text.lower().startswith(("bonjour", "bonsoir")):
            text = "Bonjour, " + text
        if st.get("warmth", 0.6) > 0.75 and not text.endswith("!"):
            text += " (je reste dispo si besoin)"
        if st.get("emoji", 0.2) > 0.6:
            text = "🙂 " + text
        return text

    def _llm_refine_output(
        self,
        *,
        base_text: str,
        current_text: str,
        semantics: Dict[str, Any],
        ctx: Dict[str, Any],
        applied_hints: List[Dict[str, Any]],
        dry_run: bool,
    ) -> str:
        if dry_run:
            return current_text
        payload = {
            "base_text": base_text,
            "current_text": current_text,
            "intent": semantics.get("intent") if isinstance(semantics, dict) else None,
            "style": getattr(self.voice, "state", {}).get("style", {}),
            "applied_hints": applied_hints,
            "last_user_message": ctx.get("last_message"),
            "topics": ctx.get("topics"),
        }
        response = try_call_llm_dict(
            "language_renderer",
            input_payload=payload,
            logger=LOGGER,
        )
        if not response:
            return current_text

        revision = response.get("revision") or response.get("text")
        if isinstance(revision, str) and revision.strip():
            current_text = revision.strip()

        lexicon_updates = response.get("lexicon_updates")
        if isinstance(lexicon_updates, list) and hasattr(self.lex, "prefer"):
            for entry in lexicon_updates:
                if isinstance(entry, str) and entry.strip():
                    try:
                        self.lex.prefer(entry.strip())
                    except Exception:
                        LOGGER.debug("Failed to register lexicon update", exc_info=True)

        notes = response.get("notes")
        if isinstance(notes, str) and notes.strip():
            ctx.setdefault("llm_renderer_notes", []).append(notes.strip())

        return current_text

    def _normalize_semantic_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple, set)):
            parts: List[str] = []
            for item in value:
                normalised = self._normalize_semantic_value(item)
                if normalised:
                    parts.append(normalised)
            return "\n".join(parts).strip()
        if isinstance(value, dict):
            for key in ("text", "raw", "summary"):
                candidate = self._normalize_semantic_value(value.get(key))
                if candidate:
                    return candidate
            return ""
        return str(value).strip()

    def _semantics_text(self, semantics: Any) -> str:
        if isinstance(semantics, str):
            return semantics.strip()
        if semantics is None:
            return ""
        if not isinstance(semantics, dict):
            return self._normalize_semantic_value(semantics)

        for key in ("text", "raw", "summary"):
            candidate = self._normalize_semantic_value(semantics.get(key))
            if candidate:
                return candidate

        bullets_text = self._normalize_semantic_value(semantics.get("bullets"))
        if bullets_text:
            return bullets_text

        title_text = self._normalize_semantic_value(semantics.get("title"))
        if title_text:
            return title_text

        return ""

    def render_reply(self, semantics: Dict[str, Any], ctx: Dict[str, Any], *, dry_run: bool = False) -> str:
        """
        Règle : on n’ajoute un lien au passé / une collocation QUE si :
        - confiance >= conf_min
        - pertinence >= seuils
        - budget > 0
        - cooldown OK et pas de doublon
        Et au plus 1 ornement par réponse (priorité au lien passé).
        """
        ctx = ctx or {}
        if dry_run:
            ctx = dict(ctx)

        cooldown = dict(self._cooldown)
        last_used = dict(self._last_used)

        self._decrease_cooldowns(cooldown)
        base_text = self._semantics_text(semantics)
        normalized_text, diagnostics = humanize_reasoning_block(base_text)
        effective_text = normalized_text or base_text
        base = effective_text.strip() or "Je te réponds en tenant compte de notre historique."
        if (
            not dry_run
            and diagnostics
            and normalized_text
            and normalized_text.strip() != base_text.strip()
        ):
            ctx.setdefault("reasoning_trace", diagnostics)
        arch = getattr(getattr(self.voice, "self_model", None), "arch", None)
        policy = getattr(arch, "policy", None) if arch else None
        state_snapshot: Dict[str, Any] = {}
        predicate_registry: Dict[str, Any] = {}
        applicable_mais: List[MAI] = []
        if arch and policy and hasattr(policy, "build_predicate_registry"):
            try:
                state_snapshot = ctx.get("state_snapshot") or _build_language_state_snapshot(arch, ctx)
                predicate_registry = policy.build_predicate_registry(state_snapshot)
            except Exception:
                state_snapshot = {}
                predicate_registry = {}

        nlg_context = NLGContext(base, _apply_action_hint)
        try:
            applicable_mais = apply_mai_bids_to_nlg(nlg_context, state_snapshot, predicate_registry)
        except Exception:
            applicable_mais = []
        base = nlg_context.text
        applied_hints = nlg_context.applied_hints()
        if applied_hints and not dry_run:
            ctx.setdefault("applied_action_hints", []).extend(applied_hints)
        conf = self._confidence()
        budget = self._budget_chars(ctx)

        # Si la question est directe / l’utilisateur veut court → pas d’ornement
        last_message = ctx.get("last_message") or ""
        is_direct_question = bool(DIRECT_QUESTION_PATTERN.search(last_message))
        openness_prob = self._text_classifier.predict(last_message)
        if conf < self.THRESH["conf_min"] or budget < 60 or (ctx.get("user_style") or {}).get("prefers_long") is False:
            return self._decorate_with_voice(base)

        # 1) Candidat lien au passé
        past_txt, past_rel = self._pick_relevant_moment(last_message, ctx)
        past_features = {
            "relevance": float(past_rel),
            "confidence": float(conf),
            "budget_ratio": min(1.0, budget / 160.0),
            "cooldown": float(cooldown["past"]),
            "direct_question": 1.0 if is_direct_question else 0.0,
            "duplication": 1.0 if past_txt == last_used["past"] else 0.0,
            "openness": float(openness_prob),
        }
        past_policy_score = self._past_policy.predict(past_features) if past_txt else 0.0
        use_past = (
            past_txt
            and past_rel >= self.THRESH["past_relevance"] * 0.8
            and cooldown["past"] <= 0.0
            and past_policy_score >= 0.5
        )

        # 2) Candidat collocation aimée (faible proba, modulée par confiance)
        colloc_txt, colloc_rel = self._pick_collocation(ctx)
        p_try = self.THRESH["chance_colloc"] * (0.6 + 0.6 * conf)
        colloc_features = {
            "relevance": float(colloc_rel),
            "confidence": float(conf),
            "budget_ratio": min(1.0, budget / 160.0),
            "cooldown": float(cooldown["colloc"]),
            "randomized": self.rand(),
            "openness": float(openness_prob),
        }
        colloc_policy_score = self._colloc_policy.predict(colloc_features) if colloc_txt else 0.0
        use_colloc = (
            colloc_txt
            and (self.rand() < p_try or colloc_policy_score >= 0.65)
            and colloc_rel >= self.THRESH["colloc_relevance"] * 0.85
            and cooldown["colloc"] <= 0.0
            and colloc_policy_score >= 0.5
        )

        # Toujours max 1 ornement, priorité au passé s’il est pertinent
        if arch and getattr(arch, "memory", None):
            try:
                arch.tactic_selector = getattr(arch, "tactic_selector", TacticSelector(arch))
                selector_ctx = ContextBuilder.build(
                    arch,
                    extra={
                        "pending_questions_count": len(
                            getattr(getattr(arch, "question_manager", None), "pending_questions", [])
                            or []
                        )
                    },
                )
                rule, why = arch.tactic_selector.pick(selector_ctx)
            except Exception:
                rule, why = (None, None)
            if rule:
                if not dry_run:
                    rule["last_used_ts"] = time.time()
                    rule["usage_count"] = int(rule.get("usage_count", 0)) + 1
                    if hasattr(arch.memory, "update_memory"):
                        arch.memory.update_memory(rule)
                    else:
                        arch.memory.add_memory(rule)

                    arch.memory.add_memory(
                        {
                            "kind": "decision_trace",
                            "rule_id": rule["id"],
                            "tactic": (rule.get("tactic") or {}).get("name"),
                            "ctx_snapshot": selector_ctx,
                            "why": why,
                            "ts": time.time(),
                        }
                    )

                tac = (rule.get("tactic") or {}).get("name", "")
                if tac == "banter_leger" and "?" not in base:
                    base = base + " (clin d’œil)"
                elif tac == "ack_grateful":
                    base = base + " Merci, je le note."
                elif tac == "reformulation_empathique":
                    params = (rule.get("tactic") or {}).get("params") or {}
                    try:
                        ratio = float(params.get("mirror_ratio", 0.5))
                    except Exception:
                        ratio = 0.5
                    ratio = max(0.1, min(0.9, ratio))
                    user_msg = str(ctx.get("last_message") or "")
                    snippet = ""
                    if user_msg:
                        tokens = re.findall(r"[A-Za-zÀ-ÿ'’]+", user_msg)
                        if tokens:
                            take = max(3, int(len(tokens) * ratio))
                            snippet = join_tokens(tokens[:take])
                            if len(snippet) > 120:
                                snippet = snippet[:117].rstrip() + "…"
                    if snippet:
                        empathy = f"Si je comprends bien, tu parles de « {snippet} »."
                    else:
                        empathy = "Si je comprends bien, ce sujet te tient à cœur."
                    if empathy.lower() not in base.lower():
                        base = f"{empathy} {base}".strip()
                elif tac == "clarify_definition":
                    params = (rule.get("tactic") or {}).get("params") or {}
                    ensure_example = bool(params.get("ensure_example"))
                    state_snapshot = ctx.get("state_snapshot") or {}
                    dialogue_state = state_snapshot.get("dialogue") if isinstance(state_snapshot, dict) else None

                    def _iter_unknown_terms(dialogue: Any):
                        if dialogue is None:
                            return
                        if isinstance(dialogue, dict):
                            frames = dialogue.get("recent_frames", []) or []
                            for frame in reversed(frames):
                                if not isinstance(frame, dict):
                                    continue
                                for key in ("unknown_terms", "terms_need_definition", "unknowns"):
                                    vals = frame.get(key) or []
                                    for val in vals:
                                        if val:
                                            yield str(val)
                            profile = dialogue.get("user_profile", {})
                            if isinstance(profile, dict):
                                for val in reversed(profile.get("unknown_terms", []) or []):
                                    if val:
                                        yield str(val)
                            return
                        frames = list(getattr(dialogue, "recent_frames", []) or [])
                        for frame in reversed(frames):
                            if isinstance(frame, dict):
                                sources = [
                                    frame.get("unknown_terms"),
                                    frame.get("terms_need_definition"),
                                    frame.get("unknowns"),
                                ]
                            else:
                                sources = [
                                    getattr(frame, "unknown_terms", None),
                                    getattr(frame, "terms_need_definition", None),
                                    getattr(frame, "unknowns", None),
                                ]
                            for vals in sources:
                                if not vals:
                                    continue
                                for val in vals:
                                    if val:
                                        yield str(val)
                        profile = getattr(dialogue, "user_profile", None)
                        if isinstance(profile, dict):
                            for val in reversed(profile.get("unknown_terms", []) or []):
                                if val:
                                    yield str(val)

                    term = None
                    for candidate in _iter_unknown_terms(dialogue_state):
                        candidate = candidate.strip()
                        if candidate:
                            term = candidate
                            break
                    if term is None:
                        topics = ctx.get("topics") or []
                        if topics:
                            term = str(topics[0]).strip()
                    if term:
                        intro = f"Pour clarifier, quand je parle de « {term} », j'entends ceci :"
                        if intro.lower() not in base.lower():
                            base = f"{intro} {base}".strip()
                    if ensure_example and "exemple" not in base.lower():
                        last_msg = str(ctx.get("last_message") or "")
                        example_snippet = ""
                        if last_msg:
                            tokens = re.findall(r"[A-Za-zÀ-ÿ'’]+", last_msg)
                            if tokens:
                                limit = min(len(tokens), 12)
                                example_snippet = join_tokens(tokens[:limit])
                                if len(example_snippet) > 80:
                                    example_snippet = example_snippet[:77].rstrip() + "…"
                        if example_snippet:
                            base += (
                                f" Par exemple, dans ce que tu viens d'évoquer (« {example_snippet} »),"
                                " cette définition s'applique."
                            )
                        else:
                            base += " Par exemple, on peut penser à une situation concrète pour l'illustrer."

        out = self._decorate_with_voice(base)
        out = self._llm_refine_output(
            base_text=base,
            current_text=out,
            semantics=semantics,
            ctx=ctx,
            applied_hints=applied_hints,
            dry_run=dry_run,
        )
        if use_past:
            snippet = f"\n\n↪ En lien : {past_txt}"
            if len(snippet) <= budget:
                out += snippet
                cooldown["past"] = 2.0
                last_used["past"] = past_txt
                if not dry_run:
                    self._cooldown = cooldown
                    self._last_used = last_used
                    self._register_decision(
                        OrnamentDecision(
                            kind="past",
                            decided=True,
                            probability=past_policy_score,
                            features=past_features,
                            meta={"snippet_len": len(snippet)},
                        ),
                        ctx,
                    )
                return out  # on s’arrête ici

        if use_colloc:
            snippet = f"\n\n({colloc_txt})"
            if len(snippet) <= budget:
                out += snippet
                cooldown["colloc"] = 2.0
                last_used["colloc"] = colloc_txt
                if not dry_run:
                    self._cooldown = cooldown
                    self._last_used = last_used
                    self._register_decision(
                        OrnamentDecision(
                            kind="colloc",
                            decided=True,
                            probability=colloc_policy_score,
                            features=colloc_features,
                            meta={"snippet_len": len(snippet)},
                        ),
                        ctx,
                    )
            elif not dry_run:
                self._register_decision(
                    OrnamentDecision(
                        kind="colloc",
                        decided=False,
                        probability=colloc_policy_score,
                        features=colloc_features,
                        meta={"reason": "budget", "snippet_len": len(snippet)},
                    ),
                    ctx,
                )
        elif not dry_run and colloc_txt:
            self._register_decision(
                OrnamentDecision(
                    kind="colloc",
                    decided=False,
                    probability=colloc_policy_score,
                    features=colloc_features,
                    meta={"reason": "policy_block"},
                ),
                ctx,
            )

        if not dry_run and past_txt and not use_past:
            self._register_decision(
                OrnamentDecision(
                    kind="past",
                    decided=False,
                    probability=past_policy_score,
                    features=past_features,
                    meta={"reason": "policy_block"},
                ),
                ctx,
            )

        if not dry_run and ctx.get("omitted_content") and arch:
            payload = {
                "reason": "MAI-driven",
                "mai_ids": [mai.id for mai in applicable_mais],
                "evidence": [
                    asdict(doc)
                    for mai in applicable_mais
                    for doc in getattr(mai, "provenance_docs", [])
                ],
            }
            audit = getattr(arch, "audit", None)
            if audit and hasattr(audit, "log_omission_justifiee"):
                try:
                    audit.log_omission_justifiee(**payload)
                except Exception:
                    pass
            else:
                logger = getattr(arch, "logger", None)
                if logger and hasattr(logger, "write"):
                    try:
                        logger.write("nlg.omission", **payload)
                    except Exception:
                        pass

        if not dry_run:
            self._cooldown = cooldown
            self._last_used = last_used
            self._register_decision(
                OrnamentDecision(
                    kind="baseline",
                    decided=True,
                    probability=openness_prob,
                    features=OnlineTextClassifier.extract_features(last_message),
                    meta={"reason": "no_ornament"},
                ),
                ctx,
            )
        return out

    # --- Génère K variantes (macros + paraphrases légères) ---
    def render_reply_candidates(self, ctx: Dict[str, Any], base_plan: Dict[str, Any], K: int = 4) -> List[str]:
        plan = dict(base_plan or {})
        ctx = dict(ctx or {})

        neutral = self._render_once(ctx, plan, macro=None)
        candidates: List[str] = [neutral]

        macros = ["taquin", "coach", "sobre", "deadpan"]
        self._rand.shuffle(macros)
        for macro in macros:
            if len(candidates) >= max(1, K):
                break
            variant = self._render_once(ctx, plan, macro=macro)
            if variant and variant not in candidates:
                candidates.append(variant)

        out: List[str] = []
        for i, cand in enumerate(candidates):
            if i == 0:
                out.append(cand)
            else:
                out.append(paraphrase_light(cand, prob=0.30))
        return out[: max(1, K)]

    def _render_once(self, ctx: Dict[str, Any], plan: Dict[str, Any], macro: Optional[str] = None) -> str:
        sp = getattr(self.voice, "style_policy", None)
        snapshot = None
        mode_snapshot = None
        macro = (macro or "").lower() or None
        if sp and macro and hasattr(sp, "apply_macro"):
            try:
                snapshot = dict(getattr(sp, "params", {}))
                mode_snapshot = getattr(sp, "current_mode", None)
                sp.apply_macro(macro)
            except Exception:
                snapshot = None
                mode_snapshot = None

        try:
            text = self.render_reply(plan, ctx, dry_run=True)
        finally:
            if sp and snapshot is not None:
                try:
                    sp.params.update(snapshot)
                    if mode_snapshot is not None:
                        sp.current_mode = mode_snapshot
                except Exception:
                    pass

        macro_text = text
        if macro:
            if macro == "taquin" and "😉" not in macro_text:
                macro_text = macro_text.strip() + " 😉"
            elif macro == "coach":
                prefix = "Coach mode ▶ "
                if not macro_text.lower().startswith(prefix.lower()):
                    macro_text = f"{prefix}{macro_text}" if macro_text else prefix.rstrip()
            elif macro == "sobre":
                macro_text = re.sub(r"[🙂😉😊]+\s*", "", macro_text).strip()
            elif macro == "deadpan":
                lines: List[str] = []
                for line in macro_text.splitlines():
                    normalized = line.replace("!", ".").replace("…", ".")
                    tokens = normalized.split()
                    lines.append(join_tokens(tokens))
                macro_text = "\n".join(lines).strip()

        return macro_text.strip()

    def render_final(self, ctx: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        ctx_local = dict(ctx or {})
        plan_local = dict(plan or {})
        candidates = self.render_reply_candidates(ctx_local, plan_local, K=4)

        local_ctx = {
            "style": dict(getattr(getattr(self.voice, "style_policy", None), "params", {}))
        }
        scored: List[Tuple[float, int, str]] = []
        for idx, candidate in enumerate(candidates):
            try:
                score = (
                    float(self.ranker.score(local_ctx, candidate))
                    if self.ranker is not None
                    else 0.5
                )
            except Exception:
                score = 0.5
            scored.append((score, idx, candidate))

        if not scored:
            return {"text": "", "chosen": None, "alts": []}

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, chosen_idx, best_text = scored[0]

        analysis = self.critic.analyze(best_text)
        rewritten = self.critic.rewrite(best_text)
        best_text = rewritten

        final_text, quote_meta = self._maybe_add_quote(best_text, ctx_local)

        # ingest éventuel feedback post-rendu
        feedback = ctx_local.get("ornament_feedback") or {}
        if feedback:
            self._apply_feedback(feedback, ctx_local)

        return {
            "text": final_text,
            "chosen": {
                "idx": chosen_idx,
                "score": best_score,
                "text": final_text,
                "analysis": analysis,
                "quote": quote_meta,
            },
            "alts": [
                {"idx": idx, "score": score, "text": text}
                for score, idx, text in scored[1:]
            ],
        }

    def _ctx_is_safe_for_aside(self, ctx: Dict[str, Any]) -> bool:
        user_msg = (ctx.get("last_user_msg") or ctx.get("last_message") or "").lower()
        for bad in ["décès", "urgence", "mauvaise nouvelle", "licenciement", "plainte"]:
            if bad in user_msg:
                return False
        return True

    def _maybe_add_quote(self, text: str, ctx: Dict[str, Any]):
        quote_meta = None
        qm = getattr(self.voice, "quote_memory", None) or getattr(self, "quote_memory", None)
        if not qm:
            return text, quote_meta
        if (
            len(text) < 900
            and self._ctx_is_safe_for_aside(ctx)
            and self.rand() < self.THRESH["quote_prob"]
        ):
            try:
                context_seed = text + " " + (ctx.get("last_user_msg") or "")
                quote = qm.sample(context=context_seed)
            except Exception:
                quote = None
            if quote:
                quote_used = quote
                if len(quote.split()) > 20:
                    quote_used = paraphrase_light(quote, prob=0.45)
                text = f"{text}\n\n(Clin d’œil) {quote_used}".strip()
                quote_meta = {"len": len(quote_used), "raw_len": len(quote)}
        return text, quote_meta

    # ----- Adaptation & feedback -----
    def _register_decision(self, decision: OrnamentDecision, ctx: Dict[str, Any]) -> None:
        trace = {
            "kind": decision.kind,
            "decided": decision.decided,
            "probability": decision.probability,
            "features": decision.features,
            "meta": decision.meta,
            "ts": time.time(),
        }
        ctx.setdefault("language_trace", []).append(trace)
        self._pending_adaptive_events.append(decision)
        if len(self._pending_adaptive_events) > 64:
            self._pending_adaptive_events = self._pending_adaptive_events[-32:]

    def _apply_feedback(self, feedback: Dict[str, Any], ctx: Dict[str, Any]) -> None:
        if not self._pending_adaptive_events:
            return

        decay = float(feedback.get("decay", 0.9))
        satisfaction = feedback.get("satisfaction")
        liked_override = feedback.get("liked")
        user_message = ctx.get("last_message") or ""

        weight = 1.0
        if isinstance(satisfaction, (int, float)):
            weight = max(0.1, min(2.0, (float(satisfaction) - 2.5) / 2.5))
        if not math.isfinite(weight):
            weight = 1.0

        for event in self._pending_adaptive_events:
            decision_feedback = feedback.get(event.kind)
            if isinstance(decision_feedback, dict):
                liked = decision_feedback.get("liked")
                event_weight = float(decision_feedback.get("weight", weight))
            else:
                liked = decision_feedback
                event_weight = weight

            if liked is None and liked_override is not None:
                liked = liked_override

            if liked is None:
                continue

            if event.kind == "past":
                self._past_policy.update(event.features, 1.0 if liked else 0.0, sample_weight=event_weight)
            elif event.kind == "colloc":
                self._colloc_policy.update(event.features, 1.0 if liked else 0.0, sample_weight=event_weight)

        if liked_override is not None:
            self._text_classifier.update(user_message, bool(liked_override), weight)

        # légère décroissance des poids pour éviter dérive
        if decay < 1.0:
            for weights in (self._past_policy.weights, self._colloc_policy.weights):
                for key in list(weights.keys()):
                    weights[key] *= decay

        self._pending_adaptive_events.clear()

    def ingest_feedback(self, feedback: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> None:
        """API externe pour appliquer un feedback manuel."""

        context = ctx or {}
        context.setdefault("last_message", context.get("last_user_msg", ""))
        self._apply_feedback(feedback, context)
