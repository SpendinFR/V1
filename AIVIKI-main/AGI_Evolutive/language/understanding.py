"""
Language v2: NLU à cadres + état de dialogue + self-ask
- Pas d'appel LLM. Heuristiques hybrides (patterns, mots-clés, scores).
- Génère des questions ciblées si incertitude élevée.
- Retourne une frame riche + propose une réponse non-générique.
"""
import logging
import math
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Mapping, Optional, Tuple

from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)

from AGI_Evolutive.models.intent import IntentModel
from .dialogue_state import DialogueState
from .frames import DialogueAct, UtteranceFrame


LOGGER = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    return text


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _canonicalize(text: str) -> str:
    return _strip_accents(text).lower().strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


STOPWORDS = {
    _canonicalize(word)
    for word in {
        "bonjour",
        "salut",
        "merci",
        "ok",
        "fais",
        "peux",
        "pourquoi",
        "comment",
        "quoi",
        "quand",
        "où",
        "ou",
        "but",
        "objectif",
        "goal",
        "plan",
        "aide",
        "help",
        "bug",
        "erreur",
        "super",
        "parfait",
        "nul",
        "ciao",
        "bye",
        "stp",
    }
}


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()


RE_HEDGES = re.compile(r"\b(peut[- ]?etre|peux[- ]?etre|je crois|je pense|il me semble|maybe)\b", re.IGNORECASE)
RE_HAS_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
RE_TERM_DEFINITION = re.compile(
    r"\b(?P<term>[a-z0-9àâäéèêëîïôöùûüç'\- ]{2,60})\s+est\s+(?:un|une|le|la|l['’])\s+(?P<desc>[a-z0-9àâäéèêëîïôöùûüç'\- ]{3,160})",
    re.IGNORECASE,
)
RE_QUICK_DEF = re.compile(
    r"\b(?:c['’]?est|cest)\s+(?:un|une|le|la|l['’])\s+(?P<desc>[a-z0-9àâäéèêëîïôöùûüç'\- ]{3,160})",
    re.IGNORECASE,
)


class AdaptiveEMA:
    """EMA adaptative dont le beta est choisi par Thompson Sampling."""

    def __init__(self, betas: Optional[Tuple[float, ...]] = None, max_step: float = 0.2):
        self.betas = tuple(sorted(betas or (0.2, 0.4, 0.6, 0.8)))
        if not self.betas:
            raise ValueError("AdaptiveEMA requires at least one beta value")
        self.max_step = float(max_step)
        self.state = {
            beta: {"value": 0.5, "alpha": 1.0, "beta": 1.0, "last_error": None}
            for beta in self.betas
        }
        self.selected_beta = self.betas[0]
        self.drift_log: deque = deque(maxlen=64)

    def update(self, observation: float, target: float) -> float:
        observation = max(0.0, min(1.0, float(observation)))
        target = max(0.0, min(1.0, float(target)))

        for beta, slot in self.state.items():
            previous = slot["value"]
            delta = observation - previous
            step = beta * delta
            step = max(-self.max_step, min(self.max_step, step))
            new_value = previous + step
            last_error = slot["last_error"]
            prev_error = abs(target - previous) if last_error is None else last_error
            new_error = abs(target - new_value)
            reward = 1.0 if new_error < prev_error else 0.5 if math.isclose(new_error, prev_error, abs_tol=1e-6) else 0.0
            slot["value"] = max(0.0, min(1.0, new_value))
            slot["last_error"] = new_error
            slot["alpha"] += reward
            slot["beta"] += 1.0 - reward
            if abs(step) > self.max_step * 0.75:
                self.drift_log.append({
                    "beta": beta,
                    "from": previous,
                    "to": slot["value"],
                    "target": target,
                })

        best_sample = -1.0
        for beta, slot in self.state.items():
            sample = random.betavariate(slot["alpha"], slot["beta"])
            if sample > best_sample:
                best_sample = sample
                self.selected_beta = beta

        return self.state[self.selected_beta]["value"]

    def current(self) -> float:
        return self.state[self.selected_beta]["value"]


class OnlineIntentClassifier:
    """Classifieur de secours entraînable en ligne (n-grammes, ponctuation, emoji)."""

    def __init__(self, learning_rate: float = 0.25, ngram_range: Tuple[int, int] = (3, 5)):
        self.learning_rate = float(learning_rate)
        self.ngram_range = (max(1, int(ngram_range[0])), max(int(ngram_range[0]), int(ngram_range[1])))
        self.weights: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.bias: Dict[str, float] = defaultdict(float)
        self.class_counts: Counter[str] = Counter()
        self.total_updates = 0

    def _extract_features(self, text: str) -> Counter[str]:
        canonical = _canonicalize(text)
        features: Counter[str] = Counter()
        compact = re.sub(r"\s+", " ", canonical)
        min_n, max_n = self.ngram_range
        cleaned = re.sub(r"[^a-z0-9 !?]+", " ", compact)
        cleaned = re.sub(r"\s+", " ", cleaned)
        for n in range(min_n, max_n + 1):
            if n <= 0:
                continue
            for idx in range(0, len(cleaned) - n + 1):
                ngram = cleaned[idx:idx + n]
                if ngram.strip():
                    features[f"char:{ngram}"] += 1.0
        for token in re.findall(r"[a-z0-9]{2,}", canonical):
            features[f"tok:{token}"] += 1.0
        if "!" in text:
            features["has:exclam"] = 1.0
        if "?" in text:
            features["has:question"] = 1.0
        if RE_HAS_EMOJI.search(text):
            features["has:emoji"] = 1.0
        return features

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        if not self.class_counts:
            return None, 0.0
        feats = self._extract_features(text)
        scores: Dict[str, float] = {}
        for label in self.class_counts:
            score = self.bias[label]
            weights = self.weights[label]
            for feat, value in feats.items():
                score += weights.get(feat, 0.0) * value
            scores[label] = score
        if not scores:
            return None, 0.0
        max_score = max(scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        probabilities = {label: exp_val / total for label, exp_val in exp_scores.items()}
        label = max(probabilities.items(), key=lambda kv: kv[1])[0]
        return label, probabilities[label]

    def partial_fit(self, text: str, label: str) -> None:
        if not text or not label:
            return
        feats = self._extract_features(text)
        self.class_counts[label] += 1
        self.weights.setdefault(label, {})
        self.bias.setdefault(label, 0.0)
        scores: Dict[str, float] = {}
        for cls in self.class_counts:
            score = self.bias[cls]
            weights = self.weights[cls]
            for feat, value in feats.items():
                score += weights.get(feat, 0.0) * value
            scores[cls] = score
        max_score = max(scores.values()) if scores else 0.0
        exp_scores = {cls: math.exp(score - max_score) for cls, score in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        probabilities = {cls: exp_val / total for cls, exp_val in exp_scores.items()}
        lr = self.learning_rate
        for cls in self.class_counts:
            target = 1.0 if cls == label else 0.0
            prob = probabilities.get(cls, 0.0)
            error = target - prob
            self.bias[cls] += lr * error
            weights = self.weights.setdefault(cls, {})
            for feat, value in feats.items():
                weights[feat] = weights.get(feat, 0.0) + lr * error * value
        self.total_updates += 1


class SemanticUnderstanding:
    """
    Variante orientée dialogue avec suivi d'état et self-ask.

    Elle suppose la présence des sous-modules ``dialogue_state`` et
    ``frames`` et complète l'implémentation monolithique disponible via
    ``language.__init__`` en ajoutant la génération de questions et un
    suivi fin de l'incertitude.
    """

    def __init__(self, architecture=None, memory=None, intent_model=None, **_ignored):
        self.arch = architecture
        self.memory = memory
        self.state = DialogueState()
        if intent_model is None and architecture is not None:
            intent_model = getattr(architecture, "intent_model", None)
        self.intent_model = intent_model
        # Mini-lexique interne pour "mémoriser" des termes
        self.lexicon: Dict[str, Dict[str, Any]] = {}
        self._auto_patterns: deque = deque(maxlen=60)
        # Seuils
        self.min_intent_conf = 0.45
        self.self_ask_uncertainty = 0.35  # plus c'est grand, plus l'IA posera des questions
        self.intent_classifier = OnlineIntentClassifier()
        self.uncertainty_tracker = AdaptiveEMA()
        self._uncertainty_history: deque = deque(maxlen=64)
        self.uncertainty_correlation: Optional[float] = None
        self._last_llm_understanding: Optional[Mapping[str, Any]] = None

    # ---------- PUBLIC API ----------
    def _invoke_llm_understanding(
        self,
        text: str,
        context: Optional[Dict[str, Any]],
        norm: str,
        canonical: str,
    ) -> Optional[Mapping[str, Any]]:
        if not _llm_enabled():
            return None

        try:
            state_snapshot = self.state.to_dict()
        except Exception:
            state_snapshot = {}

        payload = {
            "utterance": text,
            "normalized": norm,
            "canonical": canonical,
            "dialogue_state": state_snapshot,
            "context": context or {},
            "recent_unknown_terms": list(self.state.user_profile.get("unknown_terms", []))[-10:],
        }

        try:
            response = _llm_manager().call_dict(
                "language_understanding",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM understanding unavailable", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        self._last_llm_understanding = dict(response)
        return self._last_llm_understanding

    def parse_utterance(self, text: str, context: Optional[Dict[str, Any]] = None) -> UtteranceFrame:
        context = context or {}
        raw = text or ""
        norm = _normalize(raw)
        canonical = _canonicalize(raw)

        llm_bundle = self._invoke_llm_understanding(raw, context, norm, canonical)
        llm_confidence = 0.0
        if llm_bundle:
            llm_confidence = _safe_float(
                llm_bundle.get("confidence", llm_bundle.get("confidence_score", 0.0)),
                0.0,
            )

        intent, conf = self._classify_intent(raw, norm, canonical)
        if llm_bundle:
            llm_intent = llm_bundle.get("intent")
            if isinstance(llm_intent, str) and llm_intent.strip():
                intent = llm_intent.strip()
                if llm_confidence <= 0.0:
                    llm_confidence = 0.72
                conf = max(conf, max(0.0, min(1.0, llm_confidence)))

        acts = self._guess_dialogue_acts(intent, norm)
        slots, unknowns = self._extract_slots(intent, norm, canonical)

        llm_needs_buffer: List[str] = []
        if llm_bundle:
            llm_slots = llm_bundle.get("slots")
            if isinstance(llm_slots, Mapping):
                for key, value in llm_slots.items():
                    if value is not None:
                        slots.setdefault(key, value)
            llm_entities = llm_bundle.get("entities")
            if isinstance(llm_entities, list) and llm_entities:
                slots.setdefault("entities", llm_entities)
            follow_ups = llm_bundle.get("follow_up_questions")
            if isinstance(follow_ups, list):
                for question in follow_ups:
                    if isinstance(question, str) and question.strip():
                        self.state.add_pending_question(question.strip())
            llm_unknowns = llm_bundle.get("unknown_terms")
            if isinstance(llm_unknowns, list):
                for term in llm_unknowns:
                    if isinstance(term, str) and term and term not in unknowns:
                        unknowns.append(term)
            llm_needs = llm_bundle.get("needs")
            if isinstance(llm_needs, list):
                llm_needs_buffer.extend(str(item) for item in llm_needs if isinstance(item, str))

        # Signaux d'incertitude
        uncertainty = self._compute_uncertainty(conf, unknowns, norm, canonical)

        # Enregistrer termes inconnus (pour apprentissage lexical)
        for t in unknowns:
            self._memorize_term(t)

        feedback_intent = context.get("expected_intent")
        if feedback_intent:
            self.intent_classifier.partial_fit(raw, feedback_intent)
        elif conf >= self.min_intent_conf:
            self.intent_classifier.partial_fit(raw, intent)

        frame = UtteranceFrame(
            text=raw,
            normalized_text=norm,
            intent=intent,
            confidence=round(conf, 3),
            uncertainty=round(uncertainty, 3),
            acts=acts,
            slots=slots,
            unknown_terms=list(unknowns),
            needs=[],
        )

        if llm_bundle:
            frame.meta["llm_understanding"] = llm_bundle
            canonical_query = llm_bundle.get("canonical_query")
            if isinstance(canonical_query, str) and canonical_query:
                frame.meta.setdefault("canonical_query", canonical_query)
            tone = llm_bundle.get("tone")
            if isinstance(tone, str) and tone:
                frame.meta.setdefault("tone", tone)
            if llm_needs_buffer:
                frame.needs.extend(llm_needs_buffer)

        # Mettre à jour l'état de dialogue
        self.state.update_with_frame(frame.to_dict())
        for t in unknowns:
            self.state.remember_unknown_term(t)

        # Besoins d'info (explicites)
        frame.needs = self._derive_needs(frame)
        if llm_needs_buffer:
            merged = list(frame.needs)
            merged.extend(item for item in llm_needs_buffer if item not in merged)
            frame.needs = merged

        return frame

    def respond(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Produit une réponse non-générique, introspective, basée sur la frame + self-ask si besoin."""
        frame = self.parse_utterance(text, context)

        # 1) Résumé de compréhension
        summary = self._summarize_understanding(frame)

        # 2) Questions auto-générées (self-ask) si incertitude
        questions = self._self_ask(frame)
        if not questions:
            # Si on a encore des questions en attente dans l'état, on en ressort 1-2
            questions = self.state.consume_pending_questions(2)

        # 3) Micro-plan d'action (optionnel)
        plan = self._suggest_next_action(frame)

        parts = [summary]
        if questions:
            parts.append("Pour avancer, j'ai besoin de précisions : " + " | ".join(f"• {q}" for q in questions))
        if plan:
            parts.append(f"Prochain pas (proposé) : {plan}")

        # --- RAG 5★ : réponse appuyée par sources ---
        try:
            arch = getattr(self, "arch", None) or getattr(self, "architecture", None) or getattr(getattr(self, "state", None), "architecture", None)
            if arch and getattr(arch, "rag", None):
                rag_context = None
                try:
                    if hasattr(arch, "prepare_rag_query"):
                        rag_context = arch.prepare_rag_query(text)
                except Exception:
                    rag_context = None
                rag_out = arch.rag.ask(text)
                try:
                    if hasattr(arch, "observe_rag_outcome"):
                        arch.observe_rag_outcome(text, rag_out, context=rag_context)
                except Exception:
                    pass
                if rag_out.get("status") == "ok":
                    parts.append("Réponse appuyée par sources :")
                    parts.append(rag_out["answer"])
                    cites = rag_out.get("citations") or []
                    if cites:
                        parts.append("Sources: " + " | ".join(f"[{c['doc_id']} {c['start']}–{c['end']}]" for c in cites[:3]))
                else:
                    parts.append("⚠️ Support insuffisant. Peux-tu préciser la question ou partager un lien ?")
        except Exception:
            pass

        return "\n".join(parts)

    def on_auto_intention_promoted(
        self,
        event: Mapping[str, Any],
        evaluation: Optional[Mapping[str, Any]] = None,
        self_assessment: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(event, Mapping):
            return
        requirements = event.get("requirements") or []
        for term in requirements:
            canonical = _canonicalize(str(term))
            if not canonical:
                continue
            self.lexicon.setdefault(
                canonical,
                {
                    "definition": event.get("description", ""),
                    "source": "auto_evolution",
                    "signals": event.get("signals", []),
                },
            )
        pattern = {
            "ts": time.time(),
            "action_type": event.get("action_type"),
            "score": (evaluation or {}).get("score"),
            "keywords": list(event.get("keywords", [])),
        }
        self._auto_patterns.append(pattern)
        auto_registry = getattr(self.state, "auto_intentions", None)
        if not isinstance(auto_registry, deque):
            auto_registry = deque(maxlen=50)
            setattr(self.state, "auto_intentions", auto_registry)
        auto_registry.append(pattern)
        if self_assessment and isinstance(self_assessment, Mapping):
            confidence_target = float(self_assessment.get("composite_target", 0.6) or 0.6)
            self.min_intent_conf = max(0.25, min(0.9, self.min_intent_conf - 0.05 * (confidence_target - 0.5)))

    # ---------- INTENT & ACTS ----------
    def _classify_intent(self, text: str, norm: Optional[str] = None, canonical: Optional[str] = None) -> Tuple[str, float]:
        norm_text = norm if norm is not None else _normalize(text or "")

        if self.intent_model and hasattr(self.intent_model, "predict"):
            try:
                intent, conf = self.intent_model.predict(text)
                return intent, float(conf)
            except Exception:
                pass

        base_intent, base_conf = IntentModel.rule_predict(norm_text)
        fallback_intent, fallback_conf = self.intent_classifier.predict(text)

        if fallback_intent and (fallback_conf > base_conf + 0.05 or (base_conf < self.min_intent_conf and fallback_conf >= base_conf)):
            return fallback_intent, min(0.99, max(base_conf, fallback_conf))

        if base_conf < self.min_intent_conf and fallback_intent and fallback_conf >= 0.45:
            blended = (base_conf + fallback_conf) / 2
            return fallback_intent, min(0.95, blended)

        return base_intent, base_conf

    def _guess_dialogue_acts(self, intent: str, norm: str) -> List[DialogueAct]:
        acts: List[DialogueAct] = []
        if intent == "greet":
            acts.append(DialogueAct.GREET)
        if intent == "thanks":
            acts.append(DialogueAct.THANKS)
        if intent == "bye":
            acts.append(DialogueAct.BYE)
        if intent in {"request", "create", "send", "plan", "summarize", "classify"}:
            acts.append(DialogueAct.REQUEST)
        if intent in {"ask", "ask_info"}:
            acts.append(DialogueAct.ASK)
        if intent == "meta_help":
            acts.append(DialogueAct.META_HELP)
        if intent == "set_goal":
            acts.append(DialogueAct.INFORM)
        if intent == "reflect":
            acts.append(DialogueAct.REFLECT)
        if not acts:
            # heuristique : phrase déclarative
            acts.append(DialogueAct.INFORM)
        return acts

    # ---------- SLOTS & UNKNOWN TERMS ----------
    def _extract_slots(self, intent: str, norm: str, canonical: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        slots: Dict[str, Any] = {}
        unknowns: List[str] = []
        canonical = canonical if canonical is not None else _canonicalize(norm)

        # termes entre guillemets → slot "quoted"
        quoted = re.findall(r"[\"""''](.*?)[\"""'']", norm)
        if quoted:
            slots["quoted"] = quoted[-1]

        # numéros / quantités
        numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", norm)
        if numbers:
            slots["numbers"] = numbers

        # terme après "définis/defini/definition de <concept>"
        mdef = re.search(r"(?:d[eé]finis?|d[eé]finition\s+de)\s+([a-z0-9\-_ ]{2,50})", norm, re.IGNORECASE)
        if mdef:
            slots["term_to_define"] = mdef.group(1).strip()

        # définitions implicites "X est un/une ..."
        for match in RE_TERM_DEFINITION.finditer(norm):
            term = match.group("term").strip(" ' \"")
            desc = match.group("desc").strip(" ' \"")
            if term:
                slots.setdefault("definition_candidates", []).append({
                    "term": term,
                    "description": desc,
                })
        if "definition_candidates" not in slots:
            quick = RE_QUICK_DEF.search(norm)
            if quick:
                desc = quick.group("desc").strip(" ' \"")
                slots.setdefault("definition_candidates", []).append({
                    "term": None,
                    "description": desc,
                })

        # objectifs : rudimentaire
        if intent in ("set_goal", "request") and "goal" not in slots:
            mg = re.search(r"(?:objectif|goal|but)\s*:\s*([a-z0-9\-_ ,]{3,120})", norm, re.IGNORECASE)
            if mg:
                slots["goal"] = mg.group(1).strip()

        # inconnus lexicaux simples : tokens + non présents dans lexicon
        seen_unknowns = set()
        for match in re.finditer(r"[a-zA-Zàâäéèêëîïôöùûüç0-9\-\_]{3,}", norm):
            token = match.group()
            token_canon = _canonicalize(token)
            if not token_canon or len(token_canon) > 24:
                continue
            if token_canon in STOPWORDS:
                continue
            if token_canon in self.lexicon or token_canon in seen_unknowns:
                continue
            seen_unknowns.add(token_canon)
            unknowns.append(token.lower())

        # éviter l'explosion : top 3
        unknowns = unknowns[:3]

        return slots, unknowns

    # ---------- INCERTITUDE, BESOINS, SELF-ASK ----------
    def _compute_uncertainty(self, intent_conf: float, unknowns: List[str], norm: str, canonical: str) -> float:
        u = max(0.0, 1.0 - intent_conf)
        if unknowns:
            u += 0.2 + 0.1 * (len(unknowns) - 1)
        if len(norm) < 4:
            u += 0.2
        if RE_HEDGES.search(norm):
            u += 0.15
        u = min(1.0, u)

        hedge_penalty = 0.1 if RE_HEDGES.search(norm) else 0.0
        target_quality = max(0.0, 1.0 - (0.35 * len(unknowns) + hedge_penalty + (0.2 if len(norm) < 4 else 0.0)))
        smoothed_conf = self.uncertainty_tracker.update(intent_conf, target_quality)
        smoothed_uncertainty = min(1.0, 1.0 - smoothed_conf)
        blended = min(1.0, max(0.0, 0.5 * u + 0.5 * smoothed_uncertainty))

        self._uncertainty_history.append((smoothed_conf, target_quality))
        xs = [x for x, _ in self._uncertainty_history]
        ys = [y for _, y in self._uncertainty_history]
        corr = _pearson(xs, ys)
        if corr is not None:
            self.uncertainty_correlation = corr

        return blended

    def _derive_needs(self, frame: UtteranceFrame) -> List[str]:
        needs: List[str] = []
        if frame.uncertainty > 0.6:
            needs.append("clarifier_objectif")
        if frame.unknown_terms:
            needs.append("definitions_termes")
        if frame.intent in ("request", "set_goal") and "goal" not in frame.slots:
            needs.append("preciser_goal")
        return needs

    def _self_ask(self, frame: UtteranceFrame) -> List[str]:
        qs: List[str] = []
        if frame.uncertainty >= self.self_ask_uncertainty:
            # Priorité aux inconnus
            for t in frame.unknown_terms:
                qs.append(f'Que signifie "{t}" dans ton contexte ?')
            # Clarif objectifs
            if "clarifier_objectif" in frame.needs:
                qs.append("Quel est le résultat concret que tu souhaites obtenir ?")
            if frame.intent == "request" and "quoted" in frame.slots:
                qs.append(f"Tu veux que j'agisse sur \"{frame.slots['quoted']}\" précisément ?")
        # Mémoriser 1-2 questions en attente
        for q in qs[:2]:
            self.state.add_pending_question(q)
        return qs[:2]

    # ---------- MÉMOIRE LEXICALE LOCALE ----------
    def _memorize_term(self, term: str):
        canonical = _canonicalize(term)
        if not canonical:
            return
        entry = self.lexicon.get(canonical)
        if not entry:
            self.lexicon[canonical] = {
                "forms": [term],
                "definition": None,
                "examples": [],
                "confidence": 0.0,
                "first_seen": self.state.turn_index,
                "last_seen": self.state.turn_index,
            }
        else:
            if term not in entry.get("forms", []):
                entry.setdefault("forms", []).append(term)
            entry["last_seen"] = self.state.turn_index

    # ---------- RÉSUMÉS & PLAN D'ACTION ----------
    def _summarize_understanding(self, frame: UtteranceFrame) -> str:
        parts: List[str] = []
        parts.append(f"Je pense que ton intention est **{frame.intent}** (confiance {frame.confidence:.2f}).")
        if frame.slots:
            expl = ", ".join(f"{k}={v}" for k, v in frame.slots.items())
            parts.append(f"J'ai identifié: {expl}.")
        if frame.unknown_terms:
            parts.append("J'ai des inconnus: " + ", ".join(f'"{t}"' for t in frame.unknown_terms) + ".")
        if frame.needs:
            parts.append("Besoins détectés: " + ", ".join(frame.needs) + ".")
        parts.append(f"Incertitude globale {frame.uncertainty:.2f} - je préfère vérifier avant d'agir.")
        return " ".join(parts)

    def _suggest_next_action(self, frame: UtteranceFrame) -> Optional[str]:
        # Micro plan cohérent avec l'intent
        if frame.intent == "greet":
            return "te saluer et te demander ton objectif prioritaire actuel."
        if frame.intent == "set_goal":
            return "créer/mettre à jour un objectif dans mon système si tu confirmes la formulation."
        if frame.intent == "request":
            return "décomposer la demande en étapes et vérifier que j'ai toutes les contraintes nécessaires."
        if frame.intent in {"ask", "ask_info"}:
            return "proposer une hypothèse courte puis demander validation."
        return None
