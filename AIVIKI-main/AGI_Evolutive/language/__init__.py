# language/__init__.py
"""
Module Language - auto-contenu, optimisé et complet pour ton AGI.
Contient :
  - SemanticUnderstanding  : parsing sémantique (frames, intents, slots, entities)
  - PragmaticReasoning     : contexte, intentions, implicatures, actes de langage
  - DiscourseProcessing    : gestion de la cohérence inter-tour, anaphores simples
  - LanguageGeneration     : génération textuelle contrôlée par but/tonalité/style

Objectifs :
  - AUCUN import de sous-fichier (fini les ModuleNotFoundError)
  - Auto-wiring doux via cognitive_architecture (getattr, sans import croisé)
  - Persistance (to_state / from_state) pour travailler offline et reprendre
  - Standard library only (pas de dépendances externes)

N.B. : Ce module est "optimisé mais complet" : toute la logique utile est là,
avec des implémentations compactes et robustes (pas de verbiage inutile).
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from AGI_Evolutive.models.intent import IntentModel
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)
LLM_SPEC_KEY = "language_semantic_understanding"

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
LANGUAGE_DATA_DIR = os.path.join(DATA_DIR, "language")
os.makedirs(LANGUAGE_DATA_DIR, exist_ok=True)
CLASSIFIER_STATE_PATH = os.path.join(LANGUAGE_DATA_DIR, "intent_classifier.json")
TRACKER_STATE_PATH = os.path.join(LANGUAGE_DATA_DIR, "intent_tracker.json")


# --- Compat: alias éventuels d'anciens namespaces ---
ALIASES = {
    "AGI_Evolutive.models.intent": "language.models.intent",
    "AGI_Evolutive.models.user": "language.models.user",
    "AGI_Evolutive.language.nlg": "language.nlg",
    "AGI_Evolutive.language.renderer": "language.renderer",
    "AGI_Evolutive.language.style_profiler": "language.style_policy",
    "AGI_Evolutive.social.tactic_selector": "language.social.tactic_selector",
    "AGI_Evolutive.social.social_critic": "language.style_critic",
    "AGI_Evolutive.core.structures.mai": "language.structures.mai",
}
for old, new in ALIASES.items():
    try:
        sys.modules.setdefault(old, importlib.import_module(new))
    except Exception:
        pass


# --- Utilitaires communs ---


def _json_load(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _json_save(path: str, obj: Any) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


# ============================================================
# Utilitaires
# ============================================================

def _now() -> float:
    return time.time()

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _mean(xs: List[float], default: float = 0.0) -> float:
    return sum(xs) / len(xs) if xs else default


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _normalize_text(text: str) -> str:
    return _strip_accents(text).lower()


class AdaptiveEMA:
    """EMA adaptative choisie par Thompson Sampling parmi plusieurs betas."""

    def __init__(self, betas: Optional[Iterable[float]] = None, max_step: float = 0.25):
        self.betas = tuple(sorted(betas or (0.2, 0.4, 0.6, 0.8)))
        if not self.betas:
            raise ValueError("At least one beta is required")
        self.max_step = float(max_step)
        self.state = {
            beta: {"value": 0.5, "alpha": 1.0, "beta": 1.0, "last_error": None}
            for beta in self.betas
        }
        self.selected_beta = self.betas[0]
        self.last_update_ts = 0.0
        self.drift_log: deque = deque(maxlen=50)

    def update(self, value: float, target: float) -> float:
        value = float(value)
        target = float(target)
        self.last_update_ts = _now()

        for beta in self.betas:
            s = self.state[beta]
            prev = s["value"]
            delta = value - prev
            step = beta * delta
            step = max(-self.max_step, min(self.max_step, step))
            new_val = prev + step
            error_prev = abs(target - prev) if s["last_error"] is None else s["last_error"]
            error_new = abs(target - new_val)
            reward = 1.0 if error_new < error_prev else 0.5 if math.isclose(error_new, error_prev, abs_tol=1e-6) else 0.0
            s["value"] = new_val
            s["last_error"] = error_new
            s["alpha"] += reward
            s["beta"] += 1.0 - reward
            if abs(step) > 0.15:
                self.drift_log.append({
                    "beta": beta,
                    "timestamp": self.last_update_ts,
                    "from": prev,
                    "to": new_val,
                    "target": target,
                })

        # Thompson sampling to select the most promising beta
        best_sample = -1.0
        chosen_beta = self.selected_beta
        for beta in self.betas:
            s = self.state[beta]
            sample = random.betavariate(s["alpha"], s["beta"])
            if sample > best_sample:
                best_sample = sample
                chosen_beta = beta
        self.selected_beta = chosen_beta
        return self.state[chosen_beta]["value"]

    def to_state(self) -> Dict[str, Any]:
        return {
            "betas": list(self.betas),
            "max_step": self.max_step,
            "state": {
                str(beta): {
                    "value": data["value"],
                    "alpha": data["alpha"],
                    "beta": data["beta"],
                    "last_error": data["last_error"],
                }
                for beta, data in self.state.items()
            },
            "selected_beta": self.selected_beta,
            "drift_log": list(self.drift_log),
        }

    def from_state(self, state: Dict[str, Any]):
        if not state:
            return
        betas = state.get("betas")
        if betas:
            self.betas = tuple(sorted(float(b) for b in betas))
        self.max_step = float(state.get("max_step", self.max_step))
        existing = state.get("state", {})
        for beta in self.betas:
            key = str(beta)
            data = existing.get(key)
            if data:
                slot = self.state.setdefault(beta, {"value": 0.5, "alpha": 1.0, "beta": 1.0, "last_error": None})
                slot.update({
                    "value": float(data.get("value", slot["value"])),
                    "alpha": float(data.get("alpha", slot["alpha"])),
                    "beta": float(data.get("beta", slot["beta"])),
                    "last_error": data.get("last_error", slot["last_error"]),
                })
        self.selected_beta = float(state.get("selected_beta", self.selected_beta))
        self.drift_log = deque(state.get("drift_log", []), maxlen=50)


class OnlineNgramClassifier:
    """Classifieur texte léger, entraînable en ligne avec des n-grammes de caractères."""

    def __init__(self, labels: Optional[Iterable[str]] = None, learning_rate: float = 0.8, max_features: int = 180):
        self.learning_rate = float(learning_rate)
        self.max_features = int(max_features)
        self.labels = set(labels or [])
        self.weights: Dict[str, defaultdict] = {label: defaultdict(float) for label in self.labels}
        self.biases: Dict[str, float] = {label: 0.0 for label in self.labels}
        self.class_counts: Dict[str, int] = {label: 0 for label in self.labels}
        self.last_update_ts: float = 0.0

    # ----- Features -----
    def _extract_features(self, text: str) -> Counter:
        normalized = _normalize_text(text)
        feats: Counter = Counter()
        tokens = normalized.split()
        for tok in tokens:
            max_n = min(3, len(tok))
            for n in range(1, max_n + 1):
                for i in range(len(tok) - n + 1):
                    feats[f"c{n}:{tok[i:i+n]}"] += 1
        feats["bias"] = 1
        feats["len"] = len(normalized)
        feats["has_question"] = 1 if "?" in text else 0
        feats["has_exclaim"] = 1 if "!" in text else 0
        feats["emoji"] = sum(1 for ch in text if ord(ch) > 0x1F000)
        if len(tokens) <= 6:
            feats["short"] = 1
        if any(tok.endswith("er") for tok in tokens):
            feats["verb_inf"] = 1
        if any(tok.startswith("svp") or tok.startswith("merci") for tok in tokens):
            feats["polite"] = 1
        # limit feature explosion
        if len(feats) > self.max_features:
            for key in list(feats.keys())[self.max_features:]:
                del feats[key]
        return feats

    # ----- API -----
    def predict(self, text: str) -> Tuple[str, float]:
        if not self.labels:
            return "inform", 0.4
        feats = self._extract_features(text)
        scores = {}
        for label in self.labels:
            weight = self.weights.setdefault(label, defaultdict(float))
            bias = self.biases.setdefault(label, 0.0)
            score = bias
            for feat, val in feats.items():
                score += weight.get(feat, 0.0) * val
            scores[label] = score

        max_score = max(scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        probs = {label: val / total for label, val in exp_scores.items()}
        best_label = max(probs, key=probs.get)
        return best_label, _clip(probs[best_label], 0.0, 1.0)

    def partial_fit(self, text: str, label: str):
        if not label:
            return
        self.labels.add(label)
        weight = self.weights.setdefault(label, defaultdict(float))
        self.biases.setdefault(label, 0.0)
        self.class_counts[label] = self.class_counts.get(label, 0) + 1
        feats = self._extract_features(text)
        pred_label, _ = self.predict(text)
        lr = self.learning_rate / max(1, self.class_counts[label])
        for feat, val in feats.items():
            if feat == "bias":
                continue
            weight[feat] += lr * val
            if pred_label != label:
                self.weights[pred_label][feat] -= lr * val
        self.biases[label] += lr
        if pred_label != label:
            self.biases[pred_label] -= lr
        self.last_update_ts = _now()

    # ----- Persistence -----
    def to_state(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_features": self.max_features,
            "labels": list(self.labels),
            "weights": {label: dict(weights) for label, weights in self.weights.items()},
            "biases": dict(self.biases),
            "class_counts": dict(self.class_counts),
            "last_update_ts": self.last_update_ts,
        }

    def from_state(self, state: Dict[str, Any]):
        if not state:
            return
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.max_features = int(state.get("max_features", self.max_features))
        self.labels = set(state.get("labels", self.labels))
        self.weights = {
            label: defaultdict(float, (state.get("weights", {}).get(label) or {}))
            for label in self.labels
        }
        self.biases = dict(state.get("biases", self.biases))
        self.class_counts = {label: int(val) for label, val in (state.get("class_counts", {}) or {}).items()}
        self.last_update_ts = float(state.get("last_update_ts", self.last_update_ts))


# ============================================================
# Types et structures de base
# ============================================================

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    canonical: Optional[str] = None

@dataclass
class Frame:
    intent: str
    slots: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.6

@dataclass
class Utterance:
    surface_form: str
    lang: str = "fr"
    tokens: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    frame: Optional[Frame] = None
    pragmatics: Dict[str, Any] = field(default_factory=dict)  # act, politeness, uncertainty
    timestamp: float = field(default_factory=_now)
    normalized_text: str = ""


# ============================================================
# 1) SemanticUnderstanding
# ============================================================

class SemanticUnderstanding:
    """
    Compréhension sémantique compacte :
     - tokenisation simple, NER par regex (dates, nombres, emails, urls, montants)
     - frame intent+slots : détecte objectifs fréquents (demander, informer, créer, planifier, envoyer)
     - calcul d'incertitude : hedges ("peut-être", "je crois"), tournures interrogatives

    Cette implémentation monolithique permet d'importer ``AGI_Evolutive.language``
    sans dépendances supplémentaires.  Le fichier
    :mod:`AGI_Evolutive.language.understanding` propose, lui, une variante
    plus modulaire centrée sur l'état de dialogue et les questions de
    clarification.
    """
    RE_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    RE_DATE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
    RE_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
    RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
    RE_MONEY = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:€|eur|euros|\$|usd)\b", re.I)

    HEDGES = {"peut-être", "je crois", "il me semble", "probablement", "possiblement", "éventuellement"}
    HEDGES_NORMALIZED = {_normalize_text(h) for h in HEDGES}
    INTENT_PATTERNS = [
        ("ask_info", re.compile(
            r"(?:"
            r"\b(?:quoi|quel(?:le|s)?|comment|pourquoi|combien|ou|quand)\b"
            r"|\b(?:est\s+(?:ce|ce\s+que)\s+(?:tu|vous|on|il|elle|nous|ils|elles|je)|peux?\s+(?:tu|vous)\s+(?:me\s+)?dire|serait\s+il\s+possible|aurais\s+tu|dis\s+moi)"
            r"|\?\s*$"
            r"|\b(?:donne|donne\s+moi)\s+plus\s+d?infos?\b"
            r")"
        )),
        ("create", re.compile(r"\b(?:cree|creer|ajoute|ajouter|enregistre|note|construis|fabrique|produis|genere)\b")),
        ("send", re.compile(r"\b(?:envoie|envoyer|partage|transmets?|forwarde|expedie)\b")),
        ("plan", re.compile(r"\b(?:planifie|planifier|prevois|organise|agenda|programme|roadmap|calendrier|preparer)\b")),
        ("inform", re.compile(r"\b(?:j\s?informe|voici|sache\s+que|pour\s+info|fyi)\b")),
        ("summarize", re.compile(r"\b(?:resume|resumer|synthese|synthetise|fait\s+un\s+resume|condense)\b")),
        ("classify", re.compile(r"\b(?:classe|categorise|tague|etiquette|est\s+(?:un|une|le|la|l)\s+(?:quel|quelle)\s+categorie)\b")),
    ]

    def __init__(
        self,
        cognitive_architecture: Any = None,
        memory_system: Any = None,
        intent_model: Optional[IntentModel] = None,
    ):
        self.cognitive_arch = cognitive_architecture
        self.arch = cognitive_architecture
        self.memory_system = memory_system
        self.memory = memory_system  # alias pour compatibilité/reflexive replies
        self.history: List[Utterance] = []
        self.lang = "fr"

        if intent_model is None and cognitive_architecture is not None:
            intent_model = getattr(cognitive_architecture, "intent_model", None)
        self.intent_model: IntentModel = intent_model or IntentModel()

        classifier_state = _json_load(CLASSIFIER_STATE_PATH, {})
        model_labels = getattr(self.intent_model, "LABELS", None)
        self.intent_classifier = OnlineNgramClassifier(labels=model_labels)
        self.intent_classifier.from_state(classifier_state)
        tracker_state = _json_load(TRACKER_STATE_PATH, {})
        self.intent_conf_tracker = AdaptiveEMA()
        self.intent_conf_tracker.from_state(tracker_state)
        self._classifier_state_dirty = False
        self._tracker_state_dirty = False
        self._last_persist_ts = _now()

        # auto-wiring : accès doux aux autres modules si dispos
        ca = self.cognitive_arch
        if ca:
            self.reasoning = getattr(ca, "reasoning", None)
            self.goals = getattr(ca, "goals", None)
            self.emotions = getattr(ca, "emotions", None)
            self.metacognition = getattr(ca, "metacognition", None)
            self.world_model = getattr(ca, "world_model", None)
            self.perception = getattr(ca, "perception", None)
            self.creativity = getattr(ca, "creativity", None)

    # --------- Pipeline principal ---------

    def parse_utterance(self, text: str, context: Optional[Dict[str, Any]] = None) -> Utterance:
        context = context or {}
        toks = self._tokenize(text)
        ents = self._ner(text)
        normalized = re.sub(r"\s+", " ", _normalize_text(text)).strip()
        frame = self._frame(text, toks, ents, normalized)
        prag = self._pragmatics(text, toks, normalized)

        quality = self._estimate_intent_quality(frame, prag)
        smoothed = self.intent_conf_tracker.update(frame.confidence, quality)
        if not math.isclose(smoothed, frame.confidence, abs_tol=1e-6):
            self._tracker_state_dirty = True
        frame.confidence = _clip((frame.confidence + smoothed) / 2.0, 0.0, 1.0)
        prag["uncertainty"] = max(prag.get("uncertainty", 0.0), 1.0 - frame.confidence)

        utt = Utterance(
            surface_form=text,
            lang=self.lang,
            tokens=toks,
            entities=ents,
            frame=frame,
            pragmatics=prag,
            normalized_text=normalized,
        )
        self.history.append(utt)
        if len(self.history) > 200:
            self.history.pop(0)
        # informer la méta du niveau d'incertitude
        if getattr(self, "metacognition", None) and hasattr(
            self.metacognition, "register_language_parse"
        ):
            try:
                self.metacognition.register_language_parse(
                    utt.frame.confidence if utt.frame else 0.3,
                    prag.get("uncertainty", 0.0),
                )
            except Exception:
                pass
        self._persist_learning_state()
        return utt

    def respond(self, user_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Produit une réponse textuelle tout en fournissant un résumé de l'analyse."""
        context = context or {}
        parsed = self.parse_utterance(user_text, context)
        frame = getattr(parsed, "frame", None)

        arch = getattr(self, "arch", None) or getattr(self, "cognitive_arch", None)
        reasoner = getattr(arch, "reasoning", None) if arch else None
        if reasoner is None:
            reasoner = getattr(self, "reasoning", None)

        reasoned = None
        if reasoner and hasattr(reasoner, "reason"):
            try:
                question = (
                    getattr(parsed, "surface_form", None)
                    or getattr(frame, "normalized_text", None)
                    or user_text
                )
                reasoned = reasoner.reason(
                    question,
                    context={"frame_intent": getattr(frame, "intent", "")},
                )
            except Exception:
                reasoned = None

        contexts = self._retrieve_context(parsed)

        if frame is None and isinstance(parsed, Frame):
            frame = parsed

        intent = getattr(frame, "intent", "inform")
        confidence = float(getattr(frame, "confidence", 0.5) or 0.0)
        slots = getattr(frame, "slots", {}) or {}

        slot_bits: List[str] = []
        for key, val in slots.items():
            if isinstance(val, (list, tuple)):
                slot_bits.append(f"{key}=" + ", ".join(str(v) for v in val))
            else:
                slot_bits.append(f"{key}={val}")

        summary = f"Intent détecté: {intent} (confiance {confidence:.2f})."
        if slot_bits:
            summary += " Slots: " + "; ".join(slot_bits)

        context_lines: List[str] = []
        if contexts:
            for c in contexts:
                label = c.get("title") or "mémoire"
                context_lines.append(f"• ({c['score']:.2f}) {label}: {c['snippet']}")

        contextualised_summary_parts = [summary]
        if context_lines:
            contextualised_summary_parts.append(
                "Contexte pertinent retrouvé:\n" + "\n".join(context_lines)
            )
        contextualised_summary = "\n\n".join(contextualised_summary_parts)

        if reasoned:
            steps_lines = [
                f"- {t['strategy']}: {t['notes']}"
                for t in reasoned.get("trace", [])
                if t.get("notes")
            ]
            steps = "\n".join(steps_lines) if steps_lines else "- (aucune note)"
            support = reasoned.get("support", [])
            support_lines = (
                "\nContexte mémoire:\n" + "\n".join([f"• {s}" for s in support])
            ) if support else ""
            meta = reasoned.get("meta") or {}
            confidence = float(reasoned.get("confidence", 0.0))
            complexity = float(meta.get("complexity", 0.0))
            answer = (
                f"{reasoned.get('answer', '')}\n"
                f"(confiance ~ {confidence:.2f}, complexité {complexity:.2f})\n"
                f"Démarche:\n{steps}{support_lines}"
            )
            if contextualised_summary:
                return f"{answer}\n\n{contextualised_summary}"
            return answer

        fallback = getattr(parsed, "surface_form", user_text)
        base_response = f"Reçu: {fallback}"
        if contextualised_summary:
            return f"{base_response}\n\n{contextualised_summary}"
        return base_response

    def _retrieve_context(self, frame) -> List[dict]:
        """
        Cherche 3 éléments pertinents (interactions + docs) à partir du texte normalisé
        et des slots importants. Tolérant si memory absent.
        """
        results: List[dict] = []
        try:
            arch = getattr(self, "arch", None)
            mem = getattr(self, "memory_system", None) or (getattr(arch, "memory", None) if arch else None)
            retr = getattr(mem, "retrieval", None)
            if not retr:
                return results

            q = getattr(frame, "normalized_text", "") or getattr(frame, "surface_form", "")
            extras: List[str] = []

            slots_obj = getattr(frame, "slots", None)
            if slots_obj is None and getattr(frame, "frame", None):
                slots_obj = getattr(frame.frame, "slots", None)

            if isinstance(slots_obj, dict):
                for k in ("quoted", "goal", "term_to_define"):
                    if k in slots_obj:
                        val = slots_obj[k]
                        if isinstance(val, (list, tuple)):
                            extras.extend([str(v) for v in val])
                        else:
                            extras.append(str(val))
            if extras:
                q = (q + " " + " ".join(extras)).strip()

            if not q:
                return results

            hits = retr.search_text(q, top_k=3)
            for h in hits:
                txt = h.get("text", "")
                meta = h.get("meta", {})
                snippet = (txt[:180] + "…") if len(txt) > 180 else txt
                title = meta.get("title") or meta.get("source") or meta.get("type", "")
                results.append({
                    "score": h.get("score", 0.0),
                    "title": title,
                    "snippet": snippet,
                })
        except Exception:
            return []
        return results

    def generate_reflective_reply(self, arch, user_msg: str) -> str:
        """Génère une réponse réflexive courte et lisible."""
        status = {}
        try:
            status = arch.get_cognitive_status()
        except Exception:
            status = {}

        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", user_msg.lower())
        unknown = []
        try:
            memory = getattr(self, "memory", None) or getattr(self, "memory_system", None)
            if memory and hasattr(memory, "knows"):
                unknown = [t for t in tokens if not memory.knows(t)]
            elif memory and hasattr(memory, "retrieve"):
                for t in tokens:
                    try:
                        res = memory.retrieve(t, top_k=1)
                        if not res:
                            unknown.append(t)
                    except Exception:
                        pass
            else:
                unknown = [t for t in tokens if len(t) > 12]
        except Exception:
            pass
        unknown = list(dict.fromkeys(unknown))[:3]

        reasoning = status.get("reasoning", {})
        creativity = status.get("creativity", {})
        metacog = status.get("metacognition", {})
        activation = status.get("global_activation", 0.5)

        doing = []
        if reasoning.get("recent_inferences", 0) > 0:
            doing.append("j'analyse des inférences récentes")
        if creativity.get("recent_ideas", 0) > 0:
            doing.append("je génère/évalue de nouvelles idées")
        if metacog.get("events", 0) > 0:
            doing.append("je surveille mes performances et erreurs")
        if not doing:
            doing.append("je collecte des repères pour mieux te comprendre")

        confusion = []
        if unknown:
            confusion.append("je ne suis pas sûr de bien cerner: " + ", ".join(unknown))
        avg_conf = reasoning.get("avg_confidence", 0.5)
        if isinstance(avg_conf, (int, float)) and avg_conf < 0.45:
            confusion.append("ma confiance de raisonnement est un peu basse sur ce sujet")

        next_steps = []
        if unknown:
            next_steps.append("tu peux m'expliquer ce que tu entends par " + ", ".join(unknown) + " ?")
        else:
            next_steps.append("je peux tenter un exemple concret ou reformuler si tu veux")
        if isinstance(activation, (int, float)) and activation < 0.4:
            next_steps.append("je vais réduire la complexité pour m'aligner")

        lines = []
        lines.append("• Ce que je fais: " + "; ".join(doing))
        if confusion:
            lines.append("• Ce que je ne comprends pas: " + "; ".join(confusion))
        lines.append("• Ce que je propose: " + "; ".join(next_steps))

        try:
            gist = user_msg.strip()
            if len(gist) > 120:
                gist = gist[:117] + "…"
            lines.append(f"• Ta demande (j'ai bien reçu): \"{gist}\"")
        except Exception:
            pass

        return "\n".join(lines)

    # --------- Étapes internes ---------

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE) if t.strip()]

    def _ner(self, text: str) -> List[Entity]:
        ents: List[Entity] = []
        for r, label in ((self.RE_DATE, "DATE"), (self.RE_EMAIL, "EMAIL"), (self.RE_URL, "URL"),
                         (self.RE_MONEY, "MONEY"), (self.RE_NUMBER, "NUMBER")):
            for m in r.finditer(text):
                ents.append(Entity(text=m.group(0), label=label, start=m.start(), end=m.end()))
        return ents

    def _frame(self, text: str, toks: List[str], ents: List[Entity], normalized: str) -> Frame:
        intent = "inform"
        conf = 0.55
        if self.intent_model and hasattr(self.intent_model, "predict"):
            try:
                intent, conf = self.intent_model.predict(text)
            except Exception:
                intent, conf = IntentModel.rule_predict(text)
        else:
            intent, conf = IntentModel.rule_predict(text)

        try:
            conf = float(conf)
        except Exception:
            conf = 0.55
        conf = _clip(conf, 0.0, 1.0)

        norm_plain = re.sub(r"[^a-z0-9\s\?\!]", " ", normalized)
        norm_plain = re.sub(r"\s+", " ", norm_plain)

        heuristic_intent = intent
        heuristic_conf = conf
        for label, pattern in self.INTENT_PATTERNS:
            if pattern.search(norm_plain):
                heuristic_intent = label
                heuristic_conf = max(heuristic_conf, 0.72 if label != "inform" else heuristic_conf)
                break

        classifier_intent, classifier_conf = self.intent_classifier.predict(text)
        used_classifier = False
        if (
            classifier_conf >= heuristic_conf + 0.05
            or (heuristic_intent == "inform" and classifier_conf >= 0.55)
        ):
            intent = classifier_intent
            conf = max(conf, classifier_conf)
            used_classifier = True
        else:
            intent = heuristic_intent
            conf = max(conf, heuristic_conf)

        if intent != "inform" and conf >= 0.7:
            self.intent_classifier.partial_fit(text, intent)
            self._classifier_state_dirty = True
        elif used_classifier and classifier_conf >= 0.65:
            self.intent_classifier.partial_fit(text, classifier_intent)
            self._classifier_state_dirty = True

        slots: Dict[str, Any] = {}
        for e in ents:
            if e.label in ("DATE", "MONEY", "URL", "EMAIL", "NUMBER"):
                slots.setdefault(e.label.lower() + "s", []).append(e.text)
        topic_match = re.search(r"\b(?:sur|a propos de|concernant)\s+(.+)$", norm_plain)
        if topic_match:
            slots["topic"] = topic_match.group(1).strip()[:120]

        frame = Frame(intent=intent, slots=slots, confidence=_clip(conf, 0.0, 1.0))
        self._llm_refine_frame(
            frame,
            text=text,
            normalized=normalized,
            tokens=toks,
            entities=ents,
        )
        return frame

    def _pragmatics(self, text: str, toks: List[str], normalized: str) -> Dict[str, Any]:
        stripped = text.strip()
        act = "statement"
        if stripped.endswith("?") or re.search(r"\b(peux\s+tu|pourrais\s+tu|est\s+ce\s+que)\b", normalized):
            act = "question"
        elif stripped.endswith("!"):
            act = "exclaim"
        polite = any(p in normalized for p in ("sil te plait", "svp", "merci"))
        hedge_hits = sum(1 for h in self.HEDGES_NORMALIZED if h in normalized)
        uncertainty = min(0.6, 0.2 * hedge_hits)
        return {"speech_act": act, "politeness": polite, "uncertainty": uncertainty}

    def _estimate_intent_quality(self, frame: Frame, prag: Dict[str, Any]) -> float:
        intent = getattr(frame, "intent", "inform")
        act = prag.get("speech_act", "statement")
        slots = getattr(frame, "slots", {}) or {}
        quality = 0.5

        if act == "question":
            quality = 0.9 if intent == "ask_info" else 0.3
        elif intent in {"plan", "create"}:
            if any(slots.get(key) for key in ("dates", "numbers")):
                quality = 0.75
            else:
                quality = 0.6
        elif intent == "send":
            quality = 0.7 if slots.get("emails") or slots.get("urls") else 0.5
        elif intent == "summarize":
            quality = 0.65
        elif intent == "classify":
            quality = 0.6 if slots else 0.55
        elif intent == "inform":
            quality = 0.5

        if prag.get("politeness") and intent == "ask_info":
            quality = min(1.0, quality + 0.05)

        hedge_uncertainty = prag.get("uncertainty", 0.0)
        if hedge_uncertainty >= 0.4:
            quality = max(0.2, quality - 0.15)
        return _clip(quality, 0.0, 1.0)

    def _llm_refine_frame(
        self,
        frame: Frame,
        *,
        text: str,
        normalized: str,
        tokens: List[str],
        entities: List[Entity],
    ) -> None:
        payload = {
            "text": text,
            "normalized": normalized,
            "heuristic_intent": frame.intent,
            "heuristic_confidence": frame.confidence,
            "tokens": tokens[:64],
            "entities": [
                {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                for e in entities[:16]
            ],
        }
        response = try_call_llm_dict(LLM_SPEC_KEY, input_payload=payload, logger=LOGGER)
        if not isinstance(response, dict):
            return

        new_intent = response.get("intent")
        if isinstance(new_intent, str) and new_intent:
            frame.intent = new_intent

        new_conf = response.get("confidence")
        if isinstance(new_conf, (int, float)):
            frame.confidence = _clip(float(new_conf), 0.0, 1.0)

        slots = response.get("slots")
        if isinstance(slots, dict):
            frame.slots.update(slots)

        notes = response.get("notes")
        if notes:
            frame.slots.setdefault("_llm_notes", str(notes))

    def _persist_learning_state(self, force: bool = False):
        if not (self._classifier_state_dirty or self._tracker_state_dirty or force):
            if force:
                self._last_persist_ts = _now()
            return
        if not force and (_now() - self._last_persist_ts) < 2.0:
            return
        if self._classifier_state_dirty:
            _json_save(CLASSIFIER_STATE_PATH, self.intent_classifier.to_state())
            self._classifier_state_dirty = False
        if self._tracker_state_dirty:
            _json_save(TRACKER_STATE_PATH, self.intent_conf_tracker.to_state())
            self._tracker_state_dirty = False
        self._last_persist_ts = _now()

    # --------- Persistance ---------

    def to_state(self) -> Dict[str, Any]:
        self._persist_learning_state(force=True)
        return {
            "lang": self.lang,
            "history": [{
                "surface": u.surface_form,
                "lang": u.lang,
                "tokens": u.tokens,
                "entities": [e.__dict__ for e in u.entities],
                "frame": u.frame.__dict__ if u.frame else None,
                "pragmatics": u.pragmatics,
                "timestamp": u.timestamp,
                "normalized_text": u.normalized_text,
            } for u in self.history[-100:]],
            "intent_classifier": self.intent_classifier.to_state(),
            "intent_tracker": self.intent_conf_tracker.to_state(),
        }

    def from_state(self, state: Dict[str, Any]):
        self.lang = state.get("lang", "fr")
        self.history = []
        for d in state.get("history", []):
            ents = [Entity(**e) for e in d.get("entities", [])]
            fr = d.get("frame")
            fr = Frame(**fr) if fr else None
            self.history.append(Utterance(
                surface_form=d.get("surface", ""),
                lang=d.get("lang", "fr"),
                tokens=d.get("tokens", []),
                entities=ents,
                frame=fr,
                pragmatics=d.get("pragmatics", {}),
                timestamp=d.get("timestamp", _now()),
                normalized_text=d.get("normalized_text", ""),
            ))
        classifier_state = state.get("intent_classifier")
        if classifier_state:
            self.intent_classifier.from_state(classifier_state)
        tracker_state = state.get("intent_tracker")
        if tracker_state:
            self.intent_conf_tracker.from_state(tracker_state)
        self._classifier_state_dirty = False
        self._tracker_state_dirty = False
        self._last_persist_ts = _now()


# ============================================================
# 2) PragmaticReasoning
# ============================================================

class PragmaticReasoning:
    """
    Raisonner au-delà du littéral :
     - infère l'intention et les présupposés (ex : "tu peux… ?" = requête)
     - ajuste au contexte (état émotionnel, objectifs actifs, normes sociales)
     - propose des actes de langage appropriés (répondre, demander précision, proposer plan)
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.context: Dict[str, Any] = {
            "tone": "neutral",
            "formality": 0.5,
            "cooperation": 0.7,
            "confidence": 0.6,
        }
        # auto-wiring
        ca = self.cognitive_arch
        if ca:
            self.emotions = getattr(ca, "emotions", None)
            self.goals = getattr(ca, "goals", None)
            self.world_model = getattr(ca, "world_model", None)
            self.metacognition = getattr(ca, "metacognition", None)

    def infer_intent(self, utt: Utterance) -> Dict[str, Any]:
        # base sur frame + ajustements pragm.
        intent = utt.frame.intent if utt.frame else "inform"
        act = utt.pragmatics.get("speech_act", "statement")
        uncertainty = utt.pragmatics.get("uncertainty", 0.0)
        # Implicature simple : question indirecte
        if re.search(r"\b(peux-tu|pourrais-tu|tu peux|tu pourrais)\b", utt.surface_form.lower()):
            intent, act = "request", "question"
        # Ajustement par émotions globales si dispo
        if getattr(self, "emotions", None) and hasattr(self.emotions, "current_valence"):
            try:
                val = float(self.emotions.current_valence)
                self.context["tone"] = "upbeat" if val > 0.3 else ("down" if val < -0.3 else "neutral")
            except Exception:
                pass
        return {"intent": intent, "speech_act": act, "uncertainty": uncertainty, "context": dict(self.context)}

    def next_action(self, pragmatic: Dict[str, Any]) -> str:
        intent, act, uncert = pragmatic["intent"], pragmatic["speech_act"], pragmatic["uncertainty"]
        if act == "question":
            return "answer_or_ask_clarification"
        if intent in ("create", "plan", "send"):
            return "propose_plan"
        if uncert > 0.3:
            return "ask_clarification"
        return "answer"

    def to_state(self) -> Dict[str, Any]:
        return {"context": dict(self.context)}

    def from_state(self, state: Dict[str, Any]):
        self.context.update(state.get("context", {}))


# ============================================================
# 3) DiscourseProcessing
# ============================================================

@dataclass
class DiscourseState:
    last_entities: List[Entity] = field(default_factory=list)
    last_topics: List[str] = field(default_factory=list)
    turn_index: int = 0

class DiscourseProcessing:
    """
    Gestion de discours minimale :
     - suit les entités/objets récurrents (anaphores très simples)
     - maintient un "topic stack" de 5 éléments max
     - fournit contexte de réponse ("réutiliser la dernière URL", etc.)
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.state = DiscourseState()

    def update(self, utt: Utterance):
        self.state.turn_index += 1
        ents = [e for e in utt.entities if e.label in ("URL", "EMAIL", "MONEY", "DATE", "NUMBER")]
        topics = []
        if utt.frame and "topic" in utt.frame.slots:
            topics.append(utt.frame.slots["topic"])
        # push
        self.state.last_entities = (self.state.last_entities + ents)[-10:]
        self.state.last_topics = (self.state.last_topics + topics)[-5:]

    def resolve_reference(self, text: str) -> Dict[str, Any]:
        # "celui-ci", "ça", "ce document", "la même url", etc.
        t = text.lower()
        ref_ent = None
        if "même url" in t or "la même url" in t:
            ref_ent = next((e for e in reversed(self.state.last_entities) if e.label == "URL"), None)
        return {"resolved_entity": ref_ent.__dict__ if ref_ent else None, "topic": (self.state.last_topics[-1] if self.state.last_topics else None)}

    def to_state(self) -> Dict[str, Any]:
        return {
            "turn_index": self.state.turn_index,
            "last_entities": [e.__dict__ for e in self.state.last_entities],
            "last_topics": list(self.state.last_topics),
        }

    def from_state(self, state: Dict[str, Any]):
        self.state.turn_index = int(state.get("turn_index", 0))
        self.state.last_entities = [Entity(**e) for e in state.get("last_entities", [])]
        self.state.last_topics = list(state.get("last_topics", []))


# ============================================================
# 4) LanguageGeneration
# ============================================================

class LanguageGeneration:
    """
    Génération textuelle pragmatique :
     - templates intelligents + mix lexical (tonalité, style, concision)
     - prise en compte de l'incertitude et du besoin de clarification
     - formats utilitaires : listes, plans d'action, réponses directes
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.style = {"formality": 0.5, "warmth": 0.6, "conciseness": 0.7}
        # auto-wiring
        ca = self.cognitive_arch
        if ca:
            self.language = getattr(ca, "language", None)  # self-ref possible
            self.goals = getattr(ca, "goals", None)
            self.metacognition = getattr(ca, "metacognition", None)

    def reply(self, intent: str, data: Dict[str, Any], pragmatic: Dict[str, Any]) -> str:
        act = pragmatic.get("speech_act", "statement")
        ctx = pragmatic.get("context", {})
        tone = ctx.get("tone", "neutral")
        uncertainty = pragmatic.get("uncertainty", 0.0)
        topic = data.get("topic")
        # Stratégies simples mais efficaces
        if intent == "ask_info" or act == "question":
            return self._answer_question(data, tone=tone)
        if intent in ("create", "plan", "send"):
            return self._propose_plan(intent, data, tone=tone)
        if uncertainty > 0.3:
            return self._ask_clarification(topic, tone=tone)
        # par défaut : réponse informative
        return self._inform(data, tone=tone)

    # ----- patterns -----

    def _answer_question(self, data: Dict[str, Any], tone: str = "neutral") -> str:
        topic = data.get("topic")
        if topic:
            base = f"Pour {topic}, voici ce que je vois."
        else:
            base = "Voici ce que je peux te dire."
        hints = data.get("hints", [])
        if hints:
            base += " " + " ".join(hints[:3])
        return self._tone(base, tone)

    def _propose_plan(self, intent: str, data: Dict[str, Any], tone: str = "neutral") -> str:
        topic = data.get("topic", "la tâche")
        steps = data.get("steps") or ["Clarifier l'objectif", "Rassembler les données utiles", "Proposer une ébauche", "Itérer avec ton feedback"]
        head = f"Je te propose un mini-plan pour {topic} :"
        bullet = "\n- " + "\n- ".join(steps[:6])
        return self._tone(head + bullet, tone)

    def _ask_clarification(self, topic: Optional[str], tone: str = "neutral") -> str:
        if topic:
            msg = f"Peux-tu préciser les contraintes principales pour {topic} ?"
        else:
            msg = "Peux-tu préciser le contexte ou les contraintes principales ?"
        return self._tone(msg, tone)

    def _inform(self, data: Dict[str, Any], tone: str = "neutral") -> str:
        parts = []
        if "topic" in data:
            parts.append(f"À propos de {data['topic']},")
        if "summary" in data:
            parts.append(str(data["summary"]))
        elif "text" in data:
            parts.append(str(data["text"])[:300])
        else:
            parts.append("c'est noté.")
        return self._tone(" ".join(parts), tone)

    # ----- utils -----

    def _tone(self, text: str, tone: str) -> str:
        if tone == "upbeat":
            return "🙂 " + text
        if tone == "down":
            return "… " + text
        return text

    def to_state(self) -> Dict[str, Any]:
        return {"style": dict(self.style)}

    def from_state(self, state: Dict[str, Any]):
        self.style.update(state.get("style", {}))


# ============================================================
# Export public
# ============================================================

__all__ = [
    "SemanticUnderstanding",
    "LanguageGeneration",
    "PragmaticReasoning",
    "DiscourseProcessing",
    "Utterance",
    "Frame",
    "Entity",
]
