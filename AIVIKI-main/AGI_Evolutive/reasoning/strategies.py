from __future__ import annotations

import abc
import logging
import math
import time
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import re

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


class ReasoningStrategy(abc.ABC):
    """Interface minimale pour une stratégie de raisonnement.

    Toutes les stratégies doivent retourner un dictionnaire respectant le contrat
    documenté ci-dessous. La méthode :meth:`apply` reste abstraite pour forcer
    les sous-classes à fournir une implémentation explicite et éviter les
    silences en production.
    """

    name: str = "base"
    REQUIRED_FIELDS = {"notes", "proposals", "questions", "cost", "time"}

    @abc.abstractmethod
    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute la stratégie et retourne un dictionnaire normalisé.

        Le dictionnaire doit contenir au minimum les clés suivantes :

        * ``notes`` (``str``)
        * ``proposals`` (``List[Dict[str, Any]]``)
        * ``questions`` (``List[str]``)
        * ``cost`` (``float``)
        * ``time`` (``float``)
        """

    def validate_output(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        """Valide la sortie d'une stratégie et renvoie une copie mutable.

        Lève une ``ValueError`` si une clé obligatoire est absente afin de
        faciliter le débogage en cas de régression.
        """

        missing = self.REQUIRED_FIELDS.difference(result.keys())
        if missing:
            raise ValueError(
                f"La stratégie '{self.name}' doit retourner les clés {self.REQUIRED_FIELDS}, "
                f"clés manquantes: {sorted(missing)}"
            )
        out = dict(result)
        # Garantit des types simples pour les champs principaux.
        out["notes"] = str(out.get("notes", ""))
        out.setdefault("proposals", [])
        out.setdefault("questions", [])
        out.setdefault("cost", 0.0)
        out.setdefault("time", 0.0)
        return out


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "").lower()
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", _normalize_text(text))


class OnlineIntentClassifier:
    """Classifieur texte léger mis à jour en ligne.

    Il sert de repli pour générer des questions pertinentes lorsque les règles
    déterministes ne suffisent pas.
    """

    LABEL_QUESTIONS = {
        "definition": ["Quels éléments définissent précisément le concept ?"],
        "process": ["Quelles étapes composent le processus ?"],
        "cause": ["Quelles sont les causes ou facteurs déclencheurs ?"],
        "decision": ["Quels critères permettent de trancher ?"],
    }

    def __init__(self) -> None:
        self._weights: Dict[str, defaultdict[str, float]] = {
            label: defaultdict(float) for label in self.LABEL_QUESTIONS
        }
        self._bias: Dict[str, float] = {label: 0.0 for label in self.LABEL_QUESTIONS}

    def _features(self, text: str) -> Dict[str, float]:
        tokens = _tokenize(text)
        feats: Dict[str, float] = defaultdict(float)
        for token in tokens:
            feats[f"tok::{token}"] += 1.0
        for i in range(len(tokens) - 1):
            feats[f"bi::{tokens[i]}_{tokens[i+1]}"] += 1.0
        # Signaux légers : ponctuation / longueur
        length_bucket = f"len::{len(text.strip()) // 20}"
        feats[length_bucket] += 1.0
        if "?" in text:
            feats["has_question"] = 1.0
        if "!" in text:
            feats["has_exclamation"] = 1.0
        return feats

    def _score(self, feats: Mapping[str, float], label: str) -> float:
        weight = self._weights[label]
        total = self._bias[label]
        total += sum(weight[f] * value for f, value in feats.items())
        return total

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        feats = self._features(text)
        scores = {label: self._score(feats, label) for label in self.LABEL_QUESTIONS}
        best_label, raw_score = max(scores.items(), key=lambda item: item[1])
        # Normalisation douce via sigmoïde
        confidence = 1.0 / (1.0 + math.exp(-raw_score))
        return best_label, confidence, feats

    def update(self, text: str, label: str, lr: float = 0.1) -> None:
        if label not in self.LABEL_QUESTIONS:
            return
        predicted, _, feats = self.predict(text)
        target = 1.0
        for lbl in self.LABEL_QUESTIONS:
            tgt = target if lbl == label else 0.0
            score = self._score(feats, lbl)
            pred = 1.0 / (1.0 + math.exp(-score))
            error = tgt - pred
            for feature, value in feats.items():
                self._weights[lbl][feature] += lr * error * value
            self._bias[lbl] += lr * error
        logger.debug("Intent classifier updated towards %s (predicted %s)", label, predicted)


class OnlineWeightedScorer:
    """Petite régression logistique en ligne pour pondérer les hypothèses."""

    def __init__(self) -> None:
        self._weights: Dict[str, float] = defaultdict(float)
        self._bias: float = 0.2  # encourage légèrement les synthèses appuyées

    def featurize(
        self,
        prompt: str,
        support: Iterable[str],
        proposal: Mapping[str, Any],
    ) -> Dict[str, float]:
        support_list = list(support)
        features: Dict[str, float] = {
            "support_count": float(len(support_list)),
            "support_chars": float(sum(len(s) for s in support_list)) / 100.0,
            "answer_chars": float(len((proposal.get("answer") or "").strip())) / 80.0,
            "has_support": 1.0 if support_list else 0.0,
            "question_marks": (proposal.get("answer") or "").count("?") * -0.5,
        }
        prompt_tokens = _tokenize(prompt)
        features["prompt_len"] = float(len(prompt_tokens)) / 10.0
        if "plan" in prompt_tokens:
            features["topic_plan"] = 1.0
        if "pourquoi" in prompt_tokens or "why" in prompt_tokens:
            features["topic_cause"] = 1.0
        return features

    def _linear(self, features: Mapping[str, float]) -> float:
        score = self._bias
        for key, value in features.items():
            score += self._weights[key] * value
        return score

    def score(self, features: Mapping[str, float]) -> float:
        return 1.0 / (1.0 + math.exp(-self._linear(features)))

    def update(self, features: Mapping[str, float], accepted: bool, lr: float = 0.15) -> None:
        prediction = self.score(features)
        target = 1.0 if accepted else 0.0
        error = target - prediction
        for key, value in features.items():
            self._weights[key] += lr * error * value
        self._bias += lr * error
        logger.debug(
            "Hypothesis scorer update: accepted=%s, error=%.4f, bias=%.4f",
            accepted,
            error,
            self._bias,
        )


def _get_intent_classifier(toolkit: MutableMapping[str, Any]) -> OnlineIntentClassifier:
    classifier = toolkit.get("intent_classifier")
    if not isinstance(classifier, OnlineIntentClassifier):
        classifier = OnlineIntentClassifier()
        toolkit["intent_classifier"] = classifier
    return classifier


def _get_hypothesis_scorer(toolkit: MutableMapping[str, Any]) -> OnlineWeightedScorer:
    scorer = toolkit.get("hypothesis_scorer")
    if not isinstance(scorer, OnlineWeightedScorer):
        scorer = OnlineWeightedScorer()
        toolkit["hypothesis_scorer"] = scorer
    return scorer


# ---------- 1) Décomposition (sous-problèmes) ----------
class DecompositionStrategy(ReasoningStrategy):
    name = "décomposition"

    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        questions: List[str] = []
        txt = (prompt or "").strip()

        classifier = _get_intent_classifier(toolkit)
        feedback_label = context.get("intent_feedback")
        if isinstance(feedback_label, str):
            classifier.update(txt, feedback_label)

        lower = _normalize_text(txt).replace("'", "")
        tokens = _tokenize(txt)

        # Heuristiques robustifiées
        if re.search(r"\b(pourquoi|why)\b", lower):
            questions.append("Quelles sont les causes plausibles ?")
            questions.append("Quelles preuves observez-vous ?")
        if re.search(r"\b(comment|how)\b", lower):
            questions.append("Quelles sont les étapes pour y parvenir ?")
            questions.append("Quels obstacles et hypothèses ?")
        if re.search(r"\best\s+(?:un|une|le|la|l)\b", lower) or (
            "est" in tokens and {"un", "une", "le", "la", "l"}.intersection(tokens)
        ):
            questions.append("Quelles sont les caractéristiques essentielles du concept ?")

        key = list(dict.fromkeys([t for t in tokens if len(t) > 4]))[:5]
        if key:
            questions.append(f"Définir/clarifier: {', '.join(key[:3])}")

        if not questions:
            predicted, confidence, _ = classifier.predict(txt)
            generated = classifier.LABEL_QUESTIONS.get(predicted, [])
            questions.extend(generated)
            if confidence < 0.55:
                questions.append("Quel est l'objectif précis à atteindre ?")

        notes = "Sous-problèmes identifiés" if questions else "Pas de sous-problèmes saillants"
        return self.validate_output(
            {
                "notes": notes,
                "proposals": [],
                "questions": questions,
                "cost": 0.5,
                "time": time.time() - t0,
            }
        )


# ---------- 2) Récupération d'évidence (mémoire/doc) ----------
class EvidenceRetrievalStrategy(ReasoningStrategy):
    name = "récupération"

    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        retrieve = toolkit.get("retrieve_fn")
        supports: List[str] = []

        if retrieve:
            # cherche 4 items (interactions/docs)
            hits = retrieve(prompt, top_k=4)
            for h in hits:
                title = h.get("meta", {}).get("title") or h.get("meta", {}).get("type", "")
                snippet = h.get("text", "")
                if len(snippet) > 220:
                    snippet = snippet[:220] + "…"
                label = f"{title}: {snippet}" if title else snippet
                supports.append(label)

        notes = "Évidence récupérée depuis la mémoire" if supports else "Aucune évidence en mémoire"
        return self.validate_output(
            {
                "notes": notes,
                "proposals": [],  # pas de proposition finale ici, juste du support
                "questions": [],
                "cost": 1.0,
                "time": time.time() - t0,
                "support": supports,
            }
        )


# ---------- 3) Génération / ranking d'hypothèses ----------
class HypothesisRankingStrategy(ReasoningStrategy):
    name = "hypothèses"

    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        support_snippets: List[str] = list(context.get("support", []))
        proposals: List[Dict[str, Any]] = []

        scorer = _get_hypothesis_scorer(toolkit)
        memory: Dict[str, Dict[str, float]] = toolkit.setdefault("hypothesis_memory", {})

        feedback_items = context.get("hypothesis_feedback")
        if isinstance(feedback_items, list):
            for item in feedback_items:
                if not isinstance(item, Mapping):
                    continue
                answer = item.get("answer")
                if not isinstance(answer, str):
                    continue
                features = memory.get(answer)
                if not features:
                    continue
                accepted = bool(item.get("accepted"))
                scorer.update(features, accepted)

        if support_snippets:
            joined = " | ".join(support_snippets[:3])
            ans = f"Synthèse appuyée sur mémoire: {joined}"
            proposals.append({"answer": ans, "support": support_snippets[:3]})

        if not proposals:
            classifier = _get_intent_classifier(toolkit)
            predicted, _, _ = classifier.predict(prompt)
            if predicted == "process":
                fallback = "Sans éléments externes, détaillons un plan en étapes et validons chaque contrainte."
            elif predicted == "cause":
                fallback = "Faute d'évidence, listons les causes plausibles et vérifions les indices."
            else:
                fallback = "Je manque d'évidence directe. Je propose d'éclaircir le but et les contraintes."
            proposals.append({"answer": fallback, "support": []})

        for proposal in proposals:
            features = scorer.featurize(prompt, support_snippets, proposal)
            memory[proposal["answer"]] = features
            confidence = 0.1 + 0.8 * scorer.score(features)
            proposal["confidence"] = max(0.0, min(1.0, confidence))

        return self.validate_output(
            {
                "notes": "Hypothèses construites et pondérées",
                "proposals": proposals,
                "questions": [],
                "cost": 0.8,
                "time": time.time() - t0,
            }
        )


# ---------- 4) Auto-vérification légère ----------
class SelfCheckStrategy(ReasoningStrategy):
    name = "auto-vérification"

    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        proposals: List[Dict[str, Any]] = context.get("proposals", [])
        last_answer = context.get("last_answer", "")
        notes = "Aucune contradiction apparente"

        normalized_last = _normalize_text(last_answer)
        contradictions = [
            ("oui", "non"),
            ("vrai", "faux"),
            ("possible", "impossible"),
            ("peut", "nepeutpas"),
            ("autorise", "interdit"),
        ]
        for p in proposals:
            answer_norm = _normalize_text(p.get("answer") or "")
            for x, y in contradictions:
                if x in answer_norm and y in normalized_last:
                    p["confidence"] *= 0.8
                    notes = "Contradiction détectée avec l'itération précédente (pondération réduite)"
                if y in answer_norm and x in normalized_last:
                    p["confidence"] *= 0.8
                    notes = "Contradiction détectée avec l'itération précédente (pondération réduite)"

        # clamp
        for p in proposals:
            p["confidence"] = max(0.0, min(1.0, float(p["confidence"])))

        return self.validate_output(
            {
                "notes": notes,
                "proposals": proposals,
                "questions": [],
                "cost": 0.2,
                "time": time.time() - t0,
            }
        )


def plan_reasoning_strategy(
    prompt: str,
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Choisit une stratégie de raisonnement adaptée en combinant LLM et heuristiques."""

    context_snapshot = {}
    if isinstance(context, Mapping):
        context_snapshot = {
            key: value
            for key, value in context.items()
            if isinstance(value, (str, int, float, bool))
        }

    llm_plan = try_call_llm_dict(
        "reasoning_strategies",
        input_payload={"prompt": prompt, "context": context_snapshot},
        logger=logger,
    )
    if llm_plan:
        strategy = str(llm_plan.get("strategy") or "").strip()
        steps_raw = llm_plan.get("steps")
        steps: List[Dict[str, Any]] = []
        if isinstance(steps_raw, list):
            for item in steps_raw:
                if not isinstance(item, Mapping):
                    continue
                description = str(item.get("description") or "").strip()
                confidence = item.get("confidence")
                try:
                    numeric_conf = max(0.0, min(1.0, float(confidence)))
                except (TypeError, ValueError):
                    numeric_conf = 0.5
                if description:
                    steps.append({"description": description, "confidence": numeric_conf})
        notes = str(llm_plan.get("notes") or "")
        if strategy or steps:
            return {
                "strategy": strategy or _fallback_plan_key(prompt),
                "steps": steps or _fallback_plan_steps(prompt),
                "notes": notes,
            }

    return {
        "strategy": _fallback_plan_key(prompt),
        "steps": _fallback_plan_steps(prompt),
        "notes": "plan issu des heuristiques locales",
    }


def _fallback_plan_key(prompt: str) -> str:
    normalized = _normalize_text(prompt)
    if any(token in normalized for token in ("pourquoi", "cause", "why")):
        return "analyse_causale"
    if any(token in normalized for token in ("comment", "plan", "how", "étapes", "etapes")):
        return "planification"
    if "verifier" in normalized or "check" in normalized:
        return "auto_verification"
    return "clarification"


def _fallback_plan_steps(prompt: str) -> List[Dict[str, Any]]:
    key = _fallback_plan_key(prompt)
    if key == "analyse_causale":
        return [
            {"description": "Lister les causes plausibles", "confidence": 0.7},
            {"description": "Comparer avec les observations connues", "confidence": 0.6},
        ]
    if key == "planification":
        return [
            {"description": "Décomposer la tâche en étapes", "confidence": 0.65},
            {"description": "Identifier obstacles et ressources", "confidence": 0.55},
        ]
    if key == "auto_verification":
        return [
            {"description": "Comparer avec la réponse précédente", "confidence": 0.6},
            {"description": "Chercher contradictions majeures", "confidence": 0.5},
        ]
    return [
        {"description": "Clarifier l'objectif et les contraintes", "confidence": 0.6},
        {"description": "Identifier les informations manquantes", "confidence": 0.5},
    ]
