from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

import logging


LOGGER = logging.getLogger(__name__)

try:
    from AGI_Evolutive.cognition.meta_cognition import OnlineLinear
except Exception:  # pragma: no cover - fallback for minimal environments
    OnlineLinear = None  # type: ignore[assignment]

try:
    from AGI_Evolutive.learning import ThompsonBandit
except Exception:  # pragma: no cover - fallback if learning module unavailable
    ThompsonBandit = None  # type: ignore[assignment]


TTL_CANDIDATES = (3, 7, 14, 30)


_WORD_PATTERN = re.compile(r"[\wÀ-ÖØ-öø-ÿ'-]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _WORD_PATTERN.findall(text.lower())


def _has_emoji(text: str) -> bool:
    return bool(re.search(r"[\U0001F300-\U0001FAFF]", text))


def _text_features(text: str, base_conf: float) -> Dict[str, float]:
    clean = text.strip()
    tokens = _tokenize(clean)
    token_count = max(1, len(tokens))
    char_count = max(1, len(clean))
    features: Dict[str, float] = {
        "rule_confidence": max(0.0, min(1.0, base_conf)),
        "token_len": min(1.5, len(tokens) / 12.0),
        "char_len": min(1.5, len(clean) / 80.0),
        "question_mark": 1.0 if "?" in clean else 0.0,
        "exclamation_mark": 1.0 if "!" in clean else 0.0,
        "contains_emoji": 1.0 if _has_emoji(clean) else 0.0,
        "numeric_ratio": sum(ch.isdigit() for ch in clean) / char_count,
        "uppercase_ratio": sum(ch.isupper() for ch in clean) / char_count,
        "punct_ratio": sum(1 for ch in clean if ch in ",.;:") / char_count,
    }
    # Normalise token weights
    scale = 1.0 / math.sqrt(token_count)
    for idx, token in enumerate(tokens[:24]):
        features[f"tok::{token}"] = scale
        if idx < len(tokens) - 1:
            bigram = f"{token}|{tokens[idx + 1]}"
            features[f"bi::{bigram}"] = 0.75 * scale
    return features


@dataclass
class Intent:
    label: str
    description: str
    horizon: str = "mid"
    confidence: float = 0.6
    last_seen: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    ttl_days: float = 7.0
    ttl_choice: Optional[str] = None

    def score(self, now: Optional[float] = None) -> float:
        now = now or time.time()
        ttl = max(1.0, float(self.ttl_days or 7.0))
        freshness = max(0.15, 1.0 - max(0.0, now - self.last_seen) / (ttl * 24 * 3600))
        return max(0.0, min(1.0, self.confidence * freshness))


class OnlineIntentClassifier:
    """Simple online multi-label classifier based on per-label logistic models."""

    def __init__(
        self,
        bounds: Tuple[float, float] = (0.0, 1.5),
        lr: float = 0.05,
        l2: float = 0.0005,
        max_grad: float = 0.25,
        warmup: int = 16,
        init_weight: float = 0.05,
    ) -> None:
        self._bounds = bounds
        self._lr = lr
        self._l2 = l2
        self._max_grad = max_grad
        self._warmup = warmup
        self._init_weight = init_weight
        self.models: Dict[str, OnlineLinear] = {}
        self._disabled = OnlineLinear is None

    def _ensure(self, label: str) -> OnlineLinear:
        if label not in self.models:
            if self._disabled:  # pragma: no cover - degraded mode
                raise RuntimeError("OnlineLinear is unavailable")
            self.models[label] = OnlineLinear(
                [],
                bounds=self._bounds,
                lr=self._lr,
                l2=self._l2,
                max_grad=self._max_grad,
                warmup=self._warmup,
                init_weight=self._init_weight,
            )
        return self.models[label]

    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        if self._disabled:
            return "", 0.0
        best_label = ""
        best_score = 0.0
        for label, model in self.models.items():
            score = model.predict(features)
            if score > best_score:
                best_label = label
                best_score = score
        return best_label, best_score

    def update(self, label: str, features: Dict[str, float], target: float) -> None:
        if self._disabled:
            return
        model = self._ensure(label)
        model.update(features, target)

    def to_state(self) -> Dict[str, Any]:
        if self._disabled:
            return {}
        return {label: model.to_state() for label, model in self.models.items()}

    @classmethod
    def from_state(cls, payload: Optional[Dict[str, Any]]) -> "OnlineIntentClassifier":
        instance = cls()
        if instance._disabled:  # pragma: no cover
            return instance
        if isinstance(payload, dict):
            for label, model_state in payload.items():
                instance.models[label] = OnlineLinear.from_state(
                    model_state,
                    bounds=instance._bounds,
                    lr=instance._lr,
                    l2=instance._l2,
                    max_grad=instance._max_grad,
                    warmup=instance._warmup,
                    init_weight=instance._init_weight,
                )
        return instance


class IntentModel:
    """Modélise les objectifs récurrents exprimés par l'utilisateur."""

    INTENT_PATTERNS: Tuple[Tuple[str, float], ...] = (
        (r"(?:mon|ma|mes)\s+objectif(?:s)?(?:\s+(?:principal|prioritaire))?\s+(?:est|reste)\s+(?:de\s+|d')?(.+)", 0.82),
        (r"je\s+(?:veux|souhaite|voudrais|aimerais)\s+(?:que\s+)?([^.?!]+)", 0.72),
        (r"je\s+(?:compte|prévois|projette)\s+([^.?!]+)", 0.7),
        (r"ma\s+priorité\s+(?:actuelle|principale)\s+(?:est|reste)\s+(?:de\s+|d')?(.+)", 0.85),
        (r"sur\s+le\s+long\s+terme[, ]+je\s+(?:veux|voudrais|souhaite)\s+(?:de\s+|d')?(.+)", 0.76),
        (r"c'est\s+(?:un|une|le|la|l')\s*(?:objectif|but|priorité)\s+(?:pour\s+moi\s+)?(?:de\s+|d')?(.+)", 0.78),
    )

    BASIC_RULES: Tuple[Tuple[str, Tuple[str, ...], float], ...] = (
        (
            "greet",
            (r"\b(bonjour|salut|coucou|hello|hi|hey|yo)\b",),
            0.85,
        ),
        (
            "thanks",
            (r"\b(merci|thanks|thx|merciiiii)\b",),
            0.8,
        ),
        (
            "bye",
            (r"\b(au\s*revoir|a\+|ciao|bye|salut\s+!|see\s+ya)\b",),
            0.8,
        ),
        (
            "meta_help",
            (r"^/(?:help|aide)\b", r"\b(aide|help)\b", r"supporte\s*moi"),
            0.75,
        ),
        (
            "ask_info",
            (r"\?$", r"^(pourquoi|comment|quand|quoi|combien|où|ou|quel|quelle|quels|quelles)\b"),
            0.7,
        ),
        (
            "plan",
            (r"\b(planifie|planifier|prévois|organise|agenda|programme|planifions)\b",),
            0.72,
        ),
        (
            "create",
            (r"\b(cr[ée]e|creer|ajoute|ajouter|produis|fabrique|g[ée]n[èe]re)\b",),
            0.7,
        ),
        (
            "send",
            (r"\b(envoie|envoi|envoyer|partage|transmets?|forwarde)\b",),
            0.7,
        ),
        (
            "summarize",
            (r"\b(r[ée]sume|resume|synth[ée]tise|synthese|summary|fait un r[ée]sum[ée])\b",),
            0.7,
        ),
        (
            "classify",
            (r"\b(classe|cat[ée]gorise|tague|[ée]tiquette|range)\b",),
            0.68,
        ),
        (
            "request",
            (r"\b(peux[- ]?tu|pourrais[- ]?tu|fais|fait|g[ée]n[èe]re|montre|montrez|donne|donnez)\b",),
            0.7,
        ),
        (
            "set_goal",
            (
                r"\b(mon|ma|mes)?\s*(objectif|goal|but|priorit[ée])\b",
                r"est\s+(?:un|une|le|la|l')?\s*(objectif|but)\b",
            ),
            0.76,
        ),
        (
            "feedback",
            (r"\b(bien|parfait|ok|super|top|mauvais|nul|bug|b[ée]mol)\b",),
            0.65,
        ),
        (
            "reflect",
            (r"\b(r[ée]fl[ée]chis|je pense|hypoth[èe]se|intuition|m'interroge)\b",),
            0.65,
        ),
        (
            "inform",
            (r".*",),
            0.55,
        ),
    )

    def __init__(self, path: str = "data/intent_model.json") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._intents: Dict[str, Intent] = {}
        self._history: List[Dict[str, float]] = []
        self._feature_stats: Dict[str, float] = defaultdict(float)
        self._goal_model: Optional[OnlineLinear] = None
        self._fallback_classifier: OnlineIntentClassifier = OnlineIntentClassifier()
        self._ttl_bandits: Dict[str, ThompsonBandit] = {}
        self._load()
        if self._goal_model is None and OnlineLinear is not None:
            self._goal_model = OnlineLinear(
                [],
                bounds=(0.0, 1.2),
                lr=0.045,
                l2=0.0008,
                max_grad=0.2,
                warmup=20,
                init_weight=0.04,
            )

    # ------------------------------------------------------------------
    # Persistence
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return
        models_state = raw.get("models") if isinstance(raw, dict) else {}
        if OnlineLinear is not None and isinstance(models_state, dict):
            self._goal_model = OnlineLinear.from_state(
                models_state.get("goal"),
                bounds=(0.0, 1.2),
                lr=0.045,
                l2=0.0008,
                max_grad=0.2,
                warmup=20,
                init_weight=0.04,
            )
        fallback_state = raw.get("fallback_classifier") if isinstance(raw, dict) else {}
        self._fallback_classifier = OnlineIntentClassifier.from_state(fallback_state)
        feature_stats = raw.get("feature_stats") if isinstance(raw, dict) else {}
        if isinstance(feature_stats, dict):
            for key, value in feature_stats.items():
                try:
                    self._feature_stats[key] = float(value)
                except (TypeError, ValueError):
                    continue
        for item in raw.get("intents", []):
            try:
                intent = Intent(
                    label=item["label"],
                    description=item.get("description", item["label"]),
                    horizon=item.get("horizon", "mid"),
                    confidence=float(item.get("confidence", 0.6)),
                    last_seen=float(item.get("last_seen", time.time())),
                    evidence=list(item.get("evidence", [])),
                    tags=list(item.get("tags", [])),
                    ttl_days=float(item.get("ttl_days", 7.0)),
                    ttl_choice=item.get("ttl_choice"),
                )
                self._intents[intent.label] = intent
            except Exception:
                continue
        self._history = list(raw.get("history", []))[-200:]
        ttl_state = raw.get("ttl_bandits") if isinstance(raw, dict) else {}
        self._load_bandits(ttl_state)

    def save(self) -> None:
        payload = {
            "intents": [
                {
                    "label": intent.label,
                    "description": intent.description,
                    "horizon": intent.horizon,
                    "confidence": intent.confidence,
                    "last_seen": intent.last_seen,
                    "evidence": intent.evidence,
                    "tags": intent.tags,
                    "ttl_days": intent.ttl_days,
                    "ttl_choice": intent.ttl_choice,
                }
                for intent in self._intents.values()
            ],
            "history": self._history[-200:],
            "models": {
                "goal": self._goal_model.to_state() if self._goal_model else {},
            },
            "fallback_classifier": self._fallback_classifier.to_state(),
            "ttl_bandits": self._bandit_state(),
            "feature_stats": dict(self._feature_stats),
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(payload), handle, ensure_ascii=False, indent=2)

    def _load_bandits(self, payload: Any) -> None:
        self._ttl_bandits = {}
        if ThompsonBandit is None or not isinstance(payload, dict):
            return
        for label, params in payload.items():
            if not isinstance(params, dict):
                continue
            bandit = ThompsonBandit()
            bandit.parameters = {}
            for action, values in params.items():
                if isinstance(values, (list, tuple)) and len(values) == 2:
                    try:
                        bandit.parameters[str(action)] = (float(values[0]), float(values[1]))
                    except (TypeError, ValueError):
                        continue
            if bandit.parameters:
                self._ttl_bandits[label] = bandit

    def _bandit_state(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        state: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for label, bandit in self._ttl_bandits.items():
            entries: Dict[str, Tuple[float, float]] = {}
            for action, values in getattr(bandit, "parameters", {}).items():
                try:
                    entries[str(action)] = (float(values[0]), float(values[1]))
                except (TypeError, ValueError, IndexError):
                    continue
            if entries:
                state[label] = entries
        return state

    # ------------------------------------------------------------------
    # Predictions & heuristics
    @classmethod
    def rule_predict(cls, text: str) -> Tuple[str, float]:
        low = (text or "").strip().lower()
        for intent, patterns, base_conf in cls.BASIC_RULES:
            hits = 0
            for pattern in patterns:
                if re.search(pattern, low):
                    hits += 1
            if hits:
                conf = min(0.95, base_conf + 0.05 * (hits - 1))
                return intent, conf
        return "inform", 0.4

    def predict(self, text: str) -> Tuple[str, float]:
        """Retourne l'intention la plus probable pour un message utilisateur."""

        if not text:
            return "inform", 0.0

        base_intent, base_conf = self.rule_predict(text)
        features = _text_features(text, base_conf)
        goal_prob = self._goal_model.predict(features) if self._goal_model else 0.0
        fallback_label, fallback_conf = self._fallback_classifier.predict(features)

        try:
            findings = self.observe_user_message(text, source="nlu_predict")
        except Exception:
            findings = []

        if findings:
            top = max(findings, key=lambda it: it.confidence)
            goal_conf = max(base_conf, min(0.99, float(top.confidence)))
            self._log_prediction(
                text,
                {
                    "intent": "set_goal",
                    "confidence": goal_conf,
                    "rule_intent": base_intent,
                    "rule_conf": base_conf,
                    "goal_prob": goal_prob,
                    "fallback": fallback_label,
                    "fallback_conf": fallback_conf,
                },
            )
            return "set_goal", goal_conf

        candidates: Dict[str, float] = {base_intent: base_conf}
        if goal_prob >= 0.55:
            candidates["set_goal"] = max(goal_prob, candidates.get("set_goal", 0.0))
        if fallback_label:
            candidates[fallback_label] = max(fallback_conf, candidates.get(fallback_label, 0.0))

        final_intent, final_conf = max(candidates.items(), key=lambda item: item[1])
        final_conf = max(0.0, min(0.99, final_conf))

        if final_intent == "set_goal" and self._goal_model is not None:
            self._train_goal_model(features, max(goal_prob, final_conf))
        elif self._goal_model is not None and base_intent != "set_goal" and base_conf >= 0.65:
            self._train_goal_model(features, 0.0)

        self._update_fallback_learning(final_intent, features, final_conf)

        self._log_prediction(
            text,
            {
                "intent": final_intent,
                "confidence": final_conf,
                "rule_intent": base_intent,
                "rule_conf": base_conf,
                "goal_prob": goal_prob,
                "fallback": fallback_label,
                "fallback_conf": fallback_conf,
            },
        )

        return final_intent, final_conf

    # ------------------------------------------------------------------
    # Update cycle
    def observe_user_message(self, text: str, *, source: str = "dialogue") -> List[Intent]:
        """Analyse un message pour détecter des intentions explicites."""

        findings: List[Intent] = []
        if not text:
            return findings
        low = text.lower()
        features_cache: Optional[Dict[str, float]] = None
        for pattern, prior in self.INTENT_PATTERNS:
            match = re.search(pattern, low)
            if not match:
                continue
            description = match.group(1).strip().rstrip(".,")
            label = self._normalize_label(description)
            intent = self._intents.get(label) or Intent(label=label, description=description)
            intent.confidence = max(intent.confidence, prior)
            intent.last_seen = time.time()
            if source not in intent.tags:
                intent.tags.append(source)
            snippet = text.strip()
            if snippet and snippet not in intent.evidence:
                intent.evidence.append(snippet)
                intent.evidence = intent.evidence[-5:]
            if features_cache is None:
                features_cache = _text_features(text, prior)
            self._assign_ttl(intent)
            self._train_goal_model(features_cache, 0.95)
            self._update_fallback_learning("set_goal", features_cache, max(0.75, prior))
            self._reward_ttl(intent, 0.9)
            self._feature_stats["goal_detected"] += 1.0
            self._intents[label] = intent
            findings.append(intent)
        if findings:
            self.save()
        return findings

    def decay(self, factor: float = 0.97) -> None:
        now = time.time()
        for intent in self._intents.values():
            ttl = self._assign_ttl(intent)
            elapsed = max(0.0, now - intent.last_seen)
            elapsed_days = elapsed / (24 * 3600)
            freshness = max(0.0, 1.0 - elapsed_days / ttl)
            intent.confidence *= factor
            if elapsed_days > ttl * 1.5:
                intent.confidence *= 0.7
            intent.confidence = max(0.1, min(1.0, intent.confidence))
            self._reward_ttl(intent, 0.2 + 0.6 * freshness)
        self.save()

    def reinforce(self, label: str, delta: float = 0.1) -> None:
        intent = self._intents.get(label)
        if not intent:
            return
        intent.confidence = max(0.0, min(1.0, intent.confidence + delta))
        intent.last_seen = time.time()
        self._reward_ttl(intent, 0.8)
        self.save()

    # ------------------------------------------------------------------
    # Queries
    def as_constraints(self, top_k: int = 3) -> List[Dict[str, str]]:
        now = time.time()
        ranked = sorted(self._intents.values(), key=lambda it: it.score(now), reverse=True)
        constraints = []
        for intent in ranked[:top_k]:
            constraints.append(
                {
                    "label": intent.label,
                    "description": intent.description,
                    "horizon": intent.horizon,
                    "confidence": f"{intent.confidence:.2f}",
                }
            )
        return constraints

    def pending_clarifications(self, threshold: float = 0.55) -> List[Tuple[str, float, Dict[str, str]]]:
        """Retourne les questions prioritaires pour confirmer les intentions."""

        now = time.time()
        prompts: List[Tuple[str, float, Dict[str, str]]] = []
        for intent in self._intents.values():
            score = intent.score(now)
            if score >= threshold:
                continue
            question = f"Est-ce que {intent.description} reste une priorité importante ?"
            prompts.append((question, max(0.4, 1.0 - score), {"id": intent.label, "type": "intent_confirmation"}))
        return prompts

    def describe(self) -> Dict[str, Dict[str, float]]:
        now = time.time()
        return {intent.label: {"score": intent.score(now)} for intent in self._intents.values()}

    def _top_intents(self, limit: int = 3) -> List[Intent]:
        now = time.time()
        return sorted(self._intents.values(), key=lambda it: it.score(now), reverse=True)[:limit]

    def _fallback_summary(self, recent_message: Optional[str]) -> Dict[str, Any]:
        ranked = self._top_intents(limit=3)
        if not ranked:
            return {
                "intent": "inform",
                "horizon": "immédiat",
                "justification": "Aucune intention saillante détectée, utilisation du mode heuristique.",
                "notes": "fallback",
                "source": "fallback",
                "candidates": [],
            }

        best = ranked[0]
        justification = best.evidence[-1] if best.evidence else best.description
        candidates = [
            {
                "label": intent.label,
                "horizon": intent.horizon,
                "confidence": round(intent.score(time.time()), 3),
            }
            for intent in ranked
        ]
        payload = {
            "intent": best.label,
            "horizon": best.horizon,
            "justification": justification,
            "notes": "",
            "source": "fallback",
            "candidates": candidates,
        }
        if recent_message:
            payload["recent_message"] = recent_message
        return payload

    def llm_summary(self, recent_message: Optional[str] = None) -> Dict[str, Any]:
        fallback = self._fallback_summary(recent_message)
        top_intents_payload = [
            {
                "label": intent.label,
                "description": intent.description,
                "horizon": intent.horizon,
                "confidence": round(intent.score(time.time()), 3),
                "evidence": list(intent.evidence),
            }
            for intent in self._top_intents(limit=5)
        ]
        payload = {
            "recent_message": recent_message,
            "top_intents": top_intents_payload,
            "history": self._history[-8:],
        }
        try:
            response = try_call_llm_dict(
                "models_intent",
                input_payload=payload,
                logger=LOGGER,
            )
        except Exception:
            LOGGER.debug("LLM intent summary failed", exc_info=True)
            response = None

        result = dict(fallback)
        if response:
            if isinstance(response.get("intent"), str):
                result["intent"] = response["intent"]
            if isinstance(response.get("horizon"), str):
                result["horizon"] = response["horizon"]
            if isinstance(response.get("justification"), str):
                result["justification"] = response["justification"]
            if isinstance(response.get("notes"), str):
                result["notes"] = response["notes"]
            candidates = response.get("candidates")
            if isinstance(candidates, list):
                result["candidates"] = candidates
            result["source"] = "llm"
            result["llm_payload"] = response
        return result

    # ------------------------------------------------------------------
    # Helpers
    def _assign_ttl(self, intent: Intent) -> float:
        if ThompsonBandit is None:
            intent.ttl_days = intent.ttl_days or 7.0
            return intent.ttl_days
        bandit = self._ttl_bandits.get(intent.label)
        if bandit is None:
            bandit = ThompsonBandit(prior_alpha=1.4, prior_beta=1.4)
            self._ttl_bandits[intent.label] = bandit
        actions = [str(days) for days in TTL_CANDIDATES]
        priors: Optional[Dict[str, float]] = None
        if intent.ttl_choice:
            priors = {intent.ttl_choice: 0.8}
        choice = bandit.select(actions, priors=priors, fallback=lambda: str(TTL_CANDIDATES[1]))
        try:
            intent.ttl_days = float(choice)
        except (TypeError, ValueError):
            intent.ttl_days = intent.ttl_days or float(TTL_CANDIDATES[1])
        intent.ttl_choice = choice
        self._feature_stats["ttl_assignments"] += 1.0
        return intent.ttl_days

    def _reward_ttl(self, intent: Intent, reward: float) -> None:
        if ThompsonBandit is None or not intent.ttl_choice:
            return
        bandit = self._ttl_bandits.get(intent.label)
        if not bandit:
            return
        bandit.update(intent.ttl_choice, max(0.0, min(1.0, reward)))
        self._feature_stats["ttl_updates"] += 1.0

    def _train_goal_model(self, features: Dict[str, float], target: float) -> None:
        if self._goal_model is None:
            return
        self._goal_model.update(features, max(0.0, min(1.0, target)))
        self._feature_stats["goal_updates"] += 1.0

    def _update_fallback_learning(self, label: str, features: Dict[str, float], confidence: float) -> None:
        if not label or getattr(self._fallback_classifier, "_disabled", False):
            return
        conf = max(0.0, min(1.0, confidence))
        self._fallback_classifier.update(label, features, conf)
        if label != "inform":
            self._fallback_classifier.update("inform", features, 1.0 - conf)
        self._feature_stats[f"clf_{label}"] += 1.0

    def _log_prediction(self, text: str, payload: Dict[str, Any]) -> None:
        entry: Dict[str, Any] = {"ts": time.time(), "text": text[:160]}
        for key, value in payload.items():
            if isinstance(value, (str, float, int)) or value is None:
                entry[key] = value
            else:
                entry[key] = str(value)
        self._history.append(entry)
        self._history = self._history[-200:]
        self._feature_stats["logged_predictions"] += 1.0

    @staticmethod
    def _normalize_label(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug or f"intent-{int(time.time())}"
