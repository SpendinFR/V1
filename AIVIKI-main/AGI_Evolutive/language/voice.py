from __future__ import annotations
import json, os, time, math, re
import logging
from typing import Dict, Any, List, Tuple, Iterable

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


STYLE_KNOBS = (
    "formality",
    "warmth",
    "humor",
    "emoji",
    "directness",
    "analytical",
    "storytelling",
    "conciseness",
)


def _default_style() -> Dict[str, float]:
    return {
        "formality": 0.4,
        "warmth": 0.6,
        "humor": 0.2,
        "emoji": 0.2,
        "directness": 0.6,
        "analytical": 0.6,
        "storytelling": 0.3,
        "conciseness": 0.6,
    }


class OnlineTextClassifier:
    """Petite régression logistique en ligne sur n-grammes."""

    def __init__(
        self,
        labels: Iterable[str],
        state: Dict[str, Any] | None = None,
        learning_rate: float = 0.12,
    ) -> None:
        self.labels = list(labels)
        self.learning_rate = learning_rate
        self.weights: Dict[str, Dict[str, float]] = {}
        self.bias: Dict[str, float] = {}
        if state:
            self.weights = {
                label: dict(features) for label, features in state.get("weights", {}).items()
                if label in self.labels
            }
            self.bias = {
                label: float(value) for label, value in state.get("bias", {}).items() if label in self.labels
            }

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _tokenize(text: str) -> Dict[str, float]:
        tokens = re.findall(r"[\wÀ-ÿ']+", text.lower())
        features: Dict[str, float] = {}
        for token in tokens:
            features[token] = features.get(token, 0.0) + 1.0
        if "!" in text:
            features["__exclamation__"] = text.count("!")
        if "?" in text:
            features["__question__"] = text.count("?")
        emoji_matches = re.findall(r"[\U0001F300-\U0001FAFF]", text)
        if emoji_matches:
            features["__emoji__"] = float(len(emoji_matches))
        features["__bias__"] = 1.0
        return features

    def predict(self, text: str) -> Dict[str, float]:
        if not self.labels:
            return {}
        feats = self._tokenize(text)
        scores: Dict[str, float] = {}
        for label in self.labels:
            weights = self.weights.get(label, {})
            bias = self.bias.get(label, 0.0)
            score = bias
            for token, value in feats.items():
                score += weights.get(token, 0.0) * value
            scores[label] = self._sigmoid(score)
        return scores

    def update_many(self, text: str, targets: Dict[str, float]) -> None:
        if not self.labels:
            return
        feats = self._tokenize(text)
        for label in self.labels:
            target = float(max(0.0, min(1.0, targets.get(label, 0.0))))
            weights = self.weights.setdefault(label, {})
            bias = self.bias.setdefault(label, 0.0)
            score = bias
            for token, value in feats.items():
                score += weights.get(token, 0.0) * value
            prob = self._sigmoid(score)
            error = target - prob
            step = self.learning_rate * error
            self.bias[label] = bias + step
            for token, value in feats.items():
                weights[token] = weights.get(token, 0.0) + step * value

    def serialize(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "bias": self.bias,
        }

    @property
    def is_trained(self) -> bool:
        return any(self.weights.get(label) for label in self.labels)

class VoiceProfile:
    """
    Profil de voix persistant et vraiment évolutif.

    Les curseurs sont pilotés par:
    - la persona initiale
    - les feedbacks textuels (regex + classifieur online)
    - les lectures aimées
    Avec inertie temporelle + audit trail pour éviter les ruptures.
    """

    REGEX_LIBRARY: Dict[str, Dict[str, List[Tuple[re.Pattern, float]]]] = {}

    def __init__(self, self_model, user_model=None, path: str = "data/voice_profile.json"):
        self.self_model = self_model
        self.user_model = user_model
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.config = {
            "momentum_gain": 0.08,
            "max_step": 0.25,
            "inertia_decay": 4 * 3600.0,
            "positive_scale": 0.6,
            "negative_scale": 1.0,
            "classifier_threshold": 0.65,
            "metric_decay": 6 * 3600.0,
            "audit_log_size": 60,
        }
        self.state: Dict[str, Any] = {
            "style": _default_style(),
            "style_inertia": {},
            "metrics": {},
            "audit_log": [],
            "register_blacklist": [],
            "register_whitelist": [],
            "liked_sources": [],
            "last_update": time.time(),
        }
        self._load()
        self._ensure_state_schema()
        self.classifier = OnlineTextClassifier(
            self._all_signal_labels(),
            state=self.state.get("classifier_state"),
        )
        self._init_regex_library()
        self._init_from_persona()

    def _init_regex_library(self) -> None:
        if VoiceProfile.REGEX_LIBRARY:
            return

        def patterns(items: Iterable[Tuple[str, float]]) -> List[Tuple[re.Pattern, float]]:
            compiled: List[Tuple[re.Pattern, float]] = []
            for pattern, confidence in items:
                compiled.append((re.compile(pattern, re.IGNORECASE | re.UNICODE), confidence))
            return compiled

        VoiceProfile.REGEX_LIBRARY = {
            "conciseness": {
                "increase": patterns(
                    [
                        (r"\btrop\s+(long|d[ée]taill[ée]?)\b", 1.0),
                        (r"\b(resume|synth[ée]tise|plus\s+court)\b", 0.9),
                    ]
                ),
                "decrease": patterns(
                    [
                        (r"\b(plus\s+de\s+d[ée]tails?|argumente|d[ée]veloppe)\b", 1.0),
                        (r"\bexplique\s+(davantage|mieux)\b", 0.9),
                    ]
                ),
            },
            "warmth": {
                "increase": patterns(
                    [
                        (r"\b(trop\s+(froid|sec)|plus\s+chaleureux)\b", 1.0),
                        (r"\b(sois|être)\s+plus\s+(gentil|amical)\b", 0.8),
                    ]
                ),
                "decrease": patterns(
                    [
                        (r"\btrop\s+(chaleureux|enthousiaste)\b", 0.8),
                        (r"\b(plus\s+neutre|reste\s+pro)\b", 0.7),
                    ]
                ),
            },
            "formality": {
                "increase": patterns(
                    [
                        (r"\b(trop\s+(familier|casual)|plus\s+pro)\b", 1.0),
                        (r"\b(plus\s+formel|langage\s+pro)\b", 0.9),
                    ]
                ),
                "decrease": patterns(
                    [
                        (r"\b(d[ée]tends\-?toi|moins\s+formel(le)?)\b", 0.9),
                        (r"\b(parle|fais)\s+plus\s+(simple|cool)\b", 0.8),
                    ]
                ),
            },
            "emoji": {
                "increase": patterns(
                    [
                        (r"\bajoute\s+des?\s+emoji?s?\b", 1.0),
                        (r"\bplus\s+d'?emojis?\b", 0.9),
                    ]
                ),
                "decrease": patterns(
                    [
                        (r"\btrop\s+d'?emojis?\b", 1.0),
                        (r"\b(enl[eè]ve|retire)\s+les?\s+emoji?s?\b", 0.9),
                    ]
                ),
            },
            "analytical": {
                "increase": patterns(
                    [
                        (r"\b(plus\s+d'analyse|structure|logique)\b", 0.9),
                        (r"\b(argumente|d[ée]montre)\b", 0.9),
                    ]
                ),
                "decrease": patterns(
                    [
                        (r"\btrop\s+technique\b", 0.9),
                        (r"\bsimplifie\b", 0.8),
                    ]
                ),
            },
            "humor": {
                "increase": patterns(
                    [
                        (r"\b(plus\s+d'humour|sois\s+dr[ôo]le)\b", 0.9),
                        (r"\bune\s+blague\b", 0.7),
                    ]
                ),
                "decrease": patterns(
                    [
                        (r"\btrop\s+de\s+blagues?\b", 0.8),
                        (r"\b(plus\s+s[ée]rieux|reste\s+pro)\b", 0.9),
                    ]
                ),
            },
            "storytelling": {
                "increase": patterns(
                    [
                        (r"\b(raconte|story|exemple concret)\b", 0.8),
                        (r"\b(plus\s+d'histoires|m[ée]taphore)\b", 0.8),
                    ]
                ),
                "decrease": patterns(
                    [
                        (r"\btrop\s+romanc[ée]\b", 0.7),
                        (r"\bva\s+au\s+fait\b", 0.8),
                    ]
                ),
            },
            "directness": {
                "increase": patterns(
                    [
                        (r"\b(plus\s+direct|va\s+droit\s+au\s+but)\b", 0.9),
                        (r"\b(r[ée]ponds|attaque)\s+la\s+question\b", 0.8),
                    ]
                ),
                "decrease": patterns(
                    [
                        (r"\btrop\s+cash\b", 0.8),
                        (r"\b(m[ée]nage|enrobe)\b", 0.7),
                    ]
                ),
            },
        }

    @staticmethod
    def _all_signal_labels() -> List[str]:
        labels: List[str] = []
        for knob in STYLE_KNOBS:
            labels.append(f"{knob}:increase")
            labels.append(f"{knob}:decrease")
        return labels

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _extract_signals(self, text: str) -> List[Tuple[str, str, float, str]]:
        signals: List[Tuple[str, str, float, str]] = []
        for knob, directions in VoiceProfile.REGEX_LIBRARY.items():
            for direction, patterns in directions.items():
                for pattern, confidence in patterns:
                    if pattern.search(text):
                        signals.append((knob, direction, confidence, f"regex:{pattern.pattern}"))
                        break
        coupled: List[Tuple[str, str, float, str]] = []
        for knob, direction, confidence, source in signals:
            if knob == "formality" and direction == "increase":
                coupled.append(("emoji", "decrease", min(0.7, confidence * 0.6), "coupled:formality"))
            if knob == "warmth" and direction == "increase":
                coupled.append(("emoji", "increase", min(0.6, confidence * 0.5), "coupled:warmth"))
            if knob == "conciseness" and direction == "decrease":
                coupled.append(("storytelling", "increase", min(0.6, confidence * 0.5), "coupled:conciseness"))
        signals.extend(coupled)
        return signals

    def _record_metric(self, metric: str, positive: bool, now: float | None = None) -> None:
        now = now or time.time()
        metrics = self.state.setdefault("metrics", {})
        current = metrics.setdefault(metric, {"score": 0.0, "timestamp": now})
        elapsed = max(0.0, now - float(current.get("timestamp", now)))
        decay = math.exp(-elapsed / max(1.0, self.config["metric_decay"]))
        score = float(current.get("score", 0.0)) * decay
        score += 1.0 if positive else -1.0
        metrics[metric] = {"score": score, "timestamp": now}

    def _log_event(self, event: str, payload: Dict[str, Any]) -> None:
        audit = self.state.setdefault("audit_log", [])
        audit.append({"time": time.time(), "event": event, "data": payload})
        if len(audit) > self.config["audit_log_size"]:
            del audit[: len(audit) - self.config["audit_log_size"]]

    def _apply_signal(
        self,
        knob: str,
        impact: float,
        source: str,
        now: float | None = None,
    ) -> None:
        if knob not in STYLE_KNOBS:
            return
        now = now or time.time()
        inertia = self.state.setdefault("style_inertia", {}).setdefault(
            knob, {"momentum": 0.0, "timestamp": now}
        )
        elapsed = max(0.0, now - float(inertia.get("timestamp", now)))
        if elapsed > 0.0:
            inertia["momentum"] = float(inertia.get("momentum", 0.0)) * math.exp(
                -elapsed / max(1.0, self.config["inertia_decay"])
            )
        inertia["momentum"] = float(inertia.get("momentum", 0.0)) + impact
        inertia["timestamp"] = now
        delta = max(
            -self.config["max_step"],
            min(self.config["max_step"], inertia["momentum"] * self.config["momentum_gain"]),
        )
        style = self.state.setdefault("style", _default_style())
        before = float(style.get(knob, _default_style()[knob]))
        style[knob] = max(0.0, min(1.0, before + delta))
        self.state["style"] = style
        self.state.setdefault("style_inertia", {})[knob] = inertia
        self._log_event(
            "style_adjustment",
            {
                "knob": knob,
                "delta": delta,
                "momentum": inertia["momentum"],
                "source": source,
                "before": before,
                "after": style[knob],
            },
        )

    def _analyze_phrases(self, phrases: Iterable[str]) -> List[Tuple[str, str, float, str]]:
        signals: List[Tuple[str, str, float, str]] = []
        for phrase in phrases:
            text = phrase or ""
            lower = text.lower()
            length = len(lower.split())
            if any(ch in text for ch in "😊😀😄😁😂🤣😍❤️🥰👍"):
                signals.append(("emoji", "increase", 0.7, "liked:emoji"))
                signals.append(("warmth", "increase", 0.5, "liked:emoji"))
            if "!" in text and length >= 6:
                signals.append(("warmth", "increase", 0.4, "liked:exclamation"))
            if length >= 18:
                signals.append(("storytelling", "increase", 0.6, "liked:length"))
                signals.append(("conciseness", "decrease", 0.3, "liked:length"))
            if re.search(r"\b(parce que|ainsi|donc|car)\b", lower):
                signals.append(("analytical", "increase", 0.3, "liked:logic"))
            if re.search(r"\bpar exemple\b", lower):
                signals.append(("storytelling", "increase", 0.4, "liked:example"))
        return signals


    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state.update(json.load(f))
        except Exception:
            pass

    def save(self):
        if hasattr(self, "classifier"):
            self.state["classifier_state"] = self.classifier.serialize()
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _ensure_state_schema(self) -> None:
        defaults = _default_style()
        style = self.state.setdefault("style", defaults.copy())
        for knob in STYLE_KNOBS:
            style.setdefault(knob, defaults[knob])
        self.state["style"] = style
        self.state.setdefault("style_inertia", {})
        self.state.setdefault("metrics", {})
        self.state.setdefault("audit_log", [])
        self.state.setdefault("register_blacklist", [])
        self.state.setdefault("register_whitelist", [])
        self.state.setdefault("liked_sources", [])
        self.state.setdefault("last_update", time.time())
        if "persona_bootstrapped" not in self.state:
            persona_bootstrapped = False
            audit = self.state.get("audit_log")
            if isinstance(audit, list):
                for entry in audit:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("event") != "style_adjustment":
                        continue
                    data = entry.get("data")
                    if not isinstance(data, dict):
                        continue
                    source = data.get("source")
                    if isinstance(source, str) and source.startswith("persona:"):
                        persona_bootstrapped = True
                        break
            if not persona_bootstrapped:
                for knob in STYLE_KNOBS:
                    before = float(style.get(knob, defaults[knob]))
                    if abs(before - defaults[knob]) > 1e-3:
                        persona_bootstrapped = True
                        break
            self.state["persona_bootstrapped"] = persona_bootstrapped
        else:
            self.state.setdefault("persona_bootstrapped", False)
        # purge audit log size if legacy file trop gros
        audit = self.state.get("audit_log")
        if isinstance(audit, list) and len(audit) > self.config["audit_log_size"]:
            self.state["audit_log"] = audit[-self.config["audit_log_size"] :]

    def _init_from_persona(self):
        if self.state.get("persona_bootstrapped"):
            return
        persona = self._persona_profile()
        tone = (persona.get("tone") or "").lower()
        raw_vals = persona.get("values", [])
        if isinstance(raw_vals, dict):
            vals = [key.lower() for key, level in raw_vals.items() if level]
        else:
            vals = [v.lower() for v in raw_vals]
        now = time.time()
        if "inquisitive" in tone or "analytical" in tone:
            self._apply_signal("analytical", 1.2, "persona:tone", now)
            self._apply_signal("directness", 0.6, "persona:tone", now)
        if any(word in tone for word in ("friendly", "warm", "helpful")):
            self._apply_signal("warmth", 1.0, "persona:tone", now)
            self._apply_signal("emoji", 0.5, "persona:tone", now)
        if "formal" in tone:
            self._apply_signal("formality", 1.2, "persona:tone", now)
            self._apply_signal("emoji", -0.6, "persona:tone", now)
        if "precision" in vals or "rigor" in vals:
            self._apply_signal("conciseness", 0.8, "persona:values", now)
        self.state["persona_bootstrapped"] = True
        self.save()

    # ---------- Apprentissage ----------
    def update_from_feedback(self, feedback_text: str, positive: bool):
        """Interprète feedback textuel avec inertie + classifieur."""
        now = time.time()
        normalized = self._normalize_text(feedback_text)
        llm_response = try_call_llm_dict(
            "language_voice",
            input_payload={
                "feedback": feedback_text,
                "positive": positive,
                "style": self.state.get("style", _default_style()),
            },
            logger=logger,
        )
        if llm_response and isinstance(llm_response.get("adjustments"), dict):
            adjustments = llm_response["adjustments"]
            for knob, delta in adjustments.items():
                try:
                    self._apply_signal(knob, float(delta), "llm_voice", now)
                except (TypeError, ValueError):
                    logger.debug("Invalid LLM adjustment for knob %s: %r", knob, delta)
            self._log_event(
                "llm_voice_guidance",
                {"feedback": feedback_text, "adjustments": adjustments, "positive": positive},
            )
        if not normalized:
            self._record_metric("feedback", positive, now)
            self.state["last_update"] = now
            self._log_event("feedback_processed", {"text": "", "positive": positive, "signals": {}, "register": []})
            self.save()
            return
        signals = self._extract_signals(normalized)
        register_matches = re.findall(
            r"\b(?:[ée]vite|ne\s+dis\s+pas)\s+(?:les?\s+)?([\wÀ-ÿ\-']+)",
            normalized.lower(),
        )
        blacklist = self.state.setdefault("register_blacklist", [])
        for word in register_matches:
            lower_word = word.lower()
            if lower_word and lower_word not in blacklist:
                blacklist.append(lower_word)
                self._log_event("register_blacklist", {"token": lower_word, "source": "feedback"})

        scale = self.config["positive_scale"] if positive else self.config["negative_scale"]
        classifier_targets: Dict[str, float] = {}
        for knob, direction, confidence, source in signals:
            label = f"{knob}:{direction}"
            classifier_targets[label] = max(classifier_targets.get(label, 0.0), confidence)
            signed = 1.0 if direction == "increase" else -1.0
            self._apply_signal(knob, signed * confidence * scale, source, now)

        if self.classifier.is_trained:
            predictions = self.classifier.predict(normalized)
            for label, score in predictions.items():
                if score >= self.config["classifier_threshold"] and label not in classifier_targets:
                    knob, direction = label.split(":", 1)
                    signed = 1.0 if direction == "increase" else -1.0
                    confidence = (score - self.config["classifier_threshold"]) / (
                        1.0 - self.config["classifier_threshold"]
                    )
                    confidence = max(0.2, min(0.8, confidence))
                    classifier_targets[label] = confidence
                    self._apply_signal(knob, signed * confidence * scale * 0.7, "classifier", now)

        if classifier_targets:
            self.classifier.update_many(normalized, classifier_targets)
            self.state["classifier_state"] = self.classifier.serialize()

        self._record_metric("feedback", positive, now)
        self.state["last_update"] = now
        self._log_event(
            "feedback_processed",
            {
                "text": normalized,
                "positive": positive,
                "signals": classifier_targets,
                "register": register_matches,
            },
        )
        self.save()

    def update_from_liked_source(self, inbox_path: str, phrases: List[str] | None = None):
        if inbox_path not in self.state["liked_sources"]:
            self.state["liked_sources"].append(inbox_path)
        if phrases:
            for p in phrases:
                if p not in self.state["register_whitelist"]:
                    self.state["register_whitelist"].append(p)
            now = time.time()
            derived_signals = self._analyze_phrases(phrases)
            for knob, direction, confidence, source in derived_signals:
                signed = 1.0 if direction == "increase" else -1.0
                self._apply_signal(knob, signed * confidence * self.config["positive_scale"], source, now)
            self._log_event(
                "liked_source",
                {"inbox": inbox_path, "phrases": phrases, "signals": derived_signals},
            )
            self.state["last_update"] = now
        self.save()

    # getter rapide (pour le renderer)
    def style(self) -> Dict[str, float]:
        return dict(self.state.get("style", _default_style()))

    def bump(self, knob: str, delta: float):
        if knob not in STYLE_KNOBS:
            return
        signed_delta = float(delta)
        self._apply_signal(knob, signed_delta / max(0.01, self.config["momentum_gain"]), "manual_bump")
        self.state["last_update"] = time.time()
        self.save()

    def _persona_profile(self) -> Dict[str, Any]:
        persona: Dict[str, Any] = {}
        try:
            base_state = getattr(self.self_model, "state", {}) if self.self_model is not None else {}
            persona = dict((base_state or {}).get("persona", {}) or {})
        except Exception:
            persona = {}
        if self.user_model is not None and hasattr(self.user_model, "describe"):
            try:
                user_state = self.user_model.describe() or {}
                user_persona = user_state.get("persona") or {}
                if user_persona:
                    merged = dict(persona)
                    merged.update(user_persona)
                    persona = merged
            except Exception:
                pass
        return persona
