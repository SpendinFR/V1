"""Intent classification module with heuristic patterns and ML fallback."""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)

LOGGER = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
_PATTERNS_PATH = _MODULE_DIR / "intent_patterns_fr.json"
_MODEL_PATH = _MODULE_DIR / "models" / "intent_classifier_fallback_fr.json"
_FEEDBACK_LOG_PATH = _MODULE_DIR.parents[1] / "data" / "intent_classifier_feedback.log"
_LLM_AUDIT_PATH = _MODULE_DIR.parents[1] / "data" / "intent_classifier_llm.jsonl"
_ML_CONFIDENCE_THRESHOLD = 0.45
_LLM_CONFIDENCE_THRESHOLD = 0.58


def normalize_text(text: str) -> str:
    """Return a lowercase, accent-free representation suitable for heuristics.

    The normalization keeps meaningful punctuation such as question marks and
    emojis while collapsing repeated whitespace and harmonising apostrophes.
    """

    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    harmonized = (
        without_accents.replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    lowered = harmonized.lower()
    cleaned = re.sub(r"[\u2011\u2012\u2013\u2014]", "-", lowered)
    cleaned = re.sub(r"[^\w\s\-\?\!\.,:'@#€$%&/\\\(\)\[\]{}<>\+=\*\u2600-\u27bf\U0001f300-\U0001fadf]", " ", cleaned)
    collapsed = re.sub(r"\s+", " ", cleaned)
    return collapsed.strip()


def classify(text: str) -> str:
    """Classify *text* into THREAT, QUESTION, COMMAND or INFO."""

    normalized = normalize_text(text)
    patterns = _get_patterns()

    llm_result = _classify_with_llm(text, normalized)
    if llm_result is not None:
        return llm_result

    if any(pattern.search(normalized) for pattern in patterns["THREAT"]):
        return "THREAT"
    if any(pattern.search(normalized) for pattern in patterns["QUESTION"]):
        return "QUESTION"
    if any(pattern.search(normalized) for pattern in patterns["COMMAND"]):
        return "COMMAND"

    return _classify_with_fallback(text, normalized)


def log_uncertain_intent(original: str, normalized: str, predicted_label: str, score: float) -> None:
    """Append an uncertain classification to the feedback log for later review."""

    log_entry = {
        "original": original,
        "normalized": normalized,
        "predicted_label": predicted_label,
        "score": score,
    }
    try:
        _FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as log_file:
            json.dump(log_entry, log_file, ensure_ascii=False)
            log_file.write("\n")
    except OSError:  # pragma: no cover - best effort logging
        LOGGER.debug("Unable to persist uncertain intent feedback", exc_info=True)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()


def _log_llm_decision(
    original: str,
    normalized: str,
    label: str,
    confidence: float,
    response: Mapping[str, Any],
) -> None:
    audit_payload = {
        "original": original,
        "normalized": normalized,
        "label": label,
        "confidence": round(confidence, 4),
        "class_probabilities": response.get("class_probabilities"),
        "indices": response.get("indices"),
        "notes": response.get("notes"),
    }
    try:
        _LLM_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LLM_AUDIT_PATH.open("a", encoding="utf-8") as handle:
            json.dump(audit_payload, handle, ensure_ascii=False)
            handle.write("\n")
    except OSError:  # pragma: no cover - best effort logging
        LOGGER.debug("Unable to persist LLM audit payload", exc_info=True)


def _classify_with_llm(original: str, normalized: str) -> Optional[str]:
    if not _llm_enabled():
        return None

    payload = {"utterance": original, "normalized": normalized}
    try:
        response = _llm_manager().call_dict("intent_classification", input_payload=payload)
    except (LLMUnavailableError, LLMIntegrationError):
        LOGGER.debug("LLM intent classification unavailable", exc_info=True)
        return None

    if not isinstance(response, Mapping):
        return None

    intent = response.get("intent")
    if not isinstance(intent, str):
        return None

    intent_label = intent.strip().upper()
    if not intent_label:
        return None

    try:
        confidence_raw = response.get("confidence", response.get("confidence_score", 0.0))
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if confidence < _LLM_CONFIDENCE_THRESHOLD:
        log_uncertain_intent(original, normalized, intent_label, confidence)
        return None

    _log_llm_decision(original, normalized, intent_label, confidence, response)
    return intent_label


@lru_cache()
def _get_patterns() -> Dict[str, List[re.Pattern[str]]]:
    with _PATTERNS_PATH.open(encoding="utf-8") as fh:
        config = json.load(fh)

    return {
        "THREAT": _build_threat_patterns(config["threat"]),
        "QUESTION": _build_question_patterns(config["question"]),
        "COMMAND": _build_command_patterns(config["command"]),
    }


def _build_threat_patterns(config: Dict[str, Iterable[str]]) -> List[re.Pattern[str]]:
    verb_pattern = _stem_pattern(config["verb_stems"])
    prefix_pattern = _phrase_pattern(config["prefixes"])
    future_prefix_pattern = _phrase_pattern(config["future_prefixes"])
    conditional_pattern = _phrase_pattern(config["conditional_prefixes"])
    pronouns = _phrase_pattern(config["pronouns"])
    nouns = _phrase_pattern(config["nouns"])

    patterns = [
        rf"\b{prefix_pattern}\s+{pronouns}\s+{verb_pattern}\b",
        rf"\b{future_prefix_pattern}\s+{pronouns}\s+{verb_pattern}\b",
        rf"\b{conditional_pattern}.*?{pronouns}\s+{verb_pattern}\b",
        rf"\b{pronouns}\s+{verb_pattern}\s+{nouns}\b",
    ]
    return [_compile_regex(pattern) for pattern in patterns]


def _build_question_patterns(config: Dict[str, Iterable[str]]) -> List[re.Pattern[str]]:
    interrogatives = _phrase_pattern(config["interrogatives"])
    auxiliaries = _phrase_pattern(config["auxiliaries"])
    est_variants = _phrase_pattern(config["est_variants"])

    patterns = [
        r"\?",
        rf"\b{interrogatives}\b",
        rf"\b{auxiliaries}\b",
        rf"\b{est_variants}\b",
    ]
    return [_compile_regex(pattern) for pattern in patterns]


def _build_command_patterns(config: Dict[str, Iterable[str]]) -> List[re.Pattern[str]]:
    imperatives = _phrase_pattern(config["imperatives"])
    softeners = _phrase_pattern(config["softeners"])
    structural = _phrase_pattern(config["structural"])

    patterns = [
        rf"^(?:{imperatives})\b",
        rf"\b{softeners}\b",
        rf"\b{structural}\b",
    ]
    return [_compile_regex(pattern) for pattern in patterns]


def _classify_with_fallback(original: str, normalized: str) -> str:
    model = _load_fallback_model()
    if model is None:
        return "INFO"

    prediction = _predict_with_model(model, normalized)
    if prediction is None:
        return "INFO"

    label, score = prediction
    if score >= _ML_CONFIDENCE_THRESHOLD:
        return label

    log_uncertain_intent(original, normalized, label, score)
    return "INFO"


def _predict_with_model(model: "NaiveBayesFallback", normalized: str) -> Optional[Tuple[str, float]]:
    try:
        proba = model.predict_proba(normalized)
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug("Fallback model prediction failed", exc_info=True)
        return None

    max_index = int(max(range(len(proba)), key=proba.__getitem__))
    label = model.classes[max_index]
    score = float(proba[max_index])
    return label, score


@lru_cache()
def _load_fallback_model() -> Optional["NaiveBayesFallback"]:
    if not _MODEL_PATH.exists():
        LOGGER.info("Fallback intent model not found at %s", _MODEL_PATH)
        return None

    try:
        with _MODEL_PATH.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.exception("Unable to load fallback intent model from %s", _MODEL_PATH)
        return None

    return NaiveBayesFallback.from_payload(payload)


def _compile_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def _phrase_pattern(phrases: Iterable[str]) -> str:
    escaped_phrases = []
    for phrase in phrases:
        escaped = re.escape(phrase)
        escaped = escaped.replace("\\ ", "\\s+")
        escaped = escaped.replace("\\'", "'")
        escaped = escaped.replace("\\-", "[- ]?")
        escaped_phrases.append(escaped)
    joined = "|".join(escaped_phrases)
    return f"(?:{joined})"


def _stem_pattern(stems: Iterable[str]) -> str:
    escaped = [re.escape(stem) for stem in stems]
    return f"(?:{'|'.join(escaped)})\\w*"


class NaiveBayesFallback:
    """Minimal Naive Bayes text classifier for fallback predictions."""

    def __init__(
        self,
        classes: Sequence[str],
        vocabulary: Sequence[str],
        log_prior: Dict[str, float],
        token_log_prob: Dict[str, Dict[str, float]],
        unknown_log_prob: Dict[str, float],
        token_pattern: str,
    ) -> None:
        self.classes: List[str] = list(classes)
        self._log_prior = log_prior
        self._token_log_prob = token_log_prob
        self._unknown_log_prob = unknown_log_prob
        self._token_regex = re.compile(token_pattern)
        self._vocabulary = set(vocabulary)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "NaiveBayesFallback":
        return cls(
            classes=payload["classes"],
            vocabulary=payload["vocabulary"],
            log_prior=payload["log_prior"],
            token_log_prob=payload["token_log_prob"],
            unknown_log_prob=payload["unknown_log_prob"],
            token_pattern=payload["token_pattern"],
        )

    def predict_proba(self, normalized: str) -> List[float]:
        tokens = self._token_regex.findall(normalized)
        if not tokens:
            tokens = [""]

        log_scores: List[float] = []
        for label in self.classes:
            score = float(self._log_prior[label])
            token_probs = self._token_log_prob[label]
            fallback = float(self._unknown_log_prob[label])
            for token in tokens:
                score += float(token_probs.get(token, fallback))
            log_scores.append(score)

        max_log = max(log_scores)
        exp_scores = [math.exp(score - max_log) for score in log_scores]
        total = sum(exp_scores)
        if total == 0:
            return [1.0 / len(exp_scores)] * len(exp_scores)
        return [score / total for score in exp_scores]
